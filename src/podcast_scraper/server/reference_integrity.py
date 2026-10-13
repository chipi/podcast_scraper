"""References a listener stored follow the corpus when it changes (#2345).

A listener's state stores ids: follows and "Less like this" store theme (``tc:``), storyline
(``thc:``), topic and person ids; favorites, board items and notes store those and episode
slugs; Learn paths and the last short listen store a subject id. When the corpus changes — a
re-clustering renames a storyline, a migration merges two spellings of a person — the stored id
points at nothing, and nothing ever checked.

This pass, run by the board-summarizer sidecar before its summaries (no LLM) and once by
``scripts/ops/rekey_user_references.py`` for existing state:

- **re-keys** an id that has a successor: a theme or storyline in its artifact's ``id_history``
  (``search/stable_cluster_ids.py``), a topic or person spelling the corpus maps to its canonical
  (``kg.entity_clusters``), silently;
- **flags** an id that is truly gone (``reference_flags``: the player shows it muted, "No longer
  available", with a remove action) and never deletes it;
- leaves episode references unchanged and only flags them, and never touches append-only logs.

Every write goes through the store's own lock, so it is safe beside the live app. Each change is
returned as a receipt ``{user, store, before, after}`` for the one-time backfill's undo.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

#: Subject kinds a reference can be re-keyed for, and the id prefix each carries.
SUBJECT_KINDS = frozenset({"topic", "person", "theme", "storyline"})

#: The user-state file holding the flags the player reads.
FLAGS_STORE = "reference_flags"


@dataclass
class Resolver:
    """What every stored id stands for now, built once per pass from the corpus."""

    themes: set[str] = field(default_factory=set)
    storylines: set[str] = field(default_factory=set)
    successors: dict[str, str] = field(default_factory=dict)
    gone_clusters: set[str] = field(default_factory=set)
    topics: set[str] = field(default_factory=set)
    persons: set[str] = field(default_factory=set)
    canonical: Mapping[str, str] = field(default_factory=dict)
    episodes: set[str] | None = None

    def resolve(self, ref: str) -> tuple[str, str | None]:
        """``("ok", ref)``, ``("rekey", new)``, ``("gone", None)``, or ``("unknown", ref)`` for an
        id this pass does not judge (a feed, a search, a link)."""
        if ref.startswith(("tc:", "thc:")):
            current = self.themes if ref.startswith("tc:") else self.storylines
            if ref in current:
                return "ok", ref
            hop, seen = ref, set()
            while hop in self.successors and hop not in seen:
                seen.add(hop)
                hop = self.successors[hop]
            if hop != ref and hop in current:
                return "rekey", hop
            return "gone", None
        if ref.startswith(("topic:", "person:")):
            known = self.topics if ref.startswith("topic:") else self.persons
            canon = str(self.canonical.get(ref) or ref)
            if canon in known:
                return ("ok", ref) if canon == ref else ("rekey", canon)
            return ("ok", ref) if ref in known else ("gone", None)
        return "unknown", ref

    def episode_exists(self, slug: str) -> bool:
        return self.episodes is None or slug in self.episodes


def _cluster_ids_and_history(payload: Mapping[str, Any]) -> tuple[set[str], dict[str, Any]]:
    data = payload.get("data", payload)
    if not isinstance(data, Mapping):
        return set(), {}
    ids = {
        str(c.get("graph_compound_parent_id") or c.get("cluster_id") or "")
        for c in data.get("clusters") or []
        if isinstance(c, Mapping)
    }
    history = data.get("id_history")
    return {i for i in ids if i}, dict(history) if isinstance(history, Mapping) else {}


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, Mapping) else {}


def build_resolver(root: Path, extra_history: Mapping[str, Any] | None = None) -> Resolver:
    """The resolver for corpus ``root`` (read-only). ``extra_history`` adds successors and gone
    ids from before ids were kept (the one-time backfill, ``map_prior_ids``)."""
    from podcast_scraper.kg.entity_clusters import cached_entity_id_map
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.app_kg_index import get_kg_index
    from podcast_scraper.server.app_slugs import slug_for_row

    root = Path(root)
    themes, theme_history = _cluster_ids_and_history(
        _load_json(root / "search" / "topic_clusters.json")
    )
    storylines, story_history = _cluster_ids_and_history(
        _load_json(root / "enrichments" / "topic_theme_clusters.json")
    )
    successors = {
        **dict(theme_history.get("successors") or {}),
        **dict(story_history.get("successors") or {}),
    }
    extra = extra_history or {}
    successors = {**dict(extra.get("successors") or {}), **successors}
    gone = (
        set(theme_history.get("gone") or [])
        | set(story_history.get("gone") or [])
        | set(extra.get("gone") or [])
    )
    index = get_kg_index(root)
    try:
        canonical: Mapping[str, str] = cached_entity_id_map(root)
    except Exception as exc:  # noqa: BLE001 — without the map, spellings are judged as they are
        logger.warning("reference_integrity: entity id map unavailable: %s", exc)
        canonical = {}
    return Resolver(
        themes=themes,
        storylines=storylines,
        successors={str(k): str(v) for k, v in successors.items()},
        gone_clusters={str(g) for g in gone},
        topics=set(index.topic_to_eps),
        persons=set(index.person_to_eps),
        canonical=canonical,
        episodes={slug_for_row(row) for row in cached_catalog(root)},
    )


@dataclass
class UserResult:
    """One user's pass: the receipts of what changed and the references now flagged."""

    user_id: str
    receipts: list[dict[str, Any]] = field(default_factory=list)
    flagged: list[dict[str, str]] = field(default_factory=list)


def _subject_kind(ref: str) -> str:
    for prefix, kind in (("tc:", "theme"), ("thc:", "storyline"), ("topic:", "topic")):
        if ref.startswith(prefix):
            return kind
    return "person" if ref.startswith("person:") else ""


def _fix_ref(
    resolver: Resolver, ref: str, kind: str, flag: Callable[[str, str], None]
) -> str | None:
    """The new value for ``ref`` (``None`` when unchanged); flags it when it is gone."""
    if kind == "episode":
        if ref and not resolver.episode_exists(ref):
            flag(ref, "episode")
        return None
    state, new = resolver.resolve(ref)
    if state == "rekey" and new:
        return new
    if state == "gone":
        flag(ref, kind or _subject_kind(ref))
    return None


def repair_user(
    data_dir: Path, user_id: str, resolver: Resolver, *, apply: bool = True
) -> UserResult:
    """Re-key and flag one user's references. With ``apply`` false nothing is written."""
    from podcast_scraper.server import app_collections_store as boards, app_user_state as state

    result = UserResult(user_id)
    flags: dict[str, str] = {}

    def flag(ref: str, kind: str) -> None:
        flags.setdefault(ref, kind)

    def change(store: str, before: Any, after: Any) -> None:
        result.receipts.append({"user": user_id, "store": store, "before": before, "after": after})

    def update_list(store: str, fix_row: Callable[[Any], Any]) -> None:
        with state._user_lock(data_dir, user_id, store):
            rows = state._read(data_dir, user_id, store, [])
            if not isinstance(rows, list):
                return
            out, changed = [], False
            for row in rows:
                new = fix_row(row)
                if new is not None and new != row:
                    change(store, row, new)
                    changed = True
                    out.append(new)
                else:
                    out.append(row)
            if store == "interests":
                out = list(dict.fromkeys(out))
            if changed and apply:
                state._write(data_dir, user_id, store, out)

    def fix_token(token: Any) -> Any:
        if not isinstance(token, str):
            return None
        return _fix_ref(resolver, token, _subject_kind(token), flag)

    def fix_key(key: str, kind_of: Callable[[dict[str, Any]], str]) -> Callable[[Any], Any]:
        def fix(row: Any) -> Any:
            if not isinstance(row, dict) or not isinstance(row.get(key), str):
                return None
            kind = kind_of(row)
            if kind not in SUBJECT_KINDS and kind != "episode":
                return None
            new = _fix_ref(resolver, row[key], kind, flag)
            return {**row, key: new} if new else None

        return fix

    update_list("interests", fix_token)
    update_list(
        "suggest_less", fix_key("token", lambda r: _subject_kind(str(r.get("token") or "")))
    )
    update_list("favorites", fix_key("ref", lambda r: str(r.get("kind") or "")))
    update_list("notes", fix_key("target_id", lambda r: str(r.get("target") or "")))
    subject = fix_key("subject_id", lambda r: _subject_kind(str(r.get("subject_id") or "")))
    update_list("learning_paths", subject)
    update_list("learning_history", subject)
    update_list("highlights", fix_key("episode_slug", lambda r: "episode"))

    with state._user_lock(data_dir, user_id, "reel_progress"):
        progress = state._read(data_dir, user_id, "reel_progress", None)
        new_progress = subject(progress) if isinstance(progress, dict) else None
        if new_progress is not None:
            change("reel_progress", progress, new_progress)
            if apply:
                state._write(data_dir, user_id, "reel_progress", new_progress)

    with boards._lock(data_dir, user_id):
        doc = boards._read(data_dir, user_id)
        raw_items = doc.get("items")
        items: dict[str, Any] = raw_items if isinstance(raw_items, dict) else {}
        changed = False
        fix_item = fix_key("ref", lambda r: str(r.get("kind") or ""))
        for col_id, members in items.items():
            for i, item in enumerate(members if isinstance(members, list) else []):
                new = fix_item(item)
                if new is not None:
                    change(f"collections:{col_id}", item, new)
                    members[i] = new
                    changed = True
        if changed and apply:
            boards._write(data_dir, user_id, doc)

    result.flagged = [{"ref": r, "kind": k} for r, k in sorted(flags.items())]
    if apply:
        with state._user_lock(data_dir, user_id, FLAGS_STORE):
            state._write(
                data_dir,
                user_id,
                FLAGS_STORE,
                {"gone": result.flagged, "checked_at": int(time.time())},
            )
    return result


def repair_all(
    data_dir: Path,
    root: Path,
    *,
    apply: bool = True,
    users: Iterable[str] | None = None,
    extra_history: Mapping[str, Any] | None = None,
) -> list[UserResult]:
    """Every user's pass against corpus ``root``."""
    from podcast_scraper.server.app_user_state import iter_user_ids

    resolver = build_resolver(root, extra_history)
    out = []
    for user_id in users if users is not None else iter_user_ids(data_dir):
        try:
            out.append(repair_user(data_dir, user_id, resolver, apply=apply))
        except Exception:  # noqa: BLE001 — one user's broken file never stops the rest
            logger.exception("reference_integrity: user %s failed", user_id)
    return out


def gone_references(data_dir: Path, user_id: str) -> list[dict[str, str]]:
    """The references last flagged as gone for ``user_id`` (``[]`` before the first pass)."""
    from podcast_scraper.server.app_user_state import _read

    doc = _read(data_dir, user_id, FLAGS_STORE, {})
    gone = doc.get("gone") if isinstance(doc, dict) else None
    return [g for g in gone or [] if isinstance(g, dict) and g.get("ref")]


def undo(data_dir: Path, receipts: Iterable[Mapping[str, Any]]) -> int:
    """Put back what a pass changed, where the value is still what the pass wrote. Returns how
    many it restored."""
    from podcast_scraper.server import app_collections_store as boards, app_user_state as state

    restored = 0
    for rec in receipts:
        user_id, store = str(rec["user"]), str(rec["store"])
        before, after = rec["before"], rec["after"]
        if store.startswith("collections:"):
            col_id = store.split(":", 1)[1]
            with boards._lock(data_dir, user_id):
                doc = boards._read(data_dir, user_id)
                members = (doc.get("items") or {}).get(col_id) or []
                for i, item in enumerate(members):
                    if item == after:
                        members[i] = before
                        restored += 1
                        boards._write(data_dir, user_id, doc)
                        break
            continue
        with state._user_lock(data_dir, user_id, store):
            if store == "reel_progress":
                if state._read(data_dir, user_id, store, None) == after:
                    state._write(data_dir, user_id, store, before)
                    restored += 1
                continue
            rows = state._read(data_dir, user_id, store, [])
            if isinstance(rows, list) and after in rows:
                rows[rows.index(after)] = before
                state._write(data_dir, user_id, store, rows)
                restored += 1
    return restored
