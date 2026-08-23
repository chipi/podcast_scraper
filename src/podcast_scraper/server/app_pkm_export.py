"""Graph-aware Obsidian export (RFC-113, #1472).

Emits the user's personal corpus as a **connected** Obsidian vault: each highlight becomes a note
that **wikilinks** to id-keyed `[[People/…]]` / `[[Topics/…]]` / `[[Episodes/…]]`, so the vault
mirrors the personal KG — not a flat highlight dump (the Snipd-differentiator). Extractive, no LLM
(D6); bridge-only (transcript quotes + deep-links, never audio).

**Incremental.** The server tracks a per-user vault snapshot (`path → content hash`) + a cursor.
`export_bundle(since)` returns only the **changed** notes + a `removed` tombstone list when `since`
matches the last export, else a **full** export (the fallback). The cursor advances only on real
content change. The client writes `files` and deletes `removed` under `closelistening/`.

Filenames are **canonical ids** (`person:jane` → `person_jane`), never labels: id-keyed names
survive label renames + entity merges (RFC-072 KL2 is future — labels move). Labels live in
frontmatter `aliases:` and in each link's display text (`[[People/person_jane|Jane Doe]]`).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.enrichment.enrichers._loaders import is_unresolved_speaker_placeholder
from podcast_scraper.server import app_graph_refs, app_user_state
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.atomic_write import atomic_write_text
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

_ROOT = "closelistening"


def _safe_id(entity_id: str) -> str:
    """Canonical id → filename stem: ``person:jane-doe`` → ``person_jane-doe`` (no seps).

    Also the traversal guard for any value that becomes a zip-entry path (highlight id, slug): every
    non-``[alnum-_]`` char (``/``, ``.``, ``:`` …) collapses to ``_``, so a crafted id/slug cannot
    escape the ``closelistening/`` namespace on a naive extractor (review M7).
    """
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in entity_id)


def _yaml_scalar(value: str) -> str:
    """A safe double-quoted YAML scalar — escapes ``\\``/``"``, flattens newlines, drops controls.

    Real titles/quotes routinely contain quotes and line breaks, which would otherwise produce
    invalid frontmatter (review M7).

    Control characters go too (#43). YAML forbids raw C0 controls other than tab/newline inside a
    double-quoted scalar, so a single ``\\x07`` pasted into a quote makes that note's frontmatter
    unparsable — and this is a vault the user opens in Obsidian, where a broken note is not an
    error message but a note that quietly does not work. Escaping ``\\`` and ``"`` only defends
    against the characters people type on purpose.
    """
    flat = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")
    flat = "".join(" " if (ord(c) < 0x20 or ord(c) == 0x7F) else c for c in flat)
    return f'"{flat}"'


def _wikilink_text(value: str) -> str:
    """Display text safe to put after ``|`` inside ``[[…|…]]`` (#43).

    ``]]`` ends the link and ``|`` starts a new field, so either one inside a label or an episode
    title truncates the link mid-note and spills the rest as literal text. Podcast titles really do
    contain brackets and pipes. The frontmatter path was already careful (:func:`_yaml_scalar`);
    the link BODY was emitted raw, which is the half nobody looked at.

    Replaced rather than dropped, so the reader still sees roughly the original string instead of
    two words silently fused.
    """
    flat = value.replace("\n", " ").replace("\r", " ")
    flat = "".join(" " if (ord(c) < 0x20 or ord(c) == 0x7F) else c for c in flat)
    return flat.replace("]]", "] ]").replace("[[", "[ [").replace("|", "/").strip()


def _fm_list(items: list[str]) -> str:
    """A YAML flow-list for frontmatter, e.g. ``[a, b]`` (items are ids/short strings)."""
    return "[" + ", ".join(items) + "]"


def _entity_stem(entity_id: str) -> str:
    """Filename stem for an ENTITY note — ``_safe_id`` plus a case fold (#44).

    ``_safe_id`` preserves case, so ``person:Sam`` and ``person:sam`` are distinct zip entries that
    COLLIDE on a case-insensitive filesystem — which is the default on macOS and Windows, i.e. most
    vaults. Worse than a merge: a tombstone for one would delete the file the other is using.

    The KG pipeline lowercases ids at mint, so this is belt-and-braces for pipeline artifacts. It is
    NOT belt-and-braces for `graph_refs` frozen onto a highlight at capture, which the export trusts
    verbatim — enforcing it here makes the property structural rather than inherited.
    """
    return _safe_id(entity_id).lower()


def _entity_dir(kind: str) -> str:
    return {"person": "People", "topic": "Topics"}.get(kind, "Entities")


def _entity_link(ref: dict[str, Any]) -> str:
    """A wikilink to an entity note: ``[[closelistening/People/person_x|Label]]``."""
    stem = _entity_stem(str(ref["id"]))
    label = _wikilink_text(str(ref.get("label") or ref["id"]))
    return f"[[{_ROOT}/{_entity_dir(str(ref['kind']))}/{stem}|{label}]]"


def _usable_refs(refs: Any) -> list[dict[str, Any]]:
    """Graph refs worth putting in a vault — placeholders removed (#1685).

    An unresolved person is an episode-local label, not somebody a reader can look up:
    `person:speaker-{ep}-03` (a diarization voice, #1b) or `person:unresolved-{name}-{ep}` (a
    bare first name with no surname anywhere in the episode, #1685). Every in-app surface already
    drops these via `is_unresolved_speaker_placeholder` — inside `entities_from_kg` itself — but
    the export did not, so a vault grew a `People/person_speaker-...md` note per anonymous voice
    and a wikilink pointing at it.

    Filtering HERE rather than at capture is deliberate: `graph_refs` are frozen onto a highlight
    when it is captured and the export deliberately trusts them, so highlights captured before
    this fix already carry placeholder refs. Filtering at the boundary repairs those too, and an
    exported vault cannot be migrated after the user has downloaded it.
    """
    out: list[dict[str, Any]] = []
    for ref in refs or []:
        if not isinstance(ref, dict) or not ref.get("id"):
            continue
        if is_unresolved_speaker_placeholder(str(ref["id"]), ref.get("label")):
            continue
        out.append(ref)
    return out


def _highlight_note(h: dict[str, Any], episode_title: str) -> str:
    refs = _usable_refs(h.get("graph_refs"))
    slug = str(h.get("episode_slug") or "")
    t_ms = h.get("start_ms")
    quote = str(h.get("quote_text") or "").strip()
    ent_ids = [_entity_stem(str(r["id"])) for r in refs]
    alias = _yaml_scalar(quote[:80] if quote else episode_title)
    lines = [
        "---",
        f"id: {_safe_id(str(h.get('id') or ''))}",
        f"episode: {_yaml_scalar(slug)}",
    ]
    if isinstance(t_ms, int):
        lines.append(f"t_ms: {t_ms}")
    lines.append(f"entities: {_fm_list(ent_ids)}")
    # Quoted like every other string field. No client path can set `source` today (HighlightCreate
    # has no such field, so it is always "user"), which is exactly why it is worth doing now —
    # latent, free, and the sort of thing that stops being latent without anyone revisiting it.
    lines.append(f"source: {_yaml_scalar(str(h.get('source') or 'user'))}")
    lines.append(f"aliases: [{alias}]")
    lines.append("---")
    if quote:
        # Every line prefixed, not just the first. Markdown ends a blockquote at the first
        # unprefixed line, so a captured passage spanning two lines rendered its opening line as a
        # quote and the remainder as body text attributed to nobody.
        lines.append("\n".join(f"> {line}" for line in quote.splitlines() or [quote]))
    ep_link = f"[[{_ROOT}/Episodes/{_safe_id(slug)}|{_wikilink_text(episode_title)}]]"
    deep = f"/player/{slug}" + (f"?t={t_ms // 1000}" if isinstance(t_ms, int) else "")
    lines.append(f"— {ep_link} · [▶ jump]({deep})")
    if refs:
        chips = " · ".join(_entity_link(r) for r in refs)
        lines.append(f"Discusses {chips}")
    return "\n".join(lines) + "\n"


def _entity_note(ref: dict[str, Any]) -> str:
    label = str(ref.get("label", ref["id"]))
    return (
        "---\n"
        f"id: {_yaml_scalar(str(ref['id']))}\n"
        f"kind: {ref['kind']}\n"
        f"aliases: [{_yaml_scalar(label)}]\n"
        "---\n"
        f"# {label}\n"
        f"A {ref['kind']} in your closelistening corpus.\n"
    )


def _episode_note(slug: str, title: str) -> str:
    return (
        "---\n"
        f"slug: {_yaml_scalar(slug)}\n"
        f"aliases: [{_yaml_scalar(title)}]\n"
        "---\n"
        f"# {title}\n"
        f"[Open in player](/player/{slug})\n"
    )


def _title_index(root: Path) -> dict[str, str]:
    """``slug -> episode title``, from ONE catalog walk (#42).

    This replaced a ``resolve_slug`` call per unique highlighted episode. That helper documents
    itself as O(episodes) *per call*, but each call runs ``build_catalog_rows_cumulative``, which
    walks every ``run_*/metadata/*.metadata.json`` and JSON-parses all of them — with no caching
    anywhere in that module. Highlights across 300 episodes of a 1000-episode corpus meant ~300
    full catalog walks, ~300k JSON loads, all while HOLDING the export lock, whose timeout is 5s.
    A concurrent export — the web + native-shell case that lock exists for — then raised
    ``filelock.Timeout`` and 500'd. One walk, one dict.
    """
    return {
        slug_for_row(row): (row.episode_title or "") for row in build_catalog_rows_cumulative(root)
    }


def _with_backfilled_refs(root: Path, highlight: dict[str, Any]) -> dict[str, Any]:
    """The highlight, with graph refs resolved from the episode KG if it stored none (#44)."""
    stored = highlight.get("graph_refs")
    if isinstance(stored, list) and stored:
        return highlight
    resolved = app_graph_refs.refs_for_slug(root, str(highlight.get("episode_slug") or ""))
    if not resolved:
        return highlight
    return {**highlight, "graph_refs": resolved}


def _current_vault(root: Path, data_dir: Path, user_id: str) -> dict[str, str]:
    """The full current vault as ``{path: content}`` — highlight + entity + episode notes."""
    highlights = app_user_state.get_highlights(data_dir, user_id)
    files: dict[str, str] = {}
    entity_refs: dict[str, dict[str, Any]] = {}
    episode_titles: dict[str, str] = {}
    # Built on first need, so a user with no highlights still costs zero catalog walks.
    titles: dict[str, str] | None = None

    for h in highlights:
        slug = str(h.get("episode_slug") or "")
        if not slug:
            continue
        if slug not in episode_titles:
            if titles is None:
                titles = _title_index(root)
            # An unresolvable slug falls back to itself, exactly as resolve_slug -> None did.
            episode_titles[slug] = titles.get(slug) or slug
        # Backfill refs frozen EMPTY at capture (#44). Refs are stored on the highlight when it is
        # captured, so a moment captured while its episode had no KG carried zero entity links —
        # for ever, even once the KG landed. That is the one case where "the vault mirrors your
        # graph" was quietly untrue, and it hits early captures hardest.
        #
        # Only when the stored list is EMPTY: a non-empty list is what the user actually captured,
        # and re-resolving it every export would let a later KG rewrite silently restate what an
        # old highlight was about.
        h = _with_backfilled_refs(root, h)
        hid = _safe_id(str(h.get("id") or ""))
        files[f"{_ROOT}/Highlights/{hid}.md"] = _highlight_note(h, episode_titles[slug])
        for ref in _usable_refs(h.get("graph_refs")):
            if True:
                # Last-write-wins on a repeated id with a different (stale) label. Deterministic:
                # highlights are read in a stable order, so the entity note and an older link's
                # display text can disagree, but they disagree the same way on every export.
                entity_refs[str(ref["id"])] = ref

    for ref in entity_refs.values():
        path = f"{_ROOT}/{_entity_dir(str(ref['kind']))}/{_entity_stem(str(ref['id']))}.md"
        files[path] = _entity_note(ref)
    for slug, title in episode_titles.items():
        files[f"{_ROOT}/Episodes/{_safe_id(slug)}.md"] = _episode_note(slug, title)
    return files


# --- incremental export state: per-user vault snapshot (path → content hash) + a cursor ---

_STATE_FILE = "export_state.json"
_LOCK_TIMEOUT_S = 5.0


def _state_path(data_dir: Path, user_id: str) -> Path:
    return data_dir / "users" / user_id / _STATE_FILE


def _state_lock(data_dir: Path, user_id: str) -> FileLock:
    path = _state_path(data_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_name(f".{_STATE_FILE}.lock")), timeout=_LOCK_TIMEOUT_S)


def _new_epoch() -> str:
    """A fresh vault identity, minted whenever export state is created from nothing (#41)."""
    return uuid.uuid4().hex


def _blank_state() -> dict[str, Any]:
    """State for a vault the server has never exported — or can no longer read.

    The epoch is what gives a cursor meaning. A bare integer identifies a snapshot only while the
    counter it came from still exists, and that counter restarts at 0 whenever
    ``export_state.json`` is lost OR merely unreadable — then climbs back through values a client
    may still be holding. Minting a new identity here is what lets the server notice.
    """
    return {"cursor": 0, "snapshot": {}, "epoch": _new_epoch()}


def _load_state(data_dir: Path, user_id: str) -> dict[str, Any]:
    path = _state_path(data_dir, user_id)
    if not path.is_file():
        return _blank_state()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _blank_state()
    if not isinstance(doc, dict):
        return _blank_state()
    doc.setdefault("cursor", 0)
    doc.setdefault("snapshot", {})
    # State written before epochs existed gets one now. No client can be holding it, so the next
    # request echoes nothing, correctly falls back to a full export — once — and is then in sync.
    if not isinstance(doc.get("epoch"), str) or not doc["epoch"]:
        doc["epoch"] = _new_epoch()
    return doc


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def export_bundle(
    root: Path, data_dir: Path, user_id: str, *, since: int, epoch: str | None = None
) -> dict[str, Any]:
    """Compute the export the client should apply, and advance the server's vault snapshot.

    ``since`` is the revision the client last applied (0 = never). When it matches the server's
    cursor we return an **incremental** delta (only changed files + a ``removed`` tombstone list);
    otherwise (behind / never / a new device) we fall back to a **full** export — the always-valid
    path (RFC-113). The cursor bumps only when the vault content actually changed, so re-exporting a
    static vault (or a second device) doesn't churn it. The client writes ``files`` and deletes
    ``removed`` under the ``closelistening/`` namespace. A **full** export sets
    ``replace_namespace: true`` — the client must first delete everything under ``closelistening/``
    then write ``files``, so notes that vanished while the client was behind don't linger (review
    M8: a full export can't enumerate a fallen-behind client's stale notes, so replace wholesale).
    """
    with _state_lock(data_dir, user_id):
        # Compute the vault INSIDE the lock so two concurrent exports (web + native shell) can't
        # persist snapshots out of order and desync the cursor/diff (review M5).
        current = _current_vault(root, data_dir, user_id)
        current_hashes = {p: _hash(c) for p, c in current.items()}
        state = _load_state(data_dir, user_id)
        prev_cursor = int(state["cursor"])
        state_epoch = str(state["epoch"])
        snapshot: dict[str, str] = dict(state["snapshot"])
        changed = {p: c for p, c in current.items() if current_hashes[p] != snapshot.get(p)}
        removed = sorted(p for p in snapshot if p not in current_hashes)
        content_changed = bool(changed or removed)
        cursor = prev_cursor + 1 if content_changed else prev_cursor
        # Persist the new snapshot so the next call diffs against what we just served.
        atomic_write_text(
            _state_path(data_dir, user_id),
            json.dumps(
                {"cursor": cursor, "snapshot": current_hashes, "epoch": state_epoch},
                ensure_ascii=False,
                indent=2,
            ),
        )

    # The epoch must match too (#41). `since == prev_cursor` alone treats a bare integer as
    # identifying a snapshot, but the counter restarts at 0 whenever export_state.json is lost or
    # becomes unreadable, then climbs back through values a client may still hold. On collision the
    # server served an "incremental" delta computed against ITS OWN snapshot rather than against
    # what the client actually had: the client applied a nonsense delta, kept its orphans, and —
    # because the cursors then advanced in lockstep — never asked for a full export again. Silent,
    # permanent, no self-healing. A client that sends no epoch (older build) simply gets full.
    incremental = since == prev_cursor and since != 0 and epoch == state_epoch
    files = changed if incremental else current
    return {
        "mode": "incremental" if incremental else "full",
        "revision": cursor,
        "epoch": state_epoch,
        "namespace": _ROOT,
        "files": files,
        "removed": removed if incremental else [],
        "replace_namespace": not incremental,
        "written": sorted(files.keys()),
    }
