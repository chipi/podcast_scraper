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
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.atomic_write import atomic_write_text

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
    unparseable — and this is a vault the user opens in Obsidian, where a broken note is not an
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


def _entity_dir(kind: str) -> str:
    return {"person": "People", "topic": "Topics"}.get(kind, "Entities")


def _entity_link(ref: dict[str, Any]) -> str:
    """A wikilink to an entity note: ``[[closelistening/People/person_x|Label]]``."""
    stem = _safe_id(str(ref["id"]))
    label = _wikilink_text(str(ref.get("label") or ref["id"]))
    return f"[[{_ROOT}/{_entity_dir(str(ref['kind']))}/{stem}|{label}]]"


def _highlight_note(h: dict[str, Any], episode_title: str) -> str:
    refs = [r for r in (h.get("graph_refs") or []) if isinstance(r, dict) and r.get("id")]
    slug = str(h.get("episode_slug") or "")
    t_ms = h.get("start_ms")
    quote = str(h.get("quote_text") or "").strip()
    ent_ids = [_safe_id(str(r["id"])) for r in refs]
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


def _current_vault(root: Path, data_dir: Path, user_id: str) -> dict[str, str]:
    """The full current vault as ``{path: content}`` — highlight + entity + episode notes."""
    highlights = app_user_state.get_highlights(data_dir, user_id)
    files: dict[str, str] = {}
    entity_refs: dict[str, dict[str, Any]] = {}
    episode_titles: dict[str, str] = {}

    for h in highlights:
        slug = str(h.get("episode_slug") or "")
        if not slug:
            continue
        if slug not in episode_titles:
            row = resolve_slug(root, slug)
            episode_titles[slug] = row.episode_title if row is not None else slug
        hid = _safe_id(str(h.get("id") or ""))
        files[f"{_ROOT}/Highlights/{hid}.md"] = _highlight_note(h, episode_titles[slug])
        for ref in h.get("graph_refs") or []:
            if isinstance(ref, dict) and ref.get("id"):
                entity_refs[str(ref["id"])] = ref

    for ref in entity_refs.values():
        path = f"{_ROOT}/{_entity_dir(str(ref['kind']))}/{_safe_id(str(ref['id']))}.md"
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


def _load_state(data_dir: Path, user_id: str) -> dict[str, Any]:
    path = _state_path(data_dir, user_id)
    if not path.is_file():
        return {"cursor": 0, "snapshot": {}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"cursor": 0, "snapshot": {}}
    if not isinstance(doc, dict):
        return {"cursor": 0, "snapshot": {}}
    doc.setdefault("cursor", 0)
    doc.setdefault("snapshot", {})
    return doc


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def export_bundle(root: Path, data_dir: Path, user_id: str, *, since: int) -> dict[str, Any]:
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
        snapshot: dict[str, str] = dict(state["snapshot"])
        changed = {p: c for p, c in current.items() if current_hashes[p] != snapshot.get(p)}
        removed = sorted(p for p in snapshot if p not in current_hashes)
        content_changed = bool(changed or removed)
        cursor = prev_cursor + 1 if content_changed else prev_cursor
        # Persist the new snapshot so the next call diffs against what we just served.
        atomic_write_text(
            _state_path(data_dir, user_id),
            json.dumps(
                {"cursor": cursor, "snapshot": current_hashes}, ensure_ascii=False, indent=2
            ),
        )

    incremental = since == prev_cursor and since != 0
    files = changed if incremental else current
    return {
        "mode": "incremental" if incremental else "full",
        "revision": cursor,
        "namespace": _ROOT,
        "files": files,
        "removed": removed if incremental else [],
        "replace_namespace": not incremental,
        "written": sorted(files.keys()),
    }
