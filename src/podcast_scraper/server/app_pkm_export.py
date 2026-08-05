"""Graph-aware Obsidian export (RFC-113, #1472).

Emits the user's personal corpus as a **connected** Obsidian vault: each highlight becomes a note
note that **wikilinks** to id-keyed `[[People/…]]` / `[[Topics/…]]` / `[[Episodes/…]]`, so the
vault mirrors the personal KG — not a flat highlight dump (the Snipd-differentiator). Extractive, no
LLM (D6); bridge-only (transcript quotes + deep-links, never audio).

v1 is a **full export** under a `closelistening/` namespace we own — the client replaces that folder
wholesale, which handles deletions without client-side diffing (RFC-113's endorsed v1 path). The
manifest carries the corpus `revision` so a future incremental delta can start from it.

Filenames are **canonical ids** (`person:jane` → `person_jane`), never labels: id-keyed names
survive label renames + entity merges (RFC-072 KL2 is future — labels move). Labels live in
frontmatter `aliases:` and in each link's display text (`[[People/person_jane|Jane Doe]]`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_scraper.server import app_corpus_revision, app_user_state
from podcast_scraper.server.app_slugs import resolve_slug

_ROOT = "closelistening"


def _safe_id(entity_id: str) -> str:
    """Canonical id → filename stem: ``person:jane-doe`` → ``person_jane-doe`` (no seps)."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in entity_id)


def _fm_list(items: list[str]) -> str:
    """A YAML flow-list for frontmatter, e.g. ``[a, b]`` (items are ids/short strings)."""
    return "[" + ", ".join(items) + "]"


def _entity_dir(kind: str) -> str:
    return {"person": "People", "topic": "Topics"}.get(kind, "Entities")


def _entity_link(ref: dict[str, Any]) -> str:
    """A wikilink to an entity note: ``[[closelistening/People/person_x|Label]]``."""
    stem = _safe_id(str(ref["id"]))
    return f"[[{_ROOT}/{_entity_dir(str(ref['kind']))}/{stem}|{ref.get('label', ref['id'])}]]"


def _highlight_note(h: dict[str, Any], episode_title: str) -> str:
    refs = [r for r in (h.get("graph_refs") or []) if isinstance(r, dict) and r.get("id")]
    slug = str(h.get("episode_slug") or "")
    t_ms = h.get("start_ms")
    quote = str(h.get("quote_text") or "").strip()
    ent_ids = [_safe_id(str(r["id"])) for r in refs]
    alias = f'"{quote[:80]}"' if quote else f'"{episode_title}"'
    lines = [
        "---",
        f"id: {h.get('id')}",
        f"episode: {slug}",
    ]
    if isinstance(t_ms, int):
        lines.append(f"t_ms: {t_ms}")
    lines.append(f"entities: {_fm_list(ent_ids)}")
    lines.append(f"source: {h.get('source') or 'user'}")
    lines.append(f"aliases: [{alias}]")
    lines.append("---")
    if quote:
        lines.append(f"> {quote}")
    ep_link = f"[[{_ROOT}/Episodes/{slug}|{episode_title}]]"
    deep = f"/player/{slug}" + (f"?t={t_ms // 1000}" if isinstance(t_ms, int) else "")
    lines.append(f"— {ep_link} · [▶ jump]({deep})")
    if refs:
        chips = " · ".join(_entity_link(r) for r in refs)
        lines.append(f"Discusses {chips}")
    return "\n".join(lines) + "\n"


def _entity_note(ref: dict[str, Any]) -> str:
    return (
        "---\n"
        f"id: {ref['id']}\n"
        f"kind: {ref['kind']}\n"
        f"aliases: [\"{ref.get('label', ref['id'])}\"]\n"
        "---\n"
        f"# {ref.get('label', ref['id'])}\n"
        f"A {ref['kind']} in your closelistening corpus.\n"
    )


def _episode_note(slug: str, title: str) -> str:
    return (
        "---\n"
        f"slug: {slug}\n"
        f'aliases: ["{title}"]\n'
        "---\n"
        f"# {title}\n"
        f"[Open in player](/player/{slug})\n"
    )


def build_obsidian_bundle(root: Path, data_dir: Path, user_id: str) -> dict[str, Any]:
    """Build the full Obsidian vault: ``{files: {path: content}, manifest: {...}}``.

    Files live under ``closelistening/``; the client replaces that folder wholesale (deletions
    handled by replacement). Idempotent — id-keyed filenames overwrite in place on re-export.
    """
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
        files[f"{_ROOT}/Highlights/{h.get('id')}.md"] = _highlight_note(h, episode_titles[slug])
        for ref in h.get("graph_refs") or []:
            if isinstance(ref, dict) and ref.get("id"):
                entity_refs[str(ref["id"])] = ref

    for ref in entity_refs.values():
        path = f"{_ROOT}/{_entity_dir(str(ref['kind']))}/{_safe_id(str(ref['id']))}.md"
        files[path] = _entity_note(ref)
    for slug, title in episode_titles.items():
        files[f"{_ROOT}/Episodes/{slug}.md"] = _episode_note(slug, title)

    manifest = {
        "format": "obsidian",
        "revision": app_corpus_revision.current(root, data_dir, user_id),
        "namespace": _ROOT,
        "written": sorted(files.keys()),
        "removed": [],  # full export: the client replaces the whole namespace folder
    }
    return {"files": files, "manifest": manifest}
