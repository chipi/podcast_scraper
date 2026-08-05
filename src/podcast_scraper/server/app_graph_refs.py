"""Resolve a highlight to canonical KG graph refs (#1419, PRD-046 FR6 / RFC-111 §3).

The "carry the graph" substrate: a highlight resolves to canonical person/topic references
(mirroring the shipped ``AppEntityRef`` — ``{id: "person:… | topic:…", kind, label}``) so every
outbound surface (recall, digest, share card, the next-arc export) distributes graph nodes, not
flat clips. One resolver, shared by capture (persist-at-save) and the digest assembler.

Granularity (honest, per RFC-111 §3): resolution is **episode-level** — the highlight's episode KG
entities/topics. Char-offset-precise span→entity lift (RFC-072 KL1) is future; this uses what the
shipped per-episode bridge provides and returns ``[]`` cleanly when an episode has no KG.
"""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_slugs import resolve_slug

#: A small cap keeps digests/cards legible and the persisted highlight compact.
DEFAULT_LIMIT = 3


def refs_for_slug(root: Path, slug: str, *, limit: int = DEFAULT_LIMIT) -> list[dict[str, str]]:
    """Canonical person/topic refs for an episode slug (persons first, then topics), capped."""
    if not slug:
        return []
    row = resolve_slug(root, slug)
    if row is None or not row.has_kg:
        return []
    persons, _orgs, topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
    refs: list[dict[str, str]] = [{"id": p.id, "kind": "person", "label": p.name} for p in persons]
    refs += [{"id": t.id, "kind": "topic", "label": t.label} for t in topics]
    return refs[:limit]


def refs_for_highlight(root: Path, highlight: dict) -> list[dict[str, str]]:
    """Graph refs for a highlight — its stored refs if present, else episode-level resolution."""
    stored = highlight.get("graph_refs")
    if isinstance(stored, list) and stored:
        return [r for r in stored if isinstance(r, dict) and r.get("id")]
    return refs_for_slug(root, str(highlight.get("episode_slug") or ""))
