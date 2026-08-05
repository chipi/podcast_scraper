"""Digest sections beyond revisit — new-in-follows + trending (#1413, PRD-046 FR2.1 / RFC-110 §3).

Extractive (D6, no LLM), graph-carrying builders the personal digest assembler appends after the
revisit section:

- ``new_in_follows``: recent episodes in shows the user follows that they haven't heard yet.
- ``trending_in_your_corpus``: topics in the user's heard∪captured corpus that are heating up
  (temporal_velocity enrichment, RFC-088), each anchored to a representative heard episode.

Both drop any item that can't carry the graph (the schema now enforces ``graph_refs`` non-empty).
These read the shared corpus at assembly time; that's fine in the scheduled digest job (not a
request path). Both short-circuit cheaply when the user has no follows / the corpus has no velocity
envelope, so they add no cost to users the sections don't apply to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_scraper.server import app_graph_refs, app_user_state
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_slugs import resolve_slug, slug_for_row
from podcast_scraper.server.app_user_corpus import user_episode_set
from podcast_scraper.server.corpus_catalog import build_catalog_rows

# Bound the per-user corpus scans so a large catalog / heard-set can't slow a digest assembly.
_MAX_ROWS_SCANNED = 500
_MAX_HEARD_SCANNED = 50

# The same "genuinely heating up" gate the Home trending chips use (feed_signals): velocity ≥ τ AND
# enough total mentions that the ratio isn't noise.
_MIN_VELOCITY = 1.5
_MIN_TOTAL = 3


def new_in_follows_items(
    root: Path, data_dir: Path, user_id: str, *, limit: int
) -> list[dict[str, Any]]:
    """Recent unheard episodes in the user's followed shows (newest-first), graph-carrying."""
    if limit <= 0:
        return []
    feeds = {
        str(f.get("feed_id"))
        for f in app_user_state.get_library(data_dir, user_id)
        if f.get("feed_id")
    }
    if not feeds:
        return []
    heard = user_episode_set(root, data_dir, user_id)
    rows = [r for r in build_catalog_rows(root) if r.feed_id in feeds]
    rows.sort(key=lambda r: r.sort_key())  # newest-first
    items: list[dict[str, Any]] = []
    for row in rows[:_MAX_ROWS_SCANNED]:
        slug = slug_for_row(row)
        if slug in heard:
            continue
        refs = app_graph_refs.refs_for_slug(root, slug)
        if not refs:
            continue  # no graph → drop (moat rule; schema requires non-empty graph_refs)
        items.append(
            {
                "episode_slug": slug,
                "episode_title": row.episode_title,
                "graph_refs": refs,
                "deep_link": f"/player/{slug}",
            }
        )
        if len(items) >= limit:
            break
    return items


def _topic_velocity(root: Path) -> dict[str, tuple[float, int]]:
    """``topic_id → (velocity, total)`` from the temporal_velocity envelope ({} when absent)."""
    data = load_json_artifact(root, "enrichments/temporal_velocity.json") or {}
    out: dict[str, tuple[float, int]] = {}
    for t in data.get("topics") or []:
        if isinstance(t, dict) and t.get("topic_id") is not None:
            v = t.get("velocity_last_over_6mo")
            total = t.get("total")
            if isinstance(v, (int, float)) and isinstance(total, int):
                out[str(t["topic_id"])] = (float(v), total)
    return out


def trending_items(root: Path, data_dir: Path, user_id: str, *, limit: int) -> list[dict[str, Any]]:
    """Heating-up topics in the user's corpus, anchored to a representative heard episode."""
    if limit <= 0:
        return []
    velocity = _topic_velocity(root)
    if not velocity:
        return []
    # topic_id → (label, representative heard slug) — first heard episode discussing it.
    rep: dict[str, tuple[str, str]] = {}
    for slug in sorted(user_episode_set(root, data_dir, user_id))[:_MAX_HEARD_SCANNED]:
        row = resolve_slug(root, slug)
        if row is None or not row.has_kg:
            continue
        _persons, _orgs, topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
        for topic in topics:
            rep.setdefault(topic.id, (topic.label, slug))
    ranked: list[tuple[float, str, str, str]] = []
    for topic_id, (label, slug) in rep.items():
        vel = velocity.get(topic_id)
        if vel and vel[0] >= _MIN_VELOCITY and vel[1] >= _MIN_TOTAL:
            ranked.append((vel[0], topic_id, label, slug))
    ranked.sort(reverse=True)  # hottest first
    items: list[dict[str, Any]] = []
    for _vel, topic_id, label, slug in ranked[:limit]:
        topic_slug = topic_id.split(":", 1)[-1]
        items.append(
            {
                "episode_slug": slug,
                "graph_refs": [{"id": topic_id, "kind": "topic", "label": label}],
                "deep_link": f"/topic/{topic_slug}?scope=mine",
            }
        )
    return items
