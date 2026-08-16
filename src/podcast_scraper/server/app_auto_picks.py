"""Auto-highlight seed — editor's-picks for the digest (#1416, PRD-046 FR3 / RFC-101 FR6.1).

The cold-start killer + the structural edge over tap-to-snip apps (Snipd/Podwise): a user who
listened but captured nothing still gets a valuable digest, because we seed it with the top
GI-extracted moment from each heard-but-uncaptured episode. These are **extractive** (a grounded
insight the pipeline already produced — no LLM, D6) and **graph-carrying**, and are marked
``source: "auto"`` so a renderer distinguishes them from the user's own captures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_scraper.server import app_graph_refs
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_gi_view import insights_from_gi
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.app_user_corpus import user_episode_set

# Bound the per-user episode scan so a large heard-set can't make a digest assemble slowly.
_MAX_EPISODES_SCANNED = 50


def _pick_for_slug(root: Path, slug: str) -> dict[str, Any] | None:
    """The top grounded, graph-carrying editor's-pick moment for an episode, or None."""
    row = resolve_slug(root, slug)
    if row is None or not row.has_gi:
        return None
    insights = insights_from_gi(load_json_artifact(root, row.gi_relative_path), limit=1)
    if not insights:
        return None
    ins = insights[0]
    if not ins.grounded or not ins.quotes:
        return None  # need a supporting quote for a jump-to-moment timestamp
    refs = app_graph_refs.refs_for_slug(root, slug)
    if not refs:
        return None  # carry the graph or drop it (moat rule)
    t_ms = ins.quotes[0].start_ms
    return {
        "episode_slug": slug,
        "graph_refs": refs,
        "deep_link": f"/player/{slug}" + (f"?t={t_ms // 1000}" if t_ms is not None else ""),
        "t_ms": t_ms,
        "quote": ins.text,
        "source": "auto",
    }


def auto_pick_items(
    root: Path,
    data_dir: Path,
    user_id: str,
    *,
    exclude_slugs: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Editor's-pick digest items from the user's heard-but-uncaptured episodes (source='auto').

    ``exclude_slugs`` is the set already covered by the user's own captures. Returns at most
    ``limit`` items, newest-heard first is not guaranteed — order follows the episode-set iteration
    (deterministic per corpus). Empty when the user has heard nothing new or nothing carries graph.
    """
    if limit <= 0:
        return []
    heard = user_episode_set(root, data_dir, user_id)
    candidates = sorted(heard - exclude_slugs)[:_MAX_EPISODES_SCANNED]
    items: list[dict[str, Any]] = []
    for slug in candidates:
        item = _pick_for_slug(root, slug)
        if item is not None:
            items.append(item)
        if len(items) >= limit:
            break
    return items
