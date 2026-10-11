"""Fifteen minutes on a topic — goal-shaped listening, Discover arc Step 3 (ADR-163; scan #6).

A reel across episodes (operator 2026-10-11, "cross-episode reel"): the strongest clips ABOUT one
topic, from different episodes, back to back, within a time budget. It reuses the Moments picker
(``app_moments.score_candidates``: the same candidates, clip windows and scores as an episode's
Moments) and keeps only the candidates whose insight is ABOUT the topic.

A clip must say what its insight says: its opening quote shares at least ``MIN_SUPPORT`` of its
content words with the insight (``app_reasons.quote_support``). An insight's quote list can open
on a greeting or an ad read, and a reel stitched from other episodes has no context to absorb one.

Order: breadth first — every episode's best clip on the topic before any episode's second —
episodes by that best clip's score; clips are added while they fit the budget. Each clip carries
its episode, so the player loads that episode before seeking. Built from what is on disk: no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from podcast_scraper.server.app_moments import Moment, MomentsConfig, rank_moments, score_candidates
from podcast_scraper.server.app_reasons import MIN_SUPPORT, quote_support

#: Clips taken from one episode at most, so the reel is many voices rather than one long episode.
MAX_PER_EPISODE = 2
#: Budgets offered, in minutes.
MINUTES_CHOICES = (5, 15, 30)


@dataclass(frozen=True)
class ReelClip:
    """One clip of the reel: a moment, and the episode it is in."""

    episode_slug: str
    moment: Moment


def _topic_episode_rows(root: Path, topic_id: str) -> list:
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.app_discover_view import interest_episode_index

    paths = interest_episode_index(root).get(topic_id, set())
    return [r for r in cached_catalog(root) if r.metadata_relative_path in paths and r.has_gi]


def goal_reel(
    root: Path,
    topic_id: str,
    *,
    minutes: int = 15,
    config: Optional[MomentsConfig] = None,
) -> list[ReelClip]:
    """Clips about ``topic_id`` across episodes, best and breadth first, within ``minutes``."""
    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_episode_search import episode_segments
    from podcast_scraper.server.app_moments import _topic_links
    from podcast_scraper.server.app_slugs import slug_for_row

    cfg = config or MomentsConfig.from_env()
    per_episode: list[tuple[str, list[Moment]]] = []
    for row in _topic_episode_rows(root, topic_id):
        artifact = load_json_artifact(root, row.gi_relative_path)
        if not isinstance(artifact, dict):
            continue
        about = {iid for iid, topics in _topic_links(artifact).items() if topic_id in topics}
        if not about:
            continue
        scored = [
            m
            for m in score_candidates(artifact, cfg, episode_segments(root, row))
            if m.insight_id in about and quote_support(m.quote_text, m.text) >= MIN_SUPPORT
        ]
        if scored:
            ranked = sorted(rank_moments(scored, cfg), key=lambda m: -m.score)
            per_episode.append((slug_for_row(row), ranked[:MAX_PER_EPISODE]))
    # Episodes by their best clip, strongest first; a tie keeps a stable order by slug.
    per_episode.sort(key=lambda e: (-e[1][0].score, e[0]))

    budget_ms = int(minutes * 60 * 1000)
    used = 0
    reel: list[ReelClip] = []
    for depth in range(MAX_PER_EPISODE):
        for slug, moments in per_episode:
            if depth >= len(moments):
                continue
            m = moments[depth]
            length = m.end_ms - m.start_ms
            if used + length > budget_ms:
                continue  # a shorter clip later may still fit
            reel.append(ReelClip(slug, m))
            used += length
    return reel
