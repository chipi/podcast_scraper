"""Fifteen minutes on a topic, theme or storyline — goal-shaped listening (ADR-163; scan #6).

A reel across episodes (operator 2026-10-11, "cross-episode reel"): the strongest clips ABOUT one
subject, from different episodes, back to back, within a time budget. A topic is its own subject; a
theme (``tc:``) or storyline (``thc:``) is its member topics, less any umbrella member (one that
reaches ``UMBRELLA_TOPIC_SHARE`` of the corpus would make the reel about everything). It reuses the
Moments picker (``app_moments.score_candidates``: the same candidates, clip windows and scores as an
episode's Moments) and keeps only the candidates whose insight is ABOUT one of those topics.

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
from typing import Literal, Optional

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


ReelKind = Literal["topic", "theme", "storyline"]


@dataclass(frozen=True)
class ReelSubject:
    """What a reel is about: its kind, its member topics and (theme / storyline) its label."""

    kind: ReelKind
    topics: frozenset[str]
    label: Optional[str] = None


def reel_subject(root: Path, subject_id: str) -> ReelSubject:
    """The topics a reel on ``subject_id`` is about (none for an unknown theme or storyline)."""
    if subject_id.startswith("topic:"):
        return ReelSubject("topic", frozenset({subject_id}))
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.app_discover_view import interest_episode_index
    from podcast_scraper.server.app_user_corpus import UMBRELLA_TOPIC_SHARE

    kind: ReelKind
    if subject_id.startswith("tc:"):
        from podcast_scraper.search.topic_clusters import theme_map_by_topic

        kind, by_topic = "theme", theme_map_by_topic(root)
        id_key, label_key = "cluster_id", "cluster_label"
    elif subject_id.startswith("thc:"):
        from podcast_scraper.search.storylines import storyline_map_by_topic

        kind, by_topic = "storyline", storyline_map_by_topic(root)
        id_key, label_key = "storyline_id", "storyline_label"
    else:
        return ReelSubject("topic", frozenset())
    members = {tid: info for tid, info in by_topic.items() if info.get(id_key) == subject_id}
    index = interest_episode_index(root)
    ceiling = UMBRELLA_TOPIC_SHARE * len(cached_catalog(root))
    topics = frozenset(t for t in members if len(index.get(t, set())) < ceiling)
    label = next((str(i.get(label_key) or "") for i in members.values() if i.get(label_key)), None)
    return ReelSubject(kind, topics, label)


def _episode_rows(root: Path, topics: frozenset[str]) -> list:
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.app_discover_view import interest_episode_index

    index = interest_episode_index(root)
    paths = set().union(*(index.get(t, set()) for t in topics)) if topics else set()
    return [r for r in cached_catalog(root) if r.metadata_relative_path in paths and r.has_gi]


def goal_reel(
    root: Path,
    subject_id: str,
    *,
    minutes: int = 15,
    config: Optional[MomentsConfig] = None,
) -> list[ReelClip]:
    """Clips about ``subject_id`` (a topic, or a theme's or storyline's member topics) across
    episodes, best and breadth first, within ``minutes``."""
    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_episode_search import episode_segments
    from podcast_scraper.server.app_moments import _topic_links
    from podcast_scraper.server.app_slugs import slug_for_row

    cfg = config or MomentsConfig.from_env()
    subject = reel_subject(root, subject_id)
    per_episode: list[tuple[str, list[Moment]]] = []
    for row in _episode_rows(root, subject.topics):
        artifact = load_json_artifact(root, row.gi_relative_path)
        if not isinstance(artifact, dict):
            continue
        about = {
            iid for iid, topics in _topic_links(artifact).items() if subject.topics & set(topics)
        }
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
