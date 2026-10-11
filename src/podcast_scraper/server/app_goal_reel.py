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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

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


@dataclass(frozen=True)
class ClipRef:
    """A scored clip without its text: what ranking, the time budget and a path's speaker cap
    need. Kept for every episode, so it carries no transcript; :func:`hydrate` turns the few that
    are played back into full ``Moment`` objects."""

    insight_id: str
    score: float
    #: Position in the episode's ``rank_moments`` order, the tie-break after the score.
    order: int
    start_ms: int
    end_ms: int
    speaker: Optional[str]
    topics: frozenset[str]


@dataclass(frozen=True)
class SubjectEpisode:
    """One episode of a subject, with its best clips ABOUT the subject (strongest first)."""

    slug: str
    row: Any
    clips: tuple[ClipRef, ...]


_EPISODE_CLIPS_NS = "reel_episode_clips"
_SEGMENTS_REL_NS = "reel_segments_relpath"
_SUBJECT_NS = "reel_subject_episodes"
_REEL_NS = "reel_hydrated"


def _mtime(path: str) -> float:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def _mtime_rel(root: Path, rel: Optional[str]) -> float:
    """``_mtime`` of a catalog row's corpus-relative path; 0 for none, or for one that could leave
    the corpus. Read for every episode of a subject on each corpus change, so the guard is string
    checks, not ``safe_relpath_under_corpus_root`` (which resolves the root: ~30 us a call)."""
    if not rel or rel.startswith(("/", "\\")) or ".." in rel.replace("\\", "/").split("/"):
        return 0.0
    return _mtime(os.path.join(str(root), rel))


def _segments_relpath(root: Path, row: Any) -> Optional[str]:
    """The episode's segments file, cached until its metadata file changes (the metadata names
    the transcript the segments belong to)."""
    from podcast_scraper import perf_cache
    from podcast_scraper.server.app_episode_search import episode_segments_relpath

    return (
        perf_cache.get_or_compute(  # type: ignore[no-any-return]
            _SEGMENTS_REL_NS,
            (str(root), row.metadata_relative_path),
            _mtime_rel(root, row.metadata_relative_path),
            lambda: episode_segments_relpath(root, row) or "",
        )
        or None
    )


def _episode_token(root: Path, row: Any) -> float:
    """What one episode's clips derive from: its GI, its metadata and its segments file."""
    seg = _segments_relpath(root, row)
    return float(
        hash(
            (
                _mtime_rel(root, row.gi_relative_path),
                _mtime_rel(root, row.metadata_relative_path),
                _mtime(seg) if seg else 0.0,
            )
        )
    )


def episode_clips(
    root: Path, row: Any, config: Optional[MomentsConfig] = None
) -> tuple[ClipRef, ...]:
    """Every clip of the episode a reel could play, whatever its subject: the candidates whose
    insight is ABOUT some topic and whose opening quote says what the insight says, in the
    episode's ranking order.

    Cached per episode on that episode's own files (2026-10-11, scale: 2,500 episodes on prod,
    10,000 planned). Scoring one costs ~11 ms (mostly reading its segments); keyed on the
    corpus-wide mtime instead, every ingest would rescore every episode of every subject."""
    from podcast_scraper import perf_cache

    cfg = config or MomentsConfig.from_env()
    return perf_cache.get_or_compute(  # type: ignore[no-any-return]
        _EPISODE_CLIPS_NS,
        (str(root), row.metadata_relative_path, repr(cfg)),
        _episode_token(root, row),
        lambda: _episode_clips_uncached(root, row, cfg),
    )


def _episode_clips_uncached(root: Path, row: Any, cfg: MomentsConfig) -> tuple[ClipRef, ...]:
    import sys

    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_episode_search import episode_segments
    from podcast_scraper.server.app_moments import _topic_links

    artifact = load_json_artifact(root, row.gi_relative_path)
    if not isinstance(artifact, dict):
        return ()
    links = _topic_links(artifact)
    if not links:
        return ()
    scored = [
        m
        for m in score_candidates(artifact, cfg, episode_segments(root, row))
        if links.get(m.insight_id) and quote_support(m.quote_text, m.text) >= MIN_SUPPORT
    ]
    return tuple(
        ClipRef(
            insight_id=m.insight_id,
            score=m.score,
            order=i,
            start_ms=m.start_ms,
            end_ms=m.end_ms,
            speaker=sys.intern(m.speaker) if m.speaker else m.speaker,
            topics=frozenset(sys.intern(t) for t in links[m.insight_id]),
        )
        for i, m in enumerate(rank_moments(scored, cfg))
    )


def _subject_token(root: Path) -> float:
    """What a subject's episode list derives from: the corpus (which episodes exist), the search
    sidecar (which topic is on which episode) and the storyline / theme files (its members). A
    change rebuilds the list from the per-episode cache, so only new episodes are scored."""
    from podcast_scraper import perf_cache

    stamps: list[float] = [perf_cache.corpus_mtime(root)]
    for rel in (
        "search/lance_index",
        "enrichments/topic_theme_clusters.json",
        "search/topic_clusters.json",
    ):
        stamps.append(_mtime(str(root / rel)))
    return float(hash(tuple(stamps)))


def subject_episodes(
    root: Path, subject_id: str, config: Optional[MomentsConfig] = None
) -> list[SubjectEpisode]:
    """The subject's episodes, each with its best clips ABOUT the subject (at most
    ``MAX_PER_EPISODE``), episodes by their best clip, strongest first. A clip qualifies when its
    insight is ABOUT a subject topic and its opening quote says what the insight says.

    The same for every listener, so it is built once per corpus version: Discover's "15 minutes
    on" row asks for up to 24 reels and the Learn offers for up to 12 paths on every load."""
    from podcast_scraper import perf_cache

    cfg = config or MomentsConfig.from_env()
    return list(
        perf_cache.get_or_compute(
            _SUBJECT_NS,
            (str(root), subject_id, repr(cfg)),
            _subject_token(root),
            lambda: _subject_episodes_uncached(root, subject_id, cfg),
        )
    )


def _subject_episodes_uncached(
    root: Path, subject_id: str, cfg: MomentsConfig
) -> tuple[SubjectEpisode, ...]:
    from podcast_scraper.server.app_slugs import slug_for_row

    topics = reel_subject(root, subject_id).topics
    out: list[SubjectEpisode] = []
    for row in _episode_rows(root, topics):
        about = [c for c in episode_clips(root, row, cfg) if c.topics & topics]
        if about:
            best = sorted(about, key=lambda c: (-c.score, c.order))[:MAX_PER_EPISODE]
            out.append(SubjectEpisode(slug_for_row(row), row, tuple(best)))
    # Episodes by their best clip, strongest first; a tie keeps a stable order by slug.
    out.sort(key=lambda e: (-e.clips[0].score, e.slug))
    return tuple(out)


def hydrate(
    root: Path, row: Any, refs: Sequence[ClipRef], config: Optional[MomentsConfig] = None
) -> list[Moment]:
    """The full moments (text, quote, speaker, clip) for ``refs``, all from one episode, in
    their order. A ref whose insight is gone from the episode's files is left out."""
    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_episode_search import episode_segments

    cfg = config or MomentsConfig.from_env()
    artifact = load_json_artifact(root, row.gi_relative_path)
    if not isinstance(artifact, dict) or not refs:
        return []
    by_id = {m.insight_id: m for m in score_candidates(artifact, cfg, episode_segments(root, row))}
    return [by_id[r.insight_id] for r in refs if r.insight_id in by_id]


def reel_selection(
    root: Path, subject_id: str, *, minutes: int = 15, config: Optional[MomentsConfig] = None
) -> list[tuple[SubjectEpisode, ClipRef]]:
    """Which clips the reel plays, without their text: every episode's best clip before any
    episode's second, each added while it fits ``minutes``. Enough for a chip's clip count and
    running time, so the "15 minutes on" row never reads a transcript."""
    budget_ms = int(minutes * 60 * 1000)
    used = 0
    picked: list[tuple[SubjectEpisode, ClipRef]] = []
    episodes = subject_episodes(root, subject_id, config)
    for depth in range(MAX_PER_EPISODE):
        for ep in episodes:
            if depth >= len(ep.clips):
                continue
            c = ep.clips[depth]
            length = c.end_ms - c.start_ms
            if used + length > budget_ms:
                continue  # a shorter clip later may still fit
            picked.append((ep, c))
            used += length
    return picked


def goal_reel(
    root: Path,
    subject_id: str,
    *,
    minutes: int = 15,
    config: Optional[MomentsConfig] = None,
) -> list[ReelClip]:
    """Clips about ``subject_id`` (a topic, or a theme's or storyline's member topics) across
    episodes, best and breadth first, within ``minutes``."""
    from podcast_scraper import perf_cache

    cfg = config or MomentsConfig.from_env()
    return list(
        perf_cache.get_or_compute(
            _REEL_NS,
            (str(root), subject_id, minutes, repr(cfg)),
            _subject_token(root),
            lambda: _goal_reel_uncached(root, subject_id, minutes, cfg),
        )
    )


def _goal_reel_uncached(
    root: Path, subject_id: str, minutes: int, cfg: MomentsConfig
) -> tuple[ReelClip, ...]:
    picked = reel_selection(root, subject_id, minutes=minutes, config=cfg)
    per_episode: dict[str, list[ClipRef]] = {}
    rows: dict[str, Any] = {}
    for ep, ref in picked:
        per_episode.setdefault(ep.slug, []).append(ref)
        rows[ep.slug] = ep.row
    full = {
        (slug, m.insight_id): m
        for slug, refs in per_episode.items()
        for m in hydrate(root, rows[slug], refs, cfg)
    }
    return tuple(
        ReelClip(ep.slug, full[(ep.slug, ref.insight_id)])
        for ep, ref in picked
        if (ep.slug, ref.insight_id) in full
    )
