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
from podcast_scraper.server.app_reasons import _content_words, MIN_SUPPORT, quote_support

#: Clips taken from one episode at most, so the reel is many voices rather than one long episode.
MAX_PER_EPISODE = 2
#: Two clips of one episode whose insights share at least this much of their content words (either
#: way) say the same thing: the second plays as a repeat (measured on the fixture, 2026-10-11:
#: restatements 0.75-1.00, different claims 0.30-0.53).
ECHO_SIMILARITY = 0.65
#: Across episodes only the same sentence counts as a repeat: different claims there share their
#: framing words (measured on the fixture, 2026-10-11: one identical line read in two episodes at
#: 1.0; the closest different claims at 0.79).
SAME_LINE_SIMILARITY = 0.95
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
    #: The insight's content words, so a reel can tell when two episodes say the same thing.
    words: frozenset[str] = frozenset()


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
    ranked = rank_moments(scored, cfg)
    # Strongest first, a clip that restates a stronger one is that one again (a restatement in
    # another segment). It is folded in, its topics with it, so a subject it was about still gets
    # the stronger clip. Clips that only overlap in time say different things and stay; a subject
    # never takes two that overlap (``_subject_episodes_uncached``).
    kept: dict[int, set[str]] = {}
    for i in sorted(range(len(ranked)), key=lambda i: (-ranked[i].score, i)):
        m = ranked[i]
        echo = next((k for k in kept if _echoes(ranked[k], m)), None)
        if echo is None:
            kept[i] = set(links[m.insight_id])
        else:
            kept[echo] |= links[m.insight_id]

    def ref(i: int) -> ClipRef:
        m = ranked[i]
        return ClipRef(
            insight_id=m.insight_id,
            score=m.score,
            order=i,
            start_ms=m.start_ms,
            end_ms=m.end_ms,
            speaker=sys.intern(m.speaker) if m.speaker else m.speaker,
            topics=frozenset(sys.intern(t) for t in kept[i]),
            words=frozenset(sys.intern(w) for w in _content_words(m.text)),
        )

    return tuple(ref(i) for i in sorted(kept))


def _echoes(a: Moment, b: Moment) -> bool:
    """``b`` says what ``a`` says."""
    return max(quote_support(a.text, b.text), quote_support(b.text, a.text)) >= ECHO_SIMILARITY


def _same_line(a: frozenset[str], b: frozenset[str]) -> bool:
    """Two clips (of different episodes) say the same sentence (``SAME_LINE_SIMILARITY``)."""
    if not a or not b:
        return False
    shared = len(a & b)
    return min(shared / len(a), shared / len(b)) >= SAME_LINE_SIMILARITY


def _overlap(a: ClipRef, b: ClipRef) -> bool:
    """``a`` and ``b`` play some of the same seconds (two insights grounded in one segment)."""
    return a.start_ms < b.end_ms and b.start_ms < a.end_ms


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
    candidates: list[tuple[str, Any, list[ClipRef]]] = []
    for row in _episode_rows(root, topics):
        about = sorted(
            (c for c in episode_clips(root, row, cfg) if c.topics & topics),
            key=lambda c: (-c.score, c.order),
        )
        if about:
            candidates.append((slug_for_row(row), row, about))
    # Strongest episodes first (a tie keeps a stable order by slug). Each takes its best clips that
    # neither overlap one another nor repeat the sentence of a clip already taken in another
    # episode (2026-10-11: two episodes reading the same line made a reel of one sentence, twice).
    candidates.sort(key=lambda e: (-e[2][0].score, e[0]))
    taken: list[frozenset[str]] = []
    out: list[SubjectEpisode] = []
    for slug, row, about in candidates:
        best: list[ClipRef] = []
        for c in about:
            if len(best) >= MAX_PER_EPISODE:
                break
            if any(_overlap(c, b) for b in best) or any(_same_line(c.words, w) for w in taken):
                continue
            best.append(c)
            taken.append(c.words)
        if best:
            out.append(SubjectEpisode(slug, row, tuple(best)))
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
