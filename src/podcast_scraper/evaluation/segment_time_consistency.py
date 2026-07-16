"""Ground-truth-free consistency checks for transcript segment times (#1173).

A subtitle / player seeks by each segment's ``start`` / ``end``. When those times drift from
the audio — the #1173 bug, where a provider emits coarse *segment*-level times instead of
word-accurate ones — the symptom is visible **without any reference timeline**: segments whose
text cannot physically fit their window (hundreds of chars/sec), non-monotonic starts, or times
outside ``[0, episode_end]``.

This module scores a produced corpus's ``*.segments.json`` for those properties, so a
re-transcription with the word-timestamp refinement can be checked automatically (AC3) instead
of by listening. It reuses the ``compute_*`` / ``enforce_*`` shape of
:mod:`podcast_scraper.evaluation.diarization_quality`.

Pure file scan (no DB, no audio). The impossible-rate bar is deliberately generous: sustained
human speech tops out around 25 chars/sec, so >40 chars/sec over a multi-word segment is a
timing error, not fast talking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# A segment shorter than its text can be spoken in ⇒ its times are wrong, not the speaker fast.
IMPOSSIBLE_CHARS_PER_SEC = 40.0  # human ceiling ≈ 25 cps; 40 leaves headroom for punctuation
_MIN_CHARS_FOR_RATE = 15  # ignore 1–2 word segments — rate is meaningless at that length

# Default enforcement thresholds.
MAX_IMPOSSIBLE_RATE = 0.002  # ≤0.2% of segments may exceed the impossible-cps bar
MAX_NONMONOTONIC = 0  # segment starts must never go backwards
MAX_OUT_OF_BOUNDS = 0  # no negative times, no end < start


@dataclass
class EpisodeSegmentMetrics:
    """Per-episode transcript-segment-time consistency counters."""

    episode: str
    segments_total: int = 0
    segments_rated: int = 0  # segments long enough to score a speech rate
    impossible_segments: int = 0  # chars/sec above the impossible bar
    nonmonotonic_starts: int = 0
    out_of_bounds: int = 0  # negative start/end or end < start
    worst_chars_per_sec: float = 0.0
    episode_end: float = 0.0


def _load_segments(path: Path) -> Optional[List[Dict[str, Any]]]:
    """Return the segment list from a ``.segments.json`` (list, or ``{"segments": [...]}``)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        segments = data.get("segments")
        if isinstance(segments, list):
            return segments
    return None


def discover_segment_files(corpus_root: Path) -> List[Path]:
    """Find raw ``*.segments.json`` under a corpus, excluding derived ``.adfree.segments.json``.

    The ad-free segments are a downstream re-slice of the raw ones (same times), so scoring both
    would double-count. Raw segments are the provider's direct output — the #1173 surface.
    """
    return sorted(
        p
        for p in Path(corpus_root).rglob("*.segments.json")
        if not p.name.endswith(".adfree.segments.json")
    )


def _episode_metrics(path: Path) -> EpisodeSegmentMetrics:
    m = EpisodeSegmentMetrics(episode=path.name)
    segments = _load_segments(path)
    if not segments:
        return m
    prev_start = -1.0
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        if start is None or end is None:
            continue
        m.segments_total += 1
        if start < 0 or end < 0 or end < start:
            m.out_of_bounds += 1
        if start < prev_start - 1e-6:
            m.nonmonotonic_starts += 1
        prev_start = start
        m.episode_end = max(m.episode_end, float(end))

        text = (seg.get("text") or "").strip()
        dur = end - start
        if len(text) >= _MIN_CHARS_FOR_RATE and dur > 0:
            m.segments_rated += 1
            cps = len(text) / dur
            m.worst_chars_per_sec = max(m.worst_chars_per_sec, cps)
            if cps > IMPOSSIBLE_CHARS_PER_SEC:
                m.impossible_segments += 1
    return m


def compute_segment_time_consistency(corpus_root: Path) -> Dict[str, Any]:
    """Walk a corpus and aggregate per-episode segment-time consistency metrics."""
    per_episode = [_episode_metrics(p) for p in discover_segment_files(Path(corpus_root))]
    per_episode = [e for e in per_episode if e.segments_total > 0]

    segs_total = sum(e.segments_total for e in per_episode)
    impossible = sum(e.impossible_segments for e in per_episode)
    return {
        "episodes_total": len(per_episode),
        "segments_total": segs_total,
        "impossible_segments": impossible,
        "impossible_rate": (impossible / segs_total) if segs_total else 0.0,
        "episodes_with_impossible": sum(1 for e in per_episode if e.impossible_segments),
        "episodes_nonmonotonic": sum(1 for e in per_episode if e.nonmonotonic_starts),
        "episodes_out_of_bounds": sum(1 for e in per_episode if e.out_of_bounds),
        "nonmonotonic_starts": sum(e.nonmonotonic_starts for e in per_episode),
        "out_of_bounds": sum(e.out_of_bounds for e in per_episode),
        "per_episode": [e.__dict__ for e in per_episode],
    }


def enforce_segment_time_consistency(
    metrics: Dict[str, Any],
    *,
    max_impossible_rate: float = MAX_IMPOSSIBLE_RATE,
    max_nonmonotonic: int = MAX_NONMONOTONIC,
    max_out_of_bounds: int = MAX_OUT_OF_BOUNDS,
) -> Tuple[bool, List[str]]:
    """Return ``(passed, failures)`` for the AC3 prod-safety consistency bar."""
    failures: List[str] = []
    rate = metrics.get("impossible_rate", 0.0)
    if rate > max_impossible_rate:
        failures.append(
            f"impossible-cps segments {rate:.3%} > {max_impossible_rate:.3%} "
            f"({metrics.get('impossible_segments')} segments in "
            f"{metrics.get('episodes_with_impossible')} episodes) — segment times not word-accurate"
        )
    if metrics.get("nonmonotonic_starts", 0) > max_nonmonotonic:
        failures.append(
            f"non-monotonic segment starts in {metrics.get('episodes_nonmonotonic')} episodes"
        )
    if metrics.get("out_of_bounds", 0) > max_out_of_bounds:
        failures.append(
            f"out-of-bounds segment times in {metrics.get('episodes_out_of_bounds')} episodes"
        )
    return (not failures, failures)
