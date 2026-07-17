"""Diarization-aware confidence signals for commercial detection (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

HOST_MONOLOGUE_BOOST = 0.20
TOPIC_DISCONTINUITY_BOOST = 0.15
GUEST_SPEAKER_PENALTY = 0.30
AD_DURATION_BOOST = 0.10
AD_DURATION_MIN_S = 30.0
AD_DURATION_MAX_S = 90.0

# How far (in mapped seconds) before/after a candidate to look for a speaker
# change when scoring topic discontinuity.
_CONTEXT_WINDOW_S = 20.0


@dataclass(frozen=True)
class DiarizationSignals:
    """Confidence adjustments derived from diarization context."""

    confidence_delta: float = 0.0
    disqualify: bool = False


@dataclass(frozen=True)
class _TimedSpan:
    start_s: float
    end_s: float
    speaker: Optional[str]


def _build_time_index(timed_segments: List[dict]) -> Tuple[List[_TimedSpan], float]:
    """Return speaker spans sorted by start time, plus the total covered duration.

    Unlike the previous char-concatenation index, this keeps everything in the
    *time* domain so it does not depend on character offsets matching the cleaned
    transcript (they don't — the transcript is stripped/normalised after the
    segments are produced).
    """
    spans: List[_TimedSpan] = []
    for segment in timed_segments:
        if not (segment.get("text") or "").strip():
            continue
        start_s = float(segment.get("start") or 0.0)
        end_s = float(segment.get("end") or segment.get("start") or 0.0)
        spans.append(_TimedSpan(start_s=start_s, end_s=end_s, speaker=segment.get("speaker")))
    spans.sort(key=lambda span: span.start_s)
    total_duration = max((span.end_s for span in spans), default=0.0)
    return spans, total_duration


def _speakers_in_time_range(spans: List[_TimedSpan], start_s: float, end_s: float) -> Set[str]:
    """Speaker ids whose span overlaps the [start_s, end_s) time window."""
    return {
        span.speaker
        for span in spans
        if span.speaker is not None and span.end_s > start_s and span.start_s < end_s
    }


def diarization_sponsor_signals(
    candidate_start: int,
    candidate_end: int,
    text: str,
    timed_segments: List[dict],
    host_speaker_id: str,
) -> DiarizationSignals:
    """Compute diarization-based confidence adjustments for a sponsor candidate.

    The candidate offsets are positions in the cleaned ``text``; we map them to
    the audio timeline **proportionally** (fraction of total characters -> fraction
    of total duration) rather than assuming the segment texts reproduce ``text``
    char-for-char. This is approximate but robust to the transcript cleaning that
    happens between diarization and detection, which the old absolute-offset index
    silently got wrong.
    """
    if not timed_segments or not host_speaker_id:
        return DiarizationSignals()

    text_len = max(len(text), 1)
    spans, total_duration = _build_time_index(timed_segments)
    if total_duration <= 0.0 or not spans:
        return DiarizationSignals()

    def _to_time(char_pos: int) -> float:
        return max(0.0, min(1.0, char_pos / text_len)) * total_duration

    t_start = _to_time(candidate_start)
    t_end = max(_to_time(candidate_end), t_start)

    overlapping = _speakers_in_time_range(spans, t_start, t_end)
    if not overlapping:
        return DiarizationSignals()

    if not overlapping.issubset({host_speaker_id}):
        # A non-host voice speaks inside the candidate -> it is dialogue, not a
        # host-read ad. Disqualify.
        return DiarizationSignals(disqualify=True, confidence_delta=-GUEST_SPEAKER_PENALTY)

    delta = 0.0
    duration_s = t_end - t_start
    if AD_DURATION_MIN_S <= duration_s <= AD_DURATION_MAX_S:
        delta += AD_DURATION_BOOST

    relative_pos = candidate_start / text_len
    if 0.15 <= relative_pos <= 0.85:
        delta += HOST_MONOLOGUE_BOOST

    before = _speakers_in_time_range(spans, t_start - _CONTEXT_WINDOW_S, t_start)
    after = _speakers_in_time_range(spans, t_end, t_end + _CONTEXT_WINDOW_S)
    # Require the host to RESUME after the candidate (host_speaker_id in after),
    # not just "someone speaks after" — the old condition boosted whenever any
    # non-host spoke before, regardless of who followed, over-firing the
    # topic-discontinuity signal (review 2026-07-17 low/diar-discontinuity).
    if before and before != {host_speaker_id} and host_speaker_id in after:
        delta += TOPIC_DISCONTINUITY_BOOST

    return DiarizationSignals(confidence_delta=delta)
