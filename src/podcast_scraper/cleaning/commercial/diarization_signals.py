"""Diarization-aware confidence signals for commercial detection (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

HOST_MONOLOGUE_BOOST = 0.20
TOPIC_DISCONTINUITY_BOOST = 0.15
GUEST_SPEAKER_PENALTY = 0.30
AD_DURATION_BOOST = 0.10
AD_DURATION_MIN_S = 30.0
AD_DURATION_MAX_S = 90.0


@dataclass(frozen=True)
class DiarizationSignals:
    """Confidence adjustments derived from diarization context."""

    confidence_delta: float = 0.0
    disqualify: bool = False


@dataclass(frozen=True)
class _CharTimeSpan:
    char_start: int
    char_end: int
    start_s: float
    end_s: float
    speaker: Optional[str]


def _build_char_time_index(timed_segments: List[dict]) -> List[_CharTimeSpan]:
    index: List[_CharTimeSpan] = []
    char_pos = 0
    for segment in timed_segments:
        segment_text = (segment.get("text") or "").strip()
        if not segment_text:
            continue
        char_start = char_pos
        char_end = char_pos + len(segment_text)
        index.append(
            _CharTimeSpan(
                char_start=char_start,
                char_end=char_end,
                start_s=float(segment.get("start") or 0.0),
                end_s=float(segment.get("end") or segment.get("start") or 0.0),
                speaker=segment.get("speaker"),
            )
        )
        char_pos = char_end + 1
    return index


def _spans_overlapping_char_range(
    index: List[_CharTimeSpan],
    start: int,
    end: int,
) -> List[_CharTimeSpan]:
    return [
        span
        for span in index
        if span.char_end > start and span.char_start < end and span.speaker is not None
    ]


def diarization_sponsor_signals(
    candidate_start: int,
    candidate_end: int,
    text: str,
    timed_segments: List[dict],
    host_speaker_id: str,
) -> DiarizationSignals:
    """Compute diarization-based confidence adjustments for a sponsor candidate."""
    if not timed_segments or not host_speaker_id:
        return DiarizationSignals()

    index = _build_char_time_index(timed_segments)
    overlapping = _spans_overlapping_char_range(index, candidate_start, candidate_end)
    if not overlapping:
        return DiarizationSignals()

    speakers: Set[str] = {span.speaker for span in overlapping if span.speaker}
    if not speakers.issubset({host_speaker_id}):
        return DiarizationSignals(disqualify=True, confidence_delta=-GUEST_SPEAKER_PENALTY)

    delta = 0.0
    duration_s = max(span.end_s for span in overlapping) - min(span.start_s for span in overlapping)
    if AD_DURATION_MIN_S <= duration_s <= AD_DURATION_MAX_S:
        delta += AD_DURATION_BOOST

    text_len = max(len(text), 1)
    relative_pos = candidate_start / text_len
    if 0.15 <= relative_pos <= 0.85:
        delta += HOST_MONOLOGUE_BOOST

    before = _spans_overlapping_char_range(index, max(0, candidate_start - 400), candidate_start)
    after = _spans_overlapping_char_range(index, candidate_end, min(len(text), candidate_end + 400))
    before_speakers = {span.speaker for span in before}
    after_speakers = {span.speaker for span in after}
    if before_speakers and after_speakers and before_speakers != {host_speaker_id}:
        delta += TOPIC_DISCONTINUITY_BOOST

    return DiarizationSignals(confidence_delta=delta)
