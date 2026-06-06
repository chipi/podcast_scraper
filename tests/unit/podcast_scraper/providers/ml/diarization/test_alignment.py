"""Unit tests for diarization alignment."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.alignment import align_segments_to_speakers
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment

pytestmark = pytest.mark.unit


def test_align_segments_assigns_majority_overlap_speaker() -> None:
    whisper_segments = [
        {"start": 0.0, "end": 5.0, "text": "hello"},
        {"start": 5.0, "end": 10.0, "text": "world"},
    ]
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=6.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=6.0, end=10.0, speaker="SPEAKER_01"),
        ],
        num_speakers=2,
        model_name="test",
    )

    aligned = align_segments_to_speakers(whisper_segments, diarization)

    assert aligned[0][1] == "SPEAKER_00"
    assert aligned[1][1] == "SPEAKER_01"


def test_align_segments_carries_forward_when_no_overlap() -> None:
    whisper_segments = [{"start": 12.0, "end": 14.0, "text": "after gap"}]
    diarization = DiarizationResult(
        segments=[DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00")],
        num_speakers=1,
        model_name="test",
    )

    aligned = align_segments_to_speakers(whisper_segments, diarization)

    assert aligned[0][1] == "SPEAKER_00"


def test_align_ignores_sub_epsilon_boundary_overlap() -> None:
    """A sub-microsecond boundary touch must not flip the speaker (A2 epsilon)."""
    whisper_segments = [{"start": 5.0, "end": 10.0, "text": "second"}]
    diarization = DiarizationResult(
        segments=[
            # SPEAKER_00 touches the segment start by 1e-7s — noise, not real speech.
            DiarizationSegment(start=0.0, end=5.0000001, speaker="SPEAKER_00"),
            DiarizationSegment(start=5.0, end=10.0, speaker="SPEAKER_01"),
        ],
        num_speakers=2,
        model_name="test",
    )

    aligned = align_segments_to_speakers(whisper_segments, diarization)

    assert aligned[0][1] == "SPEAKER_01"


def test_align_tie_break_is_deterministic_and_prefers_continuity() -> None:
    """Equal overlaps resolve deterministically: continuity first, else lexicographic (A2)."""
    # First segment establishes SPEAKER_01 as the running speaker.
    whisper_segments = [
        {"start": 0.0, "end": 4.0, "text": "intro"},
        {"start": 4.0, "end": 8.0, "text": "tie"},
    ]
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=4.0, speaker="SPEAKER_01"),
            # Second whisper segment overlaps SPEAKER_00 and SPEAKER_01 equally (2.0s each).
            DiarizationSegment(start=4.0, end=6.0, speaker="SPEAKER_01"),
            DiarizationSegment(start=6.0, end=8.0, speaker="SPEAKER_00"),
        ],
        num_speakers=2,
        model_name="test",
    )

    aligned = align_segments_to_speakers(whisper_segments, diarization)

    # Tie resolved to the previous speaker (continuity), not dict-insertion order.
    assert aligned[1][1] == "SPEAKER_01"
    # Determinism: same input, same output across repeated calls.
    assert align_segments_to_speakers(whisper_segments, diarization)[1][1] == "SPEAKER_01"
