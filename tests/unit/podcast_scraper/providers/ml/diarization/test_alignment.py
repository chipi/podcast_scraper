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
