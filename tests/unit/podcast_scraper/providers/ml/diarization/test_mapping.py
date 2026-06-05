"""Unit tests for diarization name mapping."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.mapping import map_speakers_to_names

pytestmark = pytest.mark.unit


def test_map_speakers_to_names_uses_intro_host_then_speaking_time() -> None:
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=60.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=60.0, end=120.0, speaker="SPEAKER_01"),
            DiarizationSegment(start=120.0, end=180.0, speaker="SPEAKER_01"),
        ],
        num_speakers=2,
        model_name="test",
    )

    mapping = map_speakers_to_names(diarization, ["Host Name", "Guest Name"])

    assert mapping["SPEAKER_00"] == "Host Name"
    assert mapping["SPEAKER_01"] == "Guest Name"
