"""Unit tests for commercial diarization cleaning context helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.cleaning.commercial.context import (
    diarization_cleaning_context,
    infer_host_speaker_id,
    load_transcript_segments,
)

pytestmark = pytest.mark.unit


def test_infer_host_speaker_id_prefers_intro_speaker() -> None:
    segments = [
        {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
        {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_01"},
        {"start": 20.0, "end": 100.0, "speaker": "SPEAKER_01"},
    ]
    assert infer_host_speaker_id(segments) == "SPEAKER_00"


def test_infer_host_weighted_by_duration_not_turn_count() -> None:
    """A guest who interjects in many short intro turns must not beat the host who
    speaks longer (B4: duration-weighted, not turn-count)."""
    segments = [
        # Guest: 3 short turns (turn-count = 3, but only 3s total).
        {"start": 0.0, "end": 1.0, "speaker": "GUEST"},
        {"start": 1.0, "end": 2.0, "speaker": "GUEST"},
        {"start": 2.0, "end": 3.0, "speaker": "GUEST"},
        # Host: 1 long intro turn (turn-count = 1, but 12s).
        {"start": 3.0, "end": 15.0, "speaker": "HOST"},
        {"start": 20.0, "end": 100.0, "speaker": "HOST"},
    ]
    assert infer_host_speaker_id(segments) == "HOST"


def test_load_transcript_segments_and_context(tmp_path: Path) -> None:
    transcript = tmp_path / "episode.txt"
    transcript.write_text("hello", encoding="utf-8")
    segments_path = tmp_path / "episode.segments.json"
    segments_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "hi"},
                {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "hey"},
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_transcript_segments(str(transcript))
    assert loaded is not None
    assert len(loaded) == 2

    segments, host_id = diarization_cleaning_context(str(transcript))
    assert segments is not None
    assert host_id == "SPEAKER_00"


def test_diarization_cleaning_context_without_speaker_ids(tmp_path: Path) -> None:
    transcript = tmp_path / "episode.txt"
    transcript.write_text("hello", encoding="utf-8")
    segments_path = tmp_path / "episode.segments.json"
    segments_path.write_text(
        json.dumps([{"start": 0.0, "end": 5.0, "text": "hi"}]),
        encoding="utf-8",
    )

    segments, host_id = diarization_cleaning_context(str(transcript))
    assert segments is None
    assert host_id is None
