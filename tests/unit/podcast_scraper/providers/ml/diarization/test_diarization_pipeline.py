"""Unit tests for diarization pipeline integration helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result

pytestmark = pytest.mark.unit


@patch("podcast_scraper.providers.ml.diarization.pipeline.create_diarization_provider")
def test_apply_diarization_enriches_segments(mock_create_provider) -> None:
    mock_provider = MagicMock()
    mock_provider.diarize.return_value = DiarizationResult(
        segments=[DiarizationSegment(start=0.0, end=2.0, speaker="SPEAKER_00")],
        num_speakers=1,
        model_name="test",
    )
    mock_create_provider.return_value = mock_provider

    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
    )
    result = {
        "text": "hello",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello"}],
    }

    enriched = apply_diarization_to_result(result, "/tmp/audio.wav", cfg, ["Host"])

    assert enriched["segments"][0]["speaker"] == "SPEAKER_00"
    assert enriched["segments"][0]["speaker_label"] == "Host"


@patch("podcast_scraper.providers.ml.diarization.pipeline.create_diarization_provider")
def test_apply_diarization_degrades_when_no_turns(mock_create_provider) -> None:
    """Zero diarization turns → segments returned unlabelled so the caller degrades (A3)."""
    mock_provider = MagicMock()
    mock_provider.diarize.return_value = DiarizationResult(
        segments=[],
        num_speakers=0,
        model_name="test",
    )
    mock_create_provider.return_value = mock_provider

    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
    )
    result = {
        "text": "hello",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello"}],
    }

    enriched = apply_diarization_to_result(result, "/tmp/audio.wav", cfg, ["Host"])

    # No phantom SPEAKER_00: segments carry no speaker_label, so the screenplay
    # gate falls back to gap-based formatting.
    assert "speaker_label" not in enriched["segments"][0]
    assert "speaker" not in enriched["segments"][0]
