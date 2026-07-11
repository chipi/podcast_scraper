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
    # SPEAKER_00 owns the intro -> host (kept raw); SPEAKER_01 talks most -> guest (#876).
    mock_provider.diarize.return_value = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=60.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=60.0, end=400.0, speaker="SPEAKER_01"),
        ],
        num_speakers=2,
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
        "text": "hello world",
        "segments": [
            {"start": 0.0, "end": 60.0, "text": "hello"},
            {"start": 60.0, "end": 400.0, "text": "world"},
        ],
    }

    # detected_speaker_names is guest-only; the guest name must land on the guest, not the host.
    enriched = apply_diarization_to_result(result, "/tmp/audio.wav", cfg, ["Guest"])

    assert enriched["segments"][0]["speaker"] == "SPEAKER_00"
    assert enriched["segments"][0]["speaker_label"] == "SPEAKER_00", "host kept raw"
    assert enriched["segments"][1]["speaker_label"] == "Guest", "guest named"
    # enrichment sidecar + role hint (harden #1170): the diagnostics dict is attached,
    # and an unnamed host segment carries speaker_role="host" (renders as "Host", not SPEAKER).
    assert "speaker_diagnostics" in enriched, "speaker_diagnostics sidecar attached"
    assert enriched["segments"][0]["speaker_role"] == "host", "unnamed host tagged for display"
    assert "speaker_role" not in enriched["segments"][1], "named guest needs no role hint"


@patch("podcast_scraper.providers.ml.diarization.pipeline.create_diarization_provider")
def test_apply_diarization_names_host_from_transcript_self_intro(mock_create_provider) -> None:
    """End-to-end (#876): host self-intro in the transcript names the diarized host voice."""
    mock_provider = MagicMock()
    mock_provider.diarize.return_value = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=60.0, speaker="SPEAKER_00"),  # intro -> host
            DiarizationSegment(start=60.0, end=400.0, speaker="SPEAKER_01"),  # guest
        ],
        num_speakers=2,
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
        "text": "Hello and welcome. I'm Patrick O'Shaughnessy. My guest is Brian Chesky.",
        "segments": [
            {"start": 0.0, "end": 60.0, "text": "Hello and welcome. I'm Patrick O'Shaughnessy."},
            {"start": 60.0, "end": 400.0, "text": "Thanks for having me."},
        ],
    }

    enriched = apply_diarization_to_result(result, "/tmp/audio.wav", cfg, ["Brian Chesky"])

    assert enriched["segments"][0]["speaker_label"] == "Patrick O'Shaughnessy", "host named"
    assert enriched["segments"][1]["speaker_label"] == "Brian Chesky", "guest named"


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
