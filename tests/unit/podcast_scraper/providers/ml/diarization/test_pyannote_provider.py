"""Unit tests for pyannote diarization provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.ml.diarization.pyannote_provider import PyAnnoteDiarizationProvider

pytestmark = pytest.mark.unit


@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._load_waveform")
@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._create_pyannote_pipeline")
def test_diarize_maps_pyannote_output(mock_create, mock_load_waveform) -> None:
    mock_pipeline = MagicMock()
    mock_create.return_value = mock_pipeline
    mock_load_waveform.return_value = (MagicMock(), 16000)

    turn_a = MagicMock(start=0.0, end=1.5)
    turn_b = MagicMock(start=1.5, end=3.0)
    mock_pipeline.return_value.itertracks.return_value = [
        (turn_a, None, "SPEAKER_00"),
        (turn_b, None, "SPEAKER_01"),
    ]

    provider = PyAnnoteDiarizationProvider("token", device="cpu")
    result = provider.diarize("/tmp/audio.wav", num_speakers=2)

    assert result.num_speakers == 2
    assert len(result.segments) == 2
    assert result.segments[0].speaker == "SPEAKER_00"


@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._load_waveform")
@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._create_pyannote_pipeline")
def test_diarize_rejects_non_positive_num_speakers(mock_create, mock_load_waveform) -> None:
    """num_speakers < 1 is rejected rather than silently bypassing the floor (A5)."""
    mock_create.return_value = MagicMock()
    mock_load_waveform.return_value = (MagicMock(), 16000)
    provider = PyAnnoteDiarizationProvider("token", device="cpu")
    with pytest.raises(ValueError):
        provider.diarize("/tmp/audio.wav", num_speakers=0)


@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._load_waveform")
@patch("podcast_scraper.providers.ml.diarization.pyannote_provider._create_pyannote_pipeline")
def test_diarize_rejects_min_greater_than_max(mock_create, mock_load_waveform) -> None:
    """min_speakers > max_speakers is an invalid bound, not silently forwarded (A5)."""
    mock_create.return_value = MagicMock()
    mock_load_waveform.return_value = (MagicMock(), 16000)
    provider = PyAnnoteDiarizationProvider("token", device="cpu")
    with pytest.raises(ValueError):
        provider.diarize("/tmp/audio.wav", num_speakers=None, min_speakers=5, max_speakers=2)
