"""Unit tests for diarization disk cache."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.cache import (
    diarization_cache_path,
    load_cached_diarization,
    save_diarization_cache,
)
from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result

pytestmark = pytest.mark.unit


def test_diarization_cache_round_trip(tmp_path: Path) -> None:
    cache_path = tmp_path / "abc_config.json"
    result = DiarizationResult(
        segments=[DiarizationSegment(start=0.0, end=1.5, speaker="SPEAKER_00")],
        num_speakers=1,
        model_name="pyannote/test",
    )
    save_diarization_cache(str(cache_path), result)
    loaded = load_cached_diarization(str(cache_path))
    assert loaded is not None
    assert loaded.num_speakers == 1
    assert loaded.segments[0].speaker == "SPEAKER_00"


@patch("podcast_scraper.providers.ml.diarization.pipeline.create_diarization_provider")
@patch("podcast_scraper.providers.ml.diarization.cache.get_audio_hash")
def test_apply_diarization_uses_cache_on_hit(
    mock_audio_hash: MagicMock,
    mock_create_provider: MagicMock,
    tmp_path: Path,
) -> None:
    mock_audio_hash.return_value = "audiohash"
    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
    )
    cache_dir = tmp_path / "cache"
    cache_path = diarization_cache_path("/tmp/audio.wav", cfg, str(cache_dir))
    save_diarization_cache(
        cache_path,
        DiarizationResult(
            segments=[DiarizationSegment(start=0.0, end=2.0, speaker="SPEAKER_01")],
            num_speakers=1,
            model_name="cached",
        ),
    )

    result = {
        "text": "hello",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello"}],
    }
    enriched = apply_diarization_to_result(
        result,
        "/tmp/audio.wav",
        cfg,
        ["Host"],
        cache_dir=str(cache_dir),
    )

    mock_create_provider.assert_not_called()
    assert enriched["segments"][0]["speaker"] == "SPEAKER_01"


@patch("podcast_scraper.providers.ml.diarization.pipeline.create_diarization_provider")
@patch("podcast_scraper.providers.ml.diarization.cache.get_audio_hash")
def test_apply_diarization_writes_cache_on_miss(
    mock_audio_hash: MagicMock,
    mock_create_provider: MagicMock,
    tmp_path: Path,
) -> None:
    mock_audio_hash.return_value = "audiohash"
    mock_provider = MagicMock()
    mock_provider.diarize.return_value = DiarizationResult(
        segments=[DiarizationSegment(start=0.0, end=2.0, speaker="SPEAKER_00")],
        num_speakers=1,
        model_name="live",
    )
    mock_create_provider.return_value = mock_provider

    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
    )
    cache_dir = tmp_path / "cache"
    cache_path = Path(diarization_cache_path("/tmp/audio.wav", cfg, str(cache_dir)))

    result = {
        "text": "hello",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello"}],
    }
    apply_diarization_to_result(
        result,
        "/tmp/audio.wav",
        cfg,
        None,
        cache_dir=str(cache_dir),
    )

    assert cache_path.is_file()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["segments"][0]["speaker"] == "SPEAKER_00"
