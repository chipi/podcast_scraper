"""MOSS transcription + diarization providers, against a mocked service (#1177).

No GPU, no live service: the point is that the *contract* is right — the joint model's single
inference feeds both stages, and its anonymous speaker labels arrive downstream in the dialect the
rest of the pipeline already speaks.
"""

from __future__ import annotations

from typing import cast, List
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization.moss_provider import (
    _normalize_speaker,
    MossDiarizationProvider,
)
from podcast_scraper.providers.moss.moss_provider import MossTranscriptionProvider

pytestmark = pytest.mark.unit

# What the DGX MOSS service returns: one inference, carrying BOTH text and speakers.
_RESPONSE = {
    "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    "text": "Welcome everyone The new pipeline is ready",
    "segments": [
        {"start": 0.48, "end": 1.66, "text": "Welcome everyone", "speaker": "S01"},
        {"start": 12.26, "end": 13.81, "text": "The new pipeline is ready", "speaker": "S02"},
    ],
    "speakers": ["S01", "S02"],
    "num_speakers": 2,
}


def _cfg() -> config.Config:
    return config.Config(
        rss="https://example.com/feed.xml",
        dgx_tailnet_host="dgx-llm-1",
    )


def _mock_client(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    client = MagicMock()
    client.post.return_value = response
    ctx = MagicMock()
    ctx.__enter__.return_value = client
    ctx.__exit__.return_value = False
    return ctx


@patch("podcast_scraper.providers.moss.moss_provider.hardened_http_client")
def test_transcription_returns_text_and_segments(mock_http, tmp_path) -> None:
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"audio")
    mock_http.return_value = _mock_client(_RESPONSE)

    provider = MossTranscriptionProvider(_cfg())
    provider.initialize()
    result, elapsed = provider.transcribe_with_segments(str(audio))
    segments = cast(List[dict], result["segments"])

    assert result["text"] == "Welcome everyone The new pipeline is ready"
    assert len(segments) == 2
    assert elapsed >= 0
    # The speaker survives on the segment — harmless here, and it is what lets the diarization
    # stage reuse this same inference instead of running the model twice.
    assert segments[0]["speaker"] == "S01"


@patch("podcast_scraper.providers.ml.diarization.moss_provider.hardened_http_client")
def test_diarization_speaks_pyannote_dialect(mock_http, tmp_path) -> None:
    """MOSS's S01/S02 must arrive as SPEAKER_01/SPEAKER_02 — the naming everything downstream
    keys off (roster, the #1167 placeholder guard, host/guest attribution)."""
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"audio")
    mock_http.return_value = _mock_client(_RESPONSE)

    provider = MossDiarizationProvider(_cfg())
    provider.initialize()
    result = provider.diarize(str(audio))

    assert result.num_speakers == 2
    assert [s.speaker for s in result.segments] == ["SPEAKER_01", "SPEAKER_02"]
    assert (result.segments[0].start, result.segments[0].end) == (0.48, 1.66)
    assert "MOSS" in result.model_name


def test_speaker_normalization() -> None:
    assert _normalize_speaker("S01") == "SPEAKER_01"
    assert _normalize_speaker("S9") == "SPEAKER_09"
    assert _normalize_speaker("S123") == "SPEAKER_123"
    assert _normalize_speaker("") == "SPEAKER_00"  # never emit an empty label
    assert _normalize_speaker("SPEAKER_02") == "SPEAKER_02"  # already normalized, left alone


@patch("podcast_scraper.providers.ml.diarization.moss_provider.hardened_http_client")
def test_malformed_turns_are_skipped_not_fatal(mock_http, tmp_path) -> None:
    """One bad turn must not cost the episode — the model is a decoder and will emit junk."""
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"audio")
    mock_http.return_value = _mock_client(
        {
            "model": "moss",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "fine", "speaker": "S01"},
                {"start": "nonsense", "end": 2.0, "text": "bad", "speaker": "S02"},
                {"end": 3.0, "text": "no start", "speaker": "S02"},
                {"start": 4.0, "end": 5.0, "text": "also fine", "speaker": "S02"},
            ],
        }
    )

    result = MossDiarizationProvider(_cfg()).diarize(str(audio))
    assert [(s.start, s.speaker) for s in result.segments] == [
        (0.0, "SPEAKER_01"),
        (4.0, "SPEAKER_02"),
    ]
    assert result.num_speakers == 2


def test_transcription_rejects_a_missing_file(tmp_path) -> None:
    provider = MossTranscriptionProvider(_cfg())
    provider.initialize()
    with pytest.raises(FileNotFoundError):
        provider.transcribe_with_segments(str(tmp_path / "nope.mp3"))


def test_both_factories_resolve_to_moss() -> None:
    """`transcription_provider: moss` + `diarization_provider: moss` must construct the two halves
    of the joint model — the wiring, not just the classes."""
    from podcast_scraper.providers.ml.diarization.factory import create_diarization_provider
    from podcast_scraper.transcription.factory import create_transcription_provider

    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="moss",
        diarization_provider="moss",
        dgx_tailnet_host="dgx-llm-1",
        transcription_fallback_provider="whisper",
    )
    assert isinstance(create_transcription_provider(cfg), MossTranscriptionProvider)
    assert isinstance(create_diarization_provider(cfg), MossDiarizationProvider)


def test_experiment_dgx_moss_profile_is_fully_local() -> None:
    """The MOSS profile must not leak to a commercial API on any stage."""
    import os

    os.environ.setdefault("DGX_TAILNET_HOST", "dgx-llm-1")
    cfg = config.Config.model_validate(
        {"profile": "experiment_dgx_moss", "rss_url": "https://example.com/feed.xml"}
    )

    assert cfg.transcription_provider == "moss"
    assert cfg.diarization_provider == "moss"
    # The LLM stages stay identical to experiment_dgx_only, so a diff isolates the audio stage.
    assert cfg.ollama_summary_model == "qwen3.5:35b"
    assert cfg.preprocessing_silence_removal is False  # the #1173 invariant

    cloud = {"openai", "gemini", "anthropic", "mistral", "deepgram", "grok", "deepseek"}
    for field in (
        "transcription_provider",
        "transcription_fallback_provider",
        "diarization_provider",
        "speaker_detector_provider",
        "summary_provider",
    ):
        assert getattr(cfg, field) not in cloud, f"{field} leaks to a cloud provider"
