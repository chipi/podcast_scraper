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
        dgx_tailnet_host="dgx.test",
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


def test_transcription_rejects_a_missing_file(tmp_path) -> None:
    provider = MossTranscriptionProvider(_cfg())
    provider.initialize()
    with pytest.raises(FileNotFoundError):
        provider.transcribe_with_segments(str(tmp_path / "nope.mp3"))


def test_experiment_dgx_moss_profile_is_fully_local() -> None:
    """The MOSS profile must not leak to a commercial API on any stage."""
    import os

    os.environ.setdefault("DGX_TAILNET_HOST", "dgx.test")
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
