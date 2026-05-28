"""Unit tests for TailnetDgxWhisperTranscriptionProvider (ADR-096)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import Config
from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
    TailnetDgxWhisperTranscriptionProvider,
)


def _dgx_cfg() -> Config:
    return Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "tailnet_dgx_whisper",
            "transcription_fallback_provider": "openai",
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "openai_api_key": "sk-test",
        }
    )


def test_config_rejects_dgx_without_fallback() -> None:
    with pytest.raises(ValueError, match="transcription_fallback_provider"):
        Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "transcription_provider": "tailnet_dgx_whisper",
                "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            }
        )


def test_nested_transcription_yaml_flattens() -> None:
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription": {
                "primary": "tailnet_dgx_whisper",
                "fallback": "openai",
            },
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "openai_api_key": "sk-test",
        }
    )
    assert cfg.transcription_provider == "tailnet_dgx_whisper"
    assert cfg.transcription_fallback_provider == "openai"


@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.emit_dgx_fallback_breadcrumb")
@patch(
    "podcast_scraper.providers.tailnet_dgx.whisper_provider.check_ollama_health", return_value=False
)
def test_falls_back_when_dgx_unhealthy(
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")

    cfg = _dgx_cfg()
    provider = TailnetDgxWhisperTranscriptionProvider(cfg)
    fallback = MagicMock()
    fallback.transcribe_with_segments.return_value = (
        {"text": "cloud text", "segments": [], "language": "en"},
        1.0,
    )
    provider._fallback = fallback
    provider._initialized = True

    text = provider.transcribe(str(audio))
    assert text == "cloud text"
    fallback.transcribe_with_segments.assert_called_once()
    _breadcrumb.assert_called_once()
