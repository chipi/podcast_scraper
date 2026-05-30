"""Transcription factory tests for tailnet_dgx_whisper (ADR-096)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import Config
from podcast_scraper.transcription.factory import create_transcription_provider


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


def test_factory_creates_tailnet_dgx_provider() -> None:
    from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
        TailnetDgxWhisperTranscriptionProvider,
    )

    provider = create_transcription_provider(_dgx_cfg())
    assert isinstance(provider, TailnetDgxWhisperTranscriptionProvider)


def test_factory_rejects_experiment_mode_for_dgx() -> None:
    from podcast_scraper.providers.params import TranscriptionParams

    params = TranscriptionParams(model_name="base.en", device="cpu", language="en")
    with pytest.raises(ValueError, match="not supported in experiment mode"):
        create_transcription_provider("tailnet_dgx_whisper", params)


@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.create_transcription_provider")
def test_whisper_provider_initialize_builds_fallback(mock_create: MagicMock) -> None:
    from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
        TailnetDgxWhisperTranscriptionProvider,
    )

    fallback = MagicMock()
    mock_create.return_value = fallback
    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    provider.initialize()
    assert provider._fallback is fallback
    fallback.initialize.assert_called_once()
    mock_create.assert_called_once()
    fb_cfg = mock_create.call_args[0][0]
    assert fb_cfg.transcription_provider == "openai"
