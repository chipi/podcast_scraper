"""Transcription factory tests for Deepgram provider."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from podcast_scraper import config
from podcast_scraper.transcription.factory import create_transcription_provider

pytestmark = pytest.mark.unit


def test_factory_creates_deepgram_provider() -> None:
    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="deepgram",
        deepgram_api_key="dg-test",
    )
    with patch(
        "podcast_scraper.providers.deepgram.deepgram_provider.DeepgramTranscriptionProvider"
    ) as mock_cls:
        mock_cls.return_value.initialize = lambda: None
        provider = create_transcription_provider(cfg)
        mock_cls.assert_called_once_with(cfg)
        assert provider is mock_cls.return_value
