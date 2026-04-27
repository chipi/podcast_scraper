"""Unit tests for SummLlama provider (#571)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from podcast_scraper.config import Config

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm]/[server]
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


def _make_config(**overrides):
    defaults = {
        "rss_url": "https://example.com/feed.xml",
        "summary_provider": "summllama",
        "generate_summaries": True,
        "generate_metadata": True,
    }
    defaults.update(overrides)
    return Config(**defaults)


class TestSummLlamaProvider:
    """Unit tests for SummLlamaProvider (no real model loading)."""

    def test_config_accepts_summllama(self):
        cfg = _make_config()
        assert cfg.summary_provider == "summllama"

    def test_factory_creates_provider(self):
        from podcast_scraper.summarization.factory import create_summarization_provider

        cfg = _make_config()
        provider = create_summarization_provider(cfg)
        assert type(provider).__name__ == "SummLlamaProvider"
        assert provider._model_id == "DISLab/SummLlama3.2-3B"
        assert provider._style == "bullets"

    def test_default_style_is_bullets(self):
        from podcast_scraper.providers.ml.summllama_provider import SummLlamaProvider

        provider = SummLlamaProvider(_make_config())
        assert provider._style == "bullets"

    def test_not_initialized_by_default(self):
        from podcast_scraper.providers.ml.summllama_provider import SummLlamaProvider

        provider = SummLlamaProvider(_make_config())
        assert not provider.is_initialized

    def test_cleanup(self):
        from podcast_scraper.providers.ml.summllama_provider import SummLlamaProvider

        provider = SummLlamaProvider(_make_config())
        provider._model = MagicMock()
        provider._tokenizer = MagicMock()
        provider._initialized = True
        provider.cleanup()
        assert not provider.is_initialized
        assert provider._model is None
        assert provider._tokenizer is None
