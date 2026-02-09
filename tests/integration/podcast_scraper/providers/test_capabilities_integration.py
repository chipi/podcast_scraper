"""Integration tests for provider capability contract in real scenarios."""

from __future__ import annotations

import unittest

import pytest

from podcast_scraper import config
from podcast_scraper.providers.capabilities import (
    get_provider_capabilities,
    is_local_provider,
)


@pytest.mark.integration
class TestCapabilitiesIntegration(unittest.TestCase):
    """Integration tests for capability contract usage."""

    def test_capability_based_provider_selection(self):
        """Test using capabilities to make provider decisions."""
        cfg = config.Config(
            rss_url="https://example.com",
            transcription_provider="whisper",
        )
        from podcast_scraper.transcription.factory import create_transcription_provider

        provider = create_transcription_provider(cfg)
        caps = get_provider_capabilities(provider)

        # Should be able to make decisions based on capabilities
        self.assertTrue(caps.supports_transcription)
        self.assertTrue(caps.supports_audio_input)
        self.assertFalse(caps.supports_json_mode)  # ML provider doesn't support JSON

    def test_capability_based_local_vs_api_detection(self):
        """Test using capabilities to detect local vs API providers."""
        cfg_ml = config.Config(
            rss_url="https://example.com",
            transcription_provider="whisper",
        )
        cfg_openai = config.Config(
            rss_url="https://example.com",
            transcription_provider="openai",
            openai_api_key="test-key",
        )

        from podcast_scraper.transcription.factory import create_transcription_provider

        ml_provider = create_transcription_provider(cfg_ml)
        openai_provider = create_transcription_provider(cfg_openai)

        # Test is_local_provider helper
        self.assertTrue(is_local_provider(ml_provider))
        self.assertFalse(is_local_provider(openai_provider))

    def test_capability_based_json_mode_detection(self):
        """Test using capabilities to detect JSON mode support."""
        cfg_openai = config.Config(
            rss_url="https://example.com",
            summary_provider="openai",
            openai_api_key="test-key",
        )
        cfg_ml = config.Config(
            rss_url="https://example.com",
            summary_provider="transformers",
        )

        from podcast_scraper.summarization.factory import create_summarization_provider

        openai_provider = create_summarization_provider(cfg_openai)
        ml_provider = create_summarization_provider(cfg_ml)

        openai_caps = get_provider_capabilities(openai_provider)
        ml_caps = get_provider_capabilities(ml_provider)

        # OpenAI should support JSON mode
        self.assertTrue(openai_caps.supports_json_mode)
        # ML provider should not support JSON mode
        self.assertFalse(ml_caps.supports_json_mode)

    def test_capability_based_transcription_support(self):
        """Test using capabilities to check transcription support."""
        cfg_anthropic = config.Config(
            rss_url="https://example.com",
            speaker_detector_provider="anthropic",
            anthropic_api_key="test-key",
        )

        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        anthropic_provider = create_speaker_detector(cfg_anthropic)
        caps = get_provider_capabilities(anthropic_provider)

        # Anthropic doesn't support transcription
        self.assertFalse(caps.supports_transcription)
        # But supports speaker detection and summarization
        self.assertTrue(caps.supports_speaker_detection)
        self.assertTrue(caps.supports_summarization)

    def test_capability_based_max_context_tokens(self):
        """Test using capabilities to check context window size."""
        cfg_gemini = config.Config(
            rss_url="https://example.com",
            summary_provider="gemini",
            gemini_api_key="test-key",
        )
        cfg_openai = config.Config(
            rss_url="https://example.com",
            summary_provider="openai",
            openai_api_key="test-key",
        )

        from podcast_scraper.summarization.factory import create_summarization_provider

        gemini_provider = create_summarization_provider(cfg_gemini)
        openai_provider = create_summarization_provider(cfg_openai)

        gemini_caps = get_provider_capabilities(gemini_provider)
        openai_caps = get_provider_capabilities(openai_provider)

        # Gemini has larger context window
        self.assertGreater(gemini_caps.max_context_tokens, openai_caps.max_context_tokens)


if __name__ == "__main__":
    unittest.main()
