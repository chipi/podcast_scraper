#!/usr/bin/env python3
"""Integration tests for unified providers (MLProvider and OpenAIProvider).

These tests verify that unified providers work correctly with other components
and can be used together in realistic scenarios.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.transcription.factory import create_transcription_provider


def setUpModule():
    """Restore the REAL ``openai`` SDK for this module's real-client assertions.

    Many unit-test modules mock ``sys.modules['openai']`` at import and deliberately never stop
    it (so ``find_spec('openai')`` stays happy without the SDK). In a single-process run that
    Mock leaks into this module, where ``test_openai_providers_share_same_client`` needs the real
    SDK to read ``client.api_key``. CI's xdist isolates workers so it only bit combined local
    runs. ``openai`` is a core dependency (always installed), so reimporting it is safe.
    """
    import importlib
    import sys

    sys.modules.pop("openai", None)
    importlib.import_module("openai")


@pytest.mark.integration
class TestUnifiedProvidersIntegration(unittest.TestCase):
    """Integration tests for unified providers working with components."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="spacy",
            summary_provider="transformers",
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
        )

    def test_providers_share_same_config(self):
        """Test that providers created from same config share configuration."""
        cfg1 = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Use test default (tiny.en)
        )
        cfg2 = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            whisper_model="small",  # Use different model to test that providers use their respective configs
        )

        provider1 = create_transcription_provider(cfg1)
        provider2 = create_transcription_provider(cfg2)

        # Providers should use their respective configs
        self.assertEqual(provider1.cfg.whisper_model, config.TEST_DEFAULT_WHISPER_MODEL)
        self.assertEqual(provider2.cfg.whisper_model, "small")


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.openai
class TestUnifiedOpenAIProvidersIntegration(unittest.TestCase):
    """Integration tests for unified OpenAI providers."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            speaker_detector_provider="openai",
            summary_provider="openai",
            openai_api_key="sk-test123",
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
        )

    def test_openai_providers_support_custom_base_url(self):
        """Test that OpenAI providers support custom base_url for E2E testing."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
            openai_api_base="http://localhost:8000/v1",
            transcribe_missing=True,
        )

        provider = create_transcription_provider(cfg)

        # Client should be created with custom base_url
        self.assertIsNotNone(provider.client)


@pytest.mark.integration
class TestProviderSwitchingIntegration(unittest.TestCase):
    """Integration tests for switching between provider types."""

    def test_switch_from_ml_to_openai(self):
        """Test switching from ML providers to OpenAI providers."""
        ml_cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="spacy",
            summary_provider="transformers",
        )

        openai_cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            speaker_detector_provider="openai",
            summary_provider="openai",
            openai_api_key="sk-test123",
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
        )

        ml_transcription = create_transcription_provider(ml_cfg)
        openai_transcription = create_transcription_provider(openai_cfg)

        self.assertEqual(ml_transcription.__class__.__name__, "MLProvider")
        self.assertEqual(openai_transcription.__class__.__name__, "OpenAIProvider")

        # Verify protocol compliance for both
        self.assertTrue(hasattr(ml_transcription, "transcribe"))
        self.assertTrue(hasattr(openai_transcription, "transcribe"))


@pytest.mark.integration
class TestProviderErrorHandlingIntegration(unittest.TestCase):
    """Integration tests for provider error handling."""

    def test_provider_initialization_failure_handling(self):
        """Test that provider initialization failures are handled gracefully.

        MLProvider.initialize() is resilient - it logs warnings but doesn't raise,
        allowing other components to initialize even if one fails.
        """
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=True,
        )

        provider = create_transcription_provider(cfg)

        # Mock initialization failure
        with patch(
            "podcast_scraper.providers.ml.ml_provider._import_third_party_whisper"
        ) as mock_import:
            mock_import.side_effect = ImportError("Whisper not available")

            # initialize() should not raise - it logs warnings and continues
            provider.initialize()
            # Verify Whisper is not initialized (but provider is still usable)
            if hasattr(provider, "_whisper_initialized"):
                self.assertFalse(provider._whisper_initialized)

    def test_provider_method_failure_handling(self):
        """Test that provider method failures are handled gracefully."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=True,
        )

        provider = create_transcription_provider(cfg)

        # Initialize provider
        with patch(
            "podcast_scraper.providers.ml.ml_provider._import_third_party_whisper"
        ) as mock_import:

            mock_model = Mock()
            mock_model.device.type = "cpu"
            mock_whisper_lib = Mock()
            mock_whisper_lib.load_model.return_value = mock_model
            mock_import.return_value = mock_whisper_lib

            provider.initialize()

        # Mock transcription failure
        provider._whisper_model.transcribe.side_effect = Exception("Transcription failed")

        # The exception propagates from _transcribe_with_whisper
        with self.assertRaises(Exception) as context:
            provider.transcribe("/path/to/audio.mp3")
        self.assertIn("Transcription failed", str(context.exception))
