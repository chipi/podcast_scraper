#!/usr/bin/env python3
"""Lifecycle and edge case tests for GeminiProvider.

These tests focus on provider lifecycle management, error recovery,
and edge cases that strengthen the refactor.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock google.generativeai before importing modules that require it
mock_google = MagicMock()
mock_genai_module = MagicMock()
mock_genai_module.configure = Mock()
mock_genai_module.GenerativeModel = Mock()
mock_api_core = MagicMock()
mock_api_core.exceptions = MagicMock()
_patch_google = patch.dict(
    "sys.modules",
    {
        "google": mock_google,
        "google.generativeai": mock_genai_module,
        "google.api_core": mock_api_core,
    },
)
_patch_google.start()

from podcast_scraper import config
from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider


@pytest.mark.unit
class TestGeminiProviderLifecycle(unittest.TestCase):
    """Test provider lifecycle edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="gemini",
            speaker_detector_provider="gemini",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_multiple_initialize_calls_idempotent(self, mock_genai):
        """Test that multiple initialize() calls are idempotent."""
        mock_genai.configure = Mock()

        # Create config with at least one capability enabled
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            gemini_api_key=self.cfg.gemini_api_key,
            transcription_provider="gemini",
            transcribe_missing=True,  # Enable transcription
        )
        provider = GeminiProvider(cfg)

        # First initialization
        provider.initialize()
        self.assertTrue(
            provider._transcription_initialized
            or provider._speaker_detection_initialized
            or provider._summarization_initialized
        )

        # Second initialization should not cause issues
        provider.initialize()
        # State should remain consistent
        self.assertTrue(provider.is_initialized)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_cleanup_when_not_initialized(self, mock_genai):
        """Test that cleanup() works even when not initialized."""
        mock_genai.configure = Mock()

        provider = GeminiProvider(self.cfg)

        # Cleanup should not raise
        provider.cleanup()
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_partial_initialization_cleanup(self, mock_genai):
        """Test cleanup when only some capabilities are initialized."""
        mock_genai.configure = Mock()

        # Initialize only transcription
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            gemini_api_key=self.cfg.gemini_api_key,
            transcription_provider="gemini",
            transcribe_missing=True,
            auto_speakers=False,
        )
        provider = GeminiProvider(cfg)
        provider.initialize()
        self.assertTrue(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)

        # Cleanup should work
        provider.cleanup()
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_requires_separate_instances_false(self, mock_genai):
        """Test that GeminiProvider does not require separate instances."""
        mock_genai.configure = Mock()

        provider = GeminiProvider(self.cfg)

        # GeminiProvider should not require separate instances (API clients are thread-safe)
        self.assertFalse(provider._requires_separate_instances)

        # Verify attribute exists and is accessible via getattr (workflow pattern)
        requires_separate = getattr(provider, "_requires_separate_instances", False)
        self.assertFalse(requires_separate)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_requires_separate_instances_accessible_via_getattr(self, mock_genai):
        """Test that _requires_separate_instances is accessible via getattr (workflow pattern)."""
        mock_genai.configure = Mock()

        provider = GeminiProvider(self.cfg)

        # Workflow uses getattr() to check this attribute
        requires_separate = getattr(provider, "_requires_separate_instances", False)
        self.assertFalse(requires_separate)

        # Should default to False if attribute doesn't exist (defensive)
        fake_provider = Mock(spec=[])
        default_value = getattr(fake_provider, "_requires_separate_instances", False)
        self.assertFalse(default_value)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_state_after_cleanup(self, mock_genai):
        """Test that state is clean after cleanup."""
        mock_genai.configure = Mock()

        # Initialize all capabilities
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            gemini_api_key=self.cfg.gemini_api_key,
            transcription_provider="gemini",
            speaker_detector_provider="gemini",
            summary_provider="gemini",
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
        )
        provider = GeminiProvider(cfg)
        provider.initialize()
        self.assertTrue(provider.is_initialized)

        # Cleanup
        provider.cleanup()

        # State should be clean
        self.assertFalse(provider.is_initialized)
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
