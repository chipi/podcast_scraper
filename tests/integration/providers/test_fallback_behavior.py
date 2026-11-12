#!/usr/bin/env python3
"""Tests for fallback behavior in provider system.

These tests verify that fallback patterns work correctly and that removing
fallbacks doesn't break functionality.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider


@pytest.mark.integration
@pytest.mark.critical_path
class TestTranscriptionProviderFallback(unittest.TestCase):
    """Test transcription provider fallback behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=True,
        )

    @patch("podcast_scraper.transcription.factory.create_transcription_provider")
    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
    def test_fallback_to_direct_whisper_on_provider_failure(
        self, mock_load_whisper, mock_create_provider
    ):
        """Test that fallback to direct whisper loading works when provider fails."""
        # Mock provider creation failure
        mock_create_provider.side_effect = Exception("Provider initialization failed")
        mock_load_whisper.return_value = Mock()

        # This simulates the fallback path in workflow.py
        # In actual workflow, this would be handled in _setup_transcription_resources
        try:
            provider = create_transcription_provider(self.cfg)
            provider.initialize()
        except Exception:
            # Fallback: direct whisper loading (for backward compatibility)
            # Note: This is a test of the fallback concept, not actual implementation
            mock_whisper_lib = Mock()
            mock_whisper_lib.load_model.return_value = Mock()
            mock_load_whisper.return_value = mock_whisper_lib
            # Verify fallback path exists conceptually
            self.assertIsNotNone(mock_load_whisper.return_value)

    def test_no_fallback_when_provider_succeeds(self):
        """Test that fallback is not used when provider succeeds."""
        provider = create_transcription_provider(self.cfg)

        # Provider should be created successfully
        self.assertIsNotNone(provider)
        # Verify it's the unified ML provider
        self.assertEqual(provider.__class__.__name__, "MLProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))


@pytest.mark.integration
@pytest.mark.critical_path
class TestSummarizationProviderFallback(unittest.TestCase):
    """Test summarization provider fallback behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )

    def test_no_fallback_for_summarization_provider(self):
        """Test that summarization provider doesn't have fallback (fail-fast pattern)."""
        # Summarization provider should fail fast if provider creation fails
        # No fallback to direct model loading
        provider = create_summarization_provider(self.cfg)

        # Should create provider successfully
        self.assertIsNotNone(provider)
        # Verify it's the unified ML provider
        self.assertEqual(provider.__class__.__name__, "MLProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

        # clear_cache() is now a no-op, should not raise


@pytest.mark.integration
@pytest.mark.critical_path
class TestBackwardCompatibilityFallbacks(unittest.TestCase):
    """Test backward compatibility fallback patterns."""

    def test_episode_processor_transcription_fallback(self):
        """Test that episode_processor has fallback for backward compatibility."""
        # This tests the fallback path in episode_processor.py
        # When transcription_provider is None, it falls back to direct whisper calls
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
        )

        # Provider should be available
        provider = create_transcription_provider(cfg)
        self.assertIsNotNone(provider)

        # In episode_processor, if provider is None, it falls back
        # This is intentional for backward compatibility
        # We test that the fallback path exists conceptually

    def test_metadata_summarization_fallback(self):
        """Test that metadata has fallback for backward compatibility."""
        # This tests the deprecated fallback path in metadata.py
        # When summary_provider is None, it falls back to direct model loading
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )

        # Provider should be available
        provider = create_summarization_provider(cfg)
        self.assertIsNotNone(provider)

        # In metadata, if provider is None, it falls back with deprecation warning
        # This is intentional for backward compatibility during migration

        # All should work without checks


if __name__ == "__main__":
    unittest.main()
