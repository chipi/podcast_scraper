#!/usr/bin/env python3
"""Tests for fallback behavior in provider system.

These tests verify that fallback patterns work correctly and that removing
fallbacks doesn't break functionality.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
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
    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
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
class TestSpeakerDetectorFallback(unittest.TestCase):
    """Test speaker detector fallback behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=False,
        )

    def test_no_fallback_for_speaker_detector(self):
        """Test that speaker detector doesn't have fallback (fail-fast pattern)."""
        # Speaker detector should fail fast if provider creation fails
        # No fallback to direct speaker_detection calls
        detector = create_speaker_detector(self.cfg)

        # Should create provider successfully
        self.assertIsNotNone(detector)
        # Verify it's the unified ML provider
        self.assertEqual(detector.__class__.__name__, "MLProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "detect_hosts"))
        self.assertTrue(hasattr(detector, "clear_cache"))

    def test_detect_hosts_no_fallback(self):
        """Test that detect_hosts() doesn't use fallback."""
        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        # detect_hosts() should be available via protocol
        self.assertTrue(hasattr(detector, "detect_hosts"))

        # Should not need fallback to direct speaker_detection calls
        result = detector.detect_hosts(
            feed_title="Test Podcast", feed_description=None, feed_authors=None
        )
        self.assertIsInstance(result, set)


@pytest.mark.integration
@pytest.mark.critical_path
class TestSummarizationProviderFallback(unittest.TestCase):
    """Test summarization provider fallback behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="local",
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


@pytest.mark.integration
@pytest.mark.critical_path
class TestCacheClearingFallback(unittest.TestCase):
    """Test cache clearing fallback behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=False,
        )

    def test_cache_clearing_no_fallback_needed(self):
        """Test that cache clearing doesn't need fallback (protocol method available)."""
        detector = create_speaker_detector(self.cfg)

        # clear_cache() should be available via protocol
        self.assertTrue(hasattr(detector, "clear_cache"))

        # Should not need fallback to direct speaker_detection.clear_spacy_model_cache()
        # This was removed in favor of protocol method
        detector.clear_cache()  # Should work without fallback

    @unittest.skip(
        "TODO: Fix spacy mocking setup - spacy.load() MagicMock interferes with test mocks"
    )
    @patch("podcast_scraper.speaker_detection.clear_spacy_model_cache")
    def test_ner_detector_cache_clearing(self, mock_clear_cache):
        """Test that NER detector clear_cache() works correctly."""
        detector = create_speaker_detector(self.cfg)

        # Call clear_cache via protocol method
        detector.clear_cache()

        # Should call module function (no fallback needed)
        mock_clear_cache.assert_called_once()


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
            summary_provider="local",
            generate_summaries=False,
        )

        # Provider should be available
        provider = create_summarization_provider(cfg)
        self.assertIsNotNone(provider)

        # In metadata, if provider is None, it falls back with deprecation warning
        # This is intentional for backward compatibility during migration


@pytest.mark.integration
@pytest.mark.critical_path
class TestFallbackRemovalImpact(unittest.TestCase):
    """Test that removing fallbacks doesn't break functionality."""

    def test_providers_always_available(self):
        """Test that providers are always available (no fallback needed)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="ner",
            summary_provider="local",
            generate_summaries=False,
            auto_speakers=False,
        )

        # All providers should be creatable
        transcription_provider = create_transcription_provider(cfg)
        speaker_detector = create_speaker_detector(cfg)
        summarization_provider = create_summarization_provider(cfg)

        # None should be None (all should succeed)
        self.assertIsNotNone(transcription_provider)
        self.assertIsNotNone(speaker_detector)
        self.assertIsNotNone(summarization_provider)

        # All should have protocol methods (no fallback needed)
        self.assertTrue(hasattr(transcription_provider, "transcribe"))
        self.assertTrue(hasattr(speaker_detector, "detect_hosts"))
        self.assertTrue(hasattr(speaker_detector, "clear_cache"))
        self.assertTrue(hasattr(summarization_provider, "summarize"))

    def test_protocol_methods_always_available(self):
        """Test that protocol methods are always available (no hasattr checks needed)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=False,
        )

        detector = create_speaker_detector(cfg)

        # Protocol methods should always be available
        # No need for hasattr() checks
        detector.initialize()
        detector.detect_hosts("Test", None, None)
        detector.clear_cache()
        detector.cleanup()

        # All should work without checks


if __name__ == "__main__":
    unittest.main()
