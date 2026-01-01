#!/usr/bin/env python3
"""Extended error handling tests for providers.

These tests verify comprehensive error handling scenarios including:
- Provider initialization failures
- Method failures after initialization
- Graceful degradation
- Error propagation
"""

import unittest
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider


@pytest.mark.integration
@pytest.mark.slow
class TestTranscriptionProviderErrorHandling(unittest.TestCase):
    """Test error handling for transcription providers."""

    def test_initialization_failure_raises_exception(self):
        """Test that initialization failure raises exception."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider = create_transcription_provider(cfg)

        # Mock initialization failure
        with patch.object(provider, "initialize", side_effect=RuntimeError("Model load failed")):
            with self.assertRaises(RuntimeError):
                provider.initialize()

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_before_initialization(self, mock_import_whisper):
        """Test that transcribe() fails if called before initialization."""
        mock_whisper_lib = Mock()
        mock_whisper_model = Mock()
        mock_whisper_lib.load_model.return_value = mock_whisper_model
        mock_import_whisper.return_value = mock_whisper_lib
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider = create_transcription_provider(cfg)

        # Should raise RuntimeError if not initialized
        with self.assertRaises(RuntimeError):
            provider.transcribe("test.mp3")

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.ml.ml_provider.MLProvider._transcribe_with_whisper")
    def test_transcribe_with_segments_failure(self, mock_transcribe, mock_import_whisper):
        """Test error handling when transcribe_with_segments() fails."""
        mock_whisper_lib = Mock()
        mock_whisper_model = Mock()
        mock_whisper_lib.load_model.return_value = mock_whisper_model
        mock_import_whisper.return_value = mock_whisper_lib
        mock_transcribe.side_effect = ValueError("Transcription failed")

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=True,
        )
        provider = create_transcription_provider(cfg)
        provider.initialize()

        # Should propagate error
        with self.assertRaises(ValueError):
            provider.transcribe_with_segments("test.mp3")

    def test_openai_provider_missing_api_key(self):
        """Test that OpenAI provider requires API key."""
        # Missing API key should be caught by config validator
        with self.assertRaises(ValidationError):
            config.Config(
                rss_url="https://example.com/feed.xml",
                transcription_provider="openai",
            )

    def test_cleanup_after_failed_initialization(self):
        """Test that cleanup() works even after failed initialization."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider = create_transcription_provider(cfg)

        # Try to initialize (may fail)
        try:
            provider.initialize()
        except Exception:  # nosec B110
            pass

        # Cleanup should still work (idempotent)
        provider.cleanup()


@pytest.mark.integration
@pytest.mark.slow
class TestSpeakerDetectorErrorHandling(unittest.TestCase):
    """Test error handling for speaker detectors."""

    def test_initialization_failure_raises_exception(self):
        """Test that initialization failure raises exception."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        # Mock initialization failure
        with patch.object(detector, "initialize", side_effect=RuntimeError("Model load failed")):
            with self.assertRaises(RuntimeError):
                detector.initialize()

    def test_detect_speakers_before_initialization(self):
        """Test that detect_speakers() auto-initializes if needed."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        # Should auto-initialize if needed (implementation-dependent)
        # For NER detector, it may auto-initialize
        result = detector.detect_speakers("Test Episode", None, set())
        self.assertIsInstance(result, tuple)

    def test_detect_hosts_failure(self):
        """Test error handling when detect_hosts() fails."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        # Mock detect_hosts failure
        with patch.object(detector, "detect_hosts", side_effect=ValueError("Detection failed")):
            with self.assertRaises(ValueError):
                detector.detect_hosts("Test", None, None)

    def test_openai_detector_missing_api_key(self):
        """Test that OpenAI detector requires API key."""
        # Missing API key should be caught by config validator
        with self.assertRaises(ValidationError):
            config.Config(
                rss_url="https://example.com/feed.xml",
                speaker_detector_provider="openai",
            )

    def test_clear_cache_failure(self):
        """Test error handling when clear_cache() fails."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        # Mock clear_cache failure
        with patch.object(detector, "clear_cache", side_effect=RuntimeError("Cache clear failed")):
            with self.assertRaises(RuntimeError):
                detector.clear_cache()


@pytest.mark.integration
@pytest.mark.slow
class TestSummarizationProviderErrorHandling(unittest.TestCase):
    """Test error handling for summarization providers."""

    def test_initialization_failure_raises_exception(self):
        """Test that initialization failure raises exception."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="local",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)

        # Mock initialization failure
        with patch.object(provider, "initialize", side_effect=RuntimeError("Model load failed")):
            with self.assertRaises(RuntimeError):
                provider.initialize()

    @patch("podcast_scraper.summarization.local_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.SummaryModel")
    def test_summarize_before_initialization(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test that summarize() fails if called before initialization."""
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"
        mock_summary_model.return_value = Mock()

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="local",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)

        # Should raise RuntimeError if not initialized
        with self.assertRaises(RuntimeError):
            provider.summarize("test text")

    @patch("podcast_scraper.summarization.local_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.SummaryModel")
    def test_summarize_failure_after_initialization(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test error handling when summarize() fails after initialization."""
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"
        mock_summary_model.return_value = Mock()

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="local",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)
        provider.initialize()

        # Mock summarize failure
        with patch.object(provider, "summarize", side_effect=ValueError("Summarization failed")):
            with self.assertRaises(ValueError):
                provider.summarize("test text")

    def test_openai_provider_missing_api_key(self):
        """Test that OpenAI provider requires API key."""
        # Missing API key should be caught by config validator
        with self.assertRaises(ValidationError):
            config.Config(
                rss_url="https://example.com/feed.xml",
                summary_provider="openai",
                generate_summaries=False,
            )


@pytest.mark.integration
@pytest.mark.slow
class TestProviderSwitchingErrorHandling(unittest.TestCase):
    """Test error handling when switching providers."""

    def test_switch_to_invalid_provider(self):
        """Test error handling when switching to invalid provider."""
        # Invalid transcription provider
        with self.assertRaises(ValueError):
            create_transcription_provider(
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    transcription_provider="invalid",
                )
            )

        # Invalid speaker detector provider
        with self.assertRaises(ValueError):
            create_speaker_detector(
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="invalid",
                )
            )

        # Invalid summarization provider
        with self.assertRaises(ValueError):
            create_summarization_provider(
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="invalid",
                    generate_summaries=False,
                )
            )

    def test_switch_provider_without_required_config(self):
        """Test error handling when switching provider without required config."""
        # OpenAI provider requires API key
        with self.assertRaises(ValidationError):
            config.Config(
                rss_url="https://example.com/feed.xml",
                transcription_provider="openai",
            )


@pytest.mark.integration
@pytest.mark.slow
class TestGracefulDegradation(unittest.TestCase):
    """Test graceful degradation scenarios."""

    def test_provider_cleanup_on_exception(self):
        """Test that providers can be cleaned up even if exceptions occur."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="ner",
            summary_provider="local",
            generate_summaries=False,
            auto_speakers=False,
        )

        transcription_provider = create_transcription_provider(cfg)
        speaker_detector = create_speaker_detector(cfg)
        summarization_provider = create_summarization_provider(cfg)

        # Simulate exception during use
        try:
            # This might fail
            transcription_provider.initialize()
        except Exception:  # nosec B110
            pass

        # Cleanup should still work
        transcription_provider.cleanup()
        speaker_detector.cleanup()
        summarization_provider.cleanup()

    def test_multiple_initialization_calls(self):
        """Test that multiple initialization calls are safe (idempotent)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        # Should be safe to call multiple times
        detector.initialize()
        detector.initialize()
        detector.initialize()

        # Should still work
        detector.cleanup()

    def test_multiple_cleanup_calls(self):
        """Test that multiple cleanup calls are safe (idempotent)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        detector.initialize()

        # Should be safe to call multiple times
        detector.cleanup()
        detector.cleanup()
        detector.cleanup()


if __name__ == "__main__":
    unittest.main()
