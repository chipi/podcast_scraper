#!/usr/bin/env python3
"""Extended protocol compliance tests for providers.

These tests verify that all providers correctly implement ALL protocol methods,
including initialize(), cleanup(), clear_cache(), detect_hosts(), and transcribe_with_segments().
"""

import inspect
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider


@pytest.mark.integration
class TestProtocolLifecycleMethods(unittest.TestCase):
    """Test initialize() and cleanup() methods for all providers."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="ner",
            summary_provider="local",
            generate_summaries=False,
            auto_speakers=False,
        )

    def test_transcription_provider_has_initialize(self):
        """Test that TranscriptionProvider has initialize() method."""
        provider = create_transcription_provider(self.cfg)
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(callable(getattr(provider, "initialize")))

    def test_transcription_provider_has_cleanup(self):
        """Test that TranscriptionProvider has cleanup() method."""
        provider = create_transcription_provider(self.cfg)
        self.assertTrue(hasattr(provider, "cleanup"))
        self.assertTrue(callable(getattr(provider, "cleanup")))

    def test_speaker_detector_has_initialize(self):
        """Test that SpeakerDetector has initialize() method."""
        detector = create_speaker_detector(self.cfg)
        self.assertTrue(hasattr(detector, "initialize"))
        self.assertTrue(callable(getattr(detector, "initialize")))

    def test_speaker_detector_has_cleanup(self):
        """Test that SpeakerDetector has cleanup() method."""
        detector = create_speaker_detector(self.cfg)
        self.assertTrue(hasattr(detector, "cleanup"))
        self.assertTrue(callable(getattr(detector, "cleanup")))

    def test_summarization_provider_has_initialize(self):
        """Test that SummarizationProvider has initialize() method."""
        provider = create_summarization_provider(self.cfg)
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(callable(getattr(provider, "initialize")))

    def test_summarization_provider_has_cleanup(self):
        """Test that SummarizationProvider has cleanup() method."""
        provider = create_summarization_provider(self.cfg)
        self.assertTrue(hasattr(provider, "cleanup"))
        self.assertTrue(callable(getattr(provider, "cleanup")))

    @patch("podcast_scraper.transcription.whisper_provider.whisper_integration.load_whisper_model")
    def test_transcription_provider_initialize_cleanup_cycle(self, mock_load_model):
        """Test that transcription provider can be initialized and cleaned up."""
        mock_load_model.return_value = Mock()
        provider = create_transcription_provider(self.cfg)

        # Should not be initialized initially
        if hasattr(provider, "is_initialized"):
            self.assertFalse(provider.is_initialized)

        # Initialize
        provider.initialize()
        if hasattr(provider, "is_initialized"):
            self.assertTrue(provider.is_initialized)

        # Cleanup
        provider.cleanup()
        if hasattr(provider, "is_initialized"):
            self.assertFalse(provider.is_initialized)

    def test_speaker_detector_initialize_cleanup_cycle(self):
        """Test that speaker detector can be initialized and cleaned up."""
        detector = create_speaker_detector(self.cfg)

        # Initialize (may be no-op if auto_speakers is False)
        detector.initialize()

        # Cleanup should always work
        detector.cleanup()

        # Verify cleanup worked
        if hasattr(detector, "is_initialized"):
            self.assertFalse(detector.is_initialized)

    @patch("podcast_scraper.summarization.local_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.SummaryModel")
    def test_summarization_provider_initialize_cleanup_cycle(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test that summarization provider can be initialized and cleaned up."""
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"
        mock_summary_model.return_value = Mock()

        # Use config with generate_summaries=True to ensure initialization
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="local",
            generate_metadata=True,
            generate_summaries=True,
        )
        provider = create_summarization_provider(cfg)

        # Initialize
        provider.initialize()
        if hasattr(provider, "is_initialized"):
            self.assertTrue(provider.is_initialized)

        # Cleanup
        provider.cleanup()
        if hasattr(provider, "is_initialized"):
            self.assertFalse(provider.is_initialized)


@pytest.mark.integration
class TestSpeakerDetectorDetectHosts(unittest.TestCase):
    """Test detect_hosts() method for SpeakerDetector protocol."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=False,
        )

    def test_detector_has_detect_hosts_method(self):
        """Test that detector implements detect_hosts() method."""
        detector = create_speaker_detector(self.cfg)
        self.assertTrue(hasattr(detector, "detect_hosts"))
        self.assertTrue(callable(getattr(detector, "detect_hosts")))

    def test_detect_hosts_method_signature(self):
        """Test that detect_hosts() has correct signature."""
        detector = create_speaker_detector(self.cfg)
        sig = inspect.signature(detector.detect_hosts)
        params = list(sig.parameters.keys())

        # Should have feed_title, feed_description, feed_authors
        self.assertIn("feed_title", params)
        self.assertIn("feed_description", params)
        self.assertIn("feed_authors", params)

    def test_detect_hosts_returns_set(self):
        """Test that detect_hosts() returns a Set[str]."""
        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        # Call with minimal inputs
        result = detector.detect_hosts(
            feed_title="Test Podcast", feed_description=None, feed_authors=None
        )

        # Should return a set
        self.assertIsInstance(result, set)
        # All items should be strings
        for item in result:
            self.assertIsInstance(item, str)

    def test_detect_hosts_with_authors(self):
        """Test detect_hosts() with feed_authors (preferred source)."""
        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        # Test with authors (should prefer this source)
        result = detector.detect_hosts(
            feed_title="Test Podcast",
            feed_description="A test podcast",
            feed_authors=["John Doe", "Jane Smith"],
        )

        # Should return authors as hosts
        self.assertIsInstance(result, set)
        self.assertIn("John Doe", result)
        self.assertIn("Jane Smith", result)


@pytest.mark.integration
class TestTranscriptionProviderTranscribeWithSegments(unittest.TestCase):
    """Test transcribe_with_segments() method for TranscriptionProvider protocol."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )

    def test_provider_has_transcribe_with_segments_method(self):
        """Test that provider implements transcribe_with_segments() method."""
        provider = create_transcription_provider(self.cfg)
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))
        self.assertTrue(callable(getattr(provider, "transcribe_with_segments")))

    def test_transcribe_with_segments_method_signature(self):
        """Test that transcribe_with_segments() has correct signature."""
        provider = create_transcription_provider(self.cfg)
        sig = inspect.signature(provider.transcribe_with_segments)
        params = list(sig.parameters.keys())

        # Should have audio_path and optional language
        self.assertIn("audio_path", params)
        self.assertIn("language", params)

    def test_transcribe_with_segments_return_type(self):
        """Test that transcribe_with_segments() returns tuple[dict, float]."""
        provider = create_transcription_provider(self.cfg)
        sig = inspect.signature(provider.transcribe_with_segments)
        return_annotation = str(sig.return_annotation)

        # Should return a tuple
        self.assertIn("tuple", return_annotation.lower())

    @patch("podcast_scraper.transcription.whisper_provider.whisper_integration.load_whisper_model")
    @patch(
        "podcast_scraper.transcription.whisper_provider.whisper_integration.transcribe_with_whisper"
    )
    def test_transcribe_with_segments_returns_expected_structure(
        self, mock_transcribe, mock_load_model
    ):
        """Test that transcribe_with_segments() returns expected structure."""
        # Mock transcription result
        mock_result = {
            "text": "Hello world",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello"},
                {"start": 1.0, "end": 2.0, "text": "world"},
            ],
        }
        mock_transcribe.return_value = (mock_result, 1.5)
        mock_load_model.return_value = Mock()

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # Call transcribe_with_segments
        result_dict, elapsed = provider.transcribe_with_segments("test.mp3")

        # Should return tuple
        self.assertIsInstance(result_dict, dict)
        self.assertIsInstance(elapsed, float)

        # Result dict should have text and segments
        self.assertIn("text", result_dict)
        self.assertIn("segments", result_dict)
        self.assertIsInstance(result_dict["segments"], list)


@pytest.mark.integration
class TestSpeakerDetectorClearCache(unittest.TestCase):
    """Test clear_cache() method for SpeakerDetector protocol."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ner",
            auto_speakers=False,
        )

    def test_detector_has_clear_cache_method(self):
        """Test that detector implements clear_cache() method."""
        detector = create_speaker_detector(self.cfg)
        self.assertTrue(hasattr(detector, "clear_cache"))
        self.assertTrue(callable(getattr(detector, "clear_cache")))

    def test_clear_cache_method_signature(self):
        """Test that clear_cache() has correct signature (no parameters)."""
        detector = create_speaker_detector(self.cfg)
        sig = inspect.signature(detector.clear_cache)
        params = list(sig.parameters.keys())

        # Should have no required parameters (only self)
        # Parameters should be empty or only have optional params
        self.assertEqual(len(params), 0)

    @patch("podcast_scraper.speaker_detection.clear_spacy_model_cache")
    def test_ner_detector_clear_cache_calls_module_function(self, mock_clear_cache):
        """Test that NER detector clear_cache() calls module function."""
        detector = create_speaker_detector(self.cfg)

        # Call clear_cache
        detector.clear_cache()

        # Should call module function
        mock_clear_cache.assert_called_once()

    def test_openai_detector_clear_cache_is_noop(self):
        """Test that OpenAI detector clear_cache() is a no-op."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        # Should not raise exception
        detector.clear_cache()

        # Can be called multiple times
        detector.clear_cache()
        detector.clear_cache()


@pytest.mark.integration
class TestProtocolMethodCompleteness(unittest.TestCase):
    """Test that all protocol methods are implemented by all providers."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="ner",
            summary_provider="local",
            generate_summaries=False,
            auto_speakers=False,
        )

    def test_transcription_provider_all_methods(self):
        """Test that TranscriptionProvider implements all required methods."""
        provider = create_transcription_provider(self.cfg)

        required_methods = [
            "initialize",
            "cleanup",
            "transcribe",
            "transcribe_with_segments",
        ]

        for method_name in required_methods:
            self.assertTrue(
                hasattr(provider, method_name),
                f"TranscriptionProvider missing method: {method_name}",
            )
            self.assertTrue(
                callable(getattr(provider, method_name)),
                f"TranscriptionProvider method not callable: {method_name}",
            )

    def test_speaker_detector_all_methods(self):
        """Test that SpeakerDetector implements all required methods."""
        detector = create_speaker_detector(self.cfg)

        required_methods = [
            "initialize",
            "cleanup",
            "clear_cache",
            "detect_hosts",
            "detect_speakers",
            "analyze_patterns",
        ]

        for method_name in required_methods:
            self.assertTrue(
                hasattr(detector, method_name),
                f"SpeakerDetector missing method: {method_name}",
            )
            self.assertTrue(
                callable(getattr(detector, method_name)),
                f"SpeakerDetector method not callable: {method_name}",
            )

    def test_summarization_provider_all_methods(self):
        """Test that SummarizationProvider implements all required methods."""
        provider = create_summarization_provider(self.cfg)

        required_methods = [
            "initialize",
            "cleanup",
            "summarize",
        ]

        for method_name in required_methods:
            self.assertTrue(
                hasattr(provider, method_name),
                f"SummarizationProvider missing method: {method_name}",
            )
            self.assertTrue(
                callable(getattr(provider, method_name)),
                f"SummarizationProvider method not callable: {method_name}",
            )


if __name__ == "__main__":
    unittest.main()
