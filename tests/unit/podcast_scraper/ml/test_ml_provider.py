#!/usr/bin/env python3
"""Standalone unit tests for unified ML provider.

These tests verify that MLProvider correctly implements all three protocols
(TranscriptionProvider, SpeakerDetector, SummarizationProvider) using
Whisper, spaCy, and Transformers.

These are standalone provider tests - they test the provider itself,
not its integration with the app.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.ml.ml_provider import MLProvider


@pytest.mark.unit
class TestMLProviderStandalone(unittest.TestCase):
    """Standalone tests for MLProvider - testing the provider itself."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="spacy",
            summary_provider="transformers",
            transcribe_missing=False,  # Disable to avoid loading models
            auto_speakers=False,  # Disable to avoid loading spaCy
            generate_summaries=False,  # Disable to avoid loading Transformers
        )

    def test_provider_creation(self):
        """Test that MLProvider can be created."""
        provider = MLProvider(self.cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "MLProvider")

    def test_provider_implements_all_protocols(self):
        """Test that MLProvider implements all three protocols."""
        provider = MLProvider(self.cfg)

        # TranscriptionProvider protocol
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

        # SpeakerDetector protocol
        self.assertTrue(hasattr(provider, "detect_speakers"))
        self.assertTrue(hasattr(provider, "detect_hosts"))
        self.assertTrue(hasattr(provider, "analyze_patterns"))
        self.assertTrue(hasattr(provider, "clear_cache"))

        # SummarizationProvider protocol
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

    def test_provider_initialization_state(self):
        """Test that provider tracks initialization state for each capability."""
        provider = MLProvider(self.cfg)

        # Initially not initialized
        self.assertFalse(provider._whisper_initialized)
        self.assertFalse(provider._spacy_initialized)
        self.assertFalse(provider._transformers_initialized)
        self.assertFalse(provider.is_initialized)

    def test_provider_requires_separate_instances(self):
        """Test that provider marks itself as requiring separate instances."""
        provider = MLProvider(self.cfg)
        self.assertTrue(provider._requires_separate_instances)

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_whisper_initialization(self, mock_import_whisper):
        """Test that Whisper can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            whisper_model=self.cfg.whisper_model,
            auto_speakers=False,  # Disable to avoid loading spaCy
        )
        provider = MLProvider(cfg)

        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib

        provider.initialize()

        self.assertTrue(provider._whisper_initialized)
        self.assertEqual(provider._whisper_model, mock_model)
        # Other capabilities should not be initialized
        self.assertFalse(provider._spacy_initialized)
        self.assertFalse(provider._transformers_initialized)

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_spacy_initialization(self, mock_get_ner):
        """Test that spaCy can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            auto_speakers=True,
            whisper_model=self.cfg.whisper_model,
        )
        provider = MLProvider(cfg)

        mock_nlp = Mock()
        mock_get_ner.return_value = mock_nlp

        provider.initialize()

        self.assertTrue(provider._spacy_initialized)
        self.assertEqual(provider._spacy_nlp, mock_nlp)
        # Other capabilities should not be initialized
        self.assertFalse(provider._whisper_initialized)
        self.assertFalse(provider._transformers_initialized)

    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_transformers_initialization(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test that Transformers can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
            whisper_model=self.cfg.whisper_model,
            auto_speakers=False,  # Disable to avoid loading spaCy
        )
        provider = MLProvider(cfg)

        mock_model = Mock()
        mock_model.model_name = "facebook/bart-base"
        mock_model.device = "cpu"
        mock_summary_model.return_value = mock_model
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"

        provider.initialize()

        self.assertTrue(provider._transformers_initialized)
        self.assertEqual(provider._map_model, mock_model)
        # Other capabilities should not be initialized
        self.assertFalse(provider._whisper_initialized)
        self.assertFalse(provider._spacy_initialized)

    def test_unified_initialization(self):
        """Test that all capabilities can be initialized together."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            transcription_provider=self.cfg.transcription_provider,
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
            whisper_model=self.cfg.whisper_model,
        )

        with (
            patch(
                "podcast_scraper.ml.ml_provider._import_third_party_whisper"
            ) as mock_import_whisper,
            patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model") as mock_get_ner,
            patch(
                "podcast_scraper.ml.ml_provider.summarizer.select_summary_model"
            ) as mock_select_map,
            patch(
                "podcast_scraper.ml.ml_provider.summarizer.select_reduce_model"
            ) as mock_select_reduce,
            patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel") as mock_summary_model,
        ):

            mock_model = Mock()
            mock_model.device.type = "cpu"
            mock_whisper_lib = Mock()
            mock_whisper_lib.load_model.return_value = mock_model
            mock_import_whisper.return_value = mock_whisper_lib

            mock_nlp = Mock()
            mock_get_ner.return_value = mock_nlp

            mock_transformers_model = Mock()
            mock_transformers_model.model_name = "facebook/bart-base"
            mock_transformers_model.device = "cpu"
            mock_summary_model.return_value = mock_transformers_model
            mock_select_map.return_value = "facebook/bart-base"
            mock_select_reduce.return_value = "facebook/bart-base"

            provider = MLProvider(cfg)
            provider.initialize()

            # All should be initialized
            self.assertTrue(provider._whisper_initialized)
            self.assertTrue(provider._spacy_initialized)
            self.assertTrue(provider._transformers_initialized)
            self.assertTrue(provider.is_initialized)

    def test_cleanup_releases_all_resources(self):
        """Test that cleanup releases all resources."""
        provider = MLProvider(self.cfg)
        provider._whisper_initialized = True
        provider._spacy_initialized = True
        provider._transformers_initialized = True

        provider.cleanup()

        self.assertFalse(provider._whisper_initialized)
        self.assertFalse(provider._spacy_initialized)
        self.assertFalse(provider._transformers_initialized)
        self.assertFalse(provider.is_initialized)

    def test_backward_compatibility_properties(self):
        """Test that backward compatibility properties exist."""
        provider = MLProvider(self.cfg)

        # Whisper properties
        self.assertTrue(hasattr(provider, "model"))
        self.assertTrue(hasattr(provider, "is_initialized"))

        # spaCy properties
        self.assertTrue(hasattr(provider, "nlp"))
        self.assertTrue(hasattr(provider, "heuristics"))

        # Transformers properties
        self.assertTrue(hasattr(provider, "map_model"))
        self.assertTrue(hasattr(provider, "reduce_model"))


@pytest.mark.unit
class TestMLProviderTranscription(unittest.TestCase):
    """Tests for MLProvider transcription methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            whisper_model="tiny",
        )

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    def test_transcribe_success(self, mock_progress, mock_import_whisper):
        """Test successful transcription."""
        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_model.transcribe.return_value = {"text": "Hello world", "segments": []}
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib
        mock_progress.return_value.__enter__.return_value = None

        provider = MLProvider(self.cfg)
        provider.initialize()

        result = provider.transcribe("/path/to/audio.mp3")

        self.assertEqual(result, "Hello world")
        mock_model.transcribe.assert_called_once()

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    def test_transcribe_with_language(self, mock_progress, mock_import_whisper):
        """Test transcription with explicit language."""
        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_model.transcribe.return_value = {"text": "Bonjour", "segments": []}
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib
        mock_progress.return_value.__enter__.return_value = None

        provider = MLProvider(self.cfg)
        provider.initialize()

        result = provider.transcribe("/path/to/audio.mp3", language="fr")

        self.assertEqual(result, "Bonjour")
        # Verify language was passed to transcribe
        call_args = mock_model.transcribe.call_args
        self.assertEqual(call_args[1]["language"], "fr")

    def test_transcribe_not_initialized(self):
        """Test transcribe raises RuntimeError if not initialized."""
        provider = MLProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    def test_transcribe_empty_text_raises_error(self, mock_progress, mock_import_whisper):
        """Test transcribe raises ValueError if transcription returns empty text."""
        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_model.transcribe.return_value = {"text": "", "segments": []}
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib
        mock_progress.return_value.__enter__.return_value = None

        provider = MLProvider(self.cfg)
        provider.initialize()

        with self.assertRaises(ValueError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("empty text", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.ml.ml_provider.progress.progress_context")
    def test_transcribe_with_segments_success(self, mock_progress, mock_import_whisper):
        """Test transcribe_with_segments returns full result."""
        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
        mock_model.transcribe.return_value = {
            "text": "Hello world",
            "segments": mock_segments,
        }
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib
        mock_progress.return_value.__enter__.return_value = None

        provider = MLProvider(self.cfg)
        provider.initialize()

        result_dict, elapsed = provider.transcribe_with_segments("/path/to/audio.mp3")

        self.assertEqual(result_dict["text"], "Hello world")
        self.assertEqual(len(result_dict["segments"]), 2)
        self.assertIsInstance(elapsed, float)
        self.assertGreater(elapsed, 0)

    def test_format_screenplay_from_segments(self):
        """Test screenplay formatting from segments."""
        provider = MLProvider(self.cfg)

        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {
                "start": 1.6,
                "end": 2.0,
                "text": "world",
            },  # Gap > 0.5s (1.6 - 1.0 = 0.6) triggers speaker change
            {"start": 2.0, "end": 3.0, "text": "test"},
        ]

        result = provider.format_screenplay_from_segments(
            segments=segments,
            num_speakers=2,
            speaker_names=["Host", "Guest"],
            gap_s=0.5,
        )

        self.assertIn("Host:", result)
        self.assertIn("Guest:", result)
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_format_screenplay_empty_segments(self):
        """Test screenplay formatting with empty segments."""
        provider = MLProvider(self.cfg)

        result = provider.format_screenplay_from_segments(
            segments=[],
            num_speakers=2,
            speaker_names=["Host", "Guest"],
            gap_s=0.5,
        )

        self.assertEqual(result, "")


@pytest.mark.unit
class TestMLProviderSpeakerDetection(unittest.TestCase):
    """Tests for MLProvider speaker detection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
            auto_speakers=True,
        )

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.detect_speaker_names")
    def test_detect_speakers_success(self, mock_detect, mock_get_ner):
        """Test successful speaker detection."""
        mock_nlp = Mock()
        mock_get_ner.return_value = mock_nlp
        mock_detect.return_value = (["Alice", "Bob"], {"Alice"}, True)

        provider = MLProvider(self.cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertEqual(speakers, ["Alice", "Bob"])
        self.assertEqual(hosts, {"Alice"})
        self.assertTrue(success)

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.detect_hosts_from_feed")
    def test_detect_hosts_success(self, mock_detect, mock_get_ner):
        """Test successful host detection."""
        mock_nlp = Mock()
        mock_get_ner.return_value = mock_nlp
        mock_detect.return_value = {"Alice", "Bob"}

        provider = MLProvider(self.cfg)
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=["Alice"],
        )

        self.assertEqual(hosts, {"Alice", "Bob"})

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.analyze_episode_patterns")
    def test_analyze_patterns_success(self, mock_analyze, mock_get_ner):
        """Test successful pattern analysis."""
        from podcast_scraper import models

        mock_nlp = Mock()
        mock_get_ner.return_value = mock_nlp
        mock_heuristics = {"pattern": "value"}
        mock_analyze.return_value = mock_heuristics

        provider = MLProvider(self.cfg)
        provider.initialize()

        episodes = [
            models.Episode(
                idx=1,
                title="Episode 1",
                title_safe="Episode_1",
                item=None,
                transcript_urls=[],
                media_url="https://example.com/1",
                media_type="audio/mpeg",
            )
        ]

        result = provider.analyze_patterns(episodes=episodes, known_hosts={"Alice"})

        self.assertEqual(result, mock_heuristics)
        self.assertEqual(provider._spacy_heuristics, mock_heuristics)

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.analyze_episode_patterns")
    def test_analyze_patterns_no_nlp(self, mock_analyze, mock_get_ner):
        """Test analyze_patterns returns None if nlp is None."""
        mock_get_ner.return_value = None

        provider = MLProvider(self.cfg)
        provider._spacy_nlp = None  # Force None

        result = provider.analyze_patterns(episodes=[], known_hosts=set())

        self.assertIsNone(result)

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.clear_spacy_model_cache")
    def test_clear_cache(self, mock_clear):
        """Test cache clearing."""
        provider = MLProvider(self.cfg)

        provider.clear_cache()

        mock_clear.assert_called_once()


@pytest.mark.unit
class TestMLProviderSummarization(unittest.TestCase):
    """Tests for MLProvider summarization methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            generate_metadata=True,  # Required when generate_summaries=True
        )

    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.summarize_long_text")
    def test_summarize_success(
        self, mock_summarize, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test successful summarization."""
        mock_model = Mock()
        mock_model.model_name = "facebook/bart-base"
        mock_model.device = "cpu"
        mock_summary_model.return_value = mock_model
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"
        mock_summarize.return_value = "This is a summary."

        provider = MLProvider(self.cfg)
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model_used"], "facebook/bart-base")

    def test_summarize_not_initialized(self):
        """Test summarize raises RuntimeError if not initialized."""
        provider = MLProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.summarize_long_text")
    def test_summarize_with_params(
        self, mock_summarize, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test summarization with custom parameters."""
        mock_model = Mock()
        mock_model.model_name = "facebook/bart-base"
        mock_model.device = "cpu"
        mock_summary_model.return_value = mock_model
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"
        mock_summarize.return_value = "Summary"

        provider = MLProvider(self.cfg)
        provider.initialize()

        params = {"max_length": 100, "min_length": 50, "chunk_size": 512}
        provider.summarize("Text", params=params)

        # Verify summarize_long_text was called with correct params
        call_kwargs = mock_summarize.call_args[1]
        self.assertEqual(call_kwargs["max_length"], 100)
        self.assertEqual(call_kwargs["min_length"], 50)

    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.summarize_long_text")
    def test_summarize_error_handling(
        self, mock_summarize, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test summarization error handling."""
        mock_model = Mock()
        mock_model.model_name = "facebook/bart-base"
        mock_model.device = "cpu"
        mock_summary_model.return_value = mock_model
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"
        mock_summarize.side_effect = Exception("Summarization failed")

        provider = MLProvider(self.cfg)
        provider.initialize()

        with self.assertRaises(ValueError) as context:
            provider.summarize("Text")

        self.assertIn("Summarization failed", str(context.exception))
