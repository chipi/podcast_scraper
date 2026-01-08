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
            transcribe_missing=False,  # Explicitly disable to avoid initializing Whisper
            generate_summaries=False,  # Explicitly disable to avoid initializing Transformers
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
            transcribe_missing=False,  # Explicitly disable to avoid initializing Whisper
        )
        provider = MLProvider(cfg)

        mock_model = Mock()
        mock_model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_model.device = "cpu"
        mock_summary_model.return_value = mock_model
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL

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
            mock_transformers_model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
            mock_transformers_model.device = "cpu"
            mock_summary_model.return_value = mock_transformers_model
            mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
            mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL

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
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
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

    def test_clear_cache(self):
        """Test clear_cache method (no-op after cache removal)."""
        provider = MLProvider(self.cfg)
        # clear_cache() is now a no-op, should not raise
        provider.clear_cache()


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
        mock_model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_model.device = "cpu"
        mock_summary_model.return_value = mock_model
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
        mock_summarize.return_value = "This is a summary."

        provider = MLProvider(self.cfg)
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model_used"], config.TEST_DEFAULT_SUMMARY_MODEL)

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
        mock_model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_model.device = "cpu"
        mock_summary_model.return_value = mock_model
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
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
        mock_model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_model.device = "cpu"
        mock_summary_model.return_value = mock_model
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
        mock_summarize.side_effect = Exception("Summarization failed")

        provider = MLProvider(self.cfg)
        provider.initialize()

        with self.assertRaises(ValueError) as context:
            provider.summarize("Text")

        self.assertIn("Summarization failed", str(context.exception))


@pytest.mark.unit
class TestMLProviderPreload(unittest.TestCase):
    """Tests for MLProvider.preload() method.

    These tests focus on the preload functionality, which is a characteristic
    of MLProvider. All preloading logic is encapsulated within the provider.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="spacy",
            summary_provider="transformers",
            preload_models=True,  # Enable preloading by default
        )

    def test_preload_has_method(self):
        """Test that MLProvider has preload() method."""
        provider = MLProvider(self.cfg)
        self.assertTrue(hasattr(provider, "preload"))
        self.assertTrue(callable(provider.preload))

    def test_preload_skips_when_disabled(self):
        """Test that preload() skips when preload_models=False."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=False,
            transcribe_missing=True,
            transcription_provider="whisper",
        )
        provider = MLProvider(cfg)

        # Should not raise and should not initialize
        provider.preload()
        self.assertFalse(provider._whisper_initialized)

    def test_preload_skips_when_dry_run(self):
        """Test that preload() skips when dry_run=True."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            dry_run=True,
            transcribe_missing=True,
            transcription_provider="whisper",
        )
        provider = MLProvider(cfg)

        # Should not raise and should not initialize
        provider.preload()
        self.assertFalse(provider._whisper_initialized)

    def test_preload_skips_when_no_models_needed(self):
        """Test that preload() skips when no models are needed."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
        )
        provider = MLProvider(cfg)

        # Should not raise and should not initialize anything
        provider.preload()
        self.assertFalse(provider._whisper_initialized)
        self.assertFalse(provider._spacy_initialized)
        self.assertFalse(provider._transformers_initialized)

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_preload_whisper_success(self, mock_import_whisper):
        """Test successful Whisper model preloading."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Use test default (tiny.en)
            auto_speakers=False,
            generate_summaries=False,
        )
        provider = MLProvider(cfg)

        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib

        provider.preload()

        # Verify model was initialized
        self.assertTrue(provider._whisper_initialized)
        self.assertEqual(provider._whisper_model, mock_model)

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_preload_spacy_success(self, mock_get_ner):
        """Test successful spaCy model preloading."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            auto_speakers=True,
            speaker_detector_provider="spacy",
            ner_model="en_core_web_sm",
            transcribe_missing=False,
            generate_summaries=False,
        )
        provider = MLProvider(cfg)

        mock_nlp = Mock()
        mock_get_ner.return_value = mock_nlp

        provider.preload()

        # Verify model was initialized
        self.assertTrue(provider._spacy_initialized)
        self.assertEqual(provider._spacy_nlp, mock_nlp)

    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_preload_transformers_success(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test successful Transformers model preloading."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            generate_summaries=True,
            summary_provider="transformers",
            generate_metadata=True,  # Required when generate_summaries=True
            transcribe_missing=False,
            auto_speakers=False,
        )
        provider = MLProvider(cfg)

        mock_model = Mock()
        mock_model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_model.device = "cpu"
        mock_summary_model.return_value = mock_model
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL

        provider.preload()

        # Verify model was initialized
        self.assertTrue(provider._transformers_initialized)
        self.assertEqual(provider._map_model, mock_model)

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_preload_whisper_fails_fast(self, mock_import_whisper):
        """Test that preload() fails fast when Whisper model cannot be loaded."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Use test default (tiny.en)
            auto_speakers=False,
            generate_summaries=False,
        )
        provider = MLProvider(cfg)

        mock_import_whisper.side_effect = ImportError("openai-whisper not installed")

        with self.assertRaises(RuntimeError) as context:
            provider.preload()

        # Verify error message
        self.assertIn("Failed to preload Whisper model", str(context.exception))
        self.assertIn("openai-whisper", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    def test_preload_spacy_fails_fast(self, mock_get_ner):
        """Test that preload() fails fast when spaCy model cannot be loaded."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            auto_speakers=True,
            speaker_detector_provider="spacy",
            ner_model="en_core_web_sm",
            transcribe_missing=False,
            generate_summaries=False,
        )
        provider = MLProvider(cfg)

        mock_get_ner.side_effect = OSError("Model not found")

        with self.assertRaises(RuntimeError) as context:
            provider.preload()

        # Verify error message
        self.assertIn("Failed to preload spaCy model", str(context.exception))
        self.assertIn("spacy download", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_preload_transformers_fails_fast(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test that preload() fails fast when Transformers model cannot be loaded."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            generate_summaries=True,
            summary_provider="transformers",
            generate_metadata=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        provider = MLProvider(cfg)

        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
        mock_summary_model.side_effect = RuntimeError("Model download failed")

        with self.assertRaises(RuntimeError) as context:
            provider.preload()

        # Verify error message
        self.assertIn("Failed to preload Transformers models", str(context.exception))
        self.assertIn("model cache", str(context.exception).lower())

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_preload_all_models_success(
        self,
        mock_summary_model,
        mock_select_map,
        mock_select_reduce,
        mock_get_ner,
        mock_import_whisper,
    ):
        """Test preloading all three model types together."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            auto_speakers=True,
            speaker_detector_provider="spacy",
            generate_summaries=True,
            summary_provider="transformers",
            generate_metadata=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Use test default (tiny.en)
        )
        provider = MLProvider(cfg)

        # Setup mocks
        mock_whisper_model = Mock()
        mock_whisper_model.device.type = "cpu"
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_whisper_model
        mock_import_whisper.return_value = mock_whisper_lib

        mock_nlp = Mock()
        mock_get_ner.return_value = mock_nlp

        mock_transformers_model = Mock()
        mock_transformers_model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_transformers_model.device = "cpu"
        mock_summary_model.return_value = mock_transformers_model
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL

        provider.preload()

        # Verify all models were initialized
        self.assertTrue(provider._whisper_initialized)
        self.assertTrue(provider._spacy_initialized)
        self.assertTrue(provider._transformers_initialized)

    def test_preload_skips_openai_providers(self):
        """Test that preload() skips when using OpenAI providers."""
        # Whisper with OpenAI provider - should skip
        cfg1 = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="openai",  # OpenAI, not whisper
            openai_api_key="sk-test123",  # Required for OpenAI provider
            auto_speakers=False,  # Disable to avoid spaCy loading
            generate_summaries=False,  # Disable to avoid Transformers loading
        )
        provider1 = MLProvider(cfg1)
        provider1.preload()
        self.assertFalse(provider1._whisper_initialized)

        # Summarization with OpenAI provider - should skip
        cfg2 = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            generate_summaries=True,
            summary_provider="openai",  # OpenAI, not transformers
            openai_api_key="sk-test123",  # Required for OpenAI provider
            generate_metadata=True,
            auto_speakers=False,  # Disable to avoid spaCy loading
            transcribe_missing=False,  # Disable to avoid Whisper loading
        )
        provider2 = MLProvider(cfg2)
        provider2.preload()
        self.assertFalse(provider2._transformers_initialized)

        # Speaker detection with OpenAI provider - should skip
        cfg3 = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            auto_speakers=True,
            speaker_detector_provider="openai",  # OpenAI, not spacy
            openai_api_key="sk-test123",  # Required for OpenAI provider
            transcribe_missing=False,  # Disable to avoid Whisper loading
            generate_summaries=False,  # Disable to avoid Transformers loading
        )
        provider3 = MLProvider(cfg3)
        provider3.preload()
        self.assertFalse(provider3._spacy_initialized)

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_preload_idempotent(self, mock_import_whisper):
        """Test that preload() is idempotent (can be called multiple times safely)."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Use test default (tiny.en)
            auto_speakers=False,  # Disable to avoid spaCy filesystem I/O
            generate_summaries=False,  # Disable to avoid Transformers loading
        )
        provider = MLProvider(cfg)

        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib

        # First preload
        provider.preload()
        self.assertTrue(provider._whisper_initialized)
        first_call_count = mock_whisper_lib.load_model.call_count

        # Second preload should not reload (initialize() is idempotent)
        provider.preload()
        self.assertTrue(provider._whisper_initialized)
        # Should still only be called once (initialize() checks _whisper_initialized)
        self.assertEqual(mock_whisper_lib.load_model.call_count, first_call_count)
