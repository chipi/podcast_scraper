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
from podcast_scraper.exceptions import (
    ProviderDependencyError,
    ProviderNotInitializedError,
    ProviderRuntimeError,
)
from podcast_scraper.providers.ml.ml_provider import MLProvider


@pytest.mark.integration
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

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
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

    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.SummaryModel")
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
            transcribe_missing=False,
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

    def test_internal_attributes_exist(self):
        """Test that internal attributes exist."""
        provider = MLProvider(self.cfg)

        # Whisper attributes
        self.assertTrue(hasattr(provider, "_whisper_model"))
        self.assertTrue(hasattr(provider, "is_initialized"))

        # spaCy attributes
        self.assertTrue(hasattr(provider, "_spacy_nlp"))

        # Transformers attributes
        self.assertTrue(hasattr(provider, "_map_model"))
        self.assertTrue(hasattr(provider, "_reduce_model"))


@pytest.mark.integration
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

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_success(self, mock_import_whisper):
        """Test successful transcription."""
        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_model.transcribe.return_value = {"text": "Hello world", "segments": []}
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib

        provider = MLProvider(self.cfg)
        provider.initialize()

        result = provider.transcribe("/path/to/audio.mp3")

        self.assertEqual(result, "Hello world")
        mock_model.transcribe.assert_called_once()

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_with_language(self, mock_import_whisper):
        """An ENGLISH language reaches the model. Non-English is refused — see below.

        This asserted `language="fr"` was passed straight through, which is the behaviour S0.7
        (#2178) deliberately removed: this provider's model default is `base.en`, and for a
        non-English language `normalize_whisper_model_name` strips the `.en` and runs the small
        multilingual ["base", "tiny"] chain, whose output is confident and wrong.
        """
        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_model.transcribe.return_value = {"text": "Hello there", "segments": []}
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib

        provider = MLProvider(self.cfg)
        provider.initialize()

        result = provider.transcribe("/path/to/audio.mp3", language="en")

        self.assertEqual(result, "Hello there")
        # Verify language was passed to transcribe
        call_args = mock_model.transcribe.call_args
        self.assertEqual(call_args[1]["language"], "en")

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_refuses_a_non_english_language(self, mock_import_whisper):
        """S0.7 (#2178): refuse rather than transcribe badly.

        The refusal must happen BEFORE the model is called — a wasted transcription that gets
        discarded is the cost, but a wrong transcript that gets KEPT is the hazard.
        """
        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib

        provider = MLProvider(self.cfg)
        provider.initialize()

        with self.assertRaises(ValueError) as ctx:
            provider.transcribe("/path/to/audio.mp3", language="fr")

        self.assertIn("supports ['en']", str(ctx.exception))
        mock_model.transcribe.assert_not_called()

    def test_transcribe_not_initialized(self):
        """Test transcribe raises ProviderNotInitializedError if not initialized."""
        provider = MLProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(ProviderNotInitializedError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_empty_text_raises_error(self, mock_import_whisper):
        """Test transcribe raises ProviderRuntimeError if transcription returns empty text."""
        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_model.transcribe.return_value = {"text": "", "segments": []}
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib

        provider = MLProvider(self.cfg)
        provider.initialize()

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("empty text", str(context.exception))

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_with_segments_success(self, mock_import_whisper):
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


@pytest.mark.integration
class TestMLProviderSpeakerDetection(unittest.TestCase):
    """Tests for MLProvider speaker detection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
            auto_speakers=True,
        )

    def test_clear_cache(self):
        """Test clear_cache method (no-op after cache removal)."""
        provider = MLProvider(self.cfg)
        # clear_cache() is now a no-op, should not raise
        provider.clear_cache()


@pytest.mark.integration
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

    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.summarize_long_text")
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
        """Test summarize raises ProviderNotInitializedError if not initialized."""
        provider = MLProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(ProviderNotInitializedError) as context:
            provider.summarize("Text")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.summarize_long_text")
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

    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.summarize_long_text")
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

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("Summarization failed", str(context.exception))


@pytest.mark.integration
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

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
    def test_preload_whisper_success(self, mock_import_whisper):
        """Test successful Whisper model preloading."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
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

    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.SummaryModel")
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

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
    def test_preload_whisper_fails_fast(self, mock_import_whisper):
        """Test that preload() fails fast when Whisper model cannot be loaded."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            auto_speakers=False,
            generate_summaries=False,
        )
        provider = MLProvider(cfg)

        mock_import_whisper.side_effect = ImportError("openai-whisper not installed")

        with self.assertRaises(ProviderDependencyError) as context:
            provider.preload()

        # Verify error message
        self.assertIn("Failed to preload Whisper model", str(context.exception))
        self.assertIn("openai-whisper", str(context.exception))

    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.providers.ml.ml_provider.summarizer.SummaryModel")
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

        with self.assertRaises(ProviderDependencyError) as context:
            provider.preload()

        # Verify error message
        self.assertIn("Failed to preload Transformers models", str(context.exception))
        self.assertIn("model cache", str(context.exception).lower())

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

    @patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper")
    def test_preload_idempotent(self, mock_import_whisper):
        """Test that preload() is idempotent (can be called multiple times safely)."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            preload_models=True,
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
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


@pytest.mark.integration
class TestMLProviderGILEvidence(unittest.TestCase):
    """Tests for MLProvider extract_quotes and score_entailment (GIL evidence)."""

    def setUp(self):
        """Set up test config."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="spacy",
            summary_provider="transformers",
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
        )

    def test_extract_kg_graph_not_implemented_returns_none(self):
        """Transformers summarization has no LLM KG path; API returns None."""
        provider = MLProvider(self.cfg)
        self.assertIsNone(provider.extract_kg_graph("any transcript text"))


@pytest.mark.integration
class TestMLProviderSummaryUseCase:
    """BART/LED take a document, not instructions, so a board summary on the
    ``transformers`` provider is the use case's material written as prose, summarised by this same
    map/reduce within the use case's budget. Without a use case nothing changes."""

    MATERIAL = {
        "episodes": [
            {
                "title": "Quokka Ledger Hour",
                "show": "Marsupial Finance",
                "summary": "Two hosts argue about small-island accounting.",
                "key_points": ["Ledgers drift when nobody reconciles them."],
                "insights": [{"speaker": "Ada Lin", "text": "Reconcile weekly, not yearly."}],
            }
        ],
        "moments": [{"text": "Counting is a moral act.", "speaker": "Ada Lin", "episode": "Q"}],
    }

    def _provider(self):
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=True,
            generate_metadata=True,
            transcribe_missing=False,
            auto_speakers=False,
        )
        provider = MLProvider(cfg)
        provider._transformers_initialized = True
        provider._map_model = Mock(model_name="facebook/bart-large-cnn", device="cpu")
        provider._reduce_model = provider._map_model
        return provider

    def _summarize(self, provider, *args, **kwargs):
        target = "podcast_scraper.providers.ml.ml_provider.summarizer.summarize_long_text"
        caps = "podcast_scraper.providers.ml.ml_provider.ModelRegistry.get_capabilities"
        with (
            patch(target, return_value="A board about ledgers.") as run,
            patch(caps, return_value=Mock(max_input_tokens=1024)),
        ):
            out = provider.summarize(*args, **kwargs)
        return out, run.call_args.kwargs

    def _use_case(self):
        from podcast_scraper.providers.common.summary_use_case import SummaryUseCase

        return SummaryUseCase(
            system_prompt="summarization/board_system_v1",
            user_prompt="summarization/board_user_v1",
            stage="board_summary",
            max_length=345,
            params=self.MATERIAL,
            document_prompt="summarization/board_document_v1",
        )

    def test_the_material_written_as_prose_is_the_document(self):
        out, sent = self._summarize(self._provider(), "", params={"use_case": self._use_case()})

        assert out["summary"] == "A board about ledgers."
        doc = sent["text"]
        assert "Quokka Ledger Hour, from Marsupial Finance." in doc
        assert "Ada Lin: Reconcile weekly, not yearly." in doc
        assert "Ada Lin: Counting is a moral act." in doc
        # No instructions and no markup: BART would copy them into the summary.
        assert "Summarise" not in doc and "<episode" not in doc
        assert "listener's board" not in doc
        assert sent["transcript_text"] == doc

    def test_the_budget_caps_both_the_single_pass_and_the_reduce(self):
        _out, sent = self._summarize(self._provider(), "", params={"use_case": self._use_case()})

        assert sent["max_length"] == 345 and sent["reduce_max_new_tokens"] == 345
        assert sent["preprocessing_profile"] == "cleaning_none"  # not a transcript

    def test_metadata_names_the_document_template(self):
        out, _sent = self._summarize(self._provider(), "", params={"use_case": self._use_case()})

        prompts = out["metadata"]["prompts"]
        assert set(prompts) == {"document"}
        assert prompts["document"]["name"] == "transformers/summarization/board_document_v1"
        assert prompts["document"]["file"] == "shared/summarization/board_document_v1.j2"
        assert prompts["document"]["params"] == {"episodes": 1, "moments": 1}

    def test_without_a_use_case_it_is_the_episode_summary(self):
        out, sent = self._summarize(self._provider(), "A long transcript about ledgers.")

        assert sent["text"] == "A long transcript about ledgers."
        assert sent["max_length"] == 150 and sent["preprocessing_profile"] != "cleaning_none"
        assert "prompts" not in out["metadata"]

    def test_a_use_case_without_a_document_template_fails_loudly(self):
        from dataclasses import replace

        uc = replace(self._use_case(), document_prompt=None)
        with pytest.raises(ValueError, match="document_prompt"):
            self._summarize(self._provider(), "", params={"use_case": uc})
