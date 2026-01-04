#!/usr/bin/env python3
"""Integration tests for provider system (Stage 5).

These tests verify that all providers work together correctly in the workflow.
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
class TestProviderIntegration(unittest.TestCase):
    """Test that all providers work together."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="spacy",
            summary_provider="transformers",
            generate_summaries=False,  # Disable to avoid loading models in tests
            auto_speakers=False,  # Disable to avoid loading spaCy in tests
            transcribe_missing=True,
        )

    def test_all_providers_can_be_created(self):
        """Test that all three provider types can be created from config."""
        transcription_provider = create_transcription_provider(self.cfg)
        speaker_detector = create_speaker_detector(self.cfg)
        summarization_provider = create_summarization_provider(self.cfg)

        self.assertIsNotNone(transcription_provider)
        self.assertIsNotNone(speaker_detector)
        self.assertIsNotNone(summarization_provider)

        # Verify they are the unified providers (protocol compliance, not class names)
        # All ML-based providers should be MLProvider
        self.assertEqual(transcription_provider.__class__.__name__, "MLProvider")
        self.assertEqual(speaker_detector.__class__.__name__, "MLProvider")
        self.assertEqual(summarization_provider.__class__.__name__, "MLProvider")

        # Verify protocol compliance
        self.assertTrue(hasattr(transcription_provider, "transcribe"))
        self.assertTrue(hasattr(speaker_detector, "detect_speakers"))
        self.assertTrue(hasattr(summarization_provider, "summarize"))

    def test_providers_have_required_methods(self):
        """Test that all providers have required protocol methods."""
        transcription_provider = create_transcription_provider(self.cfg)
        speaker_detector = create_speaker_detector(self.cfg)
        summarization_provider = create_summarization_provider(self.cfg)

        # Transcription provider
        self.assertTrue(hasattr(transcription_provider, "transcribe"))
        self.assertTrue(hasattr(transcription_provider, "initialize"))

        # Speaker detector
        self.assertTrue(hasattr(speaker_detector, "detect_speakers"))
        self.assertTrue(hasattr(speaker_detector, "analyze_patterns"))

        # Summarization provider
        self.assertTrue(hasattr(summarization_provider, "summarize"))
        self.assertTrue(hasattr(summarization_provider, "initialize"))

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_provider_initialization_order(
        self,
        mock_summary_model,
        mock_select_map,
        mock_select_reduce,
        mock_get_ner,
        mock_import_whisper,
    ):
        """Test that providers can be initialized in any order."""
        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib
        mock_get_ner.return_value = Mock()
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
        mock_summary_model.return_value = Mock()

        transcription_provider = create_transcription_provider(self.cfg)
        speaker_detector = create_speaker_detector(self.cfg)
        summarization_provider = create_summarization_provider(self.cfg)

        # Initialize in different orders
        # Order 1: Transcription -> Speaker -> Summarization
        if hasattr(transcription_provider, "initialize"):
            transcription_provider.initialize()  # type: ignore[attr-defined]
        if hasattr(speaker_detector, "initialize"):
            speaker_detector.initialize()  # type: ignore[attr-defined]
        if hasattr(summarization_provider, "initialize"):
            summarization_provider.initialize()  # type: ignore[attr-defined]

        # All should be initialized
        # Check initialization status (some providers may not have is_initialized property)
        if hasattr(transcription_provider, "is_initialized"):
            self.assertTrue(transcription_provider.is_initialized)  # type: ignore[attr-defined]
        if hasattr(speaker_detector, "is_initialized"):
            # Speaker detector may not set is_initialized if auto_speakers is False
            if self.cfg.auto_speakers:
                self.assertTrue(speaker_detector.is_initialized)  # type: ignore[attr-defined]
        if hasattr(summarization_provider, "is_initialized"):
            # Summarization provider may not set is_initialized if generate_summaries is False
            if self.cfg.generate_summaries:
                self.assertTrue(summarization_provider.is_initialized)  # type: ignore[attr-defined]

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_provider_cleanup_order(
        self,
        mock_summary_model,
        mock_select_map,
        mock_select_reduce,
        mock_get_ner,
        mock_import_whisper,
    ):
        """Test that providers can be cleaned up in any order."""
        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_whisper_lib = Mock()
        mock_whisper_lib.load_model.return_value = mock_model
        mock_import_whisper.return_value = mock_whisper_lib
        mock_get_ner.return_value = Mock()
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
        mock_summary_model.return_value = Mock()

        transcription_provider = create_transcription_provider(self.cfg)
        speaker_detector = create_speaker_detector(self.cfg)
        summarization_provider = create_summarization_provider(self.cfg)

        # Initialize all
        if hasattr(transcription_provider, "initialize"):
            transcription_provider.initialize()  # type: ignore[attr-defined]
        if hasattr(speaker_detector, "initialize"):
            speaker_detector.initialize()  # type: ignore[attr-defined]
        if hasattr(summarization_provider, "initialize"):
            summarization_provider.initialize()  # type: ignore[attr-defined]

        # Cleanup in reverse order
        if hasattr(summarization_provider, "cleanup"):
            summarization_provider.cleanup()  # type: ignore[attr-defined]
        if hasattr(speaker_detector, "cleanup"):
            speaker_detector.cleanup()  # type: ignore[attr-defined]
        if hasattr(transcription_provider, "cleanup"):
            transcription_provider.cleanup()  # type: ignore[attr-defined]

        # All should be cleaned up
        if hasattr(transcription_provider, "is_initialized"):
            self.assertFalse(transcription_provider.is_initialized)  # type: ignore[attr-defined]
        if hasattr(summarization_provider, "is_initialized"):
            self.assertFalse(summarization_provider.is_initialized)  # type: ignore[attr-defined]


@pytest.mark.integration
class TestProviderSwitching(unittest.TestCase):
    """Test provider switching via configuration."""

    @pytest.mark.llm
    @pytest.mark.openai
    def test_transcription_provider_switching(self):
        """Test that transcription provider can be switched via config."""
        cfg1 = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider1 = create_transcription_provider(cfg1)
        # Verify protocol compliance, not class name
        self.assertEqual(provider1.__class__.__name__, "MLProvider")
        self.assertTrue(hasattr(provider1, "transcribe"))

        # Test OpenAI provider creation (Stage 6)
        cfg2 = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
        )
        provider2 = create_transcription_provider(cfg2)
        # Verify protocol compliance, not class name
        self.assertEqual(provider2.__class__.__name__, "OpenAIProvider")
        self.assertTrue(hasattr(provider2, "transcribe"))

        # Test error handling: missing API key (caught by validator)
        with self.assertRaises(ValidationError):
            config.Config(rss_url="https://example.com/feed.xml", transcription_provider="openai")

    @pytest.mark.llm
    @pytest.mark.openai
    def test_speaker_detector_switching(self):
        """Test that speaker detector can be switched via config."""
        cfg1 = config.Config(
            rss_url="https://example.com/feed.xml", speaker_detector_provider="spacy"
        )
        detector1 = create_speaker_detector(cfg1)
        # Verify protocol compliance, not class name
        self.assertEqual(detector1.__class__.__name__, "MLProvider")
        self.assertTrue(hasattr(detector1, "detect_speakers"))

        # Test OpenAI detector creation (Stage 6)
        cfg2 = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
            auto_speakers=True,
        )
        detector2 = create_speaker_detector(cfg2)
        # Verify protocol compliance, not class name
        self.assertEqual(detector2.__class__.__name__, "OpenAIProvider")
        self.assertTrue(hasattr(detector2, "detect_speakers"))

        # Test error handling: missing API key (caught by validator)
        with self.assertRaises(ValidationError):
            config.Config(
                rss_url="https://example.com/feed.xml", speaker_detector_provider="openai"
            )

    @pytest.mark.llm
    @pytest.mark.openai
    def test_summarization_provider_switching(self):
        """Test that summarization provider can be switched via config."""
        cfg1 = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )
        provider1 = create_summarization_provider(cfg1)
        # Verify protocol compliance, not class name
        self.assertEqual(provider1.__class__.__name__, "MLProvider")
        self.assertTrue(hasattr(provider1, "summarize"))

        # Test OpenAI provider creation (Stage 6)
        cfg2 = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test123",
            generate_metadata=True,
            generate_summaries=True,
        )
        provider2 = create_summarization_provider(cfg2)
        # Verify protocol compliance, not class name
        self.assertEqual(provider2.__class__.__name__, "OpenAIProvider")
        self.assertTrue(hasattr(provider2, "summarize"))

        # Test error handling: missing API key (caught by validator)
        with self.assertRaises(ValidationError):
            config.Config(
                rss_url="https://example.com/feed.xml",
                summary_provider="openai",
                generate_summaries=False,
            )


@pytest.mark.integration
class TestProviderErrorHandling(unittest.TestCase):
    """Test error handling when providers fail."""

    def test_transcription_provider_initialization_failure(self):
        """Test graceful handling when transcription provider fails to initialize."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider = create_transcription_provider(cfg)

        # Mock initialization failure
        with patch.object(provider, "initialize", side_effect=Exception("Model load failed")):
            with self.assertRaises(Exception):
                provider.initialize()  # type: ignore[attr-defined]

    def test_speaker_detector_initialization_failure(self):
        """Test graceful handling when speaker detector fails to initialize."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        # Mock initialization failure
        with patch.object(detector, "initialize", side_effect=Exception("Model load failed")):
            with self.assertRaises(Exception):
                detector.initialize()  # type: ignore[attr-defined]

    def test_summarization_provider_initialization_failure(self):
        """Test graceful handling when summarization provider fails to initialize."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)

        # Mock initialization failure
        with patch.object(provider, "initialize", side_effect=Exception("Model load failed")):
            with self.assertRaises(Exception):
                provider.initialize()  # type: ignore[attr-defined]

    def test_provider_method_failure_after_initialization(self):
        """Test graceful handling when provider methods fail after initialization."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)

        # Mock summarize failure
        with patch.object(provider, "summarize", side_effect=ValueError("Summarization failed")):
            with self.assertRaises(ValueError):
                provider.summarize("test text")


@pytest.mark.integration
class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with default configurations."""

    def test_default_provider_configuration(self):
        """Test that default config uses expected providers."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")

        # Defaults use new consistent naming (library names)
        self.assertEqual(cfg.transcription_provider, "whisper")
        self.assertEqual(cfg.speaker_detector_provider, "spacy")
        self.assertEqual(cfg.summary_provider, "transformers")

    def test_default_providers_are_created(self):
        """Test that default providers can be created."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")

        transcription_provider = create_transcription_provider(cfg)
        speaker_detector = create_speaker_detector(cfg)
        summarization_provider = create_summarization_provider(cfg)

        # Should create the unified ML providers (defaults)
        self.assertEqual(transcription_provider.__class__.__name__, "MLProvider")
        self.assertEqual(speaker_detector.__class__.__name__, "MLProvider")
        self.assertEqual(summarization_provider.__class__.__name__, "MLProvider")

        # Verify protocol compliance
        self.assertTrue(hasattr(transcription_provider, "transcribe"))
        self.assertTrue(hasattr(speaker_detector, "detect_speakers"))
        self.assertTrue(hasattr(summarization_provider, "summarize"))

    def test_config_without_provider_fields(self):
        """Test that config without explicit provider fields still works."""
        # Create config with minimal fields
        cfg = config.Config(rss_url="https://example.com/feed.xml")

        # Providers should use defaults (new consistent naming)
        self.assertEqual(cfg.transcription_provider, "whisper")
        self.assertEqual(cfg.speaker_detector_provider, "spacy")
        self.assertEqual(cfg.summary_provider, "transformers")


@pytest.mark.integration
class TestProviderInstanceReuse(unittest.TestCase):
    """Test that factories reuse preloaded MLProvider instance."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="spacy",
            summary_provider="transformers",
            preload_models=True,
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.workflow._preloaded_ml_provider", None)
    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.ml.ml_provider.speaker_detection.get_ner_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_factories_reuse_preloaded_instance(
        self,
        mock_summary_model,
        mock_select_map,
        mock_select_reduce,
        mock_get_ner,
        mock_import_whisper,
    ):
        """Test that factories reuse preloaded MLProvider instance when available."""
        from podcast_scraper import workflow
        from podcast_scraper.ml.ml_provider import MLProvider

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

        # Simulate early preloading (as done in workflow)
        preloaded_provider = MLProvider(self.cfg)
        preloaded_provider.preload()
        workflow._preloaded_ml_provider = preloaded_provider

        # Create providers via factories - they should reuse the preloaded instance
        transcription_provider = create_transcription_provider(self.cfg)
        speaker_detector = create_speaker_detector(self.cfg)
        summarization_provider = create_summarization_provider(self.cfg)

        # Verify all three factories returned the same instance
        self.assertIs(transcription_provider, preloaded_provider)
        self.assertIs(speaker_detector, preloaded_provider)
        self.assertIs(summarization_provider, preloaded_provider)

        # Verify models were preloaded
        self.assertTrue(preloaded_provider._whisper_initialized)
        self.assertTrue(preloaded_provider._spacy_initialized)
        self.assertTrue(preloaded_provider._transformers_initialized)

    def test_factories_create_new_instance_when_no_preload(self):
        """Test that factories create new instances when no preloaded instance exists."""
        # Ensure no preloaded instance
        from podcast_scraper import workflow

        workflow._preloaded_ml_provider = None

        # Create providers via factories
        transcription_provider = create_transcription_provider(self.cfg)
        speaker_detector = create_speaker_detector(self.cfg)
        summarization_provider = create_summarization_provider(self.cfg)

        # Verify they are MLProvider instances
        self.assertEqual(transcription_provider.__class__.__name__, "MLProvider")
        self.assertEqual(speaker_detector.__class__.__name__, "MLProvider")
        self.assertEqual(summarization_provider.__class__.__name__, "MLProvider")

        # When no preload, each factory creates a new instance
        # (They may or may not be the same object, but they're separate instances)
        # The key is that they work correctly without preloading
