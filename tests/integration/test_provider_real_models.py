#!/usr/bin/env python3
"""Integration tests for providers with real ML models.

These tests verify that providers can load and use real ML models:
- Whisper transcription models (tiny model for speed)
- spaCy NER models (en_core_web_sm for speaker detection)
- Transformer summarization models (small models like bart-base or distilbart)

These tests are marked with @pytest.mark.slow and @pytest.mark.ml_models
because they require:
- ML dependencies installed (openai-whisper, spacy, transformers, torch)
- Real model downloads (first run only, then cached)
- Longer execution time (model loading and inference)

Note: These tests use the smallest/fastest models available to keep execution time reasonable.
"""

import os
import sys
import unittest
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import whisper_integration

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import create_test_config  # noqa: E402

# Check if ML dependencies are available
WHISPER_AVAILABLE = False
SPACY_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import whisper  # noqa: F401

    WHISPER_AVAILABLE = True
except ImportError:
    pass

try:
    import spacy  # noqa: F401

    SPACY_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.ml_models
@unittest.skipIf(not WHISPER_AVAILABLE, "Whisper dependencies not available")
class TestWhisperProviderRealModel(unittest.TestCase):
    """Test Whisper provider with real model loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            transcribe_missing=True,
            whisper_model="tiny",  # Smallest model for speed
            language="en",
        )

    def test_whisper_model_loading(self):
        """Test that Whisper model can be loaded."""
        # Load real Whisper model
        model = whisper_integration.load_whisper_model(self.cfg)

        # Verify model was loaded
        self.assertIsNotNone(model, "Whisper model should be loaded")

        # Verify model has expected attributes
        self.assertTrue(hasattr(model, "device"), "Model should have device attribute")
        self.assertTrue(hasattr(model, "transcribe"), "Model should have transcribe method")

    def test_whisper_provider_with_real_model(self):
        """Test Whisper provider initialization with real model."""
        from podcast_scraper.transcription.factory import create_transcription_provider

        # Create provider (real factory)
        provider = create_transcription_provider(self.cfg)

        # Initialize provider (loads real model)
        provider.initialize()  # type: ignore[attr-defined]

        # Verify provider is initialized
        self.assertTrue(provider._initialized)  # type: ignore[attr-defined]
        self.assertIsNotNone(provider._model)  # type: ignore[attr-defined]

        # Verify model is actually loaded (not mocked)
        model = provider._model  # type: ignore[attr-defined]
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "transcribe"))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.ml_models
@unittest.skipIf(not SPACY_AVAILABLE, "spaCy dependencies not available")
class TestSpacyProviderRealModel(unittest.TestCase):
    """Test spaCy NER provider with real model loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            auto_speakers=True,
            ner_model="en_core_web_sm",  # Smallest spaCy model
            language="en",
        )

    def test_spacy_model_loading(self):
        """Test that spaCy model can be loaded."""
        from podcast_scraper import speaker_detection

        # Load real spaCy model
        nlp = speaker_detection.get_ner_model(self.cfg)

        # Verify model was loaded
        self.assertIsNotNone(nlp, "spaCy model should be loaded")

        # Verify model has expected attributes
        self.assertTrue(hasattr(nlp, "pipe"), "Model should have pipe method")

    def test_ner_detector_with_real_model(self):
        """Test NER detector with real spaCy model."""
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        # Create detector (real factory)
        detector = create_speaker_detector(self.cfg)

        # Initialize detector (loads real model)
        detector.initialize()  # type: ignore[attr-defined]

        # Verify detector is initialized
        self.assertTrue(hasattr(detector, "_nlp"))
        self.assertIsNotNone(detector._nlp)  # type: ignore[attr-defined]

        # Test that detector can actually use the model
        # (detect_speakers uses the model internally)
        result = detector.detect_speakers(  # type: ignore[attr-defined]
            episode_title="Test Episode",
            episode_description="This is a test episode with John Smith and Jane Doe.",
            known_hosts={"John Smith"},
        )

        # Verify result is valid
        # detect_speakers returns Tuple[list[str], Set[str], bool]
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        speakers, hosts, success = result
        self.assertIsInstance(speakers, list)
        self.assertIsInstance(hosts, set)
        self.assertIsInstance(success, bool)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.ml_models
@unittest.skipIf(not TRANSFORMERS_AVAILABLE, "Transformers dependencies not available")
class TestTransformersProviderRealModel(unittest.TestCase):
    """Test Transformers summarization provider with real model loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
            summary_model="facebook/bart-base",  # Small model for speed
            summary_device="cpu",  # Use CPU to avoid GPU requirements
            language="en",
        )

    def test_transformers_model_loading(self):
        """Test that Transformers model can be loaded."""
        from podcast_scraper import summarizer

        # Load real transformer model
        model_name = summarizer.select_summary_model(self.cfg)
        model = summarizer.SummaryModel(
            model_name=model_name,
            device="cpu",
            cache_dir=None,
        )

        # Verify model was loaded
        self.assertIsNotNone(model, "Transformer model should be loaded")
        self.assertIsNotNone(model.model, "Model should have model attribute")
        self.assertIsNotNone(model.tokenizer, "Model should have tokenizer attribute")
        self.assertIsNotNone(model.pipeline, "Model should have pipeline attribute")

    def test_summarization_provider_with_real_model(self):
        """Test summarization provider with real transformer model."""
        from podcast_scraper.summarization.factory import create_summarization_provider

        # Create provider (real factory)
        provider = create_summarization_provider(self.cfg)

        # Initialize provider (loads real model)
        provider.initialize()  # type: ignore[attr-defined]

        # Verify provider is initialized
        self.assertTrue(provider._initialized)  # type: ignore[attr-defined]
        self.assertIsNotNone(provider._map_model)  # type: ignore[attr-defined]

        # Verify model is actually loaded (not mocked)
        map_model = provider._map_model  # type: ignore[attr-defined]
        self.assertIsNotNone(map_model)
        self.assertIsNotNone(map_model.pipeline)  # type: ignore[attr-defined]

        # Test that provider can actually use the model
        # (summarize uses the model internally)
        test_text = (
            "This is a test transcript. It contains multiple sentences. "
            "The purpose is to test summarization. We want to verify the model works."
        )
        result = provider.summarize(  # type: ignore[attr-defined]
            text=test_text,
            episode_title="Test Episode",
            episode_description="A test episode",
            params=None,
        )

        # Verify result is valid
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        # Result should contain summary
        self.assertIn("summary", result)
        self.assertIsInstance(result["summary"], str)
        self.assertGreater(len(result["summary"]), 0)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.ml_models
@unittest.skipIf(
    not (WHISPER_AVAILABLE and SPACY_AVAILABLE and TRANSFORMERS_AVAILABLE),
    "Not all ML dependencies available",
)
class TestAllProvidersRealModels(unittest.TestCase):
    """Test all providers together with real models."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            transcribe_missing=True,
            whisper_model="tiny",  # Smallest Whisper model
            auto_speakers=True,
            ner_model="en_core_web_sm",  # Smallest spaCy model
            generate_summaries=True,
            generate_metadata=True,
            summary_model="facebook/bart-base",  # Small transformer model
            summary_device="cpu",
            language="en",
        )

    def test_all_providers_initialize_with_real_models(self):
        """Test that all providers can be initialized with real models."""
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector
        from podcast_scraper.summarization.factory import create_summarization_provider
        from podcast_scraper.transcription.factory import create_transcription_provider

        # Create all providers (real factories)
        transcription_provider = create_transcription_provider(self.cfg)
        speaker_detector = create_speaker_detector(self.cfg)
        summarization_provider = create_summarization_provider(self.cfg)

        # Initialize all providers (loads real models)
        transcription_provider.initialize()  # type: ignore[attr-defined]
        speaker_detector.initialize()  # type: ignore[attr-defined]
        summarization_provider.initialize()  # type: ignore[attr-defined]

        # Verify all providers are initialized with real models
        self.assertTrue(transcription_provider._initialized)  # type: ignore[attr-defined]
        self.assertIsNotNone(transcription_provider._model)  # type: ignore[attr-defined]

        self.assertIsNotNone(speaker_detector._nlp)  # type: ignore[attr-defined]

        self.assertTrue(summarization_provider._initialized)  # type: ignore[attr-defined]
        self.assertIsNotNone(summarization_provider._map_model)  # type: ignore[attr-defined]

        # Verify models are actually loaded (not mocked)
        # Whisper model
        whisper_model = transcription_provider._model  # type: ignore[attr-defined]
        self.assertTrue(hasattr(whisper_model, "transcribe"))

        # spaCy model
        spacy_model = speaker_detector._nlp  # type: ignore[attr-defined]
        self.assertTrue(hasattr(spacy_model, "pipe"))

        # Transformer model
        transformer_model = summarization_provider._map_model  # type: ignore[attr-defined]
        self.assertIsNotNone(transformer_model.pipeline)  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
