#!/usr/bin/env python3
"""E2E tests for providers with real ML models.

These tests verify that providers can load and use real ML models:
- Whisper transcription models (tiny model for speed)
- spaCy NER models (en_core_web_sm for speaker detection)
- Transformer summarization models (small models like bart-base or distilbart)

These tests are marked with @pytest.mark.ml_models and @pytest.mark.e2e
because they require:
- ML dependencies installed (openai-whisper, spacy, transformers, torch)
- Real model downloads (first run only, then cached)
- Longer execution time (model loading and inference)

Note: These tests use the smallest/fastest models available to keep execution time reasonable.

Moved from tests/integration/providers/ — real ML models belong in E2E only.
"""

import os
import sys
import unittest
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import config

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import from parent conftest explicitly to avoid conflicts with infrastructure conftest
import importlib.util

# Import cache helpers from integration directory
import sys
from pathlib import Path

tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

parent_conftest_path = tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

create_test_config = parent_conftest.create_test_config

integration_dir = Path(__file__).parent.parent / "integration"
if str(integration_dir) not in sys.path:
    sys.path.insert(0, str(integration_dir))
from ml_model_cache_helpers import (  # noqa: E402
    require_transformers_model_cached,
    require_whisper_model_cached,
)

# Check if ML dependencies are available
# Use lazy imports to avoid loading heavy libraries (torch/transformers) at module import time
# This reduces initial memory footprint for test workers
WHISPER_AVAILABLE = False
SPACY_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False


def _check_whisper_available():
    """Lazy check for Whisper availability."""
    global WHISPER_AVAILABLE
    if WHISPER_AVAILABLE:
        return True
    try:
        import whisper  # noqa: F401

        WHISPER_AVAILABLE = True
        return True
    except ImportError:
        return False


def _check_spacy_available():
    """Lazy check for spaCy availability."""
    global SPACY_AVAILABLE
    if SPACY_AVAILABLE:
        return True
    try:
        import spacy  # noqa: F401

        SPACY_AVAILABLE = True
        return True
    except ImportError:
        return False


def _check_transformers_available():
    """Lazy check for Transformers availability (avoids loading torch at import time)."""
    global TRANSFORMERS_AVAILABLE
    if TRANSFORMERS_AVAILABLE:
        return True
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401

        TRANSFORMERS_AVAILABLE = True
        return True
    except ImportError:
        return False


@pytest.mark.e2e
@pytest.mark.ml_models
@unittest.skipIf(not _check_whisper_available(), "Whisper dependencies not available")
class TestWhisperProviderRealModel(unittest.TestCase):
    """Test Whisper provider with real model loading."""

    def setUp(self):
        """Set up test fixtures."""
        # Test default whisper (tiny.en), not production default (base).
        self.cfg = create_test_config(
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            language="en",
        )

    def test_whisper_model_loading(self):
        """Test that Whisper model can be loaded."""
        # Require model to be cached (fail fast if not)
        # Note: cfg.whisper_model defaults to "tiny.en" (from test defaults)
        # Check for "tiny.en" which is what's preloaded
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        from podcast_scraper.transcription.factory import create_transcription_provider

        # Create provider (real factory)
        provider = create_transcription_provider(self.cfg)

        # Initialize provider (loads real model)
        provider.initialize()  # type: ignore[attr-defined]

        # Get the model from the provider
        model = provider._whisper_model  # type: ignore[attr-defined]

        # Verify model was loaded
        self.assertIsNotNone(model, "Whisper model should be loaded")

        # Verify model has expected attributes
        self.assertTrue(hasattr(model, "device"), "Model should have device attribute")
        self.assertTrue(hasattr(model, "transcribe"), "Model should have transcribe method")

    def test_whisper_provider_with_real_model(self):
        """Test Whisper provider initialization with real model."""
        # Require model to be cached (fail fast if not)
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        from podcast_scraper.transcription.factory import create_transcription_provider

        # Create provider (real factory)
        provider = create_transcription_provider(self.cfg)

        # Initialize provider (loads real model)
        provider.initialize()  # type: ignore[attr-defined]

        # Verify provider is initialized
        self.assertTrue(provider.is_initialized)  # type: ignore[attr-defined]
        self.assertIsNotNone(provider._whisper_model)  # type: ignore[attr-defined]

        # Verify model is actually loaded (not mocked)
        model = provider._whisper_model  # type: ignore[attr-defined]
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "transcribe"))


@pytest.mark.e2e
@pytest.mark.ml_models
@unittest.skipIf(not _check_transformers_available(), "Transformers dependencies not available")
class TestTransformersProviderRealModel(unittest.TestCase):
    """Test Transformers summarization provider with real model loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
            summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
            summary_device="cpu",  # Use CPU to avoid GPU requirements
            language="en",
        )

    def test_transformers_model_loading(self):
        """Test that Transformers model can be loaded."""
        from podcast_scraper.providers.ml import summarizer

        # Load real transformer model
        model_name = summarizer.select_summary_model(self.cfg)

        # Require model to be cached (fail fast if not)
        require_transformers_model_cached(model_name, None)

        try:
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

            # Clean up model to prevent memory leak
            summarizer.unload_model(model)
        except Exception as e:
            # If network access is blocked (model files not fully cached), skip the test
            error_str = str(e).lower()
            if (
                "socket" in error_str
                or "connect" in error_str
                or "network" in error_str
                or "couldn't connect" in error_str
                or "huggingface.co" in error_str
            ):
                pytest.skip(
                    f"Model files not fully cached (network access blocked): {e}. "
                    f"Run 'make preload-ml-models' to ensure all model files are cached."
                )
            # Re-raise other errors as real failures
            raise

    def test_summarization_provider_with_real_model(self):
        """Test summarization provider with real transformer model."""
        from podcast_scraper.providers.ml import summarizer
        from podcast_scraper.summarization.factory import create_summarization_provider

        # Require model to be cached (skip if not, to avoid network downloads)
        model_name = summarizer.select_summary_model(self.cfg)
        require_transformers_model_cached(model_name, None)

        # Also check for REDUCE model if it might be used for long transcripts
        reduce_model_name = summarizer.select_reduce_model(self.cfg, model_name)
        if reduce_model_name != model_name:  # Only check if different from MAP model
            from tests.integration.ml_model_cache_helpers import _is_transformers_model_cached

            if not _is_transformers_model_cached(reduce_model_name, None):
                pytest.skip(
                    f"REDUCE model '{reduce_model_name}' is not cached. "
                    f"Run 'make preload-ml-models' to pre-cache all required models. "
                    f"This test requires cached models to avoid network downloads "
                    f"(which violates integration test rules)."
                )

        # Create provider (real factory)
        provider = create_summarization_provider(self.cfg)

        # Initialize provider (loads real model)
        provider.initialize()  # type: ignore[attr-defined]

        # Verify provider is initialized
        # MLProvider uses is_initialized property and _transformers_initialized
        self.assertTrue(provider.is_initialized)  # type: ignore[attr-defined]
        self.assertTrue(provider._transformers_initialized)  # type: ignore[attr-defined]
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

        # Clean up provider to prevent memory leak
        provider.cleanup()  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
