#!/usr/bin/env python3
"""Integration tests for summarizer model loading.

These tests verify that all defined models can actually be loaded when configured,
catching dependency issues (e.g., missing protobuf).

IMPORTANT: These tests require models to be pre-cached. Run `make preload-ml-models`
before running these tests. Tests will SKIP if models are not cached (to avoid
hanging on network downloads, which violates integration test rules).

Marked as 'slow' and 'integration' tests - skip with: pytest -m "not slow"
"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import cache helpers from same directory
import sys
from pathlib import Path

from conftest import create_test_config  # noqa: E402

integration_dir = Path(__file__).parent
if str(integration_dir) not in sys.path:
    sys.path.insert(0, str(integration_dir))
from ml_model_cache_helpers import (  # noqa: E402
    _is_transformers_model_cached,
    require_transformers_model_cached,
)

# Try to import summarizer, skip tests if dependencies not available
try:
    from podcast_scraper import summarizer

    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    summarizer = types.ModuleType("summarizer")  # type: ignore[assignment]


@pytest.mark.slow
@pytest.mark.integration
@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestModelIntegration(unittest.TestCase):
    """Integration tests to verify all defined models can be loaded.

    These tests verify that each model in DEFAULT_SUMMARY_MODELS can actually
    be loaded when configured, catching dependency issues (e.g., missing protobuf).

    IMPORTANT: These tests require models to be pre-cached. Run `make preload-ml-models`
    before running these tests. Tests will FAIL if models are not cached (to avoid
    hanging on network downloads, which violates integration test rules).

    Marked as 'slow' and 'integration' tests - skip with: pytest -m "not slow"
    """

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bart_large_model_loads(self):
        """Test that 'bart-large' model (BART-large-cnn) can be loaded."""
        cfg = create_test_config(summary_model="bart-large")
        model_name = summarizer.select_summary_model(cfg)
        self.assertEqual(model_name, summarizer.DEFAULT_SUMMARY_MODELS["bart-large"])

        # Require model to be cached (fail fast if not)
        require_transformers_model_cached(model_name, cfg.summary_cache_dir)

        # Try to actually load the model
        try:
            model = summarizer.SummaryModel(
                model_name=model_name,
                device=cfg.summary_device,
                cache_dir=cfg.summary_cache_dir,
            )
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.tokenizer)
            summarizer.unload_model(model)
        except Exception as e:
            # If it's a network error (pytest-socket blocks network), skip test
            # The model cache check might pass but loading requires additional files
            error_str = str(e)
            error_type = type(e).__name__
            # Check for pytest-socket blocking errors or any network-related errors
            if (
                "socket" in error_str.lower()
                or "connect" in error_str.lower()
                or "SocketConnectBlockedError" in error_type
                or "BlockedError" in error_type
            ):
                import pytest

                pytest.skip(
                    f"Model '{model_name}' is not fully cached (network access "
                    f"required). Run 'make preload-ml-models' to pre-cache all "
                    f"required files."
                )
            self.fail(f"Failed to load 'bart-large' model ({model_name}): {e}")

    def test_fast_model_loads(self):
        """Test that 'fast' model (distilbart) can be loaded."""
        cfg = create_test_config(summary_model="fast")
        model_name = summarizer.select_summary_model(cfg)
        self.assertEqual(model_name, summarizer.DEFAULT_SUMMARY_MODELS["fast"])

        # Require model to be cached (fail fast if not)
        require_transformers_model_cached(model_name, cfg.summary_cache_dir)

        try:
            model = summarizer.SummaryModel(
                model_name=model_name,
                device=cfg.summary_device,
                cache_dir=cfg.summary_cache_dir,
            )
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.tokenizer)
            summarizer.unload_model(model)
        except Exception as e:
            self.fail(f"Failed to load 'fast' model ({model_name}): {e}")

    def test_bart_small_model_loads(self):
        """Test that 'bart-small' model (BART-base) can be loaded."""
        cfg = create_test_config(summary_model="bart-small")
        model_name = summarizer.select_summary_model(cfg)
        self.assertEqual(model_name, summarizer.DEFAULT_SUMMARY_MODELS["bart-small"])

        # Require model to be cached (fail fast if not)
        require_transformers_model_cached(model_name, cfg.summary_cache_dir)

        try:
            model = summarizer.SummaryModel(
                model_name=model_name,
                device=cfg.summary_device,
                cache_dir=cfg.summary_cache_dir,
            )
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.tokenizer)
            summarizer.unload_model(model)
        except Exception as e:
            self.fail(f"Failed to load 'bart-small' model ({model_name}): {e}")

    def test_pegasus_model_loads(self):
        """Test that 'pegasus' model can be loaded (requires protobuf)."""
        cfg = create_test_config(summary_model="pegasus")
        model_name = summarizer.select_summary_model(cfg)
        self.assertEqual(model_name, summarizer.DEFAULT_SUMMARY_MODELS["pegasus"])

        # Require model to be cached (fail fast if not)
        require_transformers_model_cached(model_name, cfg.summary_cache_dir)

        try:
            model = summarizer.SummaryModel(
                model_name=model_name,
                device=cfg.summary_device,
                cache_dir=cfg.summary_cache_dir,
            )
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.tokenizer)
            summarizer.unload_model(model)
        except Exception as e:
            self.fail(f"Failed to load 'pegasus' model ({model_name}): {e}")

    def test_pegasus_xsum_model_loads(self):
        """Test that 'pegasus-xsum' model can be loaded (requires protobuf)."""
        cfg = create_test_config(summary_model="pegasus-xsum")
        model_name = summarizer.select_summary_model(cfg)
        self.assertEqual(model_name, summarizer.DEFAULT_SUMMARY_MODELS["pegasus-xsum"])

        # Require model to be cached (fail fast if not)
        require_transformers_model_cached(model_name, cfg.summary_cache_dir)

        try:
            model = summarizer.SummaryModel(
                model_name=model_name,
                device=cfg.summary_device,
                cache_dir=cfg.summary_cache_dir,
            )
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.tokenizer)
            summarizer.unload_model(model)
        except Exception as e:
            self.fail(f"Failed to load 'pegasus-xsum' model ({model_name}): {e}")

    def test_long_model_loads(self):
        """Test that 'long' model (LED-large) can be loaded."""
        cfg = create_test_config(summary_model="long")
        model_name = summarizer.select_summary_model(cfg)
        self.assertEqual(model_name, summarizer.DEFAULT_SUMMARY_MODELS["long"])

        # Require model to be cached (fail fast if not)
        require_transformers_model_cached(model_name, cfg.summary_cache_dir)

        try:
            model = summarizer.SummaryModel(
                model_name=model_name,
                device=cfg.summary_device,
                cache_dir=cfg.summary_cache_dir,
            )
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.tokenizer)
            summarizer.unload_model(model)
        except Exception as e:
            self.fail(f"Failed to load 'long' model ({model_name}): {e}")

    def test_long_fast_model_loads(self):
        """Test that 'long-fast' model (LED-base) can be loaded."""
        cfg = create_test_config(summary_model="long-fast")
        model_name = summarizer.select_summary_model(cfg)
        self.assertEqual(model_name, summarizer.DEFAULT_SUMMARY_MODELS["long-fast"])

        # Require model to be cached (fail fast if not)
        require_transformers_model_cached(model_name, cfg.summary_cache_dir)

        try:
            model = summarizer.SummaryModel(
                model_name=model_name,
                device=cfg.summary_device,
                cache_dir=cfg.summary_cache_dir,
            )
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.tokenizer)
            summarizer.unload_model(model)
        except Exception as e:
            # If network access is blocked (model files not fully cached), skip the test
            error_str = str(e)
            if (
                "socket" in error_str.lower()
                or "connect" in error_str.lower()
                or "network" in error_str.lower()
            ):
                pytest.skip(
                    f"Model files not fully cached (network access blocked): {e}. "
                    f"Run 'make preload-ml-models' to ensure all model files are cached."
                )
            self.fail(f"Failed to load 'long-fast' model ({model_name}): {e}")

    def test_all_models_defined_can_be_loaded(self):
        """Test that all models in DEFAULT_SUMMARY_MODELS can be loaded."""
        failed_models = []
        missing_cache_models = []
        for model_key, model_name in summarizer.DEFAULT_SUMMARY_MODELS.items():
            try:
                cfg = create_test_config(summary_model=model_key)
                resolved_model_name = summarizer.select_summary_model(cfg)

                # Check if model is cached before attempting to load
                if not _is_transformers_model_cached(resolved_model_name, cfg.summary_cache_dir):
                    missing_cache_models.append(f"{model_key} ({resolved_model_name})")
                    continue

                model = summarizer.SummaryModel(
                    model_name=resolved_model_name,
                    device=cfg.summary_device,
                    cache_dir=cfg.summary_cache_dir,
                )
                self.assertIsNotNone(model.model, f"Model {model_key} has no model")
                self.assertIsNotNone(model.tokenizer, f"Model {model_key} has no tokenizer")
                summarizer.unload_model(model)
            except Exception as e:
                failed_models.append(f"{model_key} ({model_name}): {e}")

        # Skip if models are missing from cache
        if missing_cache_models:
            pytest.skip(
                f"{len(missing_cache_models)} model(s) are not cached:\n"
                + "\n".join(f"  - {m}" for m in missing_cache_models)
                + "\n\nRun 'make preload-ml-models' to pre-cache all required models. "
                + "These tests require cached models to avoid network downloads "
                + "(which violates integration test rules)."
            )

        if failed_models:
            self.fail(f"Failed to load {len(failed_models)} model(s):\n" + "\n".join(failed_models))


if __name__ == "__main__":
    unittest.main()
