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

import importlib.util

# Import cache helpers from same directory
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.module_summarization]

# Import from parent conftest explicitly to avoid conflicts with infrastructure conftest
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

integration_dir = Path(__file__).parent
if str(integration_dir) not in sys.path:
    sys.path.insert(0, str(integration_dir))
from ml_model_cache_helpers import (  # noqa: E402
    _is_transformers_model_cached,
    require_transformers_model_cached,
)

# Try to import summarizer, skip tests if dependencies not available
try:
    from podcast_scraper import config, summarizer

    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    summarizer = types.ModuleType("summarizer")  # type: ignore[assignment]
    config = types.ModuleType("config")  # type: ignore[assignment]


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

    def test_bart_small_model_loads(self):
        """Test that test default MAP model can be loaded."""
        cfg = create_test_config(summary_model=config.TEST_DEFAULT_SUMMARY_MODEL)
        model_name = summarizer.select_summary_model(cfg)
        # config.TEST_DEFAULT_SUMMARY_MODEL is "bart-small" (alias)
        # select_summary_model resolves it to "facebook/bart-base" (actual model ID)
        # So we compare to the resolved model ID
        self.assertEqual(model_name, "facebook/bart-base")

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
            self.fail(
                f"Failed to load '{config.TEST_DEFAULT_SUMMARY_MODEL}' model "
                f"({model_name}): {e}"
            )

    def test_long_fast_model_loads(self):
        """Test that test default REDUCE model can be loaded."""
        cfg = create_test_config(summary_model=config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL)
        model_name = summarizer.select_summary_model(cfg)
        # config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL is "long-fast" (alias)
        # select_summary_model resolves it to "allenai/led-base-16384" (actual model ID)
        # So we compare to the resolved model ID
        self.assertEqual(model_name, "allenai/led-base-16384")

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
        """Test that test default models (MAP and REDUCE) can be loaded.

        This test verifies that the test default models used in integration tests
        can be loaded successfully. We only test the test defaults, not all production
        models, to keep tests fast and avoid requiring large model downloads.
        """
        from podcast_scraper import config

        # Test only the test default models (MAP and REDUCE)
        # MAP model: facebook/bart-base (config.TEST_DEFAULT_SUMMARY_MODEL)
        # REDUCE model: allenai/led-base-16384 (config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL)
        test_models = [
            ("MAP (test default)", config.TEST_DEFAULT_SUMMARY_MODEL, "summary_model"),
            (
                "REDUCE (test default)",
                config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,
                "summary_reduce_model",
            ),
        ]

        failed_models = []
        missing_cache_models = []
        for model_label, model_name, config_field in test_models:
            try:
                # Create config with the appropriate field (summary_model for MAP, summary_reduce_model for REDUCE)
                cfg = create_test_config(**{config_field: model_name})

                # Resolve model name (handles both keys and direct model IDs)
                if config_field == "summary_model":
                    resolved_model_name = summarizer.select_summary_model(cfg)
                else:  # summary_reduce_model
                    # For REDUCE, we need a MAP model name for select_reduce_model
                    map_model_name = summarizer.select_summary_model(cfg)
                    resolved_model_name = summarizer.select_reduce_model(cfg, map_model_name)

                # Check if model is cached before attempting to load
                if not _is_transformers_model_cached(resolved_model_name, cfg.summary_cache_dir):
                    missing_cache_models.append(f"{model_label} ({resolved_model_name})")
                    continue

                model = summarizer.SummaryModel(
                    model_name=resolved_model_name,
                    device=cfg.summary_device,
                    cache_dir=cfg.summary_cache_dir,
                )
                self.assertIsNotNone(model.model, f"Model {model_label} has no model")
                self.assertIsNotNone(model.tokenizer, f"Model {model_label} has no tokenizer")
                summarizer.unload_model(model)
            except Exception as e:
                failed_models.append(f"{model_label} ({model_name}): {e}")

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
