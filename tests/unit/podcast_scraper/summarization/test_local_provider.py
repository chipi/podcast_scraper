#!/usr/bin/env python3
"""Unit tests for TransformersSummarizationProvider class.

These tests verify the local transformers-based summarization provider implementation.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from parent conftest explicitly to avoid conflicts
import importlib.util
from pathlib import Path

parent_tests_dir = Path(__file__).parent.parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

parent_conftest_path = parent_tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

create_test_config = parent_conftest.create_test_config

from podcast_scraper.exceptions import (  # noqa: E402
    ProviderNotInitializedError,
    ProviderRuntimeError,
)
from podcast_scraper.summarization.factory import create_summarization_provider  # noqa: E402


class TestTransformersSummarizationProvider(unittest.TestCase):
    """Tests for Transformers summarization via MLProvider (unified provider)."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            generate_summaries=True,
            summary_provider="transformers",
            summary_model="facebook/bart-large-cnn",
            transcribe_missing=False,  # Disable to avoid loading Whisper
            auto_speakers=False,  # Disable to avoid loading spaCy
        )

    def test_init(self):
        """Test MLProvider initialization via factory."""
        provider = create_summarization_provider(self.cfg)
        self.assertEqual(provider.cfg, self.cfg)
        # MLProvider uses _map_model and _reduce_model
        self.assertIsNone(provider._map_model)
        self.assertIsNone(provider._reduce_model)
        self.assertFalse(provider._transformers_initialized)
        self.assertTrue(provider._requires_separate_instances)

    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_initialize_success(self, mock_select_model, mock_select_reduce, mock_summary_model):
        """Test successful initialization."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/bart-large-cnn"
        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-large-cnn"
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        self.assertTrue(provider._transformers_initialized)
        self.assertIsNotNone(provider._map_model)
        self.assertEqual(provider._reduce_model, provider._map_model)  # Same model
        mock_select_model.assert_called_once_with(self.cfg)
        mock_summary_model.assert_called_once()

    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_initialize_with_different_reduce_model(
        self, mock_select_model, mock_select_reduce, mock_summary_model
    ):
        """Test initialization with different reduce model."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/pegasus-xsum"
        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-large-cnn"
        mock_map_model.device = "cpu"
        mock_reduce_model = Mock()
        mock_reduce_model.model_name = "facebook/pegasus-xsum"
        mock_reduce_model.device = "cpu"
        mock_summary_model.side_effect = [mock_map_model, mock_reduce_model]

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        self.assertTrue(provider._transformers_initialized)
        self.assertIsNotNone(provider._map_model)
        self.assertIsNotNone(provider._reduce_model)
        self.assertNotEqual(provider._reduce_model, provider._map_model)
        self.assertEqual(mock_summary_model.call_count, 2)

    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_initialize_summaries_disabled(self, mock_select_model):
        """Test initialization when generate_summaries is False."""
        cfg = create_test_config(generate_summaries=False)
        provider = create_summarization_provider(cfg)
        provider.initialize()

        self.assertFalse(provider._transformers_initialized)
        self.assertIsNone(provider._map_model)
        mock_select_model.assert_not_called()

    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_initialize_already_initialized(
        self, mock_select_model, mock_select_reduce, mock_summary_model
    ):
        """Test initialization when already initialized."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/bart-large-cnn"
        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-large-cnn"
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model

        provider = create_summarization_provider(self.cfg)
        provider.initialize()
        mock_select_model.reset_mock()

        # Call again
        provider.initialize()

        # Should not call select_summary_model again
        mock_select_model.assert_not_called()

    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_initialize_handles_exception(
        self, mock_select_model, mock_select_reduce, mock_summary_model
    ):
        """Test initialization handles exceptions."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/bart-large-cnn"
        mock_summary_model.side_effect = RuntimeError("Model loading failed")

        provider = create_summarization_provider(self.cfg)

        with self.assertRaises(RuntimeError) as context:
            provider.initialize()

        self.assertIn("Model loading failed", str(context.exception))
        self.assertFalse(provider._transformers_initialized)

    @patch("podcast_scraper.ml.ml_provider.summarizer.summarize_long_text")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_summarize_success(
        self, mock_select_model, mock_select_reduce, mock_summary_model, mock_summarize
    ):
        """Test successful summarization."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/bart-large-cnn"
        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-large-cnn"
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model
        mock_summarize.return_value = "This is a summary."

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        result = provider.summarize("Long transcript text here...")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIsNone(result["summary_short"])
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model_used"], "facebook/bart-large-cnn")
        mock_summarize.assert_called_once()

    @patch("podcast_scraper.ml.ml_provider.summarizer.summarize_long_text")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_summarize_not_initialized(
        self, mock_select_model, mock_select_reduce, mock_summary_model, mock_summarize
    ):
        """Test summarize raises ProviderNotInitializedError when not initialized."""
        provider = create_summarization_provider(self.cfg)

        with self.assertRaises(ProviderNotInitializedError) as context:
            provider.summarize("Text")

        self.assertIn("not initialized", str(context.exception))
        self.assertEqual(context.exception.provider, "MLProvider/Transformers")
        mock_summarize.assert_not_called()

    @patch("podcast_scraper.ml.ml_provider.summarizer.summarize_long_text")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_summarize_with_params(
        self, mock_select_model, mock_select_reduce, mock_summary_model, mock_summarize
    ):
        """Test summarize with custom parameters."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/bart-large-cnn"
        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-large-cnn"
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model
        mock_summarize.return_value = "Summary"

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        params = {
            "max_length": 200,
            "min_length": 50,
            "chunk_size": 1024,
            "chunk_parallelism": 4,
        }
        result = provider.summarize("Text", params=params)

        self.assertEqual(result["summary"], "Summary")
        # Verify params were passed correctly
        call_args = mock_summarize.call_args
        self.assertEqual(call_args.kwargs["max_length"], 200)
        self.assertEqual(call_args.kwargs["min_length"], 50)

    @patch("podcast_scraper.ml.ml_provider.summarizer.summarize_long_text")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_summarize_handles_exception(
        self, mock_select_model, mock_select_reduce, mock_summary_model, mock_summarize
    ):
        """Test summarize handles exceptions."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/bart-large-cnn"
        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-large-cnn"
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model
        mock_summarize.side_effect = ValueError("Summarization failed")

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("Summarization failed", str(context.exception))

    @patch("podcast_scraper.ml.ml_provider.summarizer.unload_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_cleanup(self, mock_select_model, mock_select_reduce, mock_summary_model, mock_unload):
        """Test cleanup method."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/bart-large-cnn"
        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-large-cnn"
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        provider.cleanup()

        self.assertIsNone(provider._map_model)
        self.assertIsNone(provider._reduce_model)
        self.assertFalse(provider._transformers_initialized)
        mock_unload.assert_called_once()

    @patch("podcast_scraper.ml.ml_provider.summarizer.unload_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_cleanup_with_different_reduce_model(
        self, mock_select_model, mock_select_reduce, mock_summary_model, mock_unload
    ):
        """Test cleanup with different reduce model."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/pegasus-xsum"
        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-large-cnn"
        mock_map_model.device = "cpu"
        mock_reduce_model = Mock()
        mock_reduce_model.model_name = "facebook/pegasus-xsum"
        mock_reduce_model.device = "cpu"
        mock_summary_model.side_effect = [mock_map_model, mock_reduce_model]

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        provider.cleanup()

        # Should unload both models
        self.assertEqual(mock_unload.call_count, 2)

    def test_cleanup_not_initialized(self):
        """Test cleanup when not initialized."""
        provider = create_summarization_provider(self.cfg)
        provider.cleanup()  # Should not raise

    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_map_model_property(self, mock_select_model, mock_select_reduce, mock_summary_model):
        """Test map_model property."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/bart-large-cnn"
        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-large-cnn"
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model

        provider = create_summarization_provider(self.cfg)
        self.assertIsNone(provider.map_model)

        provider.initialize()
        self.assertEqual(provider.map_model, mock_map_model)

    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_reduce_model_property(self, mock_select_model, mock_select_reduce, mock_summary_model):
        """Test reduce_model property."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/bart-large-cnn"
        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-large-cnn"
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model

        provider = create_summarization_provider(self.cfg)
        self.assertIsNone(provider.reduce_model)

        provider.initialize()
        self.assertEqual(provider.reduce_model, mock_map_model)

    def test_is_initialized_property(self):
        """Test is_initialized property."""
        provider = create_summarization_provider(self.cfg)
        self.assertFalse(provider.is_initialized)

        # Set transformers_initialized to True to test the property
        provider._transformers_initialized = True
        self.assertTrue(provider.is_initialized)

    @patch("podcast_scraper.ml.ml_provider.summarizer.summarize_long_text")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_summarize_auto_detects_word_chunking(
        self, mock_select_model, mock_select_reduce, mock_summary_model, mock_summarize
    ):
        """Test summarize auto-detects word chunking for BART models."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/bart-large-cnn"
        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-large-cnn"
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model
        mock_summarize.return_value = "Summary"

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        provider.summarize("Text")

        # Verify use_word_chunking was auto-detected as True for BART
        call_args = mock_summarize.call_args
        self.assertTrue(call_args.kwargs.get("use_word_chunking", False))

    @patch("podcast_scraper.ml.ml_provider.summarizer.summarize_long_text")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    def test_summarize_with_prompt(
        self, mock_select_model, mock_select_reduce, mock_summary_model, mock_summarize
    ):
        """Test summarize with custom prompt."""
        mock_select_model.return_value = "facebook/bart-large-cnn"
        mock_select_reduce.return_value = "facebook/bart-large-cnn"
        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-large-cnn"
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model
        mock_summarize.return_value = "Summary"

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        params = {"prompt": "Summarize this podcast episode"}
        provider.summarize("Text", params=params)

        call_args = mock_summarize.call_args
        self.assertEqual(call_args.kwargs["prompt"], "Summarize this podcast episode")


if __name__ == "__main__":
    unittest.main()
