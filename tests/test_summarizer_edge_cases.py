#!/usr/bin/env python3
"""Additional tests for summarization edge cases and error conditions."""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to import summarizer, skip tests if dependencies not available
try:
    from podcast_scraper import summarizer

    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    summarizer = types.ModuleType("summarizer")  # type: ignore[assignment]


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestModelLoadingFailures(unittest.TestCase):
    """Test error conditions during model loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.summarizer.AutoTokenizer")
    @patch("podcast_scraper.summarizer.AutoModelForSeq2SeqLM")
    @patch("podcast_scraper.summarizer.torch")
    def test_tokenizer_loading_failure(self, mock_torch, mock_model_class, mock_tokenizer_class):
        """Test that tokenizer loading failure raises appropriate error."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        # Simulate tokenizer loading failure
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network timeout")

        with self.assertRaises(Exception) as context:
            summarizer.SummaryModel(
                model_name="facebook/bart-base",
                device=None,
                cache_dir=self.temp_dir,
            )
        self.assertIn("Network timeout", str(context.exception))

    @patch("podcast_scraper.summarizer.AutoTokenizer")
    @patch("podcast_scraper.summarizer.AutoModelForSeq2SeqLM")
    @patch("podcast_scraper.summarizer.torch")
    def test_model_loading_failure(self, mock_torch, mock_model_class, mock_tokenizer_class):
        """Test that model loading failure raises appropriate error."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Simulate model loading failure
        mock_model_class.from_pretrained.side_effect = OSError("Model not found")

        with self.assertRaises(OSError) as context:
            summarizer.SummaryModel(
                model_name="invalid/model-name",
                device=None,
                cache_dir=self.temp_dir,
            )
        self.assertIn("Model not found", str(context.exception))

    @patch("podcast_scraper.summarizer.AutoTokenizer")
    @patch("podcast_scraper.summarizer.AutoModelForSeq2SeqLM")
    @patch("podcast_scraper.summarizer.pipeline")
    @patch("podcast_scraper.summarizer.torch")
    def test_pipeline_creation_failure(
        self, mock_torch, mock_pipeline, mock_model_class, mock_tokenizer_class
    ):
        """Test that pipeline creation failure raises appropriate error."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Simulate pipeline creation failure
        mock_pipeline.side_effect = RuntimeError("Pipeline initialization failed")

        with self.assertRaises(RuntimeError) as context:
            summarizer.SummaryModel(
                model_name="facebook/bart-base",
                device=None,
                cache_dir=self.temp_dir,
            )
        self.assertIn("Pipeline initialization failed", str(context.exception))


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestMemoryCleanup(unittest.TestCase):
    """Test memory cleanup and model unloading."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.summarizer.AutoTokenizer")
    @patch("podcast_scraper.summarizer.AutoModelForSeq2SeqLM")
    @patch("podcast_scraper.summarizer.pipeline")
    @patch("podcast_scraper.summarizer.torch")
    def test_unload_model_sets_to_none(
        self, mock_torch, mock_pipeline, mock_model_class, mock_tokenizer_class
    ):
        """Test that unload_model() sets model attributes to None."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe

        model = summarizer.SummaryModel(
            model_name="facebook/bart-base",
            device="cpu",
            cache_dir=self.temp_dir,
        )

        # Verify model is loaded
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.tokenizer)
        self.assertIsNotNone(model.pipeline)

        # Unload model
        summarizer.unload_model(model)

        # Verify model is unloaded
        self.assertIsNone(model.model)
        self.assertIsNone(model.tokenizer)
        self.assertIsNone(model.pipeline)

    def test_unload_model_with_none(self):
        """Test that unload_model() handles None gracefully."""
        # Should not raise an exception
        summarizer.unload_model(None)

    @patch("podcast_scraper.summarizer.AutoTokenizer")
    @patch("podcast_scraper.summarizer.AutoModelForSeq2SeqLM")
    @patch("podcast_scraper.summarizer.pipeline")
    @patch("podcast_scraper.summarizer.torch")
    def test_unload_model_twice(
        self, mock_torch, mock_pipeline, mock_model_class, mock_tokenizer_class
    ):
        """Test that unload_model() can be called multiple times safely."""
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe

        model = summarizer.SummaryModel(
            model_name="facebook/bart-base",
            device="cpu",
            cache_dir=self.temp_dir,
        )

        # Unload twice - should not raise an exception
        summarizer.unload_model(model)
        summarizer.unload_model(model)

        # Verify still None
        self.assertIsNone(model.model)


if __name__ == "__main__":
    unittest.main()
