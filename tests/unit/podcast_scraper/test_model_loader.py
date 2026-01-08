"""Unit tests for podcast_scraper.model_loader module.

This module tests the centralized model download functions, which are the ONLY
place where ML models can be downloaded. All other code must use local_files_only=True.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.model_loader import (
    preload_transformers_models,
    preload_whisper_models,
)


@pytest.mark.unit
class TestModelLoaderWhisper(unittest.TestCase):
    """Tests for preload_whisper_models function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.whisper_cache = Path(self.temp_dir) / "whisper"
        self.whisper_cache.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.model_loader.get_whisper_cache_dir")
    @patch("whisper.load_model")
    def test_preload_whisper_models_success(self, mock_load_model, mock_get_cache):
        """Test successful Whisper model preloading."""
        mock_get_cache.return_value = self.whisper_cache
        mock_model = Mock()
        mock_model.dims = Mock()
        mock_load_model.return_value = mock_model

        # Test with explicit model name
        preload_whisper_models(["tiny.en"])

        mock_load_model.assert_called_once_with("tiny.en", download_root=str(self.whisper_cache))

    @patch("podcast_scraper.model_loader.get_whisper_cache_dir")
    @patch("whisper.load_model")
    def test_preload_whisper_models_from_env(self, mock_load_model, mock_get_cache):
        """Test Whisper model preloading from environment variable."""
        mock_get_cache.return_value = self.whisper_cache
        mock_model = Mock()
        mock_model.dims = Mock()
        mock_load_model.return_value = mock_model

        # Set environment variable
        os.environ["WHISPER_MODELS"] = "tiny.en,base.en"

        try:
            preload_whisper_models()
            # Should be called twice (one for each model)
            self.assertEqual(mock_load_model.call_count, 2)
        finally:
            os.environ.pop("WHISPER_MODELS", None)

    @patch("podcast_scraper.model_loader.get_whisper_cache_dir")
    @patch("whisper.load_model")
    def test_preload_whisper_models_default(self, mock_load_model, mock_get_cache):
        """Test Whisper model preloading uses default when no env var."""
        mock_get_cache.return_value = self.whisper_cache
        mock_model = Mock()
        mock_model.dims = Mock()
        mock_load_model.return_value = mock_model

        # Remove env var if it exists
        os.environ.pop("WHISPER_MODELS", None)

        preload_whisper_models()

        # Should use default test model
        mock_load_model.assert_called_once_with(
            config.TEST_DEFAULT_WHISPER_MODEL, download_root=str(self.whisper_cache)
        )

    @patch("podcast_scraper.model_loader.get_whisper_cache_dir")
    @patch("whisper.load_model")
    def test_preload_whisper_models_empty_list(self, mock_load_model, mock_get_cache):
        """Test that empty model list is skipped."""
        mock_get_cache.return_value = self.whisper_cache

        preload_whisper_models([])
        mock_load_model.assert_not_called()

    @patch("podcast_scraper.model_loader.get_whisper_cache_dir")
    def test_preload_whisper_models_import_error(self, mock_get_cache):
        """Test that ImportError is raised when whisper is not installed."""
        mock_get_cache.return_value = self.whisper_cache

        with patch.dict("sys.modules", {"whisper": None}):
            with self.assertRaises(ImportError):
                preload_whisper_models(["tiny.en"])

    @patch("podcast_scraper.model_loader.get_whisper_cache_dir")
    @patch("whisper.load_model")
    def test_preload_whisper_models_download_error(self, mock_load_model, mock_get_cache):
        """Test that download errors are raised."""
        mock_get_cache.return_value = self.whisper_cache
        mock_load_model.side_effect = Exception("Network error")

        with self.assertRaises(Exception) as context:
            preload_whisper_models(["tiny.en"])

        self.assertIn("Network error", str(context.exception))


@pytest.mark.unit
class TestModelLoaderTransformers(unittest.TestCase):
    """Tests for preload_transformers_models function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.transformers_cache = Path(self.temp_dir) / "huggingface" / "hub"
        self.transformers_cache.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.model_loader.get_transformers_cache_dir")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_preload_transformers_models_success(
        self, mock_tokenizer_class, mock_model_class, mock_get_cache
    ):
        """Test successful Transformers model preloading."""
        mock_get_cache.return_value = self.transformers_cache
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Test with explicit model name
        preload_transformers_models(["facebook/bart-base"])

        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "facebook/bart-base",
            cache_dir=str(self.transformers_cache),
            local_files_only=False,
        )
        mock_model_class.from_pretrained.assert_called_once_with(
            "facebook/bart-base",
            cache_dir=str(self.transformers_cache),
            local_files_only=False,
        )

    @patch("podcast_scraper.model_loader.get_transformers_cache_dir")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_preload_transformers_models_from_env(
        self, mock_tokenizer_class, mock_model_class, mock_get_cache
    ):
        """Test Transformers model preloading from environment variable."""
        mock_get_cache.return_value = self.transformers_cache
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Set environment variable
        os.environ["TRANSFORMERS_MODELS"] = "facebook/bart-base,allenai/led-base-16384"

        try:
            preload_transformers_models()
            # Should be called twice (one for each model)
            self.assertEqual(mock_tokenizer_class.from_pretrained.call_count, 2)
            self.assertEqual(mock_model_class.from_pretrained.call_count, 2)
        finally:
            os.environ.pop("TRANSFORMERS_MODELS", None)

    @patch("podcast_scraper.model_loader.get_transformers_cache_dir")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_preload_transformers_models_default(
        self, mock_tokenizer_class, mock_model_class, mock_get_cache
    ):
        """Test Transformers model preloading uses default when no env var."""
        mock_get_cache.return_value = self.transformers_cache
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Remove env var if it exists
        os.environ.pop("TRANSFORMERS_MODELS", None)

        preload_transformers_models()

        # Should use default test models
        self.assertEqual(mock_tokenizer_class.from_pretrained.call_count, 2)
        self.assertEqual(mock_model_class.from_pretrained.call_count, 2)

    @patch("podcast_scraper.model_loader.get_transformers_cache_dir")
    @patch("transformers.AutoTokenizer")
    def test_preload_transformers_models_empty_list(self, mock_tokenizer, mock_get_cache):
        """Test that empty model list is skipped."""
        mock_get_cache.return_value = self.transformers_cache

        preload_transformers_models([])
        mock_tokenizer.from_pretrained.assert_not_called()

    @patch("podcast_scraper.model_loader.get_transformers_cache_dir")
    def test_preload_transformers_models_import_error(self, mock_get_cache):
        """Test that ImportError is raised when transformers is not installed."""
        mock_get_cache.return_value = self.transformers_cache

        with patch.dict("sys.modules", {"transformers": None}):
            with self.assertRaises(ImportError):
                preload_transformers_models(["facebook/bart-base"])

    @patch("podcast_scraper.model_loader.get_transformers_cache_dir")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_preload_transformers_models_download_error(
        self, mock_tokenizer_class, mock_model_class, mock_get_cache
    ):
        """Test that download errors are raised."""
        mock_get_cache.return_value = self.transformers_cache
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")

        with self.assertRaises(Exception) as context:
            preload_transformers_models(["facebook/bart-base"])

        self.assertIn("Network error", str(context.exception))

    @patch("podcast_scraper.model_loader.get_transformers_cache_dir")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_preload_transformers_models_already_cached(
        self, mock_tokenizer_class, mock_model_class, mock_get_cache
    ):
        """Test that already cached models are detected."""
        mock_get_cache.return_value = self.transformers_cache
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Create fake cache directory
        model_cache_path = self.transformers_cache / "models--facebook--bart-base"
        model_cache_path.mkdir(parents=True, exist_ok=True)

        preload_transformers_models(["facebook/bart-base"])

        # Should still download (to verify cache is complete)
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
