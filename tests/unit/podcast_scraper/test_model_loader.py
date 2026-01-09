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
    _resolve_model_alias,
    preload_transformers_models,
    preload_whisper_models,
)

# Check if ML dependencies are available
try:
    import whisper  # noqa: F401

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper dependencies not available")
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

        # Create model file to test the exists() path
        model_file = self.whisper_cache / "tiny.en.pt"
        model_file.touch()

        # Test with explicit model name
        preload_whisper_models(["tiny.en"])

        mock_load_model.assert_called_once_with("tiny.en", download_root=str(self.whisper_cache))

    @patch("podcast_scraper.model_loader.get_whisper_cache_dir")
    @patch("whisper.load_model")
    @patch("podcast_scraper.model_loader.logger")
    def test_preload_whisper_models_model_file_not_exists(
        self, mock_logger, mock_load_model, mock_get_cache
    ):
        """Test Whisper model preloading when model file doesn't exist after load."""
        mock_get_cache.return_value = self.whisper_cache
        mock_model = Mock()
        mock_model.dims = Mock()
        mock_load_model.return_value = mock_model

        # Don't create model file - test the else branch
        preload_whisper_models(["tiny.en"])

        mock_load_model.assert_called_once_with("tiny.en", download_root=str(self.whisper_cache))
        # Verify info message was logged (the else branch)
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertTrue(
            any("✓ Whisper tiny.en cached" in msg and "MB" not in msg for msg in info_calls)
        )

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
    def test_preload_whisper_models_from_env_with_whitespace(self, mock_load_model, mock_get_cache):
        """Test Whisper model preloading from environment variable with whitespace."""
        mock_get_cache.return_value = self.whisper_cache
        mock_model = Mock()
        mock_model.dims = Mock()
        mock_load_model.return_value = mock_model

        # Set environment variable with whitespace
        os.environ["WHISPER_MODELS"] = " tiny.en , base.en "

        try:
            preload_whisper_models()
            # Should be called twice (one for each model, whitespace stripped)
            self.assertEqual(mock_load_model.call_count, 2)
            # Verify models were stripped
            calls = [call[0][0] for call in mock_load_model.call_args_list]
            self.assertIn("tiny.en", calls)
            self.assertIn("base.en", calls)
        finally:
            os.environ.pop("WHISPER_MODELS", None)

    @patch("podcast_scraper.model_loader.get_whisper_cache_dir")
    @patch("whisper.load_model")
    @patch("podcast_scraper.model_loader.logger")
    def test_preload_whisper_models_from_env_empty_after_strip(
        self, mock_logger, mock_load_model, mock_get_cache
    ):
        """Test Whisper model preloading from environment variable that's empty after strip."""
        mock_get_cache.return_value = self.whisper_cache

        # Set environment variable to whitespace only (after split and strip, results in empty list)
        os.environ["WHISPER_MODELS"] = "   ,  ,  "

        try:
            preload_whisper_models()
            # After splitting and filtering, list is empty, so should return early
            mock_load_model.assert_not_called()
            # Verify debug message was logged
            mock_logger.debug.assert_called_once()
            self.assertIn("Skipping Whisper model preloading", mock_logger.debug.call_args[0][0])
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
    @patch("podcast_scraper.model_loader.logger")
    def test_preload_whisper_models_empty_list(self, mock_logger, mock_load_model, mock_get_cache):
        """Test that empty model list is skipped."""
        mock_get_cache.return_value = self.whisper_cache

        preload_whisper_models([])
        mock_load_model.assert_not_called()
        # Verify debug message was logged
        mock_logger.debug.assert_called_once()
        self.assertIn("Skipping Whisper model preloading", mock_logger.debug.call_args[0][0])

    @patch("podcast_scraper.model_loader.get_whisper_cache_dir")
    @patch("podcast_scraper.model_loader.logger")
    def test_preload_whisper_models_import_error(self, mock_logger, mock_get_cache):
        """Test that ImportError is raised when whisper is not installed."""
        mock_get_cache.return_value = self.whisper_cache

        with patch.dict("sys.modules", {"whisper": None}):
            with self.assertRaises(ImportError):
                preload_whisper_models(["tiny.en"])
            # Verify error was logged
            mock_logger.error.assert_called_once()
            self.assertIn("openai-whisper not installed", mock_logger.error.call_args[0][0])

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
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers dependencies not available")
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
    @patch("podcast_scraper.model_loader.logger")
    def test_preload_transformers_models_not_cached_path(
        self, mock_logger, mock_tokenizer_class, mock_model_class, mock_get_cache
    ):
        """Test Transformers model preloading when model is not cached (else branch)."""
        mock_get_cache.return_value = self.transformers_cache
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Don't create cache directory - test the "Not cached, downloading..." path
        preload_transformers_models(["facebook/bart-base"])

        # Verify "Not cached, downloading..." message was logged
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertTrue(any("Not cached, downloading" in msg for msg in info_calls))

    @patch("podcast_scraper.model_loader.get_transformers_cache_dir")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    @patch("podcast_scraper.model_loader.logger")
    def test_preload_transformers_models_cache_not_exists_after_download(
        self, mock_logger, mock_tokenizer_class, mock_model_class, mock_get_cache
    ):
        """Test Transformers model preloading when cache doesn't exist after download."""
        mock_get_cache.return_value = self.transformers_cache
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Don't create cache directory - test the else branch after download
        preload_transformers_models(["facebook/bart-base"])

        # Verify the else branch message was logged (model name without size)
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertTrue(
            any("✓ Downloaded and cached: facebook/bart-base" in msg for msg in info_calls)
        )

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
    @patch("podcast_scraper.model_loader.logger")
    def test_preload_transformers_models_empty_list(
        self, mock_logger, mock_tokenizer, mock_get_cache
    ):
        """Test that empty model list is skipped."""
        mock_get_cache.return_value = self.transformers_cache

        preload_transformers_models([])
        mock_tokenizer.from_pretrained.assert_not_called()
        # Verify debug message was logged
        mock_logger.debug.assert_called_once()
        self.assertIn("Skipping Transformers model preloading", mock_logger.debug.call_args[0][0])

    @patch("podcast_scraper.model_loader.get_transformers_cache_dir")
    @patch("podcast_scraper.model_loader.logger")
    def test_preload_transformers_models_import_error(self, mock_logger, mock_get_cache):
        """Test that ImportError is raised when transformers is not installed."""
        mock_get_cache.return_value = self.transformers_cache

        with patch.dict("sys.modules", {"transformers": None}):
            with self.assertRaises(ImportError):
                preload_transformers_models(["facebook/bart-base"])
            # Verify error was logged
            mock_logger.error.assert_called_once()
            self.assertIn("transformers not installed", mock_logger.error.call_args[0][0])

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


@pytest.mark.unit
class TestResolveModelAlias(unittest.TestCase):
    """Tests for _resolve_model_alias function."""

    def test_resolve_bart_small_alias(self):
        """Test that bart-small alias resolves to facebook/bart-base."""
        result = _resolve_model_alias("bart-small")
        self.assertEqual(result, "facebook/bart-base")

    def test_resolve_bart_large_alias(self):
        """Test that bart-large alias resolves to facebook/bart-large-cnn."""
        result = _resolve_model_alias("bart-large")
        self.assertEqual(result, "facebook/bart-large-cnn")

    def test_resolve_long_alias(self):
        """Test that long alias resolves to allenai/led-large-16384."""
        result = _resolve_model_alias("long")
        self.assertEqual(result, "allenai/led-large-16384")

    def test_resolve_long_fast_alias(self):
        """Test that long-fast alias resolves to allenai/led-base-16384."""
        result = _resolve_model_alias("long-fast")
        self.assertEqual(result, "allenai/led-base-16384")

    def test_resolve_direct_model_id(self):
        """Test that direct HuggingFace model IDs are returned unchanged."""
        result = _resolve_model_alias("facebook/bart-base")
        self.assertEqual(result, "facebook/bart-base")

    def test_resolve_unknown_returns_unchanged(self):
        """Test that unknown model names are returned unchanged."""
        result = _resolve_model_alias("some/unknown-model")
        self.assertEqual(result, "some/unknown-model")


@pytest.mark.unit
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers dependencies not available")
class TestPreloadTransformersWithAliases(unittest.TestCase):
    """Tests for preload_transformers_models with alias resolution."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.transformers_cache = Path(self.temp_dir) / "transformers"
        self.transformers_cache.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("podcast_scraper.model_loader.get_transformers_cache_dir")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_preload_with_alias_resolves_to_real_id(
        self, mock_tokenizer_class, mock_model_class, mock_get_cache
    ):
        """Test that aliases are resolved before download."""
        mock_get_cache.return_value = self.transformers_cache
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Use alias
        preload_transformers_models(["bart-small"])

        # Should be called with resolved model ID
        mock_tokenizer_class.from_pretrained.assert_called_once()
        call_args = mock_tokenizer_class.from_pretrained.call_args
        self.assertEqual(call_args[0][0], "facebook/bart-base")

    @patch("podcast_scraper.model_loader.get_transformers_cache_dir")
    @patch("transformers.AutoModelForSeq2SeqLM")
    @patch("transformers.AutoTokenizer")
    def test_preload_deduplicates_resolved_models(
        self, mock_tokenizer_class, mock_model_class, mock_get_cache
    ):
        """Test that duplicate models (after resolution) are deduplicated."""
        mock_get_cache.return_value = self.transformers_cache
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Both resolve to same model
        preload_transformers_models(["bart-small", "facebook/bart-base"])

        # Should only download once
        self.assertEqual(mock_tokenizer_class.from_pretrained.call_count, 1)
