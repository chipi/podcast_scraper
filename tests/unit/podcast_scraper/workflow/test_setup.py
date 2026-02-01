"""Unit tests for podcast_scraper.workflow.stages.setup module.

This module tests the pipeline setup functionality, including ML model
caching and environment initialization.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from podcast_scraper.workflow.stages.setup import (
    ensure_ml_models_cached,
    should_preload_ml_models,
)


@pytest.mark.unit
class TestShouldPreloadMLModels(unittest.TestCase):
    """Tests for should_preload_ml_models function."""

    def test_returns_true_when_transcribe_missing_with_whisper(self):
        """Test returns True when transcribe_missing is True and provider is whisper."""
        cfg = Mock()
        cfg.transcribe_missing = True
        cfg.transcription_provider = "whisper"
        cfg.generate_summaries = False
        cfg.summary_provider = "openai"

        result = should_preload_ml_models(cfg)
        self.assertTrue(result)

    def test_returns_true_when_generate_summaries_with_transformers(self):
        """Test returns True when generate_summaries is True and provider is transformers."""
        cfg = Mock()
        cfg.transcribe_missing = False
        cfg.transcription_provider = "openai"
        cfg.generate_summaries = True
        cfg.summary_provider = "transformers"

        result = should_preload_ml_models(cfg)
        self.assertTrue(result)

    def test_returns_false_when_no_ml_providers(self):
        """Test returns False when no ML providers are configured."""
        cfg = Mock()
        cfg.transcribe_missing = True
        cfg.transcription_provider = "openai"
        cfg.generate_summaries = True
        cfg.summary_provider = "openai"

        result = should_preload_ml_models(cfg)
        self.assertFalse(result)

    def test_returns_false_when_features_disabled(self):
        """Test returns False when both features are disabled."""
        cfg = Mock()
        cfg.transcribe_missing = False
        cfg.transcription_provider = "whisper"
        cfg.generate_summaries = False
        cfg.summary_provider = "transformers"

        result = should_preload_ml_models(cfg)
        self.assertFalse(result)


@pytest.mark.unit
class TestEnsureMLModelsCached(unittest.TestCase):
    """Tests for ensure_ml_models_cached function."""

    def test_skips_when_preload_disabled(self):
        """Test that function returns early when preload_models is False."""
        cfg = Mock()
        cfg.preload_models = False
        cfg.dry_run = False

        # Should return without doing anything
        ensure_ml_models_cached(cfg)
        # No assertions needed - just shouldn't raise

    def test_skips_when_dry_run(self):
        """Test that function returns early during dry run."""
        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = True

        # Should return without doing anything
        ensure_ml_models_cached(cfg)
        # No assertions needed - just shouldn't raise

    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    def test_skips_when_no_ml_models_needed(self, mock_should_preload):
        """Test that function returns early when no ML models are needed."""
        mock_should_preload.return_value = False
        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False

        ensure_ml_models_cached(cfg)
        mock_should_preload.assert_called_once_with(cfg)

    @patch("podcast_scraper.workflow.stages.setup.config._is_test_environment")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    def test_skips_in_test_environment(self, mock_should_preload, mock_is_test):
        """Test that function returns early in test environment."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = True
        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False

        ensure_ml_models_cached(cfg)
        mock_is_test.assert_called_once()

    @patch("podcast_scraper.config._is_test_environment")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    @patch("podcast_scraper.cache.get_whisper_cache_dir")
    def test_checks_whisper_model_cache(self, mock_get_whisper, mock_should_preload, mock_is_test):
        """Test that function checks if Whisper model is cached."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = False

        # Create a mock path that returns False for exists()
        mock_cache_dir = MagicMock(spec=Path)
        mock_model_file = MagicMock(spec=Path)
        mock_model_file.exists.return_value = True  # Model exists, no download needed
        mock_cache_dir.__truediv__ = Mock(return_value=mock_model_file)
        mock_get_whisper.return_value = mock_cache_dir

        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False
        cfg.transcribe_missing = True
        cfg.transcription_provider = "whisper"
        cfg.whisper_model = "tiny.en"
        cfg.generate_summaries = False
        cfg.summary_provider = "openai"

        ensure_ml_models_cached(cfg)

        # Should check if model file exists
        mock_model_file.exists.assert_called()

    @patch("podcast_scraper.config._is_test_environment")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    @patch("podcast_scraper.providers.ml.model_loader.preload_whisper_models")
    @patch("podcast_scraper.cache.get_whisper_cache_dir")
    def test_downloads_missing_whisper_model(
        self, mock_get_whisper, mock_preload_whisper, mock_should_preload, mock_is_test
    ):
        """Test that function downloads missing Whisper model."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = False

        # Create a mock path that returns False for exists() - model not cached
        mock_cache_dir = MagicMock(spec=Path)
        mock_model_file = MagicMock(spec=Path)
        mock_model_file.exists.return_value = False  # Model NOT cached
        mock_cache_dir.__truediv__ = Mock(return_value=mock_model_file)
        mock_get_whisper.return_value = mock_cache_dir

        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False
        cfg.transcribe_missing = True
        cfg.transcription_provider = "whisper"
        cfg.whisper_model = "tiny.en"
        cfg.generate_summaries = False
        cfg.summary_provider = "openai"

        ensure_ml_models_cached(cfg)

        # Should call preload_whisper_models
        mock_preload_whisper.assert_called_once_with(["tiny.en"])

    @patch("podcast_scraper.config._is_test_environment")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    @patch("podcast_scraper.providers.ml.summarizer.select_reduce_model")
    @patch("podcast_scraper.providers.ml.summarizer.select_summary_model")
    @patch("podcast_scraper.cache.get_transformers_cache_dir")
    @patch("podcast_scraper.cache.get_whisper_cache_dir")
    def test_checks_transformers_model_cache(
        self,
        mock_get_whisper,
        mock_get_transformers,
        mock_select_summary,
        mock_select_reduce,
        mock_should_preload,
        mock_is_test,
    ):
        """Test that function checks if Transformers model is cached."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = False

        # Setup summarizer mocks
        mock_select_summary.return_value = "bart-small"
        mock_select_reduce.return_value = "bart-small"

        # Create mock paths
        mock_transformers_cache = MagicMock(spec=Path)
        mock_model_path = MagicMock(spec=Path)
        mock_model_path.exists.return_value = True  # Model exists
        mock_transformers_cache.__truediv__ = Mock(return_value=mock_model_path)
        mock_get_transformers.return_value = mock_transformers_cache

        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False
        cfg.transcribe_missing = False
        cfg.transcription_provider = "openai"
        cfg.generate_summaries = True
        cfg.summary_provider = "transformers"

        ensure_ml_models_cached(cfg)

        # Should check model selection
        mock_select_summary.assert_called_once_with(cfg)

    @patch("podcast_scraper.config._is_test_environment")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    @patch("podcast_scraper.providers.ml.model_loader.preload_transformers_models")
    @patch("podcast_scraper.providers.ml.summarizer.select_reduce_model")
    @patch("podcast_scraper.providers.ml.summarizer.select_summary_model")
    @patch("podcast_scraper.cache.get_transformers_cache_dir")
    @patch("podcast_scraper.cache.get_whisper_cache_dir")
    def test_downloads_missing_transformers_model(
        self,
        mock_get_whisper,
        mock_get_transformers,
        mock_select_summary,
        mock_select_reduce,
        mock_preload_transformers,
        mock_should_preload,
        mock_is_test,
    ):
        """Test that function downloads missing Transformers model."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = False

        # Setup summarizer mocks
        mock_select_summary.return_value = "bart-small"
        mock_select_reduce.return_value = "bart-small"

        # Create mock paths - model NOT cached
        mock_transformers_cache = MagicMock(spec=Path)
        mock_model_path = MagicMock(spec=Path)
        mock_model_path.exists.return_value = False  # Model NOT cached
        mock_transformers_cache.__truediv__ = Mock(return_value=mock_model_path)
        mock_get_transformers.return_value = mock_transformers_cache

        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False
        cfg.transcribe_missing = False
        cfg.transcription_provider = "openai"
        cfg.generate_summaries = True
        cfg.summary_provider = "transformers"

        ensure_ml_models_cached(cfg)

        # Should call preload_transformers_models
        mock_preload_transformers.assert_called_once_with(["bart-small"])

    @patch("podcast_scraper.config._is_test_environment")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    @patch("podcast_scraper.workflow.stages.setup.logger")
    def test_handles_import_error_gracefully(self, mock_logger, mock_should_preload, mock_is_test):
        """Test that function handles ImportError gracefully."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = False

        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False
        cfg.transcribe_missing = True
        cfg.transcription_provider = "whisper"

        # Patch to raise ImportError when importing cache module
        with patch(
            "podcast_scraper.cache.get_whisper_cache_dir",
            side_effect=ImportError("cache module not available"),
        ):
            # Should not raise
            ensure_ml_models_cached(cfg)

    @patch("podcast_scraper.config._is_test_environment")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    @patch("podcast_scraper.workflow.stages.setup.logger")
    def test_handles_general_exception_gracefully(
        self, mock_logger, mock_should_preload, mock_is_test
    ):
        """Test that function handles general exceptions gracefully."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = False

        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False
        cfg.transcribe_missing = True
        cfg.transcription_provider = "whisper"

        # Patch to raise general exception at the source module level
        with patch(
            "podcast_scraper.cache.get_whisper_cache_dir",
            side_effect=RuntimeError("Unexpected error"),
        ):
            # Should not raise
            ensure_ml_models_cached(cfg)
            # Should log debug message
            mock_logger.debug.assert_called()


if __name__ == "__main__":
    unittest.main()
