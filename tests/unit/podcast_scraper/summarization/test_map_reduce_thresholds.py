"""Unit tests for model-specific threshold logic (Issue #283).

Tests verify that LED and BART models use different thresholds and transition zones.
Updated to work with refactored code in providers.ml.summarizer module.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, Mock, patch

from podcast_scraper.providers.ml.summarizer import (
    _combine_summaries_reduce,
    LONG_CONTEXT_THRESHOLD,
    MINI_MAP_REDUCE_MAX_TOKENS,
    MINI_MAP_REDUCE_THRESHOLD,
    SUMMARY_VALIDATION_THRESHOLD,
)


def _add_summarize_lock_to_mock(mock_model):
    """Helper to add _summarize_lock to mock model that supports context manager protocol."""
    mock_model._summarize_lock = MagicMock()
    mock_model._summarize_lock.__enter__ = Mock(return_value=None)
    mock_model._summarize_lock.__exit__ = Mock(return_value=None)
    return mock_model


class TestModelSpecificThresholds(unittest.TestCase):
    """Test model-specific threshold selection and transition zones."""

    def _create_led_model_mock(self, max_position: int = 16384):
        """Create a mock model that simulates LED model config."""
        from unittest.mock import MagicMock

        from podcast_scraper.providers.ml.summarizer import SummaryModel

        mock_tokenizer = Mock()
        mock_model = Mock(spec=SummaryModel)
        mock_model.tokenizer = mock_tokenizer
        mock_model.model = Mock()
        # LED models have max_encoder_position_embeddings
        # But the code checks max_position_embeddings first, so we need both
        mock_model.model.config = Mock(
            spec=["max_position_embeddings", "max_encoder_position_embeddings"],
            max_position_embeddings=max_position,  # Code checks this first
            max_encoder_position_embeddings=max_position,
        )
        _add_summarize_lock_to_mock(mock_model)
        return mock_model, mock_tokenizer

    def _create_bart_model_mock(self, max_position: int = 1024):
        """Create a mock model that simulates BART model config."""
        from unittest.mock import MagicMock

        from podcast_scraper.providers.ml.summarizer import SummaryModel

        mock_tokenizer = Mock()
        mock_model = Mock(spec=SummaryModel)
        mock_model.tokenizer = mock_tokenizer
        mock_model.model = Mock()
        # BART models have max_position_embeddings (not max_encoder_position_embeddings)
        mock_model.model.config = Mock(
            spec=["max_position_embeddings"],
            max_position_embeddings=max_position,
        )
        _add_summarize_lock_to_mock(mock_model)
        return mock_model, mock_tokenizer

    def setUp(self):
        """Set up test fixtures."""
        # Default to LED model for backwards compatibility
        self.mock_model, self.mock_tokenizer = self._create_led_model_mock()

    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_abstractive")
    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_extractive")
    def test_led_model_uses_correct_thresholds(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test that LED models (model_max >= 4096) use correct thresholds."""
        # LED model: 16384 max encoder position embeddings
        mock_model, mock_tokenizer = self._create_led_model_mock(16384)
        # Mock tokenizer.encode to return 3000 tokens (below single_pass_limit for LED)
        # The function calls _join_summaries_with_structure which creates combined text
        # then tokenizer.encode is called on that combined text
        # For LED: model_max=16384, usable_context=16184, single_pass_limit=min(16184, 9710)=9710
        # So 3000 tokens < 9710 → should call abstractive
        mock_tokenizer.encode.return_value = list(range(3000))  # 3k tokens

        mock_abstractive.return_value = "Summary"

        _combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1", "Summary 2"],
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # For LED with 3k tokens, should use abstractive (single-pass)
        # single_pass_limit for LED (16384) is ~9710, so 3k < 9710 → abstractive
        # Verify abstractive was called
        mock_abstractive.assert_called_once()
        mock_mini_map_reduce.assert_not_called()
        mock_extractive.assert_not_called()

    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_abstractive")
    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_extractive")
    def test_bart_model_uses_correct_thresholds(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test that BART models (model_max < 4096) use correct thresholds."""
        # BART model: 1024 max position embeddings
        mock_model, mock_tokenizer = self._create_bart_model_mock(1024)
        # Create combined text that will result in ~500 tokens
        mock_tokenizer.encode.return_value = list(range(500))  # 500 tokens

        mock_abstractive.return_value = "Summary"

        _combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1", "Summary 2"],
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # For BART with 500 tokens, should use abstractive (single-pass)
        # Verify abstractive was called
        mock_abstractive.assert_called_once()

    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_abstractive")
    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_extractive")
    def test_led_model_large_input_uses_hierarchical(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test LED model with large input uses hierarchical reduce."""
        # LED model
        mock_model, mock_tokenizer = self._create_led_model_mock(16384)
        # Large input: > single_pass_limit but < mini_map_reduce_ceiling
        # For LED: single_pass_limit ~9710, mini_map_reduce_ceiling=usable_context (~16184)
        # Use 10000 tokens: > 9710 (single_pass_limit) but < 16184 (ceiling)
        mock_tokenizer.encode.return_value = list(range(10000))  # 10k tokens

        mock_mini_map_reduce.return_value = "Summary"

        _combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1"] * 10,  # Multiple summaries
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # For 10k tokens with LED, should use hierarchical reduce
        # (between single_pass_limit ~9710 and mini_map_reduce_ceiling ~16184)
        mock_mini_map_reduce.assert_called_once()

    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_abstractive")
    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_extractive")
    def test_led_model_very_large_input_uses_extractive(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test LED model with very large input uses extractive."""
        # LED model
        mock_model, mock_tokenizer = self._create_led_model_mock(16384)
        # Very large input: > mini_map_reduce_ceiling
        # For LED: mini_map_reduce_ceiling = usable_context (~16184)
        # So we need > 16184 tokens to trigger extractive
        mock_tokenizer.encode.return_value = list(range(17000))  # 17k tokens > 16184

        mock_extractive.return_value = "Summary"

        _combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1"] * 20,  # Many summaries
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # For very large input with LED (> 16184 tokens), should use extractive
        mock_extractive.assert_called_once()
        mock_mini_map_reduce.assert_not_called()
        mock_abstractive.assert_not_called()

    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_abstractive")
    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_extractive")
    def test_bart_model_large_input_uses_hierarchical(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test BART model with large input uses hierarchical reduce."""
        # BART model
        mock_model, mock_tokenizer = self._create_bart_model_mock(1024)
        # Large input: > single_pass_limit but < mini_map_reduce_ceiling
        # For BART: single_pass_limit ~600, so use 2000 tokens
        mock_tokenizer.encode.return_value = list(range(2000))  # 2k tokens

        mock_mini_map_reduce.return_value = "Summary"

        _combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1"] * 10,  # Multiple summaries
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # For 2k tokens with BART, should use hierarchical reduce
        mock_mini_map_reduce.assert_called_once()

    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_abstractive")
    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.providers.ml.summarizer._combine_summaries_extractive")
    def test_bart_model_very_large_input_uses_extractive(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test BART model with very large input uses extractive."""
        # BART model
        mock_model, mock_tokenizer = self._create_bart_model_mock(1024)
        # Very large input: > mini_map_reduce_ceiling (4000 tokens)
        mock_tokenizer.encode.return_value = list(range(5000))  # 5k tokens

        mock_extractive.return_value = "Summary"

        _combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1"] * 20,  # Many summaries
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # For very large input with BART, should use extractive
        mock_extractive.assert_called_once()

    def test_threshold_constants(self):
        """Test that threshold constants are correctly defined."""
        # Verify constants exist and have expected values
        self.assertEqual(MINI_MAP_REDUCE_THRESHOLD, 800)
        self.assertEqual(MINI_MAP_REDUCE_MAX_TOKENS, 4000)
        self.assertEqual(SUMMARY_VALIDATION_THRESHOLD, 0.6)
        self.assertEqual(LONG_CONTEXT_THRESHOLD, 4096)

        # Verify thresholds are logical
        self.assertLess(MINI_MAP_REDUCE_THRESHOLD, MINI_MAP_REDUCE_MAX_TOKENS)
        self.assertGreater(LONG_CONTEXT_THRESHOLD, 1024)  # Should be > BART max
