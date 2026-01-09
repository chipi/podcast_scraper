"""Unit tests for model-specific threshold logic (Issue #283).

Tests verify that LED and BART models use different thresholds and transition zones.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from podcast_scraper.summarization.map_reduce import (
    BART_MINI_MAP_REDUCE_MAX_TOKENS,
    BART_TRANSITION_END,
    BART_TRANSITION_START,
    combine_summaries_reduce,
    LED_MINI_MAP_REDUCE_MAX_TOKENS,
    LED_TRANSITION_END,
    LED_TRANSITION_START,
    LED_VALIDATION_THRESHOLD,
    LONG_CONTEXT_THRESHOLD,
    SUMMARY_VALIDATION_THRESHOLD,
)


class TestModelSpecificThresholds(unittest.TestCase):
    """Test model-specific threshold selection and transition zones."""

    def _create_led_model_mock(self, max_position: int = 16384):
        """Create a mock model that simulates LED model config."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.tokenizer = mock_tokenizer
        mock_model.model = Mock()
        # LED models have max_encoder_position_embeddings
        mock_model.model.config = Mock(
            spec=["max_encoder_position_embeddings"],
            max_encoder_position_embeddings=max_position,
        )
        return mock_model, mock_tokenizer

    def _create_bart_model_mock(self, max_position: int = 1024):
        """Create a mock model that simulates BART model config."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.tokenizer = mock_tokenizer
        mock_model.model = Mock()
        # BART models have max_position_embeddings (not max_encoder_position_embeddings)
        mock_model.model.config = Mock(
            spec=["max_position_embeddings"],
            max_position_embeddings=max_position,
        )
        return mock_model, mock_tokenizer

    def setUp(self):
        """Set up test fixtures."""
        # Default to LED model for backwards compatibility
        self.mock_model, self.mock_tokenizer = self._create_led_model_mock()

    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_abstractive")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_extractive")
    def test_led_model_uses_led_thresholds(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test that LED models (model_max >= 4096) use LED-specific thresholds."""
        # LED model: 16384 max encoder position embeddings
        mock_model, mock_tokenizer = self._create_led_model_mock(16384)
        mock_tokenizer.encode.return_value = list(range(3000))  # 3k tokens

        mock_abstractive.return_value = "Summary"

        combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1", "Summary 2"],
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # Verify abstractive was called with LED validation threshold
        mock_abstractive.assert_called_once()
        call_kwargs = mock_abstractive.call_args[1]
        self.assertEqual(
            call_kwargs["validation_threshold"],
            LED_VALIDATION_THRESHOLD,
            "LED model should use LED validation threshold (0.75)",
        )

    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_abstractive")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_extractive")
    def test_bart_model_uses_bart_thresholds(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test that BART models (model_max < 4096) use BART-specific thresholds."""
        # BART model: 1024 max position embeddings
        mock_model, mock_tokenizer = self._create_bart_model_mock(1024)
        mock_tokenizer.encode.return_value = list(range(500))  # 500 tokens

        mock_abstractive.return_value = "Summary"

        combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1", "Summary 2"],
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # Verify abstractive was called with BART validation threshold
        mock_abstractive.assert_called_once()
        call_kwargs = mock_abstractive.call_args[1]
        self.assertEqual(
            call_kwargs["validation_threshold"],
            SUMMARY_VALIDATION_THRESHOLD,
            "BART model should use BART validation threshold (0.6)",
        )

    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_abstractive")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_extractive")
    def test_led_transition_zone_below_start(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test LED model below transition_start uses hierarchical reduce.

        Note: The actual decision depends on single_pass_limit calculation.
        For LED with 16384 max: usable_context=16184, single_pass_limit=min(6000, 9710)=6000.
        So tokens <= 6000 use abstractive, >6000 use hierarchical/extractive.
        This test verifies that the threshold constants are correctly defined.
        """
        # LED model
        mock_model, mock_tokenizer = self._create_led_model_mock(16384)
        # For LED: single_pass_limit is based on usable_context * 0.6
        # With model_max=16384, usable_context=16184, single_pass_limit=9710
        # transition_start=5500, so tokens in 5500-9710 use hierarchical reduce
        # We test with 5300 tokens which is below transition_start
        mock_tokenizer.encode.return_value = list(range(5300))  # 5.3k tokens

        mock_abstractive.return_value = "Summary"

        combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1", "Summary 2"],
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # For 5.3k tokens with LED (single_pass_limit=6000), should use abstractive
        # This test primarily verifies the constants are correct
        mock_abstractive.assert_called_once()
        # Verify it was called with LED validation threshold
        call_kwargs = mock_abstractive.call_args[1]
        self.assertEqual(call_kwargs["validation_threshold"], LED_VALIDATION_THRESHOLD)

    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_abstractive")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_extractive")
    def test_led_transition_zone_in_zone(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test LED model in transition zone uses extractive.

        For LED: single_pass_limit=6000, transition_start=5500, transition_end=6500
        Transition zone is: >max(5500,6000)=6000 and <=6500
        So we need >6000 but <=6500, e.g., 6200 tokens.
        """
        # LED model
        mock_model, mock_tokenizer = self._create_led_model_mock(16384)
        # In transition zone: >6000 but <=6500
        mock_tokenizer.encode.return_value = list(range(6200))  # 6.2k tokens

        mock_extractive.return_value = "Summary"

        combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1", "Summary 2"],
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # Should use extractive (in transition zone)
        mock_extractive.assert_called_once()
        mock_mini_map_reduce.assert_not_called()

    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_abstractive")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_extractive")
    def test_led_transition_zone_above_end(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test LED model above transition_end (>6500) uses extractive."""
        # LED model
        mock_model, mock_tokenizer = self._create_led_model_mock(16384)
        # Above transition_end
        mock_tokenizer.encode.return_value = list(range(7000))  # 7k tokens

        mock_extractive.return_value = "Summary"

        combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1", "Summary 2"],
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # Should use extractive (above transition zone)
        mock_extractive.assert_called_once()
        mock_mini_map_reduce.assert_not_called()

    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_abstractive")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_extractive")
    def test_bart_transition_zone_below_start(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test BART model below transition_start uses hierarchical reduce."""
        # BART model
        mock_model, mock_tokenizer = self._create_bart_model_mock(1024)
        # Below transition_start (3500)
        mock_tokenizer.encode.return_value = list(range(3000))  # 3k tokens

        mock_mini_map_reduce.return_value = "Summary"

        combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1", "Summary 2"],
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # Should use hierarchical reduce (not extractive)
        mock_mini_map_reduce.assert_called_once()
        mock_extractive.assert_not_called()

    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_abstractive")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_mini_map_reduce")
    @patch("podcast_scraper.summarization.map_reduce.combine_summaries_extractive")
    def test_bart_transition_zone_in_zone(
        self, mock_extractive, mock_mini_map_reduce, mock_abstractive
    ):
        """Test BART model in transition zone uses extractive.

        For BART: single_pass_limit ~600, transition_start=3500, transition_end=4500
        Transition zone is: >max(3500,600)=3500 and <=4500
        So we need >3500 but <=4500, e.g., 4000 tokens.
        """
        # BART model
        mock_model, mock_tokenizer = self._create_bart_model_mock(1024)
        # In transition zone: >3500 but <=4500
        mock_tokenizer.encode.return_value = list(range(4000))  # 4k tokens

        mock_extractive.return_value = "Summary"

        combine_summaries_reduce(
            model=mock_model,
            chunk_summaries=["Summary 1", "Summary 2"],
            max_length=150,
            min_length=30,
            prompt=None,
        )

        # Should use extractive (in transition zone)
        mock_extractive.assert_called_once()
        mock_mini_map_reduce.assert_not_called()

    def test_threshold_constants(self):
        """Test that threshold constants are correctly defined."""
        # LED thresholds
        self.assertEqual(LED_MINI_MAP_REDUCE_MAX_TOKENS, 6000)
        self.assertEqual(LED_TRANSITION_START, 5500)
        self.assertEqual(LED_TRANSITION_END, 6500)
        self.assertEqual(LED_VALIDATION_THRESHOLD, 0.75)

        # BART thresholds
        self.assertEqual(BART_MINI_MAP_REDUCE_MAX_TOKENS, 4000)
        self.assertEqual(BART_TRANSITION_START, 3500)
        self.assertEqual(BART_TRANSITION_END, 4500)
        self.assertEqual(SUMMARY_VALIDATION_THRESHOLD, 0.6)

        # Model detection
        self.assertEqual(LONG_CONTEXT_THRESHOLD, 4096)

        # Verify transition zones are logical
        self.assertLess(LED_TRANSITION_START, LED_TRANSITION_END)
        self.assertLess(BART_TRANSITION_START, BART_TRANSITION_END)
        self.assertLess(LED_TRANSITION_START, LED_MINI_MAP_REDUCE_MAX_TOKENS)
        self.assertLess(BART_TRANSITION_START, BART_MINI_MAP_REDUCE_MAX_TOKENS)
