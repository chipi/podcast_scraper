#!/usr/bin/env python3
"""Unit tests for transcript cleaning functionality."""

import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper.cleaning import (
    HybridCleaner,
    LLMBasedCleaner,
    PatternBasedCleaner,
)
from podcast_scraper.cleaning.base import TranscriptCleaningProcessor

pytestmark = [pytest.mark.unit]


class TestPatternBasedCleaner(unittest.TestCase):
    """Tests for PatternBasedCleaner."""

    def test_cleaner_initialization(self):
        """Test cleaner initialization."""
        cleaner = PatternBasedCleaner()
        self.assertIsInstance(cleaner, TranscriptCleaningProcessor)

    @patch("podcast_scraper.preprocessing.clean_for_summarization")
    def test_clean_calls_preprocessing(self, mock_clean):
        """Test that clean() calls preprocessing function."""
        mock_clean.return_value = "cleaned text"
        cleaner = PatternBasedCleaner()

        result = cleaner.clean("raw text")
        self.assertEqual(result, "cleaned text")
        mock_clean.assert_called_once_with("raw text")

    @patch("podcast_scraper.preprocessing.remove_sponsor_blocks")
    def test_remove_sponsors(self, mock_remove):
        """Test remove_sponsors() method."""
        mock_remove.return_value = "text without sponsors"
        cleaner = PatternBasedCleaner()

        result = cleaner.remove_sponsors("text with sponsors")
        self.assertEqual(result, "text without sponsors")
        mock_remove.assert_called_once_with("text with sponsors")

    @patch("podcast_scraper.preprocessing.remove_outro_blocks")
    def test_remove_outros(self, mock_remove):
        """Test remove_outros() method."""
        mock_remove.return_value = "text without outro"
        cleaner = PatternBasedCleaner()

        result = cleaner.remove_outros("text with outro")
        self.assertEqual(result, "text without outro")
        mock_remove.assert_called_once_with("text with outro")


class TestLLMBasedCleaner(unittest.TestCase):
    """Tests for LLMBasedCleaner."""

    def test_cleaner_initialization(self):
        """Test cleaner initialization."""
        cleaner = LLMBasedCleaner()
        self.assertIsNotNone(cleaner)

    def test_clean_requires_provider_with_method(self):
        """Test that clean() requires a provider with clean_transcript method."""
        cleaner = LLMBasedCleaner()

        provider_without_method = Mock(spec=[])

        with self.assertRaises(AttributeError) as context:
            cleaner.clean("text", provider_without_method)
        self.assertIn("does not support semantic cleaning", str(context.exception))

    def test_clean_calls_provider_clean_transcript(self):
        """Test that clean() calls provider's clean_transcript method."""
        cleaner = LLMBasedCleaner()

        mock_provider = Mock()
        mock_provider.clean_transcript.return_value = "llm cleaned text"

        result = cleaner.clean("raw text", mock_provider)
        self.assertEqual(result, "llm cleaned text")
        mock_provider.clean_transcript.assert_called_once_with("raw text", pipeline_metrics=None)

    def test_clean_handles_empty_result(self):
        """Test that clean() handles empty result from provider."""
        cleaner = LLMBasedCleaner()

        mock_provider = Mock()
        mock_provider.clean_transcript.return_value = ""

        result = cleaner.clean("original text", mock_provider)
        # Should return original text if LLM returns empty
        self.assertEqual(result, "original text")

    def test_clean_handles_none_result(self):
        """Test that clean() handles None result from provider."""
        cleaner = LLMBasedCleaner()

        mock_provider = Mock()
        mock_provider.clean_transcript.return_value = None

        result = cleaner.clean("original text", mock_provider)
        # Should return original text if LLM returns None
        self.assertEqual(result, "original text")

    def test_clean_handles_exception(self):
        """Test that clean() handles exceptions from provider."""
        cleaner = LLMBasedCleaner()

        mock_provider = Mock()
        mock_provider.clean_transcript.side_effect = RuntimeError("API error")

        result = cleaner.clean("original text", mock_provider)
        # Should return original text if LLM fails
        self.assertEqual(result, "original text")

    def test_clean_rejects_excessively_short_llm_output(self):
        """If LLM returns a tiny fraction of input, use pattern-cleaned text (bad model output)."""
        cleaner = LLMBasedCleaner()
        mock_provider = Mock()
        long_input = "substantive transcript token. " * 120
        self.assertGreaterEqual(len(long_input), 2000)
        mock_provider.clean_transcript.return_value = (
            "Unrelated short paragraph about something else."
        )
        result = cleaner.clean(long_input, mock_provider)
        self.assertEqual(result, long_input)

    def test_clean_accepts_llm_output_at_or_above_ratio_threshold(self):
        """Output length >= 20% of pattern-cleaned input passes the guard."""
        cleaner = LLMBasedCleaner()
        mock_provider = Mock()
        long_input = "word " * 400
        self.assertGreaterEqual(len(long_input), 2000)
        acceptable = "y" * (len(long_input) // 5 + 50)
        self.assertGreaterEqual(len(acceptable) / len(long_input), 0.20)
        mock_provider.clean_transcript.return_value = acceptable
        result = cleaner.clean(long_input, mock_provider)
        self.assertEqual(result, acceptable)

    def test_clean_rejects_llm_output_just_below_ratio_threshold(self):
        """Output length strictly below 20% falls back to pattern-cleaned text."""
        cleaner = LLMBasedCleaner()
        mock_provider = Mock()
        long_input = "word " * 500
        self.assertGreaterEqual(len(long_input), 2000)
        too_short = "y" * int(len(long_input) * 0.15)
        self.assertLess(len(too_short) / len(long_input), 0.20)
        mock_provider.clean_transcript.return_value = too_short
        result = cleaner.clean(long_input, mock_provider)
        self.assertEqual(result, long_input)

    def test_clean_rejects_llm_output_at_ratio_0179_issue_564(self):
        """Borderline real-world ratio under 20% (e.g. ~0.179) still falls back (GitHub #564)."""
        cleaner = LLMBasedCleaner()
        mock_provider = Mock()
        long_input = "a" * 5000
        self.assertGreaterEqual(len(long_input), 2000)
        out_len = int(len(long_input) * 0.179)
        llm_out = "x" * out_len
        self.assertLess(len(llm_out) / len(long_input), 0.20)
        mock_provider.clean_transcript.return_value = llm_out
        result = cleaner.clean(long_input, mock_provider)
        self.assertEqual(result, long_input)

    def test_clean_accepts_llm_output_at_exact_twenty_percent_ratio_issue_564(self):
        """Output length exactly 20% of input passes (guard uses strictly below 0.20)."""
        cleaner = LLMBasedCleaner()
        mock_provider = Mock()
        long_input = "a" * 5000
        self.assertGreaterEqual(len(long_input), 2000)
        out_len = int(len(long_input) * 0.20)
        self.assertEqual(out_len, 1000)
        llm_out = "x" * out_len
        self.assertEqual(len(llm_out) / len(long_input), 0.20)
        mock_provider.clean_transcript.return_value = llm_out
        result = cleaner.clean(long_input, mock_provider)
        self.assertEqual(result, llm_out)


class TestHybridCleaner(unittest.TestCase):
    """Tests for HybridCleaner."""

    def test_cleaner_initialization(self):
        """Test cleaner initialization."""
        cleaner = HybridCleaner()
        self.assertIsNotNone(cleaner)
        self.assertIsNotNone(cleaner.pattern_cleaner)
        self.assertIsNotNone(cleaner.llm_cleaner)

    @patch("podcast_scraper.preprocessing.clean_for_summarization")
    def test_clean_pattern_only_when_no_provider(self, mock_clean):
        """Test that clean() uses only pattern-based when no provider provided."""
        mock_clean.return_value = "pattern cleaned text"
        cleaner = HybridCleaner()

        result = cleaner.clean("raw text", provider=None)
        self.assertEqual(result, "pattern cleaned text")
        mock_clean.assert_called_once_with("raw text")

    @patch("podcast_scraper.preprocessing.clean_for_summarization")
    def test_clean_pattern_only_when_not_needed(self, mock_clean):
        """Test that clean() uses only pattern-based when LLM not needed."""
        # Create text that won't trigger LLM cleaning (high reduction ratio)
        mock_clean.return_value = "x" * 900  # 10% reduction from 1000 chars
        cleaner = HybridCleaner()

        mock_provider = Mock()
        mock_provider.clean_transcript.return_value = "llm cleaned"

        original = "x" * 1000
        result = cleaner.clean(original, provider=mock_provider)
        # Should only use pattern cleaning (reduction >= 5%)
        self.assertEqual(result, "x" * 900)
        mock_clean.assert_called_once_with(original)
        # Should not call LLM
        mock_provider.clean_transcript.assert_not_called()

    @patch("podcast_scraper.preprocessing.clean_for_summarization")
    def test_clean_uses_llm_when_needed(self, mock_clean):
        """Test that clean() uses LLM when pattern-based insufficient."""
        # Create text that will trigger LLM cleaning (low reduction + sponsor keywords)
        mock_clean.return_value = (
            "Main content. "
            "This episode is sponsored by Acme Corp. "
            "Brought to you by Test Company."
        )
        cleaner = HybridCleaner()

        mock_provider = Mock()
        mock_provider.clean_transcript.return_value = "llm cleaned text"

        original = "Main content. " * 10
        result = cleaner.clean(original, provider=mock_provider)
        self.assertEqual(result, "llm cleaned text")
        mock_clean.assert_called_once_with(original)
        # Should call LLM after pattern cleaning (sponsor keywords detected)
        mock_provider.clean_transcript.assert_called_once()

    @patch("podcast_scraper.preprocessing.clean_for_summarization")
    def test_clean_forwards_pipeline_metrics_to_provider(self, mock_clean):
        """HybridCleaner passes pipeline_metrics through to clean_transcript."""
        mock_clean.return_value = (
            "Main content. "
            "This episode is sponsored by Acme Corp. "
            "Brought to you by Test Company."
        )
        cleaner = HybridCleaner()
        mock_provider = Mock()
        mock_provider.clean_transcript.return_value = "llm cleaned text"
        original = "Main content. " * 10
        pm = object()

        result = cleaner.clean(original, provider=mock_provider, pipeline_metrics=pm)

        self.assertEqual(result, "llm cleaned text")
        mock_provider.clean_transcript.assert_called_once()
        _args, kwargs = mock_provider.clean_transcript.call_args
        self.assertIs(kwargs.get("pipeline_metrics"), pm)

    @patch("podcast_scraper.preprocessing.clean_for_summarization")
    def test_clean_handles_llm_failure(self, mock_clean):
        """Test that clean() falls back to pattern-cleaned text if LLM fails."""
        mock_clean.return_value = "pattern cleaned text"
        cleaner = HybridCleaner()

        mock_provider = Mock()
        mock_provider.clean_transcript.side_effect = RuntimeError("LLM error")

        # Use text that triggers LLM cleaning
        original = "x" * 1000
        mock_clean.return_value = "x" * 980  # Low reduction (< 5%)
        result = cleaner.clean(original, provider=mock_provider)
        # Should return pattern-cleaned text if LLM fails
        self.assertEqual(result, "x" * 980)

    @patch("podcast_scraper.preprocessing.clean_for_summarization")
    def test_clean_falls_back_when_llm_output_too_short(self, mock_clean):
        """Hybrid path must not pass a bogus short LLM 'cleaned' transcript downstream."""
        mock_clean.side_effect = lambda x: x
        cleaner = HybridCleaner()
        mock_provider = Mock()
        mock_provider.clean_transcript.return_value = "Short unrelated model output."
        original = "episode words. " * 200
        self.assertGreaterEqual(len(original), 2000)
        result = cleaner.clean(original, provider=mock_provider)
        self.assertEqual(result, original)
        mock_provider.clean_transcript.assert_called_once()

    def test_needs_llm_cleaning_low_reduction_ratio(self):
        """Test that _needs_llm_cleaning returns True for low reduction ratio."""
        cleaner = HybridCleaner()

        # Original text: 1000 chars, cleaned: 980 chars (only 2% reduction)
        original = "x" * 1000
        cleaned = "x" * 980

        mock_provider = Mock()
        result = cleaner._needs_llm_cleaning(original, cleaned, mock_provider)
        self.assertTrue(result, "Should need LLM cleaning when reduction < 5%")

    def test_needs_llm_cleaning_high_reduction_ratio(self):
        """Test that _needs_llm_cleaning returns False for high reduction ratio."""
        cleaner = HybridCleaner()

        # Original text: 1000 chars, cleaned: 900 chars (10% reduction)
        original = "x" * 1000
        cleaned = "x" * 900

        mock_provider = Mock()
        result = cleaner._needs_llm_cleaning(original, cleaned, mock_provider)
        self.assertFalse(result, "Should not need LLM cleaning when reduction >= 5%")

    def test_needs_llm_cleaning_sponsor_keywords(self):
        """Test that _needs_llm_cleaning returns True when sponsor keywords present."""
        cleaner = HybridCleaner()

        original = "Main content here."
        # Cleaned text still has sponsor keywords (2+ matches)
        cleaned = (
            "Main content here. "
            "This episode is sponsored by Acme Corp. "
            "Brought to you by Test Company."
        )

        mock_provider = Mock()
        result = cleaner._needs_llm_cleaning(original, cleaned, mock_provider)
        self.assertTrue(result, "Should need LLM cleaning when sponsor keywords present")

    def test_needs_llm_cleaning_high_promotional_density(self):
        """Test that _needs_llm_cleaning returns True for high promotional phrase density."""
        cleaner = HybridCleaner()

        # Text with many promotional phrases (high density)
        cleaned = (
            "Check out our website. "
            "Visit our store. "
            "Go to example.com. "
            "Sign up now. "
            "Subscribe today. "
            "Follow us on social. "
            "Rate and review. "
            "Leave a review. " * 20  # Repeat to increase density
        )

        mock_provider = Mock()
        result = cleaner._needs_llm_cleaning("original", cleaned, mock_provider)
        self.assertTrue(result, "Should need LLM cleaning when promotional density is high")

    def test_needs_llm_cleaning_no_provider(self):
        """Test that _needs_llm_cleaning returns False when provider doesn't support it."""
        cleaner = HybridCleaner()

        provider_without_method = Mock(spec=[])

        result = cleaner._needs_llm_cleaning("original", "cleaned", provider_without_method)
        self.assertFalse(result, "Should not need LLM cleaning if provider doesn't support it")


class TestPatternBasedCleanerEdgeCases(unittest.TestCase):
    """Edge-case tests for PatternBasedCleaner."""

    @patch("podcast_scraper.preprocessing.clean_for_summarization")
    def test_clean_empty_string(self, mock_clean):
        """clean('') delegates to preprocessing with empty string."""
        mock_clean.return_value = ""
        result = PatternBasedCleaner().clean("")
        mock_clean.assert_called_once_with("")
        self.assertEqual(result, "")

    @patch("podcast_scraper.preprocessing.clean_for_summarization")
    def test_clean_whitespace_only(self, mock_clean):
        """clean('   ') delegates to preprocessing with whitespace."""
        mock_clean.return_value = ""
        result = PatternBasedCleaner().clean("   ")
        mock_clean.assert_called_once_with("   ")
        self.assertEqual(result, "")

    @patch("podcast_scraper.preprocessing.remove_sponsor_blocks")
    def test_remove_sponsors_empty_string(self, mock_remove):
        """remove_sponsors('') works without error."""
        mock_remove.return_value = ""
        result = PatternBasedCleaner().remove_sponsors("")
        mock_remove.assert_called_once_with("")
        self.assertEqual(result, "")

    @patch("podcast_scraper.preprocessing.remove_outro_blocks")
    def test_remove_outros_empty_string(self, mock_remove):
        """remove_outros('') works without error."""
        mock_remove.return_value = ""
        result = PatternBasedCleaner().remove_outros("")
        mock_remove.assert_called_once_with("")
        self.assertEqual(result, "")


class TestLLMBasedCleanerExtended(unittest.TestCase):
    """Extended tests for LLMBasedCleaner edge cases."""

    def test_clean_forwards_pipeline_metrics(self):
        """pipeline_metrics kwarg is forwarded to provider.clean_transcript."""
        cleaner = LLMBasedCleaner()
        mock_provider = Mock()
        mock_provider.clean_transcript.return_value = "cleaned"
        pm = object()

        cleaner.clean("some text", mock_provider, pipeline_metrics=pm)

        mock_provider.clean_transcript.assert_called_once_with("some text", pipeline_metrics=pm)

    def test_clean_non_str_return_falls_back(self):
        """Provider returning int falls back to original text."""
        cleaner = LLMBasedCleaner()
        mock_provider = Mock()
        mock_provider.clean_transcript.return_value = 123

        result = cleaner.clean("original text", mock_provider)
        self.assertEqual(result, "original text")

    def test_clean_dict_return_falls_back(self):
        """Provider returning dict falls back to original text."""
        cleaner = LLMBasedCleaner()
        mock_provider = Mock()
        mock_provider.clean_transcript.return_value = {}

        result = cleaner.clean("original text", mock_provider)
        self.assertEqual(result, "original text")

    def test_clean_short_input_skips_length_guard(self):
        """Input shorter than 2000 chars bypasses the length guard."""
        cleaner = LLMBasedCleaner()
        mock_provider = Mock()
        short_input = "a" * 500
        very_short_output = "b"
        mock_provider.clean_transcript.return_value = very_short_output

        result = cleaner.clean(short_input, mock_provider)
        self.assertEqual(result, very_short_output)

    def test_clean_whitespace_only_return_falls_back(self):
        """Provider returning whitespace-only string falls back (length guard catches it)."""
        cleaner = LLMBasedCleaner()
        mock_provider = Mock()
        mock_provider.clean_transcript.return_value = "   "
        long_input = "word " * 500
        self.assertGreaterEqual(len(long_input), 2000)

        result = cleaner.clean(long_input, mock_provider)
        self.assertEqual(result, long_input)


class TestHybridCleanerBoundaries(unittest.TestCase):
    """Boundary-condition tests for HybridCleaner heuristics."""

    def test_needs_llm_cleaning_at_exact_5_percent_boundary(self):
        """Exactly 5% reduction (reduction_ratio == 0.05) → False (not < 0.05)."""
        cleaner = HybridCleaner()
        original = "x" * 1000
        cleaned = "x" * 950
        mock_provider = Mock()

        result = cleaner._needs_llm_cleaning(original, cleaned, mock_provider)
        self.assertFalse(result)

    def test_needs_llm_cleaning_at_4_9_percent(self):
        """4.9% reduction (reduction_ratio < 0.05) → True."""
        cleaner = HybridCleaner()
        original = "x" * 1000
        cleaned = "x" * 951
        mock_provider = Mock()

        result = cleaner._needs_llm_cleaning(original, cleaned, mock_provider)
        self.assertTrue(result)

    def test_needs_llm_cleaning_one_sponsor_keyword_not_enough(self):
        """Exactly 1 sponsor keyword does not trigger LLM cleaning."""
        cleaner = HybridCleaner()
        original = "x" * 100
        cleaned = "Use the promo code SAVE20 for a discount."
        mock_provider = Mock()

        result = cleaner._needs_llm_cleaning(original, cleaned, mock_provider)
        self.assertFalse(result)

    def test_needs_llm_cleaning_two_sponsor_keywords_triggers(self):
        """Exactly 2 sponsor keywords triggers LLM cleaning."""
        cleaner = HybridCleaner()
        original = "x" * 100
        cleaned = "This episode is sponsored by Acme Corp. " "Brought to you by Test Company."
        mock_provider = Mock()

        result = cleaner._needs_llm_cleaning(original, cleaned, mock_provider)
        self.assertTrue(result)

    @patch("podcast_scraper.preprocessing.clean_for_summarization")
    def test_clean_empty_original(self, mock_clean):
        """HybridCleaner.clean('') returns pattern-cleaned result of ''."""
        mock_clean.return_value = ""
        cleaner = HybridCleaner()
        mock_provider = Mock()

        result = cleaner.clean("", provider=mock_provider)
        self.assertEqual(result, "")
        mock_clean.assert_called_once_with("")

    def test_needs_llm_cleaning_empty_original_skips_reduction_heuristic(self):
        """Empty original skips the reduction-ratio heuristic (original is falsy)."""
        cleaner = HybridCleaner()
        mock_provider = Mock()

        result = cleaner._needs_llm_cleaning("", "cleaned", mock_provider)
        self.assertFalse(result)


class TestTranscriptCleaningProcessorProtocol(unittest.TestCase):
    """Tests for the TranscriptCleaningProcessor runtime protocol."""

    def test_pattern_cleaner_satisfies_protocol(self):
        """PatternBasedCleaner is recognised as a TranscriptCleaningProcessor."""
        self.assertIsInstance(PatternBasedCleaner(), TranscriptCleaningProcessor)

    def test_non_conforming_object_fails_protocol(self):
        """A plain object does not satisfy TranscriptCleaningProcessor."""
        self.assertNotIsInstance(object(), TranscriptCleaningProcessor)
