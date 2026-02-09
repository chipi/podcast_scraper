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

        # Should raise AttributeError if provider doesn't have clean_transcript
        provider_without_method = Mock()
        # Remove clean_transcript attribute if it exists
        if hasattr(provider_without_method, "clean_transcript"):
            delattr(provider_without_method, "clean_transcript")

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
        mock_provider.clean_transcript.assert_called_once_with("raw text")

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

        provider_without_method = Mock()
        # Remove clean_transcript attribute if it exists
        if hasattr(provider_without_method, "clean_transcript"):
            delattr(provider_without_method, "clean_transcript")

        result = cleaner._needs_llm_cleaning("original", "cleaned", provider_without_method)
        self.assertFalse(result, "Should not need LLM cleaning if provider doesn't support it")
