#!/usr/bin/env python3
"""Unit tests for transcript cleaning functionality."""

import unittest
from unittest.mock import patch

import pytest

from podcast_scraper.cleaning import (
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
        mock_clean.assert_called_once_with(
            "raw text",
            diarization_segments=None,
            host_speaker_id=None,
            confidence_threshold=None,
            crosspromo_cue_patterns=None,
        )

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


class TestPatternBasedCleanerEdgeCases(unittest.TestCase):
    """Edge-case tests for PatternBasedCleaner."""

    @patch("podcast_scraper.preprocessing.clean_for_summarization")
    def test_clean_empty_string(self, mock_clean):
        """clean('') delegates to preprocessing with empty string."""
        mock_clean.return_value = ""
        result = PatternBasedCleaner().clean("")
        mock_clean.assert_called_once_with(
            "",
            diarization_segments=None,
            host_speaker_id=None,
            confidence_threshold=None,
            crosspromo_cue_patterns=None,
        )
        self.assertEqual(result, "")

    @patch("podcast_scraper.preprocessing.clean_for_summarization")
    def test_clean_whitespace_only(self, mock_clean):
        """clean('   ') delegates to preprocessing with whitespace."""
        mock_clean.return_value = ""
        result = PatternBasedCleaner().clean("   ")
        mock_clean.assert_called_once_with(
            "   ",
            diarization_segments=None,
            host_speaker_id=None,
            confidence_threshold=None,
            crosspromo_cue_patterns=None,
        )
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


class TestTranscriptCleaningProcessorProtocol(unittest.TestCase):
    """Tests for the TranscriptCleaningProcessor runtime protocol."""

    def test_pattern_cleaner_satisfies_protocol(self):
        """PatternBasedCleaner is recognised as a TranscriptCleaningProcessor."""
        self.assertIsInstance(PatternBasedCleaner(), TranscriptCleaningProcessor)

    def test_non_conforming_object_fails_protocol(self):
        """A plain object does not satisfy TranscriptCleaningProcessor."""
        self.assertNotIsInstance(object(), TranscriptCleaningProcessor)
