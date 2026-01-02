#!/usr/bin/env python3
"""Tests for Config cross-field validation."""

import os
import sys
import unittest
import warnings

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pydantic import ValidationError

from podcast_scraper import Config, config


class TestSummaryValidation(unittest.TestCase):
    """Test summary-related cross-field validation."""

    def test_summary_max_greater_than_min(self):
        """Test that summary_max_length must be greater than summary_min_length."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                summary_max_length=50,
                summary_min_length=100,
            )
        self.assertIn("must be greater than", str(context.exception))

    def test_summary_max_equal_to_min_fails(self):
        """Test that summary_max_length equal to summary_min_length fails."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                summary_max_length=100,
                summary_min_length=100,
            )
        self.assertIn("must be greater than", str(context.exception))

    def test_summary_max_greater_than_min_succeeds(self):
        """Test that valid max > min configuration succeeds."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            summary_max_length=200,
            summary_min_length=50,
        )
        self.assertEqual(cfg.summary_max_length, 200)
        self.assertEqual(cfg.summary_min_length, 50)

    def test_word_overlap_less_than_chunk_size(self):
        """Test that summary_word_overlap must be less than summary_word_chunk_size."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                summary_word_chunk_size=500,
                summary_word_overlap=600,
            )
        self.assertIn("must be less than", str(context.exception))

    def test_word_overlap_equal_to_chunk_size_fails(self):
        """Test that overlap equal to chunk size fails."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                summary_word_chunk_size=500,
                summary_word_overlap=500,
            )
        self.assertIn("must be less than", str(context.exception))

    def test_word_overlap_less_than_chunk_size_succeeds(self):
        """Test that valid overlap < chunk_size configuration succeeds."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            summary_word_chunk_size=900,
            summary_word_overlap=150,
        )
        self.assertEqual(cfg.summary_word_chunk_size, 900)
        self.assertEqual(cfg.summary_word_overlap, 150)

    def test_summaries_require_metadata(self):
        """Test that generate_summaries requires generate_metadata."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                generate_summaries=True,
                generate_metadata=False,
            )
        self.assertIn("requires generate_metadata", str(context.exception))

    def test_summaries_with_metadata_succeeds(self):
        """Test that summaries work when metadata is enabled."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=True,
            generate_metadata=True,
        )
        self.assertTrue(cfg.generate_summaries)
        self.assertTrue(cfg.generate_metadata)

    def test_word_chunk_size_outside_range_warns(self):
        """Test that word_chunk_size outside recommended range warns."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Config(
                rss_url="https://example.com/feed.xml",
                summary_word_chunk_size=500,  # Below 800
            )
            self.assertEqual(len(w), 1)
            self.assertIn("outside recommended range", str(w[0].message))
            self.assertIn("800-1200", str(w[0].message))

    def test_word_overlap_outside_range_warns(self):
        """Test that word_overlap outside recommended range warns."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Config(
                rss_url="https://example.com/feed.xml",
                summary_word_overlap=50,  # Below 100
            )
            self.assertEqual(len(w), 1)
            self.assertIn("outside recommended range", str(w[0].message))
            self.assertIn("100-200", str(w[0].message))


class TestOutputControlValidation(unittest.TestCase):
    """Test output control flag validation."""

    def test_clean_output_and_skip_existing_conflict(self):
        """Test that clean_output and skip_existing are mutually exclusive."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                clean_output=True,
                skip_existing=True,
            )
        self.assertIn("mutually exclusive", str(context.exception))
        self.assertIn("clean_output", str(context.exception))
        self.assertIn("skip_existing", str(context.exception))

    def test_clean_output_and_reuse_media_conflict(self):
        """Test that clean_output and reuse_media are mutually exclusive."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                clean_output=True,
                reuse_media=True,
            )
        self.assertIn("mutually exclusive", str(context.exception))
        self.assertIn("clean_output", str(context.exception))
        self.assertIn("reuse_media", str(context.exception))

    def test_skip_existing_and_reuse_media_compatible(self):
        """Test that skip_existing and reuse_media can be used together."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            skip_existing=True,
            reuse_media=True,
        )
        self.assertTrue(cfg.skip_existing)
        self.assertTrue(cfg.reuse_media)

    def test_clean_output_alone_succeeds(self):
        """Test that clean_output alone works fine."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            clean_output=True,
        )
        self.assertTrue(cfg.clean_output)
        self.assertFalse(cfg.skip_existing)
        self.assertFalse(cfg.reuse_media)


class TestTranscriptionValidation(unittest.TestCase):
    """Test transcription-related validation."""

    def test_transcribe_missing_requires_whisper_model(self):
        """Test that transcribe_missing requires a valid whisper_model."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                transcribe_missing=True,
                whisper_model="",
            )
        self.assertIn("requires a valid whisper_model", str(context.exception))

    def test_transcribe_missing_with_valid_model_succeeds(self):
        """Test that transcribe_missing works with valid whisper_model."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
        )
        self.assertTrue(cfg.transcribe_missing)
        self.assertEqual(cfg.whisper_model, config.TEST_DEFAULT_WHISPER_MODEL)

    def test_transcribe_missing_false_allows_empty_model(self):
        """Test that empty whisper_model is OK when transcribe_missing is False."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=False,
            whisper_model="",
        )
        self.assertFalse(cfg.transcribe_missing)


class TestValidationEdgeCases(unittest.TestCase):
    """Test edge cases in validation logic."""

    def test_multiple_validation_errors(self):
        """Test configuration with multiple validation errors."""
        with self.assertRaises(ValidationError) as context:
            Config(
                rss_url="https://example.com/feed.xml",
                summary_max_length=50,
                summary_min_length=100,  # Error: max <= min
                clean_output=True,
                skip_existing=True,  # Error: contradictory flags
            )
        # Should report validation errors
        error_str = str(context.exception)
        # At least one of the errors should be present
        self.assertTrue("must be greater than" in error_str or "mutually exclusive" in error_str)

    def test_valid_complex_configuration(self):
        """Test that a complex valid configuration succeeds."""
        cfg = Config(
            rss_url="https://example.com/feed.xml",
            output_dir="./transcripts",
            max_episodes=10,
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            screenplay=True,
            screenplay_num_speakers=2,
            auto_speakers=True,
            generate_metadata=True,
            metadata_format="yaml",
            generate_summaries=True,
            summary_max_length=200,
            summary_min_length=50,
            summary_word_chunk_size=900,
            summary_word_overlap=150,
            skip_existing=True,
            reuse_media=True,
        )
        # All settings should be applied correctly
        self.assertTrue(cfg.transcribe_missing)
        self.assertTrue(cfg.generate_summaries)
        self.assertTrue(cfg.skip_existing)
        self.assertTrue(cfg.reuse_media)
        self.assertFalse(cfg.clean_output)


if __name__ == "__main__":
    unittest.main()
