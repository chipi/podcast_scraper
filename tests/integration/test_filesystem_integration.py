#!/usr/bin/env python3
"""Integration tests for filesystem operations.

These tests verify filesystem utility functions work correctly:
- File naming (episode numbering, title sanitization, run suffixes)
- Output directory setup and validation
- Temporary file cleanup
- File path resolution and normalization
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import config, filesystem


@pytest.mark.integration
@pytest.mark.critical_path
class TestFilesystemOperations(unittest.TestCase):
    """Test filesystem utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sanitize_filename(self):
        """Test filename sanitization with various edge cases."""
        # Normal filename
        self.assertEqual(filesystem.sanitize_filename("Episode 1"), "Episode 1")

        # Special characters
        self.assertEqual(
            filesystem.sanitize_filename("Episode: Title/With\\Special*Chars?"),
            "Episode_ Title_With_Special_Chars_",
        )

        # Newlines and tabs
        self.assertEqual(filesystem.sanitize_filename("Episode\nTitle\tTest"), "Episode Title Test")

        # Empty string
        self.assertEqual(filesystem.sanitize_filename(""), "untitled")

        # Only special characters (becomes underscores, which should become "untitled")
        self.assertEqual(filesystem.sanitize_filename("!!!###"), "untitled")

        # Very long title
        long_title = "A" * 200
        sanitized = filesystem.sanitize_filename(long_title)
        self.assertEqual(len(sanitized), 200)
        self.assertTrue(all(c.isalnum() or c in {"_", "-", " ", "."} for c in sanitized))

    def test_file_naming_with_episode_numbering(self):
        """Test file naming with episode numbering."""
        # Test episode number formatting
        name1 = filesystem.build_whisper_output_name(1, "Episode_Title", None)
        self.assertTrue(name1.startswith("0001 - "), "Should include 4-digit episode number")

        name10 = filesystem.build_whisper_output_name(10, "Episode_Title", None)
        self.assertTrue(name10.startswith("0010 - "), "Should include 4-digit episode number")

        name100 = filesystem.build_whisper_output_name(100, "Episode_Title", None)
        self.assertTrue(name100.startswith("0100 - "), "Should include 4-digit episode number")

    def test_file_naming_with_run_suffix(self):
        """Test file naming with run suffix."""
        # Without run suffix
        name = filesystem.build_whisper_output_name(1, "Episode_Title", None)
        # The name contains "Episode_Title" which has "_", but that's from title_safe, not run suffix
        # Check that run suffix pattern is not present (run suffix would be "_suffix")
        name_parts = name.split(" - ")[1].split(".")[0]
        # Without run suffix, the name should just be the title (may contain _ from title_safe)
        # With run suffix, it would be "Episode_Title_test_run"
        self.assertNotIn("_test_run", name_parts, "Should not have run suffix")

        # With run suffix
        name_with_suffix = filesystem.build_whisper_output_name(1, "Episode_Title", "test_run")
        self.assertIn("test_run", name_with_suffix, "Should include run suffix")
        self.assertTrue(name_with_suffix.endswith(".txt"), "Should end with .txt")
        # Verify run suffix is properly formatted
        self.assertIn(
            "_test_run", name_with_suffix, "Run suffix should be prefixed with underscore"
        )

    def test_file_naming_with_long_title(self):
        """Test file naming with long titles (truncation)."""
        # Long title should be truncated
        long_title = "A" * 100
        name = filesystem.build_whisper_output_name(1, long_title, None)
        # Title part should be truncated to WHISPER_TITLE_MAX_CHARS
        title_part = name.split(" - ")[1].split(".")[0]
        self.assertLessEqual(
            len(title_part), filesystem.WHISPER_TITLE_MAX_CHARS, "Title should be truncated"
        )

    def test_output_directory_setup(self):
        """Test output directory setup and validation."""
        # Basic output directory (no ML features)
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            output_dir=self.temp_dir,
            transcribe_missing=False,  # Explicitly disable to test basic case
            auto_speakers=False,  # Disable to test basic case
            generate_summaries=False,  # Disable to test basic case
        )
        effective_dir, run_suffix = filesystem.setup_output_directory(cfg)
        self.assertEqual(effective_dir, self.temp_dir)
        self.assertIsNone(run_suffix)

        # With run_id (no ML features)
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            output_dir=self.temp_dir,
            run_id="test_run",
            transcribe_missing=False,  # Explicitly disable to test run_id only
            auto_speakers=False,  # Disable to test run_id only
            generate_summaries=False,  # Disable to test run_id only
        )
        effective_dir, run_suffix = filesystem.setup_output_directory(cfg)
        self.assertIn("test_run", effective_dir)
        self.assertEqual(run_suffix, "test_run")

        # With transcribe_missing (should add whisper model to run_suffix)
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            output_dir=self.temp_dir,
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            auto_speakers=False,  # Disable to test whisper only
            generate_summaries=False,  # Disable to test whisper only
        )
        effective_dir, run_suffix = filesystem.setup_output_directory(cfg)
        # Format: "w_<model>" for whisper provider
        self.assertIn("w_tiny.en", run_suffix or "")

    def test_output_directory_validation(self):
        """Test output directory validation."""
        # Valid directory (current directory)
        valid_dir = filesystem.validate_and_normalize_output_dir(".")
        self.assertIsNotNone(valid_dir)
        self.assertTrue(os.path.isabs(valid_dir) or os.path.exists(valid_dir))

        # Valid directory (home directory)
        home_dir = str(Path.home())
        valid_home = filesystem.validate_and_normalize_output_dir(home_dir)
        self.assertIsNotNone(valid_home)

        # Invalid directory (empty string)
        with self.assertRaises(ValueError):
            filesystem.validate_and_normalize_output_dir("")

        # Invalid directory (whitespace only)
        with self.assertRaises(ValueError):
            filesystem.validate_and_normalize_output_dir("   ")

    def test_write_file_creates_directories(self):
        """Test that write_file creates parent directories."""
        # Write to nested directory
        nested_path = os.path.join(self.temp_dir, "nested", "deep", "file.txt")
        filesystem.write_file(nested_path, b"test content")

        # Verify file was created
        self.assertTrue(os.path.exists(nested_path), "File should be created")

        # Verify parent directories were created
        self.assertTrue(
            os.path.exists(os.path.dirname(nested_path)), "Parent directories should be created"
        )

        # Verify content
        with open(nested_path, "rb") as f:
            content = f.read()
        self.assertEqual(content, b"test content", "File content should match")

    def test_truncate_whisper_title(self):
        """Test Whisper title truncation."""
        # Short title (no truncation)
        short = filesystem.truncate_whisper_title("Short", for_log=False)
        self.assertEqual(short, "Short")

        # Long title (truncation)
        long_title = "A" * 100
        truncated = filesystem.truncate_whisper_title(long_title, for_log=False)
        self.assertEqual(len(truncated), filesystem.WHISPER_TITLE_MAX_CHARS)

        # Long title with ellipsis for log
        truncated_log = filesystem.truncate_whisper_title(long_title, for_log=True)
        self.assertLessEqual(len(truncated_log), filesystem.WHISPER_TITLE_MAX_CHARS)
        if len(truncated_log) < len(long_title):
            self.assertIn("…", truncated_log or "", "Should include ellipsis for log")

    def test_build_whisper_output_path(self):
        """Test building full Whisper output path."""
        output_dir = self.temp_dir
        path = filesystem.build_whisper_output_path(1, "Episode_Title", None, output_dir)

        # Verify path structure
        self.assertTrue(path.startswith(output_dir), "Path should start with output directory")
        self.assertTrue(path.endswith(".txt"), "Path should end with .txt")
        self.assertIn("0001", path, "Path should include episode number")
        self.assertIn("Episode_Title", path, "Path should include episode title")
        self.assertIn(
            f"/{filesystem.TRANSCRIPTS_SUBDIR}/",
            path,
            "Path should include transcripts subdirectory",
        )

    def test_derive_output_dir(self):
        """Test deriving output directory from RSS URL."""
        # With override
        override_dir = filesystem.derive_output_dir(
            "https://example.com/feed.xml", "./custom_output"
        )
        self.assertIn("custom_output", override_dir)

        # Without override (should derive from URL)
        derived = filesystem.derive_output_dir("https://example.com/feed.xml", None)
        self.assertIn("example.com", derived or "", "Should include domain from URL")
        self.assertIn("output/rss_", derived or "", "Should have output/rss_ prefix")
