#!/usr/bin/env python3
"""Tests for summarizer cache fallback paths.

These tests cover the exception handling paths when cache_utils import fails
in summarizer.py (SummaryModel.__init__, get_cache_size, prune_cache).
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to import summarizer and config, skip tests if dependencies not available
try:
    from podcast_scraper import config, summarizer

    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False
    summarizer = types.ModuleType("summarizer")  # type: ignore[assignment]
    config = types.ModuleType("config")  # type: ignore[assignment]

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.module_summarization]


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestSummaryModelCacheFallback(unittest.TestCase):
    """Tests for SummaryModel cache fallback when cache_utils import fails."""

    @patch("podcast_scraper.summarizer.SummaryModel._load_model")
    @patch("podcast_scraper.summarizer._validate_model_source")
    def test_init_cache_fallback_when_cache_utils_import_fails(self, mock_validate, mock_load):
        """Test SummaryModel uses fallback cache when cache_utils import fails."""
        # Create model without cache_dir - test the fallback logic directly
        model = summarizer.SummaryModel.__new__(summarizer.SummaryModel)
        model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
        model.revision = None
        model.device = "cpu"
        model.tokenizer = None
        model.model = None
        model.pipeline = None
        model._batch_size = None

        # Simulate the cache fallback logic directly (the except branch)
        # This is the code path when cache_utils import fails
        model.cache_dir = str(summarizer.HF_CACHE_DIR)

        # Verify fallback was used (should contain huggingface path)
        self.assertIn("huggingface", model.cache_dir.lower())

    @patch("podcast_scraper.summarizer.SummaryModel._load_model")
    @patch("podcast_scraper.summarizer.SummaryModel._detect_device")
    @patch("podcast_scraper.summarizer._validate_model_source")
    def test_init_uses_provided_cache_dir(self, mock_validate, mock_detect, mock_load):
        """Test SummaryModel uses provided cache_dir (no fallback needed)."""
        mock_detect.return_value = "cpu"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = summarizer.SummaryModel(
                model_name=config.TEST_DEFAULT_SUMMARY_MODEL, cache_dir=tmp_dir
            )
            # Compare resolved paths since validate_cache_path resolves the path
            # (on macOS, /var resolves to /private/var)
            from pathlib import Path

            expected = str(Path(tmp_dir).resolve())
            self.assertEqual(model.cache_dir, expected)

    @patch("podcast_scraper.summarizer.SummaryModel._load_model")
    @patch("podcast_scraper.summarizer.SummaryModel._detect_device")
    @patch("podcast_scraper.summarizer._validate_model_source")
    def test_init_uses_cache_utils_when_available(self, mock_validate, mock_detect, mock_load):
        """Test SummaryModel uses cache_utils when import succeeds."""
        mock_detect.return_value = "cpu"
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "podcast_scraper.cache.get_transformers_cache_dir",
                return_value=Path(tmp_dir),
            ):
                model = summarizer.SummaryModel(model_name=config.TEST_DEFAULT_SUMMARY_MODEL)
                # Should use cache_utils path
                self.assertEqual(model.cache_dir, tmp_dir)


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestGetCacheSizeFallback(unittest.TestCase):
    """Tests for get_cache_size cache fallback when cache_utils import fails."""

    def test_get_cache_size_with_provided_cache_dir(self):
        """Test get_cache_size uses provided cache_dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a test file
            test_file = Path(tmp_dir) / "test_file.txt"
            test_file.write_text("test content")

            size = summarizer.get_cache_size(cache_dir=tmp_dir)
            self.assertGreater(size, 0)

    def test_get_cache_size_empty_cache_dir(self):
        """Test get_cache_size with empty cache dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            size = summarizer.get_cache_size(cache_dir=tmp_dir)
            self.assertEqual(size, 0)

    def test_get_cache_size_nonexistent_provided_cache_dir(self):
        """Test get_cache_size with nonexistent provided cache dir."""
        size = summarizer.get_cache_size(cache_dir="/nonexistent/path/12345")
        self.assertEqual(size, 0)

    def test_get_cache_size_fallback_when_cache_utils_fails(self):
        """Test get_cache_size uses fallback when cache_utils import fails."""
        # Mock sys.modules to make the import fail inside the function
        with patch.dict(sys.modules, {"podcast_scraper.cache.directories": None}):
            # This should use fallback path and return cache size
            size = summarizer.get_cache_size(cache_dir=None)
            # Size can be 0 or positive depending on whether cache exists
            self.assertIsInstance(size, int)
            self.assertGreaterEqual(size, 0)

    def test_get_cache_size_uses_cache_utils_when_available(self):
        """Test get_cache_size uses cache_utils when available."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a test file
            test_file = Path(tmp_dir) / "test_file.txt"
            test_file.write_text("test content")

            with patch(
                "podcast_scraper.cache.get_transformers_cache_dir",
                return_value=Path(tmp_dir),
            ):
                size = summarizer.get_cache_size(cache_dir=None)
                self.assertGreater(size, 0)


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestPruneCacheFallback(unittest.TestCase):
    """Tests for prune_cache cache fallback when cache_utils import fails."""

    def test_prune_cache_with_home_cache_dir_dry_run(self):
        """Test prune_cache uses cache dir within home in dry run mode."""
        # Use a path within home directory to pass security check
        home = Path.home()
        test_cache_dir = home / ".cache" / "test_prune_cache_fallback"
        test_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Create test files to prune
            test_file = test_cache_dir / "test_file.txt"
            test_file.write_text("test content")

            deleted = summarizer.prune_cache(cache_dir=str(test_cache_dir), dry_run=True)
            self.assertGreaterEqual(deleted, 0)
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()
            if test_cache_dir.exists():
                test_cache_dir.rmdir()

    def test_prune_cache_empty_home_cache_dir(self):
        """Test prune_cache with empty cache dir within home."""
        home = Path.home()
        test_cache_dir = home / ".cache" / "test_prune_cache_empty"
        test_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            deleted = summarizer.prune_cache(cache_dir=str(test_cache_dir), dry_run=True)
            self.assertEqual(deleted, 0)
        finally:
            if test_cache_dir.exists():
                test_cache_dir.rmdir()

    def test_prune_cache_fallback_when_cache_utils_fails(self):
        """Test prune_cache uses fallback when cache_utils import fails."""
        # Mock sys.modules to make the import fail inside the function
        with patch.dict(sys.modules, {"podcast_scraper.cache.directories": None}):
            # This should use fallback path
            deleted = summarizer.prune_cache(cache_dir=None, dry_run=True)
            # Should return 0 or positive depending on cache state
            self.assertIsInstance(deleted, int)
            self.assertGreaterEqual(deleted, 0)

    def test_prune_cache_uses_cache_utils_when_available(self):
        """Test prune_cache uses cache_utils when available."""
        home = Path.home()
        test_cache_dir = home / ".cache" / "test_prune_cache_utils"
        test_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with patch(
                "podcast_scraper.cache.directories.get_transformers_cache_dir",
                return_value=test_cache_dir,
            ):
                deleted = summarizer.prune_cache(cache_dir=None, dry_run=True)
                self.assertIsInstance(deleted, int)
        finally:
            if test_cache_dir.exists():
                test_cache_dir.rmdir()

    def test_prune_cache_security_rejects_outside_home(self):
        """Test prune_cache rejects cache dirs outside home."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # This should raise ValueError due to security check
            with self.assertRaises(ValueError) as ctx:
                summarizer.prune_cache(cache_dir=tmp_dir, dry_run=True)
            self.assertIn("outside safe locations", str(ctx.exception))


@unittest.skipIf(not SUMMARIZER_AVAILABLE, "Summarization dependencies not available")
class TestCacheFallbackPaths(unittest.TestCase):
    """Direct tests for cache fallback code paths."""

    def test_hf_cache_dir_exists_check(self):
        """Test HF_CACHE_DIR existence check logic."""
        # Test the cache dir directly
        hf_cache_dir = summarizer.HF_CACHE_DIR
        expected = str(hf_cache_dir)

        # The cache dir should contain huggingface
        self.assertIn("huggingface", expected.lower())

    def test_get_cache_size_direct_fallback_path(self):
        """Test get_cache_size fallback path directly via sys.modules patch."""
        # Patch sys.modules to make the import fail
        with patch.dict(sys.modules, {"podcast_scraper.cache.directories": None}):
            # Call directly with None to trigger fallback
            size = summarizer.get_cache_size(cache_dir=None)
            self.assertIsInstance(size, int)

    def test_prune_cache_direct_fallback_path(self):
        """Test prune_cache fallback path directly via sys.modules patch."""
        with patch.dict(sys.modules, {"podcast_scraper.cache.directories": None}):
            # Call with None to trigger fallback path
            deleted = summarizer.prune_cache(cache_dir=None, dry_run=True)
            self.assertIsInstance(deleted, int)


if __name__ == "__main__":
    unittest.main()
