#!/usr/bin/env python3
"""Tests for cache_manager module.

This module tests ML model cache management utilities.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper.cache import manager as cache_manager


class TestFormatSize(unittest.TestCase):
    """Test format_size function."""

    def test_format_size_bytes(self):
        """Test formatting bytes."""
        self.assertEqual(cache_manager.format_size(0), "0.00 B")
        self.assertEqual(cache_manager.format_size(512), "512.00 B")

    def test_format_size_kb(self):
        """Test formatting kilobytes."""
        self.assertEqual(cache_manager.format_size(1024), "1.00 KB")
        self.assertEqual(cache_manager.format_size(1536), "1.50 KB")

    def test_format_size_mb(self):
        """Test formatting megabytes."""
        self.assertEqual(cache_manager.format_size(1024 * 1024), "1.00 MB")
        self.assertEqual(cache_manager.format_size(1536 * 1024), "1.50 MB")

    def test_format_size_gb(self):
        """Test formatting gigabytes."""
        self.assertEqual(cache_manager.format_size(1024 * 1024 * 1024), "1.00 GB")
        self.assertEqual(cache_manager.format_size(1536 * 1024 * 1024), "1.50 GB")


class TestCalculateDirectorySize(unittest.TestCase):
    """Test calculate_directory_size function."""

    def test_calculate_directory_size_nonexistent(self):
        """Test calculating size of nonexistent directory."""
        size = cache_manager.calculate_directory_size(Path("/nonexistent/path"))
        self.assertEqual(size, 0)

    def test_calculate_directory_size_empty(self):
        """Test calculating size of empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size = cache_manager.calculate_directory_size(Path(tmpdir))
            self.assertEqual(size, 0)

    def test_calculate_directory_size_with_files(self):
        """Test calculating size of directory with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = Path(tmpdir) / "file1.txt"
            file1.write_text("test content")
            file2 = Path(tmpdir) / "file2.txt"
            file2.write_text("more content")

            size = cache_manager.calculate_directory_size(Path(tmpdir))
            self.assertGreater(size, 0)
            # Should be at least the size of both files
            self.assertGreaterEqual(size, len("test content") + len("more content"))

    def test_calculate_directory_size_nested(self):
        """Test calculating size of nested directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            file1 = subdir / "file1.txt"
            file1.write_text("nested content")

            size = cache_manager.calculate_directory_size(Path(tmpdir))
            self.assertGreater(size, 0)


class TestGetWhisperCacheInfo(unittest.TestCase):
    """Test get_whisper_cache_info function."""

    @patch("podcast_scraper.cache.manager.get_whisper_cache_dir")
    def test_get_whisper_cache_info_empty(self, mock_get_dir):
        """Test getting info for empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_dir.return_value = Path(tmpdir)
            cache_dir, size, models = cache_manager.get_whisper_cache_info()
            self.assertEqual(cache_dir, Path(tmpdir))
            self.assertEqual(size, 0)
            self.assertEqual(models, [])

    @patch("podcast_scraper.cache.manager.get_whisper_cache_dir")
    def test_get_whisper_cache_info_with_models(self, mock_get_dir):
        """Test getting info for cache with models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_dir.return_value = Path(tmpdir)
            # Create mock model files
            model1 = Path(tmpdir) / "base.en.pt"
            model1.write_bytes(b"x" * 100)
            model2 = Path(tmpdir) / "tiny.en.pt"
            model2.write_bytes(b"y" * 200)

            cache_dir, size, models = cache_manager.get_whisper_cache_info()
            self.assertEqual(cache_dir, Path(tmpdir))
            self.assertEqual(size, 300)
            self.assertEqual(len(models), 2)
            # Should be sorted by name
            self.assertEqual(models[0]["name"], "base.en.pt")
            self.assertEqual(models[1]["name"], "tiny.en.pt")


class TestGetTransformersCacheInfo(unittest.TestCase):
    """Test get_transformers_cache_info function."""

    @patch("podcast_scraper.cache.manager.get_transformers_cache_dir")
    def test_get_transformers_cache_info_empty(self, mock_get_dir):
        """Test getting info for empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_dir.return_value = Path(tmpdir)
            cache_dir, size, models = cache_manager.get_transformers_cache_info()
            self.assertEqual(cache_dir, Path(tmpdir))
            self.assertEqual(size, 0)
            self.assertEqual(models, [])

    @patch("podcast_scraper.cache.manager.get_transformers_cache_dir")
    def test_get_transformers_cache_info_with_models(self, mock_get_dir):
        """Test getting info for cache with models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_dir.return_value = Path(tmpdir)
            # Create mock model directories
            model1_dir = Path(tmpdir) / "models--facebook--bart-base"
            model1_dir.mkdir()
            (model1_dir / "file1.bin").write_bytes(b"x" * 100)

            model2_dir = Path(tmpdir) / "models--sshleifer--distilbart-cnn-12-6"
            model2_dir.mkdir()
            (model2_dir / "file2.bin").write_bytes(b"y" * 200)

            cache_dir, size, models = cache_manager.get_transformers_cache_info()
            self.assertEqual(cache_dir, Path(tmpdir))
            self.assertGreater(size, 0)
            self.assertEqual(len(models), 2)
            # Should convert directory names to model names
            model_names = [m["name"] for m in models]
            self.assertIn("facebook/bart-base", model_names)
            self.assertIn("sshleifer/distilbart-cnn-12-6", model_names)


class TestGetSpacyCacheInfo(unittest.TestCase):
    """Test get_spacy_cache_info function."""

    @patch("podcast_scraper.cache.manager.get_spacy_cache_dir")
    def test_get_spacy_cache_info_empty(self, mock_get_dir):
        """Test getting info for empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_dir.return_value = Path(tmpdir)
            cache_dir, size, models = cache_manager.get_spacy_cache_info()
            self.assertEqual(cache_dir, Path(tmpdir))
            self.assertEqual(size, 0)
            self.assertEqual(models, [])

    @patch("podcast_scraper.cache.manager.get_spacy_cache_dir")
    def test_get_spacy_cache_info_none(self, mock_get_dir):
        """Test getting info when cache_dir is None."""
        mock_get_dir.return_value = None
        cache_dir, size, models = cache_manager.get_spacy_cache_info()
        self.assertIsNone(cache_dir)
        self.assertEqual(size, 0)
        self.assertEqual(models, [])

    @patch("podcast_scraper.cache.manager.get_spacy_cache_dir")
    def test_get_spacy_cache_info_with_models(self, mock_get_dir):
        """Test getting info for cache with models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_dir.return_value = Path(tmpdir)
            # Create mock model directories
            model1_dir = Path(tmpdir) / "en_core_web_sm"
            model1_dir.mkdir()
            (model1_dir / "file1.bin").write_bytes(b"x" * 100)

            cache_dir, size, models = cache_manager.get_spacy_cache_info()
            self.assertEqual(cache_dir, Path(tmpdir))
            self.assertGreater(size, 0)
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["name"], "en_core_web_sm")


class TestGetAllCacheInfo(unittest.TestCase):
    """Test get_all_cache_info function."""

    @patch("podcast_scraper.cache.manager.get_spacy_cache_info")
    @patch("podcast_scraper.cache.manager.get_transformers_cache_info")
    @patch("podcast_scraper.cache.manager.get_whisper_cache_info")
    def test_get_all_cache_info(self, mock_whisper, mock_transformers, mock_spacy):
        """Test getting info for all caches."""
        mock_whisper.return_value = (Path("/whisper"), 100, [])
        mock_transformers.return_value = (Path("/transformers"), 200, [])
        mock_spacy.return_value = (Path("/spacy"), 50, [])

        info = cache_manager.get_all_cache_info()

        self.assertEqual(info["whisper"]["size"], 100)
        self.assertEqual(info["transformers"]["size"], 200)
        self.assertEqual(info["spacy"]["size"], 50)
        self.assertEqual(info["total_size"], 350)


class TestCleanWhisperCache(unittest.TestCase):
    """Test clean_whisper_cache function."""

    @patch("podcast_scraper.cache.manager.get_whisper_cache_info")
    def test_clean_whisper_cache_empty(self, mock_get_info):
        """Test cleaning empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_info.return_value = (Path(tmpdir), 0, [])
            deleted, freed = cache_manager.clean_whisper_cache(confirm=False)
            self.assertEqual(deleted, 0)
            self.assertEqual(freed, 0)

    @patch("podcast_scraper.cache.manager.get_whisper_cache_info")
    @patch("builtins.input")
    def test_clean_whisper_cache_with_confirmation(self, mock_input, mock_get_info):
        """Test cleaning cache with confirmation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = Path(tmpdir) / "base.en.pt"
            model_file.write_bytes(b"x" * 100)
            mock_get_info.return_value = (
                Path(tmpdir),
                100,
                [{"name": "base.en.pt", "size": 100, "path": model_file}],
            )
            mock_input.return_value = "yes"

            deleted, freed = cache_manager.clean_whisper_cache(confirm=True)
            self.assertEqual(deleted, 1)
            self.assertEqual(freed, 100)
            self.assertFalse(model_file.exists())

    @patch("podcast_scraper.cache.manager.get_whisper_cache_info")
    @patch("builtins.input")
    def test_clean_whisper_cache_cancelled(self, mock_input, mock_get_info):
        """Test cleaning cache when cancelled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = Path(tmpdir) / "base.en.pt"
            model_file.write_bytes(b"x" * 100)
            mock_get_info.return_value = (
                Path(tmpdir),
                100,
                [{"name": "base.en.pt", "size": 100, "path": model_file}],
            )
            mock_input.return_value = "no"

            deleted, freed = cache_manager.clean_whisper_cache(confirm=True)
            self.assertEqual(deleted, 0)
            self.assertEqual(freed, 0)
            self.assertTrue(model_file.exists())


class TestCleanTransformersCache(unittest.TestCase):
    """Test clean_transformers_cache function."""

    @patch("podcast_scraper.cache.manager.get_transformers_cache_info")
    def test_clean_transformers_cache_empty(self, mock_get_info):
        """Test cleaning empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_info.return_value = (Path(tmpdir), 0, [])
            deleted, freed = cache_manager.clean_transformers_cache(confirm=False)
            self.assertEqual(deleted, 0)
            self.assertEqual(freed, 0)


class TestCleanSpacyCache(unittest.TestCase):
    """Test clean_spacy_cache function."""

    @patch("podcast_scraper.cache.manager.get_spacy_cache_info")
    def test_clean_spacy_cache_empty(self, mock_get_info):
        """Test cleaning empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_info.return_value = (Path(tmpdir), 0, [])
            deleted, freed = cache_manager.clean_spacy_cache(confirm=False)
            self.assertEqual(deleted, 0)
            self.assertEqual(freed, 0)

    @patch("podcast_scraper.cache.manager.get_spacy_cache_info")
    def test_clean_spacy_cache_none(self, mock_get_info):
        """Test cleaning cache when cache_dir is None."""
        mock_get_info.return_value = (None, 0, [])
        deleted, freed = cache_manager.clean_spacy_cache(confirm=False)
        self.assertEqual(deleted, 0)
        self.assertEqual(freed, 0)


class TestCleanAllCaches(unittest.TestCase):
    """Test clean_all_caches function."""

    @patch("podcast_scraper.cache.manager.get_all_cache_info")
    def test_clean_all_caches_empty(self, mock_get_info):
        """Test cleaning all caches when empty."""
        mock_get_info.return_value = {"total_size": 0}
        results = cache_manager.clean_all_caches(confirm=False)
        self.assertEqual(results["whisper"], (0, 0))
        self.assertEqual(results["transformers"], (0, 0))
        self.assertEqual(results["spacy"], (0, 0))

    @patch("podcast_scraper.cache.manager.clean_spacy_cache")
    @patch("podcast_scraper.cache.manager.clean_transformers_cache")
    @patch("podcast_scraper.cache.manager.clean_whisper_cache")
    @patch("podcast_scraper.cache.manager.get_all_cache_info")
    @patch("builtins.input")
    def test_clean_all_caches_with_confirmation(
        self, mock_input, mock_get_info, mock_whisper, mock_transformers, mock_spacy
    ):
        """Test cleaning all caches with confirmation."""
        mock_get_info.return_value = {
            "whisper": {"count": 1, "size": 100},
            "transformers": {"count": 2, "size": 200},
            "spacy": {"count": 1, "size": 50},
            "total_size": 350,
        }
        mock_input.return_value = "yes"
        mock_whisper.return_value = (1, 100)
        mock_transformers.return_value = (2, 200)
        mock_spacy.return_value = (1, 50)

        results = cache_manager.clean_all_caches(confirm=True)
        self.assertEqual(results["whisper"], (1, 100))
        self.assertEqual(results["transformers"], (2, 200))
        self.assertEqual(results["spacy"], (1, 50))

    @patch("podcast_scraper.cache.manager.clean_spacy_cache")
    @patch("podcast_scraper.cache.manager.clean_transformers_cache")
    @patch("podcast_scraper.cache.manager.clean_whisper_cache")
    @patch("podcast_scraper.cache.manager.get_all_cache_info")
    @patch("builtins.input")
    def test_clean_all_caches_cancelled(
        self, mock_input, mock_get_info, mock_whisper, mock_transformers, mock_spacy
    ):
        """Test cleaning all caches when cancelled."""
        mock_get_info.return_value = {
            "whisper": {"count": 1, "size": 100},
            "transformers": {"count": 2, "size": 200},
            "spacy": {"count": 1, "size": 50},
            "total_size": 350,
        }
        mock_input.return_value = "no"

        results = cache_manager.clean_all_caches(confirm=True)
        self.assertEqual(results["whisper"], (0, 0))
        self.assertEqual(results["transformers"], (0, 0))
        self.assertEqual(results["spacy"], (0, 0))
        # Should not call clean functions when cancelled
        mock_whisper.assert_not_called()
        mock_transformers.assert_not_called()
        mock_spacy.assert_not_called()


class TestCacheManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in cache_manager."""

    @patch("podcast_scraper.cache.manager.get_whisper_cache_info")
    def test_clean_whisper_cache_permission_error(self, mock_get_info):
        """Test cleaning cache handles permission errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = Path(tmpdir) / "base.en.pt"
            model_file.write_bytes(b"x" * 100)
            mock_get_info.return_value = (
                Path(tmpdir),
                100,
                [{"name": "base.en.pt", "size": 100, "path": model_file}],
            )

            # Mock Path.unlink to raise PermissionError for the model file
            original_unlink = Path.unlink

            def mock_unlink(self):
                if self == model_file:
                    raise PermissionError("Access denied")
                return original_unlink(self)

            with patch("pathlib.Path.unlink", mock_unlink):
                deleted, freed = cache_manager.clean_whisper_cache(confirm=False)
                # Should handle error gracefully and continue
                self.assertGreaterEqual(deleted, 0)

    @patch("podcast_scraper.cache.manager.get_transformers_cache_info")
    def test_clean_transformers_cache_with_models(self, mock_get_info):
        """Test cleaning transformers cache with models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models--test--model"
            model_dir.mkdir()
            (model_dir / "file.bin").write_bytes(b"x" * 200)
            mock_get_info.return_value = (
                Path(tmpdir),
                200,
                [{"name": "test/model", "size": 200, "path": model_dir}],
            )

            deleted, freed = cache_manager.clean_transformers_cache(confirm=False)
            self.assertEqual(deleted, 1)
            self.assertEqual(freed, 200)
            self.assertFalse(model_dir.exists())

    @patch("podcast_scraper.cache.manager.get_spacy_cache_info")
    def test_clean_spacy_cache_with_models(self, mock_get_info):
        """Test cleaning spacy cache with models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "en_core_web_sm"
            model_dir.mkdir()
            (model_dir / "file.bin").write_bytes(b"x" * 150)
            mock_get_info.return_value = (
                Path(tmpdir),
                150,
                [{"name": "en_core_web_sm", "size": 150, "path": model_dir}],
            )

            deleted, freed = cache_manager.clean_spacy_cache(confirm=False)
            self.assertEqual(deleted, 1)
            self.assertEqual(freed, 150)
            self.assertFalse(model_dir.exists())

    def test_format_size_petabytes(self):
        """Test format_size with very large values."""
        size = 1024 * 1024 * 1024 * 1024 * 1024  # 1 PB
        result = cache_manager.format_size(size)
        self.assertIn("PB", result)

    def test_calculate_directory_size_permission_error(self):
        """Test calculate_directory_size handles permission errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file we can't access (simulate permission error)
            restricted_file = Path(tmpdir) / "restricted.txt"
            restricted_file.write_bytes(b"test")

            # Mock Path.stat to raise PermissionError for the restricted file
            original_stat = Path.stat

            def mock_stat(self):
                if self == restricted_file:
                    raise PermissionError("Access denied")
                return original_stat(self)

            with patch("pathlib.Path.stat", mock_stat):
                size = cache_manager.calculate_directory_size(Path(tmpdir))
                # Should handle error gracefully and return partial or 0
                self.assertGreaterEqual(size, 0)
