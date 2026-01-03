#!/usr/bin/env python3
"""Tests for cache utility functions.

This module tests cache directory resolution utilities.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper import cache_utils


class TestGetProjectRoot(unittest.TestCase):
    """Test get_project_root function."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear cached project root
        cache_utils._project_root = None

    def tearDown(self):
        """Clean up test fixtures."""
        # Clear cached project root
        cache_utils._project_root = None

    def test_get_project_root_finds_pyproject_toml(self):
        """Test project root detection via pyproject.toml."""
        # The actual project root should contain pyproject.toml
        root = cache_utils.get_project_root()
        self.assertTrue((root / "pyproject.toml").exists())
        self.assertIsInstance(root, Path)

    def test_get_project_root_cached(self):
        """Test that project root is cached after first call."""
        root1 = cache_utils.get_project_root()
        root2 = cache_utils.get_project_root()
        self.assertEqual(root1, root2)
        # Verify it's the same object (cached)
        self.assertIs(cache_utils._project_root, root1)

    def test_get_project_root_fallback(self):
        """Test project root fallback when pyproject.toml not found."""
        # Mock the file path to simulate being in a different location
        with patch.object(cache_utils, "__file__", "/some/nonexistent/path/cache_utils.py"):
            # Clear cache
            cache_utils._project_root = None
            # Should fallback to going up 2 levels from src/podcast_scraper/
            root = cache_utils.get_project_root()
            self.assertIsInstance(root, Path)


class TestGetWhisperCacheDir(unittest.TestCase):
    """Test get_whisper_cache_dir function."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear cached project root
        cache_utils._project_root = None

    def tearDown(self):
        """Clean up test fixtures."""
        # Clear cached project root
        cache_utils._project_root = None

    def test_get_whisper_cache_dir_prefers_local(self):
        """Test Whisper cache prefers local .cache/ directory."""
        import tempfile

        # Use temp directory to avoid filesystem I/O restrictions
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock project root to point to temp directory
            with patch.object(cache_utils, "get_project_root") as mock_root:
                mock_root.return_value = Path(temp_dir)
                local_cache = Path(temp_dir) / ".cache" / "whisper"
                local_cache.mkdir(parents=True, exist_ok=True)

                cache_dir = cache_utils.get_whisper_cache_dir()
                self.assertEqual(cache_dir, local_cache)

    def test_get_whisper_cache_dir_falls_back_to_home(self):
        """Test Whisper cache falls back to ~/.cache/whisper when local doesn't exist."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock project root to point to temp directory (no .cache exists)
            with patch.object(cache_utils, "get_project_root") as mock_root:
                mock_root.return_value = Path(temp_dir)
                # Mock home directory
                with patch("pathlib.Path.home") as mock_home:
                    mock_home.return_value = Path(temp_dir)
                    expected = Path(temp_dir) / ".cache" / "whisper"
                    cache_dir = cache_utils.get_whisper_cache_dir()
                    self.assertEqual(cache_dir, expected)

    def test_get_whisper_cache_dir_home_fallback(self):
        """Test Whisper cache uses home directory for fallback."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock project root and home
            with patch.object(cache_utils, "get_project_root") as mock_root:
                mock_root.return_value = Path(temp_dir)
                with patch("pathlib.Path.home") as mock_home:
                    mock_home.return_value = Path(temp_dir)
                    cache_dir = cache_utils.get_whisper_cache_dir()
                    expected = Path(temp_dir) / ".cache" / "whisper"
                    self.assertEqual(cache_dir, expected)


class TestGetTransformersCacheDir(unittest.TestCase):
    """Test get_transformers_cache_dir function."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear cached project root
        cache_utils._project_root = None

    def tearDown(self):
        """Clean up test fixtures."""
        # Clear cached project root
        cache_utils._project_root = None

    def test_get_transformers_cache_dir_prefers_local(self):
        """Test Transformers cache prefers local .cache/ directory."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock project root
            with patch.object(cache_utils, "get_project_root") as mock_root:
                mock_root.return_value = Path(temp_dir)
                local_cache = Path(temp_dir) / ".cache" / "huggingface" / "hub"
                local_cache.mkdir(parents=True, exist_ok=True)

                cache_dir = cache_utils.get_transformers_cache_dir()
                self.assertEqual(cache_dir, local_cache)

    def test_get_transformers_cache_dir_falls_back_to_default(self):
        """Test Transformers cache falls back to default when local doesn't exist."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock project root
            with patch.object(cache_utils, "get_project_root") as mock_root:
                mock_root.return_value = Path(temp_dir)
                # Mock transformers.file_utils.default_cache_path
                with patch("transformers.file_utils") as mock_file_utils:
                    mock_file_utils.default_cache_path = str(Path(temp_dir) / "transformers_cache")
                    cache_dir = cache_utils.get_transformers_cache_dir()
                    expected = Path(mock_file_utils.default_cache_path)
                    self.assertEqual(cache_dir, expected)

    def test_get_transformers_cache_dir_falls_back_when_transformers_not_installed(self):
        """Test Transformers cache falls back when transformers not installed."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock project root and home
            with patch.object(cache_utils, "get_project_root") as mock_root:
                mock_root.return_value = Path(temp_dir)
                with patch("pathlib.Path.home") as mock_home:
                    mock_home.return_value = Path(temp_dir)
                    # Mock ImportError when importing transformers
                    with patch(
                        "builtins.__import__",
                        side_effect=ImportError("No module named transformers"),
                    ):
                        cache_dir = cache_utils.get_transformers_cache_dir()
                        expected = Path(temp_dir) / ".cache" / "huggingface" / "hub"
                        self.assertEqual(cache_dir, expected)

    def test_get_transformers_cache_dir_home_fallback(self):
        """Test Transformers cache uses home directory for fallback."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock project root and home
            with patch.object(cache_utils, "get_project_root") as mock_root:
                mock_root.return_value = Path(temp_dir)
                with patch("pathlib.Path.home") as mock_home:
                    mock_home.return_value = Path(temp_dir)
                    # Mock ImportError
                    with patch(
                        "builtins.__import__",
                        side_effect=ImportError("No module named transformers"),
                    ):
                        cache_dir = cache_utils.get_transformers_cache_dir()
                        expected = Path(temp_dir) / ".cache" / "huggingface" / "hub"
                        self.assertEqual(cache_dir, expected)


class TestGetSpacyCacheDir(unittest.TestCase):
    """Test get_spacy_cache_dir function."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear cached project root
        cache_utils._project_root = None

    def tearDown(self):
        """Clean up test fixtures."""
        # Clear cached project root
        cache_utils._project_root = None

    def test_get_spacy_cache_dir_prefers_local(self):
        """Test spaCy cache prefers local .cache/ directory."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock project root
            with patch.object(cache_utils, "get_project_root") as mock_root:
                mock_root.return_value = Path(temp_dir)
                local_cache = Path(temp_dir) / ".cache" / "spacy"
                local_cache.mkdir(parents=True, exist_ok=True)

                cache_dir = cache_utils.get_spacy_cache_dir()
                self.assertEqual(cache_dir, local_cache)

    def test_get_spacy_cache_dir_falls_back_to_user_data(self):
        """Test spaCy cache falls back to user data dir when local doesn't exist."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock project root
            with patch.object(cache_utils, "get_project_root") as mock_root:
                mock_root.return_value = Path(temp_dir)
                # Mock Path.home() to return temp directory (function uses Path.home() directly)
                with patch("pathlib.Path.home") as mock_home:
                    mock_home.return_value = Path(temp_dir)
                    user_data_path = Path(temp_dir) / ".local" / "share" / "spacy"
                    user_data_path.mkdir(parents=True, exist_ok=True)

                    cache_dir = cache_utils.get_spacy_cache_dir()
                    self.assertEqual(cache_dir, user_data_path)

    def test_get_spacy_cache_dir_returns_none_when_no_cache_exists(self):
        """Test spaCy cache returns None when no cache directories exist."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock project root
            with patch.object(cache_utils, "get_project_root") as mock_root:
                mock_root.return_value = Path(temp_dir)
                # Mock platformdirs.user_data_dir to return non-existent path
                with patch("platformdirs.user_data_dir") as mock_user_data_dir:
                    mock_user_data_dir.return_value = str(Path(temp_dir) / "nonexistent" / "spacy")
                    cache_dir = cache_utils.get_spacy_cache_dir()
                    self.assertIsNone(cache_dir)

    def test_get_spacy_cache_dir_home_fallback(self):
        """Test spaCy cache uses home directory for fallback."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock project root and home
            with patch.object(cache_utils, "get_project_root") as mock_root:
                mock_root.return_value = Path(temp_dir)
                with patch("pathlib.Path.home") as mock_home:
                    mock_home.return_value = Path(temp_dir)
                    # Create user data dir
                    user_data_dir = Path(temp_dir) / ".local" / "share" / "spacy"
                    user_data_dir.mkdir(parents=True, exist_ok=True)

                    cache_dir = cache_utils.get_spacy_cache_dir()
                    self.assertEqual(cache_dir, user_data_dir)
