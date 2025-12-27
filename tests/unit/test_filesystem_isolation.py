#!/usr/bin/env python3
"""Test that filesystem I/O isolation is enforced in unit tests.

This test verifies that the filesystem I/O blocker in tests/unit/conftest.py
correctly prevents filesystem operations in unit tests (except tempfile operations).
"""

import os
import tempfile
import unittest
from pathlib import Path


class TestFilesystemIsolation(unittest.TestCase):
    """Test that filesystem I/O is blocked in unit tests."""

    def test_open_blocked_outside_temp(self):
        """Test that open() is blocked outside temp directories."""
        # Try to open a file in current directory (should be blocked)
        with self.assertRaises(Exception) as context:
            with open("test_file.txt", "w") as f:
                f.write("test")

        # Verify it's our FilesystemIODetectedError
        self.assertIn("Filesystem I/O detected", str(context.exception))
        self.assertIn("open()", str(context.exception))

    def test_open_allowed_in_temp(self):
        """Test that open() is allowed within temp directories."""
        # Create a temp file (should be allowed)
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write("test content")

        try:
            # Reading from temp file should be allowed
            with open(tmp_path, "r") as f:
                content = f.read()
            self.assertEqual(content, "test content")
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_os_makedirs_blocked(self):
        """Test that os.makedirs() is blocked outside temp directories."""
        with self.assertRaises(Exception) as context:
            os.makedirs("test_directory", exist_ok=True)

        self.assertIn("Filesystem I/O detected", str(context.exception))
        self.assertIn("os.makedirs()", str(context.exception))

    def test_os_makedirs_allowed_in_temp(self):
        """Test that os.makedirs() is allowed within temp directories."""
        # Create temp directory (should be allowed)
        temp_dir = tempfile.mkdtemp()
        try:
            test_dir = os.path.join(temp_dir, "subdir")
            os.makedirs(test_dir, exist_ok=True)
            self.assertTrue(os.path.exists(test_dir))
        finally:
            os.rmdir(test_dir)
            os.rmdir(temp_dir)

    def test_path_write_text_blocked(self):
        """Test that Path.write_text() is blocked outside temp directories."""
        path = Path("test_file.txt")
        with self.assertRaises(Exception) as context:
            path.write_text("test content")

        self.assertIn("Filesystem I/O detected", str(context.exception))
        self.assertIn("Path.write_text()", str(context.exception))

    def test_path_write_text_allowed_in_temp(self):
        """Test that Path.write_text() is allowed within temp directories."""
        # Create temp file (should be allowed)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp_path = Path(tmp.name)

        try:
            tmp_path.write_text("test content")
            content = tmp_path.read_text()
            self.assertEqual(content, "test content")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_shutil_rmtree_blocked(self):
        """Test that shutil.rmtree() is blocked outside temp directories."""
        import shutil

        with self.assertRaises(Exception) as context:
            shutil.rmtree("nonexistent_dir", ignore_errors=True)

        self.assertIn("Filesystem I/O detected", str(context.exception))
        self.assertIn("shutil.rmtree()", str(context.exception))

    def test_shutil_rmtree_allowed_in_temp(self):
        """Test that shutil.rmtree() is allowed within temp directories."""
        import shutil

        # Create temp directory (should be allowed)
        temp_dir = tempfile.mkdtemp()
        try:
            # Create subdirectory
            subdir = os.path.join(temp_dir, "subdir")
            os.makedirs(subdir)
            # Remove it (should be allowed)
            shutil.rmtree(subdir)
            self.assertFalse(os.path.exists(subdir))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
