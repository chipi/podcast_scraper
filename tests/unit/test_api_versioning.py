#!/usr/bin/env python3
"""Tests for API versioning."""

import os
import sys
import unittest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import podcast_scraper


class TestAPIVersioning(unittest.TestCase):
    """Tests for API versioning."""

    def test_api_version_exists(self):
        """__api_version__ should be accessible."""
        self.assertTrue(hasattr(podcast_scraper, "__api_version__"))
        api_version = podcast_scraper.__api_version__
        self.assertIsInstance(api_version, str)
        self.assertGreater(len(api_version), 0)

    def test_api_version_matches_module_version(self):
        """__api_version__ should match __version__."""
        api_version = podcast_scraper.__api_version__
        module_version = podcast_scraper.__version__
        self.assertEqual(api_version, module_version)

    def test_api_version_format(self):
        """__api_version__ should follow semantic versioning (major.minor.patch)."""
        api_version = podcast_scraper.__api_version__
        parts = api_version.split(".")
        self.assertEqual(len(parts), 3, f"Version should be major.minor.patch, got: {api_version}")
        for part in parts:
            self.assertTrue(part.isdigit(), f"Version part should be numeric, got: {part}")

    def test_api_version_in_all(self):
        """__api_version__ should be in __all__."""
        self.assertIn("__api_version__", podcast_scraper.__all__)

    def test_version_in_all(self):
        """__version__ should be in __all__."""
        self.assertIn("__version__", podcast_scraper.__all__)

    def test_api_version_importable(self):
        """__api_version__ should be importable from package."""
        from podcast_scraper import __api_version__, __version__

        self.assertEqual(__api_version__, __version__)
        self.assertEqual(__api_version__, podcast_scraper.__api_version__)

    def test_lazy_loaded_modules_importable(self):
        """Test lazy-loaded modules (cli, service) import without circular import issues."""
        # These use __getattr__ which previously had circular import bugs
        from podcast_scraper import cli, service

        self.assertTrue(cli is not None)
        self.assertTrue(service is not None)
        # Verify they're actual modules
        import types

        self.assertIsInstance(cli, types.ModuleType)
        self.assertIsInstance(service, types.ModuleType)
