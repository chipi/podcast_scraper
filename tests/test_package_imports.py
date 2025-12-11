#!/usr/bin/env python3
"""Tests for package import patterns to prevent regressions.

This test suite ensures that all import patterns work correctly and prevents
issues like:
- Circular imports with lazy-loaded modules (cli, service)
- Namespace package conflicts
- Missing __version__ or __api_version__
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import podcast_scraper


class TestPackageImports(unittest.TestCase):
    """Tests for package import patterns."""

    def test_version_attributes_accessible(self):
        """Test that __version__ and __api_version__ are directly accessible."""
        self.assertTrue(hasattr(podcast_scraper, "__version__"))
        self.assertTrue(hasattr(podcast_scraper, "__api_version__"))

        version = podcast_scraper.__version__
        api_version = podcast_scraper.__api_version__

        self.assertIsInstance(version, str)
        self.assertIsInstance(api_version, str)
        self.assertEqual(version, api_version)
        self.assertGreater(len(version), 0)

    def test_version_importable(self):
        """Test that __version__ can be imported directly."""
        from podcast_scraper import __api_version__, __version__

        self.assertEqual(__version__, podcast_scraper.__version__)
        self.assertEqual(__api_version__, podcast_scraper.__api_version__)

    def test_core_imports_work(self):
        """Test that core modules can be imported."""
        from podcast_scraper import Config, load_config_file, run_pipeline

        self.assertTrue(Config is not None)
        self.assertTrue(load_config_file is not None)
        self.assertTrue(run_pipeline is not None)

    def test_cli_module_import_direct(self):
        """Test direct module import: import podcast_scraper.cli (what tests use)."""
        import podcast_scraper.cli as cli

        self.assertTrue(cli is not None)
        self.assertTrue(hasattr(cli, "main"))

    def test_cli_lazy_import(self):
        """Test lazy import via __getattr__: from podcast_scraper import cli."""
        # This was the problematic pattern that caused circular imports
        from podcast_scraper import cli

        self.assertTrue(cli is not None)
        self.assertTrue(hasattr(cli, "main"))
        # Verify it's the same module
        import podcast_scraper.cli as cli_direct

        self.assertIs(cli, cli_direct)

    def test_service_module_import_direct(self):
        """Test direct module import: import podcast_scraper.service (what tests use)."""
        import podcast_scraper.service as service

        self.assertTrue(service is not None)
        self.assertTrue(hasattr(service, "run"))

    def test_service_lazy_import(self):
        """Test lazy import via __getattr__: from podcast_scraper import service."""
        # This was the problematic pattern that caused circular imports
        from podcast_scraper import service

        self.assertTrue(service is not None)
        self.assertTrue(hasattr(service, "run"))
        # Verify it's the same module
        import podcast_scraper.service as service_direct

        self.assertIs(service, service_direct)

    def test_no_circular_import_with_cli(self):
        """Test that importing cli doesn't cause circular imports."""
        # Import cli multiple times to ensure no circular import issues
        from podcast_scraper import cli as cli1, cli as cli2, cli as cli3

        # All should be the same module instance
        self.assertIs(cli1, cli2)
        self.assertIs(cli2, cli3)

    def test_no_circular_import_with_service(self):
        """Test that importing service doesn't cause circular imports."""
        # Import service multiple times to ensure no circular import issues
        from podcast_scraper import service as service1, service as service2, service as service3

        # All should be the same module instance
        self.assertIs(service1, service2)
        self.assertIs(service2, service3)

    def test_mixed_imports_no_conflict(self):
        """Test that mixing different import patterns doesn't cause issues."""
        import podcast_scraper.cli as cli_direct
        from podcast_scraper import cli as cli_lazy

        self.assertIs(cli_direct, cli_lazy)

        import podcast_scraper.service as service_direct
        from podcast_scraper import service as service_lazy

        self.assertIs(service_direct, service_lazy)

    def test_package_structure_correct(self):
        """Test that package structure is correct (no namespace package issues)."""
        spec = importlib.util.find_spec("podcast_scraper")

        self.assertIsNotNone(spec, "podcast_scraper package should be findable")
        self.assertIsNotNone(spec.origin, "Package should have an origin (__init__.py)")

        # Verify it's not a namespace package
        self.assertIsNotNone(spec.origin, "Should have __init__.py file")
        self.assertTrue(
            spec.origin.endswith("__init__.py"),
            f"Package origin should be __init__.py, got: {spec.origin}",
        )

        # Verify package path is correct
        if spec.submodule_search_locations:
            self.assertEqual(
                len(spec.submodule_search_locations), 1, "Should have exactly one package location"
            )
            package_dir = Path(spec.submodule_search_locations[0])
            self.assertTrue(package_dir.exists(), f"Package directory should exist: {package_dir}")
            # Verify there's no conflicting empty podcast_scraper subdirectory
            potential_conflict = package_dir / "podcast_scraper"
            self.assertFalse(
                potential_conflict.exists() and potential_conflict.is_dir(),
                "Should not have empty podcast_scraper/ subdirectory (namespace package conflict)",
            )

    def test_all_imports_in_sequence(self):
        """Test importing everything in sequence to catch any import-order issues."""
        # Import in various orders to ensure no dependencies on import order
        import podcast_scraper
        import podcast_scraper.cli
        import podcast_scraper.service
        from podcast_scraper import __api_version__, __version__, cli, Config, service

        # All should work without errors
        self.assertTrue(podcast_scraper is not None)
        self.assertTrue(Config is not None)
        self.assertTrue(podcast_scraper.cli is not None)
        self.assertTrue(cli is not None)
        self.assertTrue(podcast_scraper.service is not None)
        self.assertTrue(service is not None)
        self.assertTrue(__version__ is not None)
        self.assertTrue(__api_version__ is not None)

    def test_cli_import_doesnt_break_version(self):
        """Test that importing cli doesn't break version access."""
        version_before = podcast_scraper.__version__

        from podcast_scraper import cli

        # Verify cli module is accessible
        self.assertIsNotNone(cli)
        version_after = podcast_scraper.__version__

        self.assertEqual(version_before, version_after)
        self.assertEqual(version_after, "2.4.0")

    def test_service_import_doesnt_break_version(self):
        """Test that importing service doesn't break version access."""
        version_before = podcast_scraper.__version__

        from podcast_scraper import service

        # Verify service module is accessible
        self.assertIsNotNone(service)
        version_after = podcast_scraper.__version__

        self.assertEqual(version_before, version_after)
        self.assertEqual(version_after, "2.4.0")


if __name__ == "__main__":
    unittest.main()
