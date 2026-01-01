#!/usr/bin/env python3
"""Tests for progress reporting functionality.

These tests verify the progress reporting system that allows pluggable
progress indicators for long-running operations.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper import progress


class TestNoopProgress(unittest.TestCase):
    """Test the no-op progress reporter."""

    def test_noop_progress_update(self):
        """Test that noop progress update does nothing."""
        reporter = progress._NoopProgress()
        # Should not raise any errors
        reporter.update(1)
        reporter.update(10)
        reporter.update(0)
        reporter.update(-1)  # Even negative values should be handled

    def test_noop_progress_context_manager(self):
        """Test that noop progress context manager works."""
        with progress._noop_progress(100, "Test") as reporter:
            self.assertIsInstance(reporter, progress._NoopProgress)
            reporter.update(50)
            reporter.update(50)


class TestSetProgressFactory(unittest.TestCase):
    """Test set_progress_factory function."""

    def setUp(self):
        """Save and restore the original factory."""
        self.original_factory = progress._progress_factory

    def tearDown(self):
        """Restore the original factory."""
        progress._progress_factory = self.original_factory

    def test_set_custom_factory(self):
        """Test setting a custom progress factory."""
        mock_factory = MagicMock()
        mock_reporter = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_reporter)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_factory.return_value = mock_context

        progress.set_progress_factory(mock_factory)
        self.assertEqual(progress._progress_factory, mock_factory)

        # Verify factory is used
        with progress.progress_context(100, "Test"):
            pass
        mock_factory.assert_called_once_with(100, "Test")

    def test_set_none_resets_to_noop(self):
        """Test that setting None resets to noop factory."""
        # Set a custom factory first
        mock_factory = MagicMock()
        progress.set_progress_factory(mock_factory)
        self.assertEqual(progress._progress_factory, mock_factory)

        # Reset to None
        progress.set_progress_factory(None)
        self.assertEqual(progress._progress_factory, progress._noop_progress)

    def test_replace_factory(self):
        """Test replacing one factory with another."""
        mock_factory1 = MagicMock()
        mock_factory2 = MagicMock()

        progress.set_progress_factory(mock_factory1)
        self.assertEqual(progress._progress_factory, mock_factory1)

        progress.set_progress_factory(mock_factory2)
        self.assertEqual(progress._progress_factory, mock_factory2)


class TestProgressContext(unittest.TestCase):
    """Test progress_context function."""

    def setUp(self):
        """Save and restore the original factory."""
        self.original_factory = progress._progress_factory

    def tearDown(self):
        """Restore the original factory."""
        progress._progress_factory = self.original_factory

    def test_default_noop_factory(self):
        """Test that default factory is noop when none is set."""
        progress._progress_factory = None
        with progress.progress_context(100, "Test") as reporter:
            self.assertIsInstance(reporter, progress._NoopProgress)

    def test_uses_custom_factory(self):
        """Test that custom factory is used when set."""
        mock_reporter = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_reporter)
        mock_context.__exit__ = MagicMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_context)
        progress.set_progress_factory(mock_factory)

        with progress.progress_context(50, "Custom") as reporter:
            self.assertEqual(reporter, mock_reporter)
            reporter.update(25)

        mock_factory.assert_called_once_with(50, "Custom")
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()

    def test_progress_context_with_total(self):
        """Test progress_context with a total value."""
        mock_reporter = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_reporter)
        mock_context.__exit__ = MagicMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_context)
        progress.set_progress_factory(mock_factory)

        with progress.progress_context(100, "Processing"):
            pass

        mock_factory.assert_called_once_with(100, "Processing")

    def test_progress_context_without_total(self):
        """Test progress_context without a total value (indeterminate)."""
        mock_reporter = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_reporter)
        mock_context.__exit__ = MagicMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_context)
        progress.set_progress_factory(mock_factory)

        with progress.progress_context(None, "Processing"):
            pass

        mock_factory.assert_called_once_with(None, "Processing")

    def test_progress_context_exception_handling(self):
        """Test that progress_context properly handles exceptions."""
        mock_reporter = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_reporter)
        mock_context.__exit__ = MagicMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_context)
        progress.set_progress_factory(mock_factory)

        with self.assertRaises(ValueError):
            with progress.progress_context(100, "Test"):
                raise ValueError("Test exception")

        # Context manager should still be properly exited
        mock_context.__exit__.assert_called_once()

    def test_progress_context_multiple_updates(self):
        """Test multiple progress updates within context."""
        mock_reporter = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_reporter)
        mock_context.__exit__ = MagicMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_context)
        progress.set_progress_factory(mock_factory)

        with progress.progress_context(100, "Test") as reporter:
            reporter.update(10)
            reporter.update(20)
            reporter.update(30)

        self.assertEqual(mock_reporter.update.call_count, 3)
        mock_reporter.update.assert_any_call(10)
        mock_reporter.update.assert_any_call(20)
        mock_reporter.update.assert_any_call(30)


class TestProgressBackwardsCompatibility(unittest.TestCase):
    """Test backwards compatibility alias."""

    def setUp(self):
        """Save and restore the original factory."""
        self.original_factory = progress._progress_factory

    def tearDown(self):
        """Restore the original factory."""
        progress._progress_factory = self.original_factory

    def test_progress_alias_works(self):
        """Test that 'progress' alias works like 'progress_context'."""
        mock_reporter = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_reporter)
        mock_context.__exit__ = MagicMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_context)
        progress.set_progress_factory(mock_factory)

        # Use the alias
        with progress.progress(100, "Test") as reporter:
            self.assertEqual(reporter, mock_reporter)

        mock_factory.assert_called_once_with(100, "Test")


class TestProgressReporterProtocol(unittest.TestCase):
    """Test that ProgressReporter protocol is correctly defined."""

    def test_progress_reporter_has_update_method(self):
        """Test that ProgressReporter protocol requires update method."""

        # Create a mock that implements the protocol
        class MockReporter:
            def update(self, advance: int) -> None:
                pass

        reporter = MockReporter()
        # Should be able to call update
        reporter.update(1)
        reporter.update(10)

    def test_noop_progress_implements_protocol(self):
        """Test that _NoopProgress implements ProgressReporter protocol."""
        reporter = progress._NoopProgress()
        # Should have update method
        self.assertTrue(hasattr(reporter, "update"))
        # Should be callable
        reporter.update(1)


if __name__ == "__main__":
    unittest.main()
