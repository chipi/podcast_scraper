"""Unit tests for timeout configuration utilities."""

from __future__ import annotations

import unittest

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

from podcast_scraper import config
from podcast_scraper.utils.timeout_config import get_http_timeout


class TestTimeoutConfig(unittest.TestCase):
    """Test timeout configuration utilities."""

    def test_get_http_timeout_default(self):
        """Test default timeout configuration."""
        cfg = config.Config(rss_url="https://example.com")
        timeout = get_http_timeout(cfg)
        self.assertIsNotNone(timeout)
        if httpx is not None:
            # Check if httpx is actually a module (not a mock)
            # When httpx is mocked by other tests, httpx.Timeout might be a MagicMock
            try:
                is_timeout = isinstance(timeout, httpx.Timeout)
            except TypeError:
                # httpx.Timeout is a mock, check by attributes instead
                is_timeout = hasattr(timeout, "connect") and hasattr(timeout, "read")

            if is_timeout:
                # Check timeout values
                self.assertGreater(timeout.connect, 0)
                self.assertGreater(timeout.read, 0)
                self.assertLess(timeout.connect, timeout.read)  # Connect should be shorter
            else:
                # Fallback: should return float
                self.assertIsInstance(timeout, (int, float))
        else:
            # Fallback: should return float
            self.assertIsInstance(timeout, (int, float))

    def test_get_http_timeout_custom(self):
        """Test custom timeout configuration."""
        cfg = config.Config(
            rss_url="https://example.com",
            timeout=30.0,  # Overall timeout
        )
        timeout = get_http_timeout(cfg)
        if httpx is not None:
            # Check if httpx is actually a module (not a mock)
            try:
                is_timeout = isinstance(timeout, httpx.Timeout)
            except TypeError:
                # httpx.Timeout is a mock, check by attributes instead
                is_timeout = hasattr(timeout, "connect") and hasattr(timeout, "read")

            if is_timeout:
                # Connect timeout should be shorter than read timeout
                self.assertLess(timeout.connect, timeout.read)
                # Read timeout should be close to overall timeout
                self.assertLessEqual(timeout.read, 30.0)
            else:
                # Fallback: should return float
                self.assertIsInstance(timeout, (int, float))
        else:
            # Fallback: should return float
            self.assertIsInstance(timeout, (int, float))

    def test_get_http_timeout_with_stage_timeouts(self):
        """Test timeout configuration with stage-specific timeouts."""
        cfg = config.Config(
            rss_url="https://example.com",
            timeout=30.0,  # Use a timeout larger than default connect (10.0)
            transcription_timeout=60.0,  # Longer for transcription
        )
        # Default HTTP timeout (not transcription-specific)
        timeout = get_http_timeout(cfg)
        if httpx is not None:
            # Check if httpx is actually a module (not a mock)
            try:
                is_timeout = isinstance(timeout, httpx.Timeout)
            except TypeError:
                # httpx.Timeout is a mock, check by attributes instead
                is_timeout = hasattr(timeout, "connect") and hasattr(timeout, "read")

            if is_timeout:
                # Connect should be shorter than read
                self.assertLess(timeout.connect, timeout.read)
                # Read timeout should respect overall timeout, not transcription timeout
                self.assertLessEqual(timeout.read, 30.0)
                self.assertEqual(timeout.read, 30.0)  # Should use cfg.timeout
            else:
                # Fallback: should return float
                self.assertIsInstance(timeout, (int, float))
        else:
            # Fallback: should return float
            self.assertIsInstance(timeout, (int, float))

    def test_get_http_timeout_connect_shorter_than_read(self):
        """Test that connect timeout is always shorter than read timeout."""
        # Test with various timeouts, including edge cases
        for overall_timeout in [1.0, 5.0, 10.0, 30.0, 60.0]:
            cfg = config.Config(
                rss_url="https://example.com",
                timeout=overall_timeout,
            )
            timeout = get_http_timeout(cfg)
            if httpx is not None:
                # Check if httpx is actually a module (not a mock)
                try:
                    is_timeout = isinstance(timeout, httpx.Timeout)
                except TypeError:
                    # httpx.Timeout is a mock, check by attributes instead
                    is_timeout = hasattr(timeout, "connect") and hasattr(timeout, "read")

                if is_timeout:
                    self.assertLess(
                        timeout.connect,
                        timeout.read,
                        f"Connect timeout ({timeout.connect}) should be shorter than "
                        f"read timeout ({timeout.read}) for overall timeout {overall_timeout}",
                    )
                else:
                    # Fallback: should return float
                    self.assertIsInstance(timeout, (int, float))
            else:
                # Fallback: should return float
                self.assertIsInstance(timeout, (int, float))


if __name__ == "__main__":
    unittest.main()
