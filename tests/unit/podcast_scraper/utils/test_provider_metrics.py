#!/usr/bin/env python3
"""Tests for provider metrics and retry utilities.

These tests verify retry behavior, jitter, and metrics tracking.
"""

import unittest
from unittest.mock import Mock, patch

from podcast_scraper.utils.provider_metrics import (
    ProviderCallMetrics,
    retry_with_metrics,
)


class TestProviderCallMetrics(unittest.TestCase):
    """Test ProviderCallMetrics dataclass."""

    def test_initial_state(self):
        """Test that metrics start with correct initial values."""
        metrics = ProviderCallMetrics()
        self.assertEqual(metrics.retries, 0)
        self.assertEqual(metrics.rate_limit_sleep_sec, 0.0)
        self.assertIsNone(metrics.prompt_tokens)
        self.assertIsNone(metrics.completion_tokens)
        self.assertIsNone(metrics.estimated_cost)

    def test_record_retry(self):
        """Test recording retry attempts."""
        metrics = ProviderCallMetrics()
        metrics.set_provider_name("test_provider")
        metrics.record_retry(sleep_seconds=2.5, reason="429")
        metrics.finalize()
        self.assertEqual(metrics.retries, 1)
        self.assertEqual(metrics.rate_limit_sleep_sec, 2.5)

    def test_record_multiple_retries(self):
        """Test recording multiple retry attempts."""
        metrics = ProviderCallMetrics()
        metrics.record_retry(sleep_seconds=1.0, reason="500")
        metrics.record_retry(sleep_seconds=2.0, reason="429")
        metrics.record_retry(sleep_seconds=4.0, reason="429")
        metrics.finalize()
        self.assertEqual(metrics.retries, 3)
        self.assertEqual(metrics.rate_limit_sleep_sec, 7.0)

    def test_set_tokens(self):
        """Test setting token counts."""
        metrics = ProviderCallMetrics()
        metrics.set_tokens(prompt_tokens=100, completion_tokens=50)
        self.assertEqual(metrics.prompt_tokens, 100)
        self.assertEqual(metrics.completion_tokens, 50)

    def test_set_cost(self):
        """Test setting estimated cost."""
        metrics = ProviderCallMetrics()
        metrics.set_cost(0.05)
        self.assertEqual(metrics.estimated_cost, 0.05)


class TestRetryWithMetrics(unittest.TestCase):
    """Test retry_with_metrics function."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't retry."""
        func = Mock(return_value="success")
        result = retry_with_metrics(func, max_retries=3)
        self.assertEqual(result, "success")
        self.assertEqual(func.call_count, 1)

    def test_retry_on_exception(self):
        """Test that retries occur on retryable exceptions."""
        func = Mock(side_effect=[Exception("error"), "success"])
        with patch("time.sleep"):
            result = retry_with_metrics(func, max_retries=3, initial_delay=0.1)
        self.assertEqual(result, "success")
        self.assertEqual(func.call_count, 2)

    def test_max_retries_exhausted(self):
        """Test that exception is raised when max retries exhausted."""
        func = Mock(side_effect=Exception("error"))
        with patch("time.sleep"):
            with self.assertRaises(Exception) as context:
                retry_with_metrics(func, max_retries=2, initial_delay=0.1)
        self.assertEqual(str(context.exception), "error")
        self.assertEqual(func.call_count, 3)  # Initial + 2 retries

    def test_jitter_adds_variation(self):
        """Test that jitter adds random variation to delays."""
        func = Mock(side_effect=[Exception("error"), "success"])
        metrics = ProviderCallMetrics()
        with patch("time.sleep") as mock_sleep:
            with patch("random.uniform", return_value=1.05):  # 5% increase
                retry_with_metrics(
                    func, max_retries=3, initial_delay=1.0, jitter=True, metrics=metrics
                )
        # Verify sleep was called with jittered delay
        mock_sleep.assert_called_once()
        # With jitter factor 1.05, delay should be 1.0 * 1.05 = 1.05
        self.assertAlmostEqual(mock_sleep.call_args[0][0], 1.05, places=2)

    def test_jitter_disabled(self):
        """Test that jitter can be disabled."""
        func = Mock(side_effect=[Exception("error"), "success"])
        with patch("time.sleep") as mock_sleep:
            retry_with_metrics(func, max_retries=3, initial_delay=1.0, jitter=False)
        # Verify sleep was called with exact delay (no jitter)
        mock_sleep.assert_called_once()
        self.assertEqual(mock_sleep.call_args[0][0], 1.0)

    def test_jitter_respects_max_delay(self):
        """Test that jitter doesn't exceed max_delay."""
        func = Mock(side_effect=[Exception("error"), "success"])
        with patch("time.sleep") as mock_sleep:
            with patch("random.uniform", return_value=2.0):  # Would exceed max
                retry_with_metrics(
                    func,
                    max_retries=3,
                    initial_delay=20.0,
                    max_delay=30.0,
                    jitter=True,
                )
        # Verify sleep was capped at max_delay
        mock_sleep.assert_called_once()
        self.assertLessEqual(mock_sleep.call_args[0][0], 30.0)

    def test_metrics_tracking(self):
        """Test that metrics are tracked correctly."""
        func = Mock(side_effect=[Exception("error"), "success"])
        metrics = ProviderCallMetrics()
        metrics.set_provider_name("test_provider")
        with patch("time.sleep"):
            retry_with_metrics(func, max_retries=3, initial_delay=0.1, metrics=metrics)
        metrics.finalize()
        self.assertEqual(metrics.retries, 1)
        self.assertGreater(metrics.rate_limit_sleep_sec, 0)

    def test_rate_limit_detection(self):
        """Test that rate limit errors are detected correctly."""
        func = Mock(side_effect=[Exception("429 Rate limit exceeded"), "success"])
        metrics = ProviderCallMetrics()
        metrics.set_provider_name("test_provider")
        with patch("time.sleep"):
            with patch("podcast_scraper.utils.provider_metrics.logger") as mock_logger:
                retry_with_metrics(func, max_retries=3, initial_delay=0.1, metrics=metrics)
        # Verify rate limit was logged
        mock_logger.info.assert_called()
        log_call = str(mock_logger.info.call_args)
        self.assertIn("429", log_call)

    def test_exponential_backoff(self):
        """Test that delays increase exponentially."""
        func = Mock(side_effect=[Exception("error"), Exception("error"), "success"])
        delays = []

        def capture_sleep(delay):
            delays.append(delay)

        with patch("time.sleep", side_effect=capture_sleep):
            with patch("random.uniform", return_value=1.0):  # No jitter for test
                retry_with_metrics(func, max_retries=3, initial_delay=1.0, jitter=False)
        # First retry: 1.0, second retry: 2.0 (doubled)
        self.assertEqual(len(delays), 2)
        self.assertAlmostEqual(delays[0], 1.0, places=1)
        self.assertAlmostEqual(delays[1], 2.0, places=1)

    def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        func = Mock(side_effect=[Exception("error"), Exception("error"), "success"])
        delays = []

        def capture_sleep(delay):
            delays.append(delay)

        with patch("time.sleep", side_effect=capture_sleep):
            with patch("random.uniform", return_value=1.0):  # No jitter for test
                retry_with_metrics(
                    func,
                    max_retries=3,
                    initial_delay=20.0,
                    max_delay=30.0,
                    jitter=False,
                )
        # First retry: 20.0, second retry: 30.0 (capped, not 40.0)
        self.assertEqual(len(delays), 2)
        self.assertAlmostEqual(delays[0], 20.0, places=1)
        self.assertAlmostEqual(delays[1], 30.0, places=1)


if __name__ == "__main__":
    unittest.main()
