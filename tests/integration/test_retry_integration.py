"""Integration tests for retry logic with real provider scenarios."""

from __future__ import annotations

import time
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper.utils.provider_metrics import ProviderCallMetrics, retry_with_metrics


@pytest.mark.integration
class TestRetryIntegration(unittest.TestCase):
    """Integration tests for retry logic in provider scenarios."""

    def test_retry_with_jitter_prevents_thundering_herd(self):
        """Test that jitter prevents synchronized retries."""
        call_times = []
        mock_func = Mock()

        def record_call_time():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Transient error")
            return "success"

        mock_func.side_effect = record_call_time
        metrics = ProviderCallMetrics()
        metrics.set_provider_name("test")

        result = retry_with_metrics(
            mock_func,
            max_retries=2,
            initial_delay=0.1,
            metrics=metrics,
            jitter=True,  # Enable jitter
        )

        self.assertEqual(result, "success")
        self.assertEqual(len(call_times), 3)  # Initial + 2 retries

        # Check that delays have variation (jitter applied)
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            # With jitter, delay should be around 0.1 but not exactly
            # (jitter adds ±10% variation, plus system timing variance)
            self.assertGreater(delay1, 0.09)  # At least 90% of base delay
            self.assertLess(delay1, 0.13)  # At most 130% (allows for jitter + timing variance)

    def test_retry_without_jitter_has_exact_delays(self):
        """Test that retry without jitter has exact delays."""
        call_times = []
        mock_func = Mock()

        def record_call_time():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Transient error")
            return "success"

        mock_func.side_effect = record_call_time
        metrics = ProviderCallMetrics()
        metrics.set_provider_name("test")

        result = retry_with_metrics(
            mock_func,
            max_retries=2,
            initial_delay=0.1,
            metrics=metrics,
            jitter=False,  # Disable jitter
        )

        self.assertEqual(result, "success")
        # With jitter disabled, delay should be closer to exact value
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            # Should be very close to 0.1 (allowing small timing variance)
            self.assertAlmostEqual(delay1, 0.1, delta=0.02)

    def test_retry_exponential_backoff_with_jitter(self):
        """Test exponential backoff with jitter using mocked time.sleep for deterministic testing."""
        mock_func = Mock(
            side_effect=[ValueError("error"), ValueError("error"), ValueError("error"), "success"]
        )
        metrics = ProviderCallMetrics()
        metrics.set_provider_name("test")

        # Mock time.sleep to make test deterministic and avoid timing flakiness
        # Track sleep calls to verify exponential backoff and jitter
        sleep_calls = []

        def record_sleep(duration):
            sleep_calls.append(duration)

        with patch("time.sleep", side_effect=record_sleep):
            # Use deterministic jitter factors to make test predictable
            # First retry: 0.1 * 1.0 = 0.1, second: 0.2 * 0.95 = 0.19, third: 0.4 * 1.05 = 0.42
            with patch("random.uniform", side_effect=[1.0, 0.95, 1.05]):
                result = retry_with_metrics(
                    mock_func,
                    max_retries=3,
                    initial_delay=0.1,
                    metrics=metrics,
                    jitter=True,
                )

        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 4)  # Initial + 3 retries
        self.assertEqual(len(sleep_calls), 3)  # 3 sleep calls for 3 retries

        # Verify exponential backoff pattern with jitter
        # delay1 = 0.1 * 1.0 = 0.1
        # delay2 = 0.2 * 0.95 = 0.19 (exponential backoff: 0.1 * 2 = 0.2, then jitter)
        # delay3 = 0.4 * 1.05 = 0.42 (exponential backoff: 0.2 * 2 = 0.4, then jitter)
        self.assertAlmostEqual(sleep_calls[0], 0.1, places=2)  # First retry: ~0.1
        self.assertAlmostEqual(sleep_calls[1], 0.19, places=2)  # Second retry: ~0.2 * 0.95
        self.assertAlmostEqual(sleep_calls[2], 0.42, places=2)  # Third retry: ~0.4 * 1.05

        # Verify exponential backoff: each delay should be approximately double the previous
        # (accounting for jitter which can vary by ±10%)
        self.assertGreater(sleep_calls[1], sleep_calls[0] * 1.5)  # At least 1.5x
        self.assertLess(sleep_calls[1], sleep_calls[0] * 2.5)  # At most 2.5x
        self.assertGreater(sleep_calls[2], sleep_calls[1] * 1.5)  # At least 1.5x
        self.assertLess(sleep_calls[2], sleep_calls[1] * 2.5)  # At most 2.5x

    def test_retry_metrics_tracking(self):
        """Test that retry metrics are properly tracked."""
        mock_func = Mock(side_effect=[ValueError("Error 1"), ValueError("Error 2"), "success"])
        metrics = ProviderCallMetrics()
        metrics.set_provider_name("test")

        result = retry_with_metrics(
            mock_func,
            max_retries=2,
            initial_delay=0.01,
            metrics=metrics,
            jitter=False,
        )

        self.assertEqual(result, "success")
        metrics.finalize()
        self.assertEqual(metrics.retries, 2)
        self.assertGreater(metrics.rate_limit_sleep_sec, 0)


if __name__ == "__main__":
    unittest.main()
