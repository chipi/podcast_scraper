"""Unit tests for podcast_scraper.utils.retry module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from podcast_scraper.utils.retry import retry_with_exponential_backoff


@pytest.mark.unit
class TestRetryWithExponentialBackoff:
    """Tests for retry_with_exponential_backoff."""

    def test_success_on_first_call(self):
        """Function succeeds on first call returns result."""

        def func():
            return 42

        result = retry_with_exponential_backoff(func, max_retries=2)
        assert result == 42

    def test_success_on_second_call_after_retryable_exception(self):
        """Success on second call after one retryable exception."""
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("transient")
            return "ok"

        with patch("podcast_scraper.utils.retry.time.sleep"):
            result = retry_with_exponential_backoff(
                flaky, max_retries=3, retryable_exceptions=(ValueError,)
            )
        assert result == "ok"
        assert len(calls) == 2

    def test_all_retries_exhausted_raises(self):
        """All retries exhausted raises last exception."""

        def always_fail():
            raise ValueError("nope")

        with patch("podcast_scraper.utils.retry.time.sleep"):
            with pytest.raises(ValueError, match="nope"):
                retry_with_exponential_backoff(
                    always_fail, max_retries=2, retryable_exceptions=(ValueError,)
                )

    def test_non_retryable_exception_propagates_immediately(self):
        """Non-retryable exception propagates without retry."""
        calls = []

        def fail_type_error():
            calls.append(1)
            raise TypeError("not retryable")

        with patch("podcast_scraper.utils.retry.time.sleep"):
            with pytest.raises(TypeError, match="not retryable"):
                retry_with_exponential_backoff(
                    fail_type_error, max_retries=3, retryable_exceptions=(ValueError,)
                )
        assert len(calls) == 1

    def test_delay_caps_at_max_delay(self):
        """Delay is capped at max_delay."""
        delays = []

        def record_sleep(secs):
            delays.append(secs)

        def fail_twice():
            if len(delays) < 2:
                raise ValueError("retry")
            return "ok"

        with patch("podcast_scraper.utils.retry.time.sleep", side_effect=record_sleep):
            retry_with_exponential_backoff(
                fail_twice,
                max_retries=3,
                initial_delay=1.0,
                max_delay=2.0,
                retryable_exceptions=(ValueError,),
            )
        assert len(delays) == 2
        assert delays[0] == 1.0
        assert delays[1] == 2.0
