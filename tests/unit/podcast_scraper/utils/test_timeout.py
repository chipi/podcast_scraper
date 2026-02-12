"""Unit tests for podcast_scraper.utils.timeout module."""

from __future__ import annotations

import pytest

from podcast_scraper.utils.timeout import timeout_context, TimeoutError, with_timeout


@pytest.mark.unit
class TestTimeoutContext:
    """Tests for timeout_context."""

    def test_none_disables_timeout(self):
        """timeout_context(None) runs body without timeout."""
        ran = []

        with timeout_context(None, "test"):
            ran.append(1)

        assert ran == [1]

    def test_zero_disables_timeout(self):
        """timeout_context(0) runs body without timeout."""
        ran = []

        with timeout_context(0, "test"):
            ran.append(1)

        assert ran == [1]

    def test_positive_timeout_body_completes(self):
        """timeout_context(seconds) runs body when it completes within time."""
        ran = []

        with timeout_context(10, "test"):
            ran.append(1)

        assert ran == [1]

    def test_timeout_fires_raises_timeout_error(self):
        """When operation exceeds timeout, TimeoutError is raised."""
        import time

        with pytest.raises(TimeoutError, match="exceeded timeout"):
            with timeout_context(1, "slow_op"):
                time.sleep(2)  # longer than 1s timeout


@pytest.mark.unit
class TestWithTimeout:
    """Tests for with_timeout."""

    def test_none_calls_func_returns_result(self):
        """with_timeout(None, func, ...) calls func and returns result."""

        def add(a, b):
            return a + b

        result = with_timeout(add, None, "add", 2, 3)
        assert result == 5

    def test_zero_calls_func_returns_result(self):
        """with_timeout(0, func, ...) calls func and returns result."""
        result = with_timeout(lambda: 99, 0, "op")
        assert result == 99

    def test_positive_timeout_func_completes_returns_result(self):
        """With positive timeout, if func completes, returns result."""
        result = with_timeout(lambda: 42, 10, "op")
        assert result == 42

    def test_keyword_args_passed(self):
        """Keyword arguments are passed to function."""

        def f(a, b=0):
            return a + b

        result = with_timeout(f, None, "op", 1, b=2)
        assert result == 3
