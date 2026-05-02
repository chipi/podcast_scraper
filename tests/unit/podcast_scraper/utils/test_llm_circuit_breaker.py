"""Tests for the per-provider LLM circuit breaker (#697)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from podcast_scraper.utils.llm_circuit_breaker import (
    LLMCircuitBreakerConfig,
    record_failure,
    record_success,
    reset_for_test,
    stats,
    wait_if_overloaded,
)


@pytest.fixture(autouse=True)
def _reset_breaker_state() -> None:
    """Each test starts with fresh per-provider state."""
    reset_for_test()


@pytest.mark.unit
class TestLLMCircuitBreakerDisabled:
    """When ``enabled=False`` the breaker is a complete no-op."""

    def test_wait_is_noop_when_disabled(self) -> None:
        cfg = LLMCircuitBreakerConfig(enabled=False)
        # No state has been created; should not raise or sleep.
        wait_if_overloaded("gemini", cfg)
        assert stats("gemini")["trips_total"] == 0

    def test_record_failure_is_noop_when_disabled(self) -> None:
        cfg = LLMCircuitBreakerConfig(enabled=False, failure_threshold=1)
        for _ in range(10):
            record_failure("gemini", cfg, error_status=503)
        assert stats("gemini")["trips_total"] == 0


@pytest.mark.unit
class TestRecordFailureClassification:
    """Only 5xx + 429 trip the breaker; non-overload errors are ignored."""

    def test_4xx_other_than_429_does_not_trip(self) -> None:
        cfg = LLMCircuitBreakerConfig(enabled=True, failure_threshold=2)
        for status in (400, 401, 403, 404, 422):
            for _ in range(5):
                record_failure("gemini", cfg, error_status=status)
        assert stats("gemini")["trips_total"] == 0

    def test_429_counts_as_overload(self) -> None:
        cfg = LLMCircuitBreakerConfig(enabled=True, failure_threshold=2, window_seconds=60)
        record_failure("gemini", cfg, error_status=429)
        record_failure("gemini", cfg, error_status=429)
        assert stats("gemini")["trips_total"] == 1

    def test_503_storm_trips(self) -> None:
        cfg = LLMCircuitBreakerConfig(enabled=True, failure_threshold=3, window_seconds=60)
        record_failure("gemini", cfg, error_status=503)
        record_failure("gemini", cfg, error_status=503)
        assert stats("gemini")["trips_total"] == 0
        record_failure("gemini", cfg, error_status=503)
        assert stats("gemini")["trips_total"] == 1


@pytest.mark.unit
class TestWaitBehaviour:
    """``wait_if_overloaded`` sleeps when the breaker is open and returns immediately otherwise."""

    def test_no_wait_when_breaker_closed(self) -> None:
        cfg = LLMCircuitBreakerConfig(enabled=True)
        with patch("podcast_scraper.utils.llm_circuit_breaker.time.sleep") as mock_sleep:
            wait_if_overloaded("gemini", cfg)
        mock_sleep.assert_not_called()

    def test_waits_when_open(self) -> None:
        cfg = LLMCircuitBreakerConfig(
            enabled=True, failure_threshold=2, window_seconds=60, cooldown_seconds=10
        )
        record_failure("gemini", cfg, error_status=503)
        record_failure("gemini", cfg, error_status=503)
        # Now in cooldown — wait should sleep approximately 10 s.
        with patch("podcast_scraper.utils.llm_circuit_breaker.time.sleep") as mock_sleep:
            wait_if_overloaded("gemini", cfg)
        mock_sleep.assert_called_once()
        slept_seconds = mock_sleep.call_args.args[0]
        assert 9.0 <= slept_seconds <= 10.0


@pytest.mark.unit
class TestRecordSuccessClearsState:
    def test_success_clears_failure_history(self) -> None:
        cfg = LLMCircuitBreakerConfig(enabled=True, failure_threshold=3)
        record_failure("gemini", cfg, error_status=503)
        record_failure("gemini", cfg, error_status=503)
        assert stats("gemini")["recent_failures_in_window"] == 2
        record_success("gemini", cfg)
        assert stats("gemini")["recent_failures_in_window"] == 0

    def test_success_clears_active_cooldown(self) -> None:
        cfg = LLMCircuitBreakerConfig(enabled=True, failure_threshold=2, cooldown_seconds=60)
        record_failure("gemini", cfg, error_status=503)
        record_failure("gemini", cfg, error_status=503)
        assert stats("gemini")["in_cooldown"] is True
        # If a probe call somehow succeeded mid-cooldown (e.g. during a half-open
        # window), success should clear the cooldown so subsequent calls don't wait.
        record_success("gemini", cfg)
        assert stats("gemini")["in_cooldown"] is False


@pytest.mark.unit
class TestPerProviderIsolation:
    """Each provider has its own breaker — gemini outage doesn't gate openai."""

    def test_provider_state_is_isolated(self) -> None:
        cfg = LLMCircuitBreakerConfig(enabled=True, failure_threshold=2)
        record_failure("gemini", cfg, error_status=503)
        record_failure("gemini", cfg, error_status=503)
        assert stats("gemini")["trips_total"] == 1
        assert stats("openai")["trips_total"] == 0
        # openai breaker is still closed.
        with patch("podcast_scraper.utils.llm_circuit_breaker.time.sleep") as mock_sleep:
            wait_if_overloaded("openai", cfg)
        mock_sleep.assert_not_called()


@pytest.mark.unit
class TestWindowPruning:
    """Failures outside the rolling window don't count toward the trip threshold."""

    def test_old_failures_pruned(self) -> None:
        cfg = LLMCircuitBreakerConfig(enabled=True, failure_threshold=3, window_seconds=10)
        # Two failures, then advance time past the window.
        record_failure("gemini", cfg, error_status=503)
        record_failure("gemini", cfg, error_status=503)
        # Mock monotonic to return a time well past the 10 s window.
        with patch(
            "podcast_scraper.utils.llm_circuit_breaker.time.monotonic",
            return_value=time.monotonic() + 30.0,
        ):
            record_failure("gemini", cfg, error_status=503)
        # Only the latest failure is in the window — under threshold,
        # so the breaker should not have tripped.
        assert stats("gemini")["trips_total"] == 0
