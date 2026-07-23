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
class TestTripAlert:
    """ADR-122: a trip fires a guarded Sentry alert so the operator sees the fuse blow."""

    def test_trip_emits_sentry_alert_once(self) -> None:
        cfg = LLMCircuitBreakerConfig(enabled=True, failure_threshold=2, window_seconds=60)
        with patch(
            "podcast_scraper.utils.llm_circuit_breaker._emit_llm_breaker_trip_alert"
        ) as alert:
            record_failure("gemini", cfg, error_status=503)
            alert.assert_not_called()  # below threshold -> no trip, no alert
            record_failure("gemini", cfg, error_status=503)
        alert.assert_called_once_with("gemini", cfg.cooldown_seconds)

    def test_alert_helper_is_guarded(self) -> None:
        """A broken/unconfigured sentry_sdk must not propagate out of the alert helper."""
        from podcast_scraper.utils import llm_circuit_breaker as mod

        with patch("sentry_sdk.capture_message", side_effect=RuntimeError("boom")):
            mod._emit_llm_breaker_trip_alert("gemini", 60.0)  # must not raise


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
class TestHoldStrategyAbort:
    """ADR-122 hold strategy: a sustained outage aborts the batch (ResilienceFuseOpenError) rather
    than waiting forever; the default failover strategy keeps #697's wait-and-resume (never raises).
    """

    _MONO = "podcast_scraper.utils.llm_circuit_breaker.time.monotonic"

    def test_hold_raises_once_outage_exceeds_max_wait(self) -> None:
        from podcast_scraper.providers.resilience.policy import ResilienceFuseOpenError

        cfg = LLMCircuitBreakerConfig(
            enabled=True,
            failure_threshold=1,
            window_seconds=60,
            cooldown_seconds=1000,  # > max_wait so a single trip exercises the abort path
            failure_strategy="hold",
            on_open_max_wait_sec=100,
        )
        with patch(self._MONO, return_value=0.0):
            record_failure("gemini", cfg, error_status=503)  # trip at t=0, hold_started_at=0
        # 200 s later: still inside cooldown, but sustained-down past the 100 s hold budget.
        with patch(self._MONO, return_value=200.0):
            with pytest.raises(ResilienceFuseOpenError):
                wait_if_overloaded("gemini", cfg)

    def test_hold_waits_but_does_not_raise_within_budget(self) -> None:
        cfg = LLMCircuitBreakerConfig(
            enabled=True,
            failure_threshold=1,
            window_seconds=60,
            cooldown_seconds=1000,
            failure_strategy="hold",
            on_open_max_wait_sec=100,
        )
        with patch(self._MONO, return_value=0.0):
            record_failure("gemini", cfg, error_status=503)
        # 50 s in (< 100 s budget): hold mode still just waits, no abort.
        with patch(self._MONO, return_value=50.0):
            with patch("podcast_scraper.utils.llm_circuit_breaker.time.sleep") as mock_sleep:
                wait_if_overloaded("gemini", cfg)  # must NOT raise
        mock_sleep.assert_called_once()

    def test_failover_never_raises_only_waits(self) -> None:
        cfg = LLMCircuitBreakerConfig(
            enabled=True,
            failure_threshold=1,
            window_seconds=60,
            cooldown_seconds=1000,
            failure_strategy="failover",  # default: wait-and-resume even past max_wait
            on_open_max_wait_sec=100,
        )
        with patch(self._MONO, return_value=0.0):
            record_failure("gemini", cfg, error_status=503)
        with patch(self._MONO, return_value=200.0):  # past budget, but failover ignores it
            with patch("podcast_scraper.utils.llm_circuit_breaker.time.sleep") as mock_sleep:
                wait_if_overloaded("gemini", cfg)  # must NOT raise
        mock_sleep.assert_called_once()

    def test_success_resets_hold_window(self) -> None:
        """A recovery mid-outage resets the hold clock so the next outage starts fresh."""
        cfg = LLMCircuitBreakerConfig(
            enabled=True,
            failure_threshold=1,
            window_seconds=60,
            cooldown_seconds=1000,
            failure_strategy="hold",
            on_open_max_wait_sec=100,
        )
        with patch(self._MONO, return_value=0.0):
            record_failure("gemini", cfg, error_status=503)
            record_success("gemini", cfg)  # recovered -> hold_started_at cleared
        assert stats("gemini")["in_cooldown"] is False


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
