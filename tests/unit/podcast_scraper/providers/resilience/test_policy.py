"""Unit tests for the ADR-122 reprocess-mode resilience policy (backoff -> trip -> hold).

Pure-logic coverage — no mock server needed. Exercises the backoff-schedule edge case (an empty
schedule) and the fuse-open Sentry alert's guarded try/except (the "sentry_sdk unavailable /
capture_message raised" branch, which must never break the resilience path itself).
"""

from __future__ import annotations

import types

import pytest

from podcast_scraper.providers.resilience import ResilienceFuseOpenError
from podcast_scraper.providers.resilience.breakers import CircuitBreaker
from podcast_scraper.providers.resilience.policy import (
    _emit_fuse_open_alert,
    FailureStrategy,
    ResiliencePolicy,
    resolve_failure_strategy,
)


def _noop_sleep(_sec: float) -> None:
    """No-op sleep so the backoff / pause-probe loops run instantly in tests."""


def _breaker() -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=1, window_sec=60.0, cooldown_sec=60.0, name="test")


def test_backoff_for_empty_schedule_returns_zero() -> None:
    """An empty backoff schedule -> no wait between attempts (defensive default, not a crash)."""
    policy = ResiliencePolicy(breaker=_breaker(), backoff_schedule_sec=())
    assert policy._backoff_for(0) == 0.0
    assert policy._backoff_for(5) == 0.0


def test_backoff_for_repeats_last_entry_once_exhausted() -> None:
    policy = ResiliencePolicy(breaker=_breaker(), backoff_schedule_sec=(30.0, 60.0, 120.0))
    assert policy._backoff_for(0) == 30.0
    assert policy._backoff_for(2) == 120.0
    assert policy._backoff_for(10) == 120.0  # capped at the last entry


def test_emit_fuse_open_alert_swallows_sentry_capture_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``sentry_sdk.capture_message`` that raises is swallowed — alerting must never break the
    resilience path (mirrors the "sentry unavailable" degrade)."""
    import sentry_sdk

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("sentry down")

    monkeypatch.setattr(sentry_sdk, "capture_message", _boom)
    _emit_fuse_open_alert("test-endpoint", 900.0)  # must NOT raise


def test_emit_fuse_open_alert_happy_path_does_not_raise() -> None:
    """The plain success path (sentry_sdk importable + capture_message succeeds) also must not
    raise; this is the same guarded call, exercised without a forced failure."""
    _emit_fuse_open_alert("test-endpoint", 900.0)


# --- ResiliencePolicy.run: backoff -> trip -> hold execution ------------------------------------


def test_run_first_attempt_succeeds() -> None:
    """Breaker closed + call succeeds on the first attempt -> return it, record success."""
    policy = ResiliencePolicy(
        breaker=_breaker(), retries_before_trip=2, backoff_schedule_sec=(0.0,)
    )
    assert policy.run(lambda: "ok", timeout_sec=1.0, sleep=_noop_sleep) == "ok"


def test_run_retries_then_succeeds() -> None:
    """A transient failure is retried with backoff (no cross-model fallover) and then succeeds."""
    calls = {"n": 0}

    def call() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return "ok"

    policy = ResiliencePolicy(
        breaker=_breaker(), retries_before_trip=3, backoff_schedule_sec=(0.0,)
    )
    assert policy.run(call, timeout_sec=1.0, sleep=_noop_sleep) == "ok"
    assert calls["n"] == 2  # failed once, succeeded on the retry


def test_run_exhausts_retries_then_holds_and_raises_fuse_open() -> None:
    """A persistently-down endpoint: retries exhaust + trip the breaker, then pause-and-probe holds
    until on_open_max_wait and raises ResilienceFuseOpenError (never falls over)."""

    def call() -> str:
        raise RuntimeError("always down")

    policy = ResiliencePolicy(
        breaker=CircuitBreaker(failure_threshold=1, window_sec=60.0, cooldown_sec=60.0, name="t"),
        retries_before_trip=2,
        backoff_schedule_sec=(0.0,),
        on_open_max_wait_sec=1.0,
        probe_interval_sec=0.5,
    )
    with pytest.raises(ResilienceFuseOpenError):
        policy.run(call, timeout_sec=1.0, sleep=_noop_sleep)


def test_run_breaker_open_on_entry_skips_straight_to_pause_and_probe() -> None:
    """If the breaker is already open on entry, run() skips the attempt loop and holds/probes."""
    breaker = CircuitBreaker(failure_threshold=1, window_sec=60.0, cooldown_sec=60.0, name="t")
    breaker.record_failure(hard=True)  # open it up front
    assert breaker.state != "closed"

    def call() -> str:
        raise RuntimeError("still down")

    policy = ResiliencePolicy(
        breaker=breaker,
        retries_before_trip=2,
        on_open_max_wait_sec=1.0,
        probe_interval_sec=0.5,
    )
    with pytest.raises(ResilienceFuseOpenError):
        policy.run(call, timeout_sec=1.0, sleep=_noop_sleep)


def test_run_probe_recovers_after_cooldown() -> None:
    """Retries trip the breaker; once the cooldown elapses a half-open probe succeeds and the
    result is returned (the hold ends on recovery, not on fallover). Uses a real short sleep so the
    breaker's min-1s cooldown actually elapses."""
    calls = {"n": 0}

    def call() -> str:
        calls["n"] += 1
        if calls["n"] <= 2:  # both initial attempts fail -> trip
            raise RuntimeError("down")
        return "recovered"  # the half-open probe succeeds

    policy = ResiliencePolicy(
        breaker=CircuitBreaker(failure_threshold=1, window_sec=60.0, cooldown_sec=1.0, name="t"),
        retries_before_trip=2,
        backoff_schedule_sec=(0.0,),
        on_open_max_wait_sec=5.0,
        probe_interval_sec=0.05,
    )
    assert policy.run(call, timeout_sec=1.0) == "recovered"  # default (real) sleep for cooldown
    assert calls["n"] >= 3


# --- resolve_failure_strategy -------------------------------------------------------------------


def test_resolve_failure_strategy_explicit_override_wins() -> None:
    """An explicitly-set resilience_failure_strategy (in model_fields_set) wins over run context."""
    cfg = types.SimpleNamespace(
        resilience_failure_strategy="hold",
        model_fields_set={"resilience_failure_strategy"},
        resilience_run_context="serve",  # would derive FAILOVER, but the explicit override wins
    )
    assert resolve_failure_strategy(cfg) is FailureStrategy.HOLD


def test_resolve_failure_strategy_derived_from_run_context() -> None:
    """With no explicit override, the strategy derives from run context: reprocess->HOLD,
    serve->FAILOVER (keeps a hand-built Config consistent without profile resolution)."""
    reprocess = types.SimpleNamespace(model_fields_set=set(), resilience_run_context="reprocess")
    serve = types.SimpleNamespace(model_fields_set=set(), resilience_run_context="serve")
    assert resolve_failure_strategy(reprocess) is FailureStrategy.HOLD
    assert resolve_failure_strategy(serve) is FailureStrategy.FAILOVER
