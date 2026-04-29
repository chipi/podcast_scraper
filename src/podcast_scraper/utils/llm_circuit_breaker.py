"""Per-provider LLM circuit breaker for cloud-API 503 storms.

Sister to ``rss.http_policy.CircuitBreaker`` (which is fail-fast for RSS feed
downloads). LLM providers like Gemini occasionally return bursts of 503
``UNAVAILABLE`` when their backend is overloaded. The per-call retry ladder
(``retry_with_metrics``) handles individual failures with exponential backoff,
but multiple successive 503s can stack into many minutes of summed retry
sleeps and tip episodes over the summarization timeout (#697).

Design difference vs. ``rss.http_policy.CircuitBreaker``:

* RSS breaker: when "open", **raise** ``CircuitOpenError`` so the caller
  short-circuits the request. Appropriate for fetch-or-fail RSS reads.
* LLM breaker: when "open", **wait** the cooldown and then resume. The
  caller still wants the result; we're trading latency for survival
  during a burst. Appropriate for must-finish summarization / GIL stages.

Wiring: ``retry_with_metrics`` calls ``wait_if_overloaded`` before each
attempt and ``record_failure`` / ``record_success`` after. Single global
breaker keyed by provider name (``gemini`` / ``openai`` / ``anthropic`` /
``mistral`` / ``deepseek`` / ``grok``) — provider outages are typically
provider-wide, not per-host or per-route.

Defaults (3 failures in 30 s → 60 s cooldown) sized for the WSJ Journal
incident on 2026-04-28: ~5 503s in <30 s would have been smoothed by a
single 60 s cooldown that's still well inside the 1200 s summarization
timeout. Tunable via ``Config`` for operators who want tighter or looser
protection.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level singleton state. Per-provider entries; thread-safe.
_provider_state: Dict[str, "_BreakerState"] = {}
_state_lock = threading.Lock()


@dataclass
class _BreakerState:
    """Per-provider rolling state."""

    failure_times: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    cooldown_until: float = 0.0
    trips_total: int = 0
    cooldown_seconds_total: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _get_state(provider_name: str) -> _BreakerState:
    with _state_lock:
        state = _provider_state.get(provider_name)
        if state is None:
            state = _BreakerState()
            _provider_state[provider_name] = state
        return state


def reset_for_test() -> None:
    """Clear all per-provider state. Test-only."""
    with _state_lock:
        _provider_state.clear()


@dataclass
class LLMCircuitBreakerConfig:
    """Tunable thresholds for the per-provider breaker.

    Defaults sized for the Gemini 503 storm pattern observed on real
    cloud_thin runs: short bursts of 3-5 503s in <30 s, often resolving
    within ~30-60 s. A 60 s cooldown after 3 failures within 30 s buys
    enough time for the upstream to recover before the next call,
    without blowing the 1200 s summarization timeout budget.
    """

    enabled: bool = False
    failure_threshold: int = 3
    window_seconds: float = 30.0
    cooldown_seconds: float = 60.0


def _is_overload_status(error_status: int) -> bool:
    """Treat 5xx + 429 as breaker-eligible (transient upstream overload)."""
    return error_status == 429 or 500 <= error_status < 600


def wait_if_overloaded(
    provider_name: str,
    config: LLMCircuitBreakerConfig,
    metrics: Optional[Any] = None,
) -> None:
    """Sleep until the breaker's cooldown elapses; no-op if breaker is closed.

    Called BEFORE each provider call. Doesn't raise — just delays. The
    caller still gets to make the request once the cooldown ends.
    """
    if not config.enabled:
        return
    state = _get_state(provider_name)
    now = time.monotonic()
    with state.lock:
        if now >= state.cooldown_until:
            return
        wait_seconds = state.cooldown_until - now
    logger.warning(
        "llm_circuit_breaker open for provider=%s — waiting %.1fs before next call",
        provider_name,
        wait_seconds,
    )
    if metrics is not None and hasattr(metrics, "record_llm_circuit_breaker_wait"):
        metrics.record_llm_circuit_breaker_wait(provider_name, wait_seconds)
    time.sleep(wait_seconds)


def record_failure(
    provider_name: str,
    config: LLMCircuitBreakerConfig,
    error_status: int,
    metrics: Optional[Any] = None,
) -> None:
    """Record a failed call. Trip the breaker if the rolling window threshold is met."""
    if not config.enabled:
        return
    if not _is_overload_status(error_status):
        return
    state = _get_state(provider_name)
    now = time.monotonic()
    with state.lock:
        state.failure_times.append(now)
        # Prune entries outside the window.
        cutoff = now - config.window_seconds
        while state.failure_times and state.failure_times[0] < cutoff:
            state.failure_times.popleft()
        if len(state.failure_times) >= config.failure_threshold:
            # Trip — set cooldown and clear the window so we don't immediately
            # re-trip on the same burst.
            state.cooldown_until = now + config.cooldown_seconds
            state.failure_times.clear()
            state.trips_total += 1
            state.cooldown_seconds_total += config.cooldown_seconds
            logger.warning(
                "llm_circuit_breaker tripped for provider=%s "
                "(%d failures in <%.0fs window); cooldown=%.0fs",
                provider_name,
                config.failure_threshold,
                config.window_seconds,
                config.cooldown_seconds,
            )
            if metrics is not None and hasattr(metrics, "record_llm_circuit_breaker_trip"):
                metrics.record_llm_circuit_breaker_trip(provider_name, config.cooldown_seconds)


def record_success(provider_name: str, config: LLMCircuitBreakerConfig) -> None:
    """Record a successful call. Clear the breaker's state for this provider."""
    if not config.enabled:
        return
    state = _get_state(provider_name)
    with state.lock:
        state.failure_times.clear()
        state.cooldown_until = 0.0


def stats(provider_name: str) -> Dict[str, Any]:
    """Return per-provider breaker stats. For metrics + tests."""
    state = _get_state(provider_name)
    with state.lock:
        now = time.monotonic()
        return {
            "provider": provider_name,
            "trips_total": state.trips_total,
            "cooldown_seconds_total": state.cooldown_seconds_total,
            "in_cooldown": now < state.cooldown_until,
            "cooldown_remaining_seconds": max(0.0, state.cooldown_until - now),
            "recent_failures_in_window": len(state.failure_times),
        }
