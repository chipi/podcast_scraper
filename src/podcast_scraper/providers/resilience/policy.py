"""Self-hosted-model resilience policy (ADR-119): backoff -> trip -> hold, by run context.

Two run contexts share the same building blocks (:class:`~.breakers.CircuitBreaker`,
:func:`~.deadlines.run_with_watchdog`, the caller's duration-scaled timeout) but resolve a
blown fuse differently:

- ``RunContext.SERVE`` -> optimise availability (today's behaviour, unchanged): fail fast,
  trip the breaker on the first hard timeout, raise so the wrapping ``FallbackChain``
  (RFC-106 / #1198) can advance to the next model. Providers keep their own inline
  fail-fast logic for this context; :class:`ResiliencePolicy` is not involved (see
  ``whisper_provider._transcribe_via_dgx``'s serve branch).
- ``RunContext.REPROCESS`` -> optimise consistency: :meth:`ResiliencePolicy.run` retries the
  SAME callable (never a different model) with exponential backoff, trips the fuse only
  after ``retries_before_trip`` failures-despite-backoff, and on a blown fuse pauses and
  probes the endpoint (half-open) rather than raising for a fallover, up to
  ``on_open_max_wait_sec`` before surfacing :class:`ResilienceFuseOpenError` (the operator
  alert) to the caller.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence, TypeVar

from .breakers import CircuitBreaker
from .deadlines import run_with_watchdog, WATCHDOG_GRACE_SEC

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ADR-119 proposed reprocess-mode defaults (config-tunable; see Config.resilience_*).
DEFAULT_RETRIES_BEFORE_TRIP = 3
DEFAULT_BACKOFF_SCHEDULE_SEC: tuple[float, ...] = (30.0, 60.0, 120.0)
DEFAULT_ON_OPEN_MAX_WAIT_SEC = 900.0
DEFAULT_PROBE_INTERVAL_SEC = 30.0


class RunContext(str, Enum):
    """Which resilience behaviour a self-hosted-model call should use (ADR-119)."""

    SERVE = "serve"
    REPROCESS = "reprocess"


class ResilienceFuseOpenError(RuntimeError):
    """Reprocess mode exhausted pause-and-probe without the endpoint recovering.

    This is the "alert the operator" signal ADR-119 calls for — distinct from a plain
    timeout/connection error. Reprocess mode never falls over to another model, so raising
    this simply fails the current call once the endpoint has had ``on_open_max_wait_sec`` to
    recover and hasn't.
    """


@dataclass(frozen=True)
class ResiliencePolicy:
    """Backoff -> trip-after-N -> hold-and-probe, for reprocess-context calls.

    Wraps a "call the chosen model" callable. The SAME callable is retried on every
    attempt/probe — this policy never switches models. Reuses the existing
    :class:`CircuitBreaker` (callers pass in their process-wide, per-endpoint instance)
    rather than owning its own breaker state.
    """

    breaker: CircuitBreaker
    retries_before_trip: int = DEFAULT_RETRIES_BEFORE_TRIP
    backoff_schedule_sec: Sequence[float] = DEFAULT_BACKOFF_SCHEDULE_SEC
    on_open_max_wait_sec: float = DEFAULT_ON_OPEN_MAX_WAIT_SEC
    probe_interval_sec: float = DEFAULT_PROBE_INTERVAL_SEC
    name: str = "resilience"

    def _backoff_for(self, attempt_index: int) -> float:
        """Wait before the attempt after ``attempt_index`` (0-based). Repeats the
        schedule's last entry once exhausted (ADR-119: "...x2, capped")."""
        if not self.backoff_schedule_sec:
            return 0.0
        idx = min(attempt_index, len(self.backoff_schedule_sec) - 1)
        return self.backoff_schedule_sec[idx]

    def run(
        self,
        call: Callable[[], T],
        *,
        timeout_sec: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> T:
        """Run ``call`` under the reprocess policy; see the module docstring for the flow.

        Raises :class:`ResilienceFuseOpenError` if the endpoint never recovers within
        ``on_open_max_wait_sec`` of pause-and-probe. Any other exception is ``call``'s own
        (unclassified) failure, surfaced only once ``retries_before_trip`` attempts and the
        subsequent hold-and-probe are both exhausted.

        Deliberately checks ``breaker.state`` (read-only) rather than ``breaker.allow()`` to
        decide which branch to take: ``allow()`` has the side effect of consuming the single
        half-open probe, and that consumption belongs to :meth:`_pause_and_probe`'s polling
        loop, not to this dispatch check.
        """
        if self.breaker.state == "closed":
            try:
                return self._attempt_with_backoff(call, timeout_sec, sleep)
            except Exception as exc:  # noqa: BLE001 - retries exhausted; breaker now open
                logger.warning(
                    "%s: retries exhausted (%d attempts); holding and probing rather than "
                    "falling over: %s",
                    self.name,
                    self.retries_before_trip,
                    exc,
                )
        return self._pause_and_probe(call, timeout_sec, sleep)

    def _attempt_with_backoff(
        self, call: Callable[[], T], timeout_sec: float, sleep: Callable[[float], None]
    ) -> T:
        """Retry ``call`` up to ``retries_before_trip`` times with exponential backoff
        between attempts. Trips the breaker (hard) only once every attempt has failed —
        the policy's own retry count is the trip gate, bypassing the breaker's own
        rolling-window threshold (mirrors how a serve-mode hard timeout already force-trips
        bypassing that same threshold)."""
        last_exc: Exception | None = None
        for attempt in range(self.retries_before_trip):
            try:
                result = run_with_watchdog(call, timeout_sec + WATCHDOG_GRACE_SEC, label=self.name)
            except Exception as exc:  # noqa: BLE001 - retried below; re-raised once exhausted
                last_exc = exc
                logger.warning(
                    "%s: attempt %d/%d failed (%s); backing off",
                    self.name,
                    attempt + 1,
                    self.retries_before_trip,
                    exc,
                )
                if attempt < self.retries_before_trip - 1:
                    sleep(self._backoff_for(attempt))
                continue
            self.breaker.record_success()
            return result
        self.breaker.record_failure(hard=True)
        assert last_exc is not None
        raise last_exc

    def _pause_and_probe(
        self, call: Callable[[], T], timeout_sec: float, sleep: Callable[[float], None]
    ) -> T:
        """Poll the breaker until it allows a half-open probe, try exactly one probe call
        per allowed window, and keep polling until either a probe succeeds or
        ``on_open_max_wait_sec`` elapses (no cross-model fallover, ever)."""
        waited = 0.0
        while waited <= self.on_open_max_wait_sec:
            if self.breaker.allow():
                try:
                    result = run_with_watchdog(
                        call, timeout_sec + WATCHDOG_GRACE_SEC, label=self.name
                    )
                except Exception as exc:  # noqa: BLE001 - keep probing until max-wait
                    logger.warning("%s: probe failed (%s); still holding", self.name, exc)
                    self.breaker.record_failure(hard=True)
                else:
                    self.breaker.record_success()
                    return result
            sleep(self.probe_interval_sec)
            waited += self.probe_interval_sec
        raise ResilienceFuseOpenError(
            f"{self.name}: endpoint did not recover within {self.on_open_max_wait_sec:.0f}s "
            "of pause-and-probe; alerting operator (reprocess mode never falls over to "
            "another model)"
        )


__all__ = [
    "DEFAULT_BACKOFF_SCHEDULE_SEC",
    "DEFAULT_ON_OPEN_MAX_WAIT_SEC",
    "DEFAULT_PROBE_INTERVAL_SEC",
    "DEFAULT_RETRIES_BEFORE_TRIP",
    "ResilienceFuseOpenError",
    "ResiliencePolicy",
    "RunContext",
]
