"""Circuit breaker for any inference-service HTTP client.

Closed → rolling-window failures → open(cooldown) → half-open probe → closed.
A trimmed-down take on ``rss/http_policy.CircuitBreaker`` for a single
inference endpoint. While open, :meth:`allow` returns False so callers skip
the failing endpoint and go straight to their fallback — a wedged batch
isn't paced by per-request timeouts. After the cooldown one half-open
probe is allowed; its outcome closes or re-opens the breaker. A ``hard``
failure (a definitive timeout — strong evidence the endpoint is unusable
right now) trips immediately so the very first wedge spares the rest of
the batch. Thread-safe and process-wide.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """In-memory failure-rate circuit breaker for provider calls.

    Transitions ``closed → open`` once ``failure_threshold`` failures land
    within ``window_sec``; stays open for ``cooldown_sec`` before flipping
    back to ``closed`` on the next ``allow()``-then-success cycle. Thread-safe
    via an internal lock so it can be shared across worker threads.
    """

    def __init__(
        self,
        failure_threshold: int,
        window_sec: float,
        cooldown_sec: float,
        name: str = "inference",
    ) -> None:
        self._threshold = max(1, failure_threshold)
        self._window = max(1.0, window_sec)
        self._cooldown = max(1.0, cooldown_sec)
        self._failures: Deque[float] = deque(maxlen=64)
        self._state = "closed"
        self._name = name
        self._open_until = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def allow(self) -> bool:
        """False while the breaker is open (cooldown not elapsed); else True."""
        with self._lock:
            if self._state == "open":
                if time.monotonic() < self._open_until:
                    return False
                self._state = "half_open"  # allow a single probe
                logger.info("%s circuit breaker half-open: probing once", self._name)
            return True

    def record_success(self) -> None:
        """Close the breaker and clear the failure window — endpoint answered cleanly."""
        with self._lock:
            if self._state != "closed":
                logger.info("%s circuit breaker closed: endpoint recovered", self._name)
            self._state = "closed"
            self._failures.clear()
            self._open_until = 0.0

    def record_failure(self, *, hard: bool = False) -> None:
        """Record a failure; open the breaker on a ``hard`` failure (a definitive
        timeout) or a failed half-open probe immediately, else once the rolling window
        reaches the failure threshold."""
        with self._lock:
            now = time.monotonic()
            if hard or self._state == "half_open":
                if self._state != "open":
                    logger.warning(
                        "%s circuit breaker OPEN for %.0fs: %s",
                        self._name,
                        self._cooldown,
                        "hard timeout" if hard else "half-open probe failed",
                    )
                self._state = "open"
                self._open_until = now + self._cooldown
                self._failures.clear()
                return
            cutoff = now - self._window
            while self._failures and self._failures[0] < cutoff:
                self._failures.popleft()
            self._failures.append(now)
            if len(self._failures) >= self._threshold:
                logger.warning(
                    "%s circuit breaker OPEN for %.0fs: %d failures within %.0fs",
                    self._name,
                    self._cooldown,
                    len(self._failures),
                    self._window,
                )
                self._state = "open"
                self._open_until = now + self._cooldown

    def reset(self) -> None:
        """Force back to closed (test/ops hook)."""
        with self._lock:
            self._state = "closed"
            self._failures.clear()
            self._open_until = 0.0


__all__ = ["CircuitBreaker"]
