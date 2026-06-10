"""Shared resilience primitives for the DGX provider calls (#946 / #954).

The DGX Whisper (``:8002``) and pyannote diarize (``:8001``) clients both upload
an audio file to a single-GPU service on the shared GB10 box and must degrade
gracefully when that GPU is contended or wedged. They share the same defences,
collected here so the two providers stay in lockstep:

- :func:`probe_audio_duration_sec` / :func:`effective_timeout_sec` — size a
  request timeout from the audio length so long episodes don't false-fail under
  brief contention.
- :data:`TimeoutLike` — the exception classes that mean "the server is slow /
  still working" (fall over, don't re-queue a duplicate) vs a connection blip.
- :func:`run_with_watchdog` — a hard process-side wall-clock deadline. httpx's
  own read/write timeout has been observed to never fire when a co-tenant
  workload (e.g. a vLLM crash-loop, #954) intermittently stalls the GPU: the
  multipart upload trickles, each accepted chunk resets httpx's per-write
  timeout, and the request hangs indefinitely. The watchdog guarantees the
  caller regains control and fails over.
- :class:`CircuitBreaker` — a trimmed-down take on
  ``rss/http_policy.CircuitBreaker``: once DGX has failed, trip a cooldown during
  which callers skip DGX entirely (no per-request timeout tax on a wedged batch);
  a half-open probe after the cooldown re-tests so it self-heals on recovery.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, cast, Deque, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default wall-clock slack added on top of a request's duration-scaled timeout
# before the watchdog gives up (covers upload + response serialisation).
WATCHDOG_GRACE_SEC = 30.0

# Timeout-class errors: the server is slow/contended and (likely) still working
# the request, so retrying would double the load. Resolved lazily so a missing
# httpx doesn't break import.
try:  # pragma: no cover - import guard
    import httpx as _httpx

    TimeoutLike: tuple[type[Exception], ...] = (_httpx.TimeoutException, TimeoutError)
except ImportError:  # pragma: no cover
    TimeoutLike = (TimeoutError,)


def probe_audio_duration_sec(audio_path: str) -> Optional[float]:
    """Best-effort audio duration (seconds) for timeout scaling; None on failure.

    Used only to size the request timeout, so a miss is harmless — it just falls
    back to the flat base budget. Dependency-light (soundfile is already a
    transitive dep) and never raises.
    """
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001 - duration is advisory only
        return None
    return None


def effective_timeout_sec(
    base_sec: float, per_audio_min_sec: float, duration_sec: Optional[float]
) -> float:
    """Duration-scaled request budget: ``base + (audio_minutes * per_audio_min)``.

    A flat timeout false-fails long episodes whenever the shared GPU is briefly
    contended. Scaling by audio length lets a call wait the contention out instead
    of bailing prematurely.
    """
    base = float(base_sec)
    if duration_sec and duration_sec > 0 and per_audio_min_sec:
        base += (float(duration_sec) / 60.0) * float(per_audio_min_sec)
    return base


def run_with_watchdog(fn: Callable[[], T], deadline_sec: float, *, label: str) -> T:
    """Run ``fn()`` under a hard wall-clock deadline; raise ``TimeoutError`` if it
    overruns.

    Runs ``fn`` in a daemon worker thread and waits at most ``deadline_sec``. If the
    call hasn't returned by then we stop waiting and raise — guaranteeing the caller
    regains control even when ``fn`` (e.g. a trickling httpx upload) ignores its own
    timeout. The orphaned worker is a daemon: it holds at most one connection, never
    blocks process exit, and unwinds whenever the underlying call finally errors.
    Exceptions raised inside ``fn`` propagate to the caller unchanged.
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["res"] = fn()
        except BaseException as exc:  # noqa: BLE001 - propagate to caller thread
            box["err"] = exc

    worker = threading.Thread(target=_run, name=label, daemon=True)
    worker.start()
    worker.join(deadline_sec)
    if worker.is_alive():
        raise TimeoutError(
            f"{label} exceeded hard wall-clock deadline {deadline_sec:.0f}s; "
            "abandoning request and failing over"
        )
    if "err" in box:
        raise box["err"]
    return cast(T, box["res"])


class CircuitBreaker:
    """closed → rolling-window failures → open(cooldown) → half-open probe → closed.

    A trimmed-down take on ``rss/http_policy.CircuitBreaker`` for a single DGX
    endpoint. While open, :meth:`allow` returns False so callers skip DGX and go
    straight to their fallback — a wedged batch isn't paced by per-request timeouts.
    After the cooldown one half-open probe is allowed; its outcome closes or
    re-opens the breaker. A ``hard`` failure (a definitive timeout — strong evidence
    the endpoint is unusable right now) trips immediately so the very first wedge
    spares the rest of the batch. Thread-safe and process-wide.
    """

    def __init__(
        self,
        failure_threshold: int,
        window_sec: float,
        cooldown_sec: float,
        *,
        name: str = "dgx",
    ) -> None:
        self._name = name
        self._threshold = max(1, failure_threshold)
        self._window = max(1.0, window_sec)
        self._cooldown = max(1.0, cooldown_sec)
        self._failures: Deque[float] = deque(maxlen=64)
        self._state = "closed"
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
                logger.info("%s circuit breaker half-open: probing DGX once", self._name)
            return True

    def record_success(self) -> None:
        with self._lock:
            if self._state != "closed":
                logger.info("%s circuit breaker closed: DGX recovered", self._name)
            self._state = "closed"
            self._failures.clear()

    def record_failure(self, *, hard: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
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
