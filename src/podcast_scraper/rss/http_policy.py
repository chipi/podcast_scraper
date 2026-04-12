"""Per-host HTTP policy: throttling, Retry-After, circuit breaker, RSS conditional GET.

Issue #522: fair usage toward feeds and CDNs. Configured via :func:`configure_http_policy`
at pipeline start (see ``orchestration``). Metrics are process-wide and reset there,
similar to ``downloader.configure_downloader``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# --- process-wide metrics (reset in configure_http_policy) ---------------------------------
_metrics_lock = threading.Lock()
_host_throttle_wait_seconds = 0.0
_host_throttle_events = 0
_retry_after_events = 0
_retry_after_total_sleep_seconds = 0.0
_circuit_breaker_trips = 0
_circuit_breaker_open_keys: List[str] = []
_rss_conditional_hit = 0
_rss_conditional_miss = 0


def reset_http_policy_metrics() -> None:
    """Reset policy metrics (pipeline start)."""
    global _host_throttle_wait_seconds, _host_throttle_events
    global _retry_after_events, _retry_after_total_sleep_seconds
    global _circuit_breaker_trips, _circuit_breaker_open_keys
    global _rss_conditional_hit, _rss_conditional_miss
    with _metrics_lock:
        _host_throttle_wait_seconds = 0.0
        _host_throttle_events = 0
        _retry_after_events = 0
        _retry_after_total_sleep_seconds = 0.0
        _circuit_breaker_trips = 0
        _circuit_breaker_open_keys = []
        _rss_conditional_hit = 0
        _rss_conditional_miss = 0


def record_host_throttle_wait(seconds: float) -> None:
    """Record time spent waiting on per-host throttle (interval or Retry-After)."""
    global _host_throttle_wait_seconds, _host_throttle_events
    if seconds <= 0:
        return
    with _metrics_lock:
        _host_throttle_wait_seconds += seconds
        _host_throttle_events += 1


def record_retry_after_sleep(seconds: float) -> None:
    """Record Retry-After driven sleep (tracker / logging path)."""
    global _retry_after_events, _retry_after_total_sleep_seconds
    if seconds <= 0:
        return
    with _metrics_lock:
        _retry_after_events += 1
        _retry_after_total_sleep_seconds += seconds


def record_circuit_trip(key: str) -> None:
    """Increment trip counter and append ``key`` to the open-breaker list (metrics)."""
    global _circuit_breaker_trips
    with _metrics_lock:
        _circuit_breaker_trips += 1
        if key not in _circuit_breaker_open_keys:
            _circuit_breaker_open_keys.append(key)


def record_rss_conditional_hit() -> None:
    """Count RSS fetches that returned 304 (validators matched cache)."""
    global _rss_conditional_hit
    with _metrics_lock:
        _rss_conditional_hit += 1


def record_rss_conditional_miss() -> None:
    """Count RSS fetches that needed a full response (200 or non-304 path)."""
    global _rss_conditional_miss
    with _metrics_lock:
        _rss_conditional_miss += 1


def get_http_policy_metrics_snapshot() -> Dict[str, object]:
    """Return a copy of policy metrics for ``metrics.Metrics.finish()``."""
    with _metrics_lock:
        return {
            "host_throttle_wait_seconds": round(_host_throttle_wait_seconds, 4),
            "host_throttle_events": _host_throttle_events,
            "retry_after_events": _retry_after_events,
            "retry_after_total_sleep_seconds": round(_retry_after_total_sleep_seconds, 4),
            "circuit_breaker_trips": _circuit_breaker_trips,
            "circuit_breaker_open_feeds": list(_circuit_breaker_open_keys),
            "rss_conditional_hit": _rss_conditional_hit,
            "rss_conditional_miss": _rss_conditional_miss,
        }


def netloc_from_url(url: str) -> str:
    """Return lowercase netloc for host-keyed policy."""
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        return host if host else "unknown"
    except Exception:
        return "unknown"


class CircuitOpenError(Exception):
    """Raised when the circuit breaker blocks a request for a scope key."""

    def __init__(self, scope_key: str) -> None:
        self.scope_key = scope_key
        super().__init__(f"Circuit open for {scope_key!r}")


# --- Retry-After tracking (per netloc) -----------------------------------------------------


class RetryAfterTracker:
    """Sleep until ``Retry-After`` deadline per host before new requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._deadline: Dict[str, float] = {}

    def set_deadline_from_header(self, netloc: str, header_value: str) -> None:
        """Parse Retry-After (seconds or HTTP-date) and extend deadline for netloc."""
        now = time.monotonic()
        deadline = self._parse_retry_after(header_value, now)
        if deadline is None:
            return
        with self._lock:
            prev = self._deadline.get(netloc, 0.0)
            self._deadline[netloc] = max(prev, deadline)
        logger.warning(
            "HTTP Retry-After for host %s: %r until ~%.1fs from now",
            netloc,
            header_value,
            deadline - now,
        )

    @staticmethod
    def _parse_retry_after(value: str, now_mono: float) -> Optional[float]:
        value = (value or "").strip()
        if not value:
            return None
        try:
            sec = int(value)
            if sec < 0:
                return None
            return now_mono + float(sec)
        except ValueError:
            pass
        try:
            dt = parsedate_to_datetime(value)
            if dt is None:
                return None
            ts = dt.timestamp()
            wall_now = time.time()
            return now_mono + max(0.0, ts - wall_now)
        except (TypeError, ValueError, OSError):
            return None

    def wait_until_clear(self, netloc: str) -> None:
        """Block until this host is past its Retry-After deadline."""
        while True:
            with self._lock:
                deadline = self._deadline.get(netloc)
            if deadline is None:
                return
            now = time.monotonic()
            wait = deadline - now
            if wait <= 0:
                with self._lock:
                    if self._deadline.get(netloc, 0) <= now:
                        self._deadline.pop(netloc, None)
                return
            logger.debug("Retry-After wait for host %s: %.2fs", netloc, wait)
            record_retry_after_sleep(wait)
            record_host_throttle_wait(wait)
            time.sleep(wait)


def note_retry_after_from_response(response: object, request_url: Optional[str] = None) -> None:
    """If response has Retry-After, update tracker (urllib3 retry path)."""
    tracker = _STATE.retry_after_tracker
    if tracker is None:
        return
    try:
        headers = getattr(response, "headers", None)
        if headers is None:
            return
        getm = getattr(headers, "get", None)
        if not callable(getm):
            return
        raw = getm("Retry-After") or getm("retry-after")
        if not raw:
            return
        url = str(getattr(response, "url", None) or request_url or "")
        netloc = netloc_from_url(url)
        tracker.set_deadline_from_header(netloc, str(raw))
    except Exception:
        logger.debug("note_retry_after_from_response: ignored", exc_info=True)


# --- Host throttle -------------------------------------------------------------------------


@dataclass
class _ThrottleSlot:
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_finish: float = 0.0
    semaphore: Optional[threading.Semaphore] = None


class HostThrottle:
    """Minimum interval and max concurrent requests per netloc."""

    def __init__(
        self,
        interval_ms: int,
        max_concurrent: int,
        retry_after: Optional[RetryAfterTracker],
    ) -> None:
        self._interval_sec = max(0.0, interval_ms / 1000.0) if interval_ms > 0 else 0.0
        self._max_concurrent = max_concurrent if max_concurrent > 0 else 0
        self._retry_after = retry_after
        self._slots: Dict[str, _ThrottleSlot] = {}
        self._map_lock = threading.Lock()

    def _slot(self, netloc: str) -> _ThrottleSlot:
        with self._map_lock:
            if netloc not in self._slots:
                sem = (
                    threading.Semaphore(self._max_concurrent) if self._max_concurrent > 0 else None
                )
                self._slots[netloc] = _ThrottleSlot(semaphore=sem)
            return self._slots[netloc]

    def acquire(self, url: str) -> None:
        """Wait for Retry-After, concurrency slot, and per-host interval before the request."""
        netloc = netloc_from_url(url)
        if self._retry_after is not None:
            self._retry_after.wait_until_clear(netloc)
        if self._interval_sec <= 0 and self._max_concurrent <= 0:
            return
        slot = self._slot(netloc)
        if slot.semaphore is not None:
            slot.semaphore.acquire()
        if self._interval_sec > 0:
            with slot.lock:
                now = time.monotonic()
                wait = slot.last_finish + self._interval_sec - now
                if wait > 0:
                    time.sleep(wait)
                    record_host_throttle_wait(wait)
                # last_finish updated on release

    def release(self, url: str) -> None:
        """Update interval bookkeeping and release the concurrency slot for ``url``."""
        netloc = netloc_from_url(url)
        if self._interval_sec <= 0 and self._max_concurrent <= 0:
            return
        with self._map_lock:
            slot = self._slots.get(netloc)
        if slot is None:
            return
        if self._interval_sec > 0:
            with slot.lock:
                slot.last_finish = time.monotonic()
        if slot.semaphore is not None:
            slot.semaphore.release()


# --- Circuit breaker -----------------------------------------------------------------------


@dataclass
class _BreakerEntry:
    state: str = "closed"
    failure_times: Deque[float] = field(default_factory=lambda: deque(maxlen=256))
    open_until: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class CircuitBreaker:
    """Rolling-window failures then cooldown; optional feed- vs host-scoped key."""

    def __init__(
        self,
        enabled: bool,
        failure_threshold: int,
        window_seconds: float,
        cooldown_seconds: float,
        scope: str,
        rss_url: str,
    ) -> None:
        self._enabled = enabled
        self._threshold = max(1, failure_threshold)
        self._window = max(1.0, window_seconds)
        self._cooldown = max(1.0, cooldown_seconds)
        self._scope = scope if scope in ("feed", "host") else "feed"
        self._rss_url = rss_url or ""
        self._entries: Dict[str, _BreakerEntry] = {}
        self._entries_lock = threading.Lock()

    def scope_key(self, request_url: str) -> str:
        """Stable breaker key for feed- or host-scoped mode."""
        if self._scope == "feed" and self._rss_url:
            return f"feed:{self._rss_url}"
        return f"host:{netloc_from_url(request_url)}"

    def _entry(self, key: str) -> _BreakerEntry:
        with self._entries_lock:
            if key not in self._entries:
                self._entries[key] = _BreakerEntry()
            return self._entries[key]

    def check_allow(self, request_url: str) -> None:
        """Raise CircuitOpenError if the breaker blocks this request."""
        if not self._enabled:
            return
        key = self.scope_key(request_url)
        entry = self._entry(key)
        now = time.monotonic()
        with entry.lock:
            if entry.state == "open":
                if now < entry.open_until:
                    raise CircuitOpenError(key)
                entry.state = "half_open"
            # half_open: allow single probe; closed: allow

    def record_success(self, request_url: str) -> None:
        """Close the breaker or clear failure history after a successful request."""
        if not self._enabled:
            return
        key = self.scope_key(request_url)
        entry = self._entry(key)
        with entry.lock:
            if entry.state == "half_open":
                entry.state = "closed"
                entry.failure_times.clear()
            elif entry.state == "closed":
                entry.failure_times.clear()

    def record_failure(self, request_url: str, status_code: int) -> None:
        """Count qualifying failures; open the circuit when the rolling window threshold is met."""
        if not self._enabled:
            return
        if status_code not in (0, 401, 403, 429) and not (status_code >= 500):
            return
        key = self.scope_key(request_url)
        entry = self._entry(key)
        now = time.monotonic()
        with entry.lock:
            if entry.state == "half_open":
                entry.state = "open"
                entry.open_until = now + self._cooldown
                record_circuit_trip(key)
                return
            cutoff = now - self._window
            while entry.failure_times and entry.failure_times[0] < cutoff:
                entry.failure_times.popleft()
            entry.failure_times.append(now)
            if len(entry.failure_times) >= self._threshold:
                entry.state = "open"
                entry.open_until = now + self._cooldown
                record_circuit_trip(key)


# --- Conditional GET cache -----------------------------------------------------------------


class ConditionalGetCache:
    """Persist ETag / Last-Modified and RSS body per feed URL (JSON + sidecar body)."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _key(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _paths(self, url: str) -> Tuple[Path, Path]:
        h = self._key(url)
        return self._dir / f"{h}.json", self._dir / f"{h}.body"

    def conditional_headers(self, url: str) -> Dict[str, str]:
        """Return ``If-None-Match`` / ``If-Modified-Since`` from stored metadata, if any."""
        meta_path, _ = self._paths(url)
        if not meta_path.is_file():
            return {}
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        out: Dict[str, str] = {}
        etag = meta.get("etag")
        if etag:
            out["If-None-Match"] = str(etag)
        lm = meta.get("last_modified")
        if lm:
            out["If-Modified-Since"] = str(lm)
        return out

    def get_cached_body(self, url: str) -> Optional[bytes]:
        """Return the last cached RSS body bytes for ``url``, or None if missing."""
        _, body_path = self._paths(url)
        if not body_path.is_file():
            return None
        try:
            return body_path.read_bytes()
        except OSError:
            return None

    def update_from_success(
        self, url: str, etag: Optional[str], last_modified: Optional[str], body: bytes
    ) -> None:
        """Persist validators and body after HTTP 200."""
        meta_path, body_path = self._paths(url)
        with self._lock:
            try:
                body_path.write_bytes(body)
                meta = {}
                if etag:
                    meta["etag"] = etag
                if last_modified:
                    meta["last_modified"] = last_modified
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except OSError as exc:
                logger.warning("RSS conditional cache write failed: %s", exc)


# --- Module policy state -------------------------------------------------------------------


@dataclass
class _PolicyState:
    throttle: Optional[HostThrottle] = None
    retry_after_tracker: Optional[RetryAfterTracker] = None
    circuit: Optional[CircuitBreaker] = None
    conditional: Optional[ConditionalGetCache] = None
    rss_conditional_get: bool = False
    rss_url: str = ""


_STATE = _PolicyState()


def configure_http_policy(
    *,
    rss_url: str = "",
    host_request_interval_ms: int = 0,
    host_max_concurrent: int = 0,
    circuit_breaker_enabled: bool = False,
    circuit_breaker_failure_threshold: int = 5,
    circuit_breaker_window_seconds: int = 60,
    circuit_breaker_cooldown_seconds: int = 120,
    circuit_breaker_scope: str = "feed",
    rss_conditional_get: bool = False,
    rss_cache_dir: Optional[str] = None,
) -> None:
    """Initialize HTTP policy for this process / pipeline run."""
    global _STATE
    reset_http_policy_metrics()

    env_skip = (os.environ.get("PODCAST_SCRAPER_RSS_SKIP_CONDITIONAL") or "").strip().lower()
    if env_skip in ("1", "true", "yes", "on"):
        rss_conditional_get = False

    needs_policy = (
        host_request_interval_ms > 0
        or host_max_concurrent > 0
        or circuit_breaker_enabled
        or rss_conditional_get
    )
    retry_tracker: Optional[RetryAfterTracker] = RetryAfterTracker() if needs_policy else None

    throttle: Optional[HostThrottle] = None
    if retry_tracker is not None:
        throttle = HostThrottle(
            host_request_interval_ms,
            host_max_concurrent,
            retry_tracker,
        )

    circuit: Optional[CircuitBreaker] = None
    if circuit_breaker_enabled:
        circuit = CircuitBreaker(
            True,
            circuit_breaker_failure_threshold,
            float(circuit_breaker_window_seconds),
            float(circuit_breaker_cooldown_seconds),
            circuit_breaker_scope,
            rss_url,
        )

    cond: Optional[ConditionalGetCache] = None
    if rss_conditional_get:
        base = rss_cache_dir or os.environ.get("PODCAST_SCRAPER_RSS_CACHE_DIR")
        cache_path = Path(os.path.expanduser(str(base) if base else "~/.cache/podcast_scraper/rss"))
        try:
            cond = ConditionalGetCache(cache_path)
        except OSError as exc:
            logger.warning(
                "RSS conditional GET cache unusable at %s (%s); fetching RSS without validators",
                cache_path,
                exc,
            )
            cond = None

    _STATE = _PolicyState(
        throttle=throttle,
        retry_after_tracker=retry_tracker,
        circuit=circuit,
        conditional=cond,
        rss_conditional_get=rss_conditional_get,
        rss_url=rss_url or "",
    )

    logger.debug(
        "http_policy: interval_ms=%s max_conc=%s circuit=%s cond_get=%s",
        host_request_interval_ms,
        host_max_concurrent,
        circuit_breaker_enabled,
        rss_conditional_get,
    )


@contextmanager
def gated_http_request(url: str) -> Iterator[None]:
    """Acquire throttle for ``url``; release after block."""
    th = _STATE.throttle
    if th is None:
        yield
        return
    try:
        th.acquire(url)
        yield
    finally:
        th.release(url)
