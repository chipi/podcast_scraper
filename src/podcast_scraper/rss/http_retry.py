"""httpx transport wrapper that reproduces the urllib3 ``Retry`` policy the RSS
downloader used before its migration to the outbound HTTP factory (#1194).

The urllib3 policy this replaces:

- ``total=N``, ``read=N``, ``connect=N``, ``status=N`` — a single N drives retries
  across ALL failure categories.
- ``backoff_factor`` — sleep between attempts follows
  ``backoff_factor * 2 ** (attempt - 1)``, capped at urllib3's
  ``BACKOFF_MAX = 120s``.
- ``allowed_methods`` — only GET / HEAD / OPTIONS retry on status codes; other
  methods (POST, PUT, DELETE, ...) surface the response after one call.
- ``status_forcelist`` — 408 / 429 / 500 / 502 / 503 / 504 trigger a retry.
- ``respect_retry_after_header=True`` — a ``Retry-After`` header overrides the
  backoff computation.
- ``raise_on_status=False`` — the last attempt's response is returned even
  when its status is still in the forcelist, letting the caller inspect the
  final headers / body.

Every retry pumps ``_increment_http_retry_events`` in
:mod:`podcast_scraper.rss.downloader` so ``http_retry_events`` in the run
summary counts the same way it did under urllib3. A retry with a
``Retry-After`` header also feeds :func:`http_policy.note_retry_after_from_response`
so the throttle-wait accounting stays consistent.

Design notes:

- Wraps an underlying ``httpx.HTTPTransport`` — the factory can layer this on
  top of its proxy / TLS transport via the ``transport_wrapper=`` argument
  to :func:`podcast_scraper.net.outbound_http.create_client`.
- Connection-level errors (``httpx.ConnectError``, ``httpx.ReadError``,
  ``httpx.RemoteProtocolError``, ``httpx.TimeoutException``) count toward
  the same retry budget as status-code retries, mirroring urllib3's
  ``connect=`` + ``read=`` treatment.
- The retry loop is bounded by ``total`` attempts total (the first call plus
  ``total`` retries — matching urllib3 semantics for ``total=N`` where the
  Nth failure is the last one). If ``total=0``, the transport passes through
  once with no retries.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, FrozenSet, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


# urllib3's cap on the computed backoff. Preserved so a bursty retry storm
# behaves the same across the migration boundary.
_BACKOFF_MAX_SECONDS = 120.0

# Errors that count as a retryable connection-level failure. Kept aligned with
# what urllib3 considered a ``connect``/``read`` failure.
_RETRYABLE_EXCEPTIONS: Tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
)


def _default_note_retry_after(response: httpx.Response, url: str) -> None:
    """Fallback for :attr:`RetryTransport.on_retry_after` when the caller does
    not wire a real callback. No-op — the downloader wires the real one at
    module-import time.
    """


def _default_on_retry() -> None:
    """Fallback for :attr:`RetryTransport.on_retry`. No-op."""


class RetryTransport(httpx.BaseTransport):
    """Wrap a base transport with the same retry policy the urllib3 downloader used."""

    def __init__(
        self,
        base: httpx.BaseTransport,
        *,
        total: int,
        backoff_factor: float,
        status_forcelist: Tuple[int, ...],
        allowed_methods: FrozenSet[str],
        on_retry: Callable[[], None] = _default_on_retry,
        on_retry_after: Callable[[httpx.Response, str], None] = _default_note_retry_after,
    ) -> None:
        """Build a retrying wrapper.

        Args:
            base: The underlying ``httpx.BaseTransport`` — typically the
                proxy/TLS transport that :func:`create_client` builds.
            total: Max retries (attempts = 1 + total, matching urllib3).
            backoff_factor: Sleep = ``backoff_factor * 2 ** (attempt - 1)``,
                capped at 120 s.
            status_forcelist: Status codes that trigger a retry when the
                request method is retryable.
            allowed_methods: HTTP methods eligible for status-code retries.
                Others surface the response after one call.
            on_retry: Called once per retry event (counter increment).
            on_retry_after: Called with the response and URL whenever a
                retryable response carries a ``Retry-After`` header, so the
                caller can note the throttle-wait separately.
        """
        self._base = base
        self._total = int(total)
        self._backoff_factor = float(backoff_factor)
        self._status_forcelist = frozenset(status_forcelist)
        self._allowed_methods = frozenset(m.upper() for m in allowed_methods)
        self._on_retry = on_retry
        self._on_retry_after = on_retry_after

    def close(self) -> None:
        """Close the wrapped base transport."""
        self._base.close()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Send ``request`` through the base transport, retrying per the urllib3-parity policy."""
        method = request.method.upper()
        url = str(request.url)
        method_allowed = method in self._allowed_methods

        last_response: Optional[httpx.Response] = None
        last_exc: Optional[BaseException] = None

        for attempt in range(self._total + 1):
            last_response = None
            last_exc = None
            try:
                response = self._base.handle_request(request)
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt >= self._total:
                    raise
                self._on_retry()
                sleep_s = self._compute_backoff(attempt)
                logger.warning(
                    "Retrying HTTP request (attempt %d/%d) %s %s due to %s",
                    attempt + 2,
                    self._total + 1,
                    method,
                    url,
                    exc,
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue

            # Status-code retry path — only when method is retryable and the
            # response status is on the forcelist.
            if method_allowed and response.status_code in self._status_forcelist:
                last_response = response
                if attempt >= self._total:
                    # Last attempt — return the response, don't raise
                    # (raise_on_status=False in the urllib3 equivalent).
                    return response
                self._on_retry()
                try:
                    self._on_retry_after(response, url)
                except Exception:  # pragma: no cover - defensive
                    logger.debug("Retry-After hook raised", exc_info=True)
                sleep_s = self._sleep_for_response(response, attempt)
                logger.warning(
                    "Retrying HTTP request (attempt %d/%d) %s %s due to %s",
                    attempt + 2,
                    self._total + 1,
                    method,
                    url,
                    response,
                )
                # Drain the body so the connection can be reused for the retry.
                try:
                    response.read()
                    response.close()
                except Exception:  # pragma: no cover - defensive
                    pass
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue

            return response

        # Fell out of the loop without a return — surface the last observable.
        if last_response is not None:
            return last_response
        if last_exc is not None:  # pragma: no cover - unreachable in practice
            raise last_exc
        raise RuntimeError(  # pragma: no cover - unreachable in practice
            "RetryTransport exhausted retries with no response and no exception"
        )

    # ---- internals -----------------------------------------------------

    def _compute_backoff(self, attempt: int) -> float:
        """Return the sleep in seconds for the given zero-based attempt.

        Matches urllib3's exponential-backoff formula:
        ``backoff_factor * (2 ** attempt)`` where ``attempt`` is the
        zero-based retry index (attempt=0 is the sleep before the first
        retry, i.e. after the first failure), capped at
        ``BACKOFF_MAX_SECONDS``. urllib3 phrases this as
        ``backoff_factor * (2 ** (retry_number - 1))`` where ``retry_number``
        is 1-based; our attempt is (retry_number - 1) so the exponent
        matches.
        """
        if attempt < 0 or self._backoff_factor <= 0:
            return 0.0
        raw: float = self._backoff_factor * (2**attempt)
        return float(min(raw, _BACKOFF_MAX_SECONDS))

    def _sleep_for_response(self, response: httpx.Response, attempt: int) -> float:
        """Prefer a ``Retry-After`` header over the computed backoff.

        Matches urllib3's ``respect_retry_after_header=True`` semantics. Values
        larger than ``BACKOFF_MAX_SECONDS`` are honored (a server-directed
        wait is authoritative), keeping symmetry with urllib3's behaviour where
        ``Retry-After`` bypasses the cap.
        """
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                seconds = float(retry_after.strip())
                if seconds >= 0:
                    return seconds
            except ValueError:
                # HTTP-date form is uncommon on the surfaces we hit; fall through
                # to computed backoff rather than trying to parse the date.
                logger.debug(
                    "Retry-After header %r is not a numeric second count; using "
                    "computed backoff",
                    retry_after,
                )
        return self._compute_backoff(attempt)


__all__ = ["RetryTransport"]
