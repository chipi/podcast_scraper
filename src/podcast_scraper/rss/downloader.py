"""HTTP session management and download helpers for podcast_scraper.

RSS feed fetches use :func:`fetch_rss_feed_url`, which applies a dedicated
retry policy (more attempts and gentler exponential backoff than generic
:func:`fetch_url`) for flaky feed hosts. Transcripts and episode media still
use :func:`fetch_url` / :func:`http_download_to_file`.

Transport is ``httpx`` behind :func:`podcast_scraper.net.create_client` so
admin-configured outbound proxy (#1129) + TLS trust (#1130) apply to every
feed / media call (#1194). Retry semantics come from
:class:`podcast_scraper.rss.http_retry.RetryTransport` — the same
408 / 429 / 500 / 502 / 503 / 504 status list, the same
GET / HEAD / OPTIONS allowed methods, and the same exponential-backoff
formula the previous urllib3 ``Retry`` used, so
``metrics.json``'s ``http_retry_events`` counter stays comparable across the
migration.

The retry event counter (see :func:`get_http_retry_event_count`) is
**process-wide**. It is reset in :func:`configure_downloader`, which
``run_pipeline`` calls at startup. Concurrent ``run_pipeline`` invocations in
one process can interleave counts; use one pipeline at a time per process for
meaningful ``metrics.json`` export.

The counter is exposed under two keys in the run summary JSON:

- ``http_retry_events`` — canonical name. Dashboards should read this.
- ``http_urllib3_retry_events`` — DEPRECATED alias for the same value. Marked
  in :mod:`podcast_scraper.workflow.metrics`; removal is gated on no
  Grafana / alerting query referencing the alias for 30 days plus one
  release overlap. Once the alias hits its removal criterion, the RSS
  downloader's http_retry_events counter (still process-wide) stays valid;
  only the legacy name goes away.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
from typing import cast, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote

import httpx

from ..net import create_client
from ..utils import progress
from ..utils.log_redaction import format_exception_for_log, redact_for_log
from ..utils.progress import ProgressReporter
from . import http_policy
from .http_retry import RetryTransport

# Character set matches ``requests.utils.requote_uri`` so URL normalization
# behavior is unchanged across the httpx migration (#1194).
_URL_SAFE_CHARS = "!#$%&'()*+,/:;=?@[]~"

logger = logging.getLogger(__name__)

# Track if we've suppressed urllib3 logs (lazy initialization)
_urllib3_logs_suppressed = False


def _suppress_urllib3_debug_logs() -> None:
    """Suppress verbose urllib3 debug logs when root logger is DEBUG.

    This is called lazily when downloader is first used, ensuring root logger
    is already configured. This keeps our debug logs visible while hiding
    urllib3's verbose connection logs.
    """
    global _urllib3_logs_suppressed
    if _urllib3_logs_suppressed:
        return

    root_logger = logging.getLogger()
    root_level = root_logger.level if root_logger.level else logging.INFO
    if root_level <= logging.DEBUG:
        urllib3_loggers = [
            "urllib3",
            "urllib3.connectionpool",
            "urllib3.connection",
            "urllib3.util",
            "httpcore",
            "httpcore.connection",
            "httpcore.http11",
        ]
        for logger_name in urllib3_loggers:
            urllib3_logger = logging.getLogger(logger_name)
            urllib3_logger.setLevel(logging.WARNING)

    _urllib3_logs_suppressed = True


BYTES_PER_MB = 1024 * 1024
# OpenAI Whisper API file size limit (25 MB)
OPENAI_MAX_FILE_SIZE_BYTES = 25 * BYTES_PER_MB
DEFAULT_HTTP_BACKOFF_FACTOR = 1.0
DEFAULT_HTTP_RETRY_TOTAL = 8
# RSS feed XML: more attempts + slower backoff to reduce load on rate-limited hosts
RSS_FEED_HTTP_RETRY_TOTAL = 5
RSS_FEED_HTTP_BACKOFF_FACTOR = 1.0
DOWNLOAD_CHUNK_SIZE = 1024 * 256
HTTP_RETRY_ALLOWED_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# 408: some CDNs return this under load; 429/5xx: retry with backoff
HTTP_RETRY_STATUS_CODES = (408, 429, 500, 502, 503, 504)

# Runtime-configurable retry settings (set via configure_downloader).
_configured_http_retry_total: Optional[int] = None
_configured_http_backoff_factor: Optional[float] = None
_configured_rss_retry_total: Optional[int] = None
_configured_rss_backoff_factor: Optional[float] = None

# Count urllib3 retry scheduling events (LoggingRetry.increment) for metrics export.
_http_retry_events_lock = threading.Lock()
_http_retry_events_total = 0


def reset_http_retry_event_counter() -> None:
    """Reset urllib3 retry event counter (call at pipeline start with configure_downloader)."""
    global _http_retry_events_total
    with _http_retry_events_lock:
        _http_retry_events_total = 0


def get_http_retry_event_count() -> int:
    """Return total urllib3 retry events recorded since last reset (thread-safe)."""
    with _http_retry_events_lock:
        return _http_retry_events_total


def _increment_http_retry_events() -> None:
    global _http_retry_events_total
    with _http_retry_events_lock:
        _http_retry_events_total += 1


def configure_downloader(
    *,
    http_retry_total: Optional[int] = None,
    http_backoff_factor: Optional[float] = None,
    rss_retry_total: Optional[int] = None,
    rss_backoff_factor: Optional[float] = None,
) -> None:
    """Apply runtime retry settings from Config.

    Call once at pipeline start. Values override the module-level
    defaults for all subsequent sessions created on any thread.
    Existing thread-local sessions are **not** reconfigured; they
    will pick up the new values on the next thread that creates a
    fresh session.

    Args:
        http_retry_total: Max retries for media/transcript downloads.
        http_backoff_factor: Backoff factor for media/transcript.
        rss_retry_total: Max retries for RSS feed fetches.
        rss_backoff_factor: Backoff factor for RSS feed fetches.
    """
    global _configured_http_retry_total
    global _configured_http_backoff_factor
    global _configured_rss_retry_total
    global _configured_rss_backoff_factor

    reset_http_retry_event_counter()

    if http_retry_total is not None:
        _configured_http_retry_total = http_retry_total
    if http_backoff_factor is not None:
        _configured_http_backoff_factor = http_backoff_factor
    if rss_retry_total is not None:
        _configured_rss_retry_total = rss_retry_total
    if rss_backoff_factor is not None:
        _configured_rss_backoff_factor = rss_backoff_factor

    logger.debug(
        "Downloader configured: http_retry=%s/%s rss_retry=%s/%s",
        _configured_http_retry_total,
        _configured_http_backoff_factor,
        _configured_rss_retry_total,
        _configured_rss_backoff_factor,
    )


def _effective_http_retry_total() -> int:
    return (
        _configured_http_retry_total
        if _configured_http_retry_total is not None
        else DEFAULT_HTTP_RETRY_TOTAL
    )


def _effective_http_backoff_factor() -> float:
    return (
        _configured_http_backoff_factor
        if _configured_http_backoff_factor is not None
        else DEFAULT_HTTP_BACKOFF_FACTOR
    )


def _effective_rss_retry_total() -> int:
    return (
        _configured_rss_retry_total
        if _configured_rss_retry_total is not None
        else RSS_FEED_HTTP_RETRY_TOTAL
    )


def _effective_rss_backoff_factor() -> float:
    return (
        _configured_rss_backoff_factor
        if _configured_rss_backoff_factor is not None
        else RSS_FEED_HTTP_BACKOFF_FACTOR
    )


_THREAD_LOCAL = threading.local()
_SESSION_REGISTRY: List[httpx.Client] = []
_SESSION_REGISTRY_LOCK = threading.Lock()


def should_log_download_summary() -> bool:
    """Return True when explicit download summaries should be emitted."""
    try:
        return not sys.stderr.isatty()
    except AttributeError:  # pragma: no cover - very old Python
        return True


def normalize_url(url: str) -> str:
    """Normalize URLs while preserving already-encoded segments.

    Reproduces ``requests.utils.requote_uri`` semantics on the stdlib:
    unquote once (so already-encoded sequences aren't double-encoded),
    then re-quote with the URI-safe character set.
    """
    try:
        normalized = quote(unquote(url), safe=_URL_SAFE_CHARS)
    except Exception:  # pragma: no cover - malformed % triplet in caller input
        normalized = quote(url, safe=_URL_SAFE_CHARS)
    if normalized != url:
        logger.debug("Normalized URL %s -> %s", url, normalized)
    else:
        logger.debug("URL %s did not require normalization", url)
    return normalized


def _make_retry_transport_wrapper(
    total: int,
    backoff_factor: float,
) -> "callable":  # type: ignore[valid-type]
    """Build a factory-compatible ``transport_wrapper`` for :func:`create_client`.

    Returns a callable that wraps the underlying (proxy / TLS) transport in a
    :class:`RetryTransport` configured with our status list, allowed methods,
    and the process-wide retry counter + Retry-After hook.
    """

    def _on_retry_after(response: httpx.Response, url: str) -> None:
        try:
            http_policy.note_retry_after_from_response(response, request_url=url)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Retry-After note failed", exc_info=True)

    def _wrap(base: httpx.HTTPTransport) -> RetryTransport:
        return RetryTransport(
            base,
            total=total,
            backoff_factor=backoff_factor,
            status_forcelist=HTTP_RETRY_STATUS_CODES,
            allowed_methods=HTTP_RETRY_ALLOWED_METHODS,
            on_retry=_increment_http_retry_events,
            on_retry_after=_on_retry_after,
        )

    return _wrap


def _get_thread_request_client() -> httpx.Client:
    """Thread-local ``httpx.Client`` for media / transcript / generic fetches."""
    _suppress_urllib3_debug_logs()

    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = create_client(
            subsystem="rss_generic",
            transport_wrapper=_make_retry_transport_wrapper(
                _effective_http_retry_total(),
                _effective_http_backoff_factor(),
            ),
        )
        setattr(_THREAD_LOCAL, "client", client)
        with _SESSION_REGISTRY_LOCK:
            _SESSION_REGISTRY.append(client)
        logger.debug("Created new thread-local HTTP client %s", hex(id(client)))
    else:
        logger.debug("Reusing thread-local HTTP client %s", hex(id(client)))
    return client


def _get_thread_feed_request_client() -> httpx.Client:
    """Thread-local ``httpx.Client`` for RSS feed XML (RSS-tuned retry policy)."""
    _suppress_urllib3_debug_logs()

    client = getattr(_THREAD_LOCAL, "feed_client", None)
    if client is None:
        client = create_client(
            subsystem="rss_feed",
            transport_wrapper=_make_retry_transport_wrapper(
                _effective_rss_retry_total(),
                _effective_rss_backoff_factor(),
            ),
        )
        setattr(_THREAD_LOCAL, "feed_client", client)
        with _SESSION_REGISTRY_LOCK:
            _SESSION_REGISTRY.append(client)
        logger.debug("Created new thread-local RSS feed HTTP client %s", hex(id(client)))
    else:
        logger.debug("Reusing thread-local RSS feed HTTP client %s", hex(id(client)))
    return client


def _close_all_sessions() -> None:
    with _SESSION_REGISTRY_LOCK:
        for client in _SESSION_REGISTRY:
            try:
                client.close()
            # Best-effort cleanup; ignore shutdown errors
            except Exception:  # pragma: no cover  # nosec B110
                pass
        _SESSION_REGISTRY.clear()
    for attr in ("client", "feed_client"):
        try:
            delattr(_THREAD_LOCAL, attr)
        except AttributeError:
            pass


atexit.register(_close_all_sessions)


def reset_http_sessions() -> None:
    """Close thread-local HTTP sessions so the next request uses current retry adapters.

    Embedders that call :func:`configure_downloader` (or rely on new ``Config`` values)
    before further downloads should call this; existing sessions otherwise keep prior
    urllib3 ``Retry`` settings. See CONFIGURATION.md (download resilience, threading).
    """
    _close_all_sessions()


def _synthetic_rss_response(url: str, body: bytes) -> httpx.Response:
    """Build a 200 OK response with in-memory RSS body (after server 304 + cache)."""
    request = httpx.Request("GET", url)
    return httpx.Response(
        200,
        content=body,
        headers={"Content-Type": "application/rss+xml; charset=utf-8"},
        request=request,
    )


def _is_ssrf_target(url: str) -> bool:
    """True if ``url``'s host is a loopback / link-local / private LITERAL IP.

    Feed-body transcript/enclosure URLs are untrusted (review 2026-07-17 M22);
    this blocks the obvious SSRF targets — the cloud metadata API
    (169.254.169.254), localhost, and RFC-1918 literals. Hostname targets that
    resolve to private ranges are additionally covered by the host-level
    metadata-egress iptables guard.
    """
    import ipaddress
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").strip("[]")
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # a hostname, not a literal IP
    if ip.is_loopback:
        return False  # 127.0.0.0/8 is used by the e2e/acceptance mock server + local dev
    # Blocks 169.254.169.254 (link-local metadata API) + RFC-1918 internal ranges.
    return ip.is_link_local or ip.is_private or ip.is_reserved


def _open_http_request(
    url: str,
    user_agent: str,
    timeout: int,
    *,
    stream: bool = False,
    session: Optional[httpx.Client] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    accept_not_modified: bool = False,
) -> Optional[httpx.Response]:
    """Execute an HTTP GET request and return the response if successful."""
    normalized_url = normalize_url(url)
    if _is_ssrf_target(normalized_url):
        logger.warning(
            "Request refused (private/link-local target): %s", redact_for_log(normalized_url)
        )
        return None
    headers: Dict[str, str] = {"User-Agent": user_agent}
    if extra_headers:
        headers.update(extra_headers)

    br = http_policy._STATE.circuit
    if br is not None:
        try:
            br.check_allow(normalized_url)
        except http_policy.CircuitOpenError:
            logger.warning(
                "Request skipped (circuit open): %s",
                redact_for_log(normalized_url),
            )
            return None

    try:
        sess = session if session is not None else _get_thread_request_client()
        logger.debug(
            "Opening HTTP connection to %s (timeout=%s, stream=%s) via client %s",
            normalized_url,
            timeout,
            stream,
            hex(id(sess)),
        )
        with http_policy.gated_http_request(normalized_url):
            request = sess.build_request("GET", normalized_url, headers=headers, timeout=timeout)
            resp = sess.send(request, stream=stream)

        if accept_not_modified and resp.status_code == 304:
            if br is not None:
                br.record_success(normalized_url)
            logger.debug("HTTP 304 Not Modified for %s", redact_for_log(normalized_url))
            return resp

        resp.raise_for_status()
        if br is not None:
            br.record_success(normalized_url)
        logger.debug(
            "HTTP request to %s succeeded with status %s and Content-Length=%s",
            normalized_url,
            resp.status_code,
            resp.headers.get("Content-Length"),
        )
        return resp
    except httpx.HTTPError as exc:
        if br is not None:
            err_resp = getattr(exc, "response", None)
            if err_resp is not None:
                br.record_failure(normalized_url, int(err_resp.status_code))
            else:
                br.record_failure(normalized_url, 0)
        logger.warning(
            "Failed to fetch %s: %s",
            redact_for_log(url),
            format_exception_for_log(exc),
        )
        return None


def http_head(url: str, user_agent: str, timeout: int) -> Optional[httpx.Response]:
    """Execute an HTTP HEAD request to get headers without downloading the body.

    This is useful for checking Content-Length before downloading large files.

    Args:
        url: URL to check
        user_agent: User-Agent header value
        timeout: Request timeout in seconds

    Returns:
        Response object with headers, or None if request failed
    """
    normalized_url = normalize_url(url)
    headers: Dict[str, str] = {"User-Agent": user_agent}
    br = http_policy._STATE.circuit
    if br is not None:
        try:
            br.check_allow(normalized_url)
        except http_policy.CircuitOpenError:
            logger.debug("HEAD skipped (circuit open): %s", redact_for_log(normalized_url))
            return None
    try:
        client = _get_thread_request_client()
        logger.debug(
            "Checking file size via HEAD request to %s (timeout=%s) via client %s",
            normalized_url,
            timeout,
            hex(id(client)),
        )
        with http_policy.gated_http_request(normalized_url):
            request = client.build_request("HEAD", normalized_url, headers=headers, timeout=timeout)
            resp = client.send(request)
        resp.raise_for_status()
        if br is not None:
            br.record_success(normalized_url)
        content_length = resp.headers.get("Content-Length")
        logger.debug(
            "HEAD request to %s succeeded with status %s and Content-Length=%s",
            normalized_url,
            resp.status_code,
            content_length,
        )
        return resp
    except httpx.HTTPError as exc:
        if br is not None:
            err_resp = getattr(exc, "response", None)
            if err_resp is not None:
                br.record_failure(normalized_url, int(err_resp.status_code))
            else:
                br.record_failure(normalized_url, 0)
        logger.debug(f"HEAD request to {url} failed: {exc}")
        return None


def fetch_url(
    url: str, user_agent: str, timeout: int, *, stream: bool = False
) -> Optional[httpx.Response]:
    """HTTP GET with default retry/backoff (transcripts, episode media, generic fetches)."""

    return _open_http_request(url, user_agent, timeout, stream=stream)


def fetch_rss_feed_url(
    url: str, user_agent: str, timeout: int, *, stream: bool = False
) -> Optional[httpx.Response]:
    """HTTP GET for RSS feed XML with RSS-tuned retries and exponential backoff."""

    extra: Dict[str, str] = {}
    cond = http_policy._STATE.conditional
    use_cond = http_policy._STATE.rss_conditional_get and cond is not None
    if use_cond:
        assert cond is not None
        extra.update(cond.conditional_headers(url))

    resp = _open_http_request(
        url,
        user_agent,
        timeout,
        stream=stream,
        session=_get_thread_feed_request_client(),
        extra_headers=extra if extra else None,
        accept_not_modified=use_cond,
    )
    if resp is None:
        return None

    if resp.status_code == 304:
        http_policy.record_rss_conditional_hit()
        if cond is None:
            resp.close()
            return None
        body = cond.get_cached_body(url)
        if body is None:
            logger.warning(
                "RSS 304 but no cached body for %s; treat as fetch failure",
                redact_for_log(url),
            )
            resp.close()
            return None
        resp.close()
        return _synthetic_rss_response(normalize_url(url), body)

    if use_cond and cond is not None:
        http_policy.record_rss_conditional_miss()
        try:
            data = resp.content
            etag = resp.headers.get("ETag") or resp.headers.get("etag")
            lm = resp.headers.get("Last-Modified") or resp.headers.get("last-modified")
            cond.update_from_success(url, etag, lm, data)
        except Exception:
            logger.debug("RSS conditional cache update skipped", exc_info=True)

    return resp


def http_get(url: str, user_agent: str, timeout: int) -> Tuple[Optional[bytes], Optional[str]]:
    """Fetch a URL and return its content and Content-Type header."""
    resp = fetch_url(url, user_agent, timeout, stream=True)
    if resp is None:
        logger.debug("No response received for %s", url)
        return None, None
    try:
        ctype = resp.headers.get("Content-Type", "")
        content_length = resp.headers.get("Content-Length")
        try:
            total_size = int(content_length) if content_length else None
        except (TypeError, ValueError):
            total_size = None

        logger.debug(
            "Reading response body from %s (content-type=%s, content-length=%s)",
            url,
            ctype,
            content_length,
        )

        body_parts: List[bytes] = []
        with progress.progress_context(total_size, "Downloading") as reporter:
            for chunk in resp.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                body_parts.append(chunk)
                cast(ProgressReporter, reporter).update(len(chunk))

        return b"".join(body_parts), ctype
    except (httpx.HTTPError, OSError) as exc:
        logger.warning(
            "Failed to read response from %s: %s",
            url,
            format_exception_for_log(exc),
        )
        return None, None
    finally:
        resp.close()


def http_download_to_file(
    url: str, user_agent: str, timeout: int, out_path: str
) -> Tuple[bool, int]:
    """Download content directly to a file path."""
    resp = fetch_url(url, user_agent, timeout, stream=True)
    if resp is None:
        logger.debug("No response received for %s; skipping download", url)
        return False, 0
    try:
        content_length = resp.headers.get("Content-Length")
        try:
            total_size = int(content_length) if content_length else None
        except (TypeError, ValueError):
            total_size = None

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        filename = os.path.basename(out_path) or os.path.basename(url)

        logger.debug(
            "Streaming download from %s to %s (content-length=%s, chunk-size=%s)",
            url,
            out_path,
            content_length,
            DOWNLOAD_CHUNK_SIZE,
        )

        total_bytes = 0
        with (
            open(out_path, "wb") as f,
            progress.progress_context(
                total_size,
                f"Downloading {filename}" if filename else "Downloading",
            ) as reporter,
        ):
            for chunk in resp.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                chunk_size = len(chunk)
                total_bytes += chunk_size
                cast(ProgressReporter, reporter).update(chunk_size)
        logger.debug("Finished downloading %s (%s bytes written)", url, total_bytes)
        return True, total_bytes
    except (httpx.HTTPError, OSError) as exc:
        logger.warning(
            "Failed to download %s to %s: %s",
            redact_for_log(url),
            out_path,
            format_exception_for_log(exc),
        )
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
                logger.debug("Removed partial download: %s", out_path)
        except OSError:
            pass
        return False, 0
    finally:
        resp.close()
