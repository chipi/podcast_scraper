"""HTTP session management and download helpers for podcast_scraper.

RSS feed fetches use :func:`fetch_rss_feed_url`, which applies a dedicated urllib3
``Retry`` policy (more attempts and gentler exponential backoff than generic
:func:`fetch_url`) for flaky feed hosts. Transcripts and episode media still use
:func:`fetch_url` / :func:`http_download_to_file`.

The urllib3 retry event counter (see :func:`get_http_retry_event_count`) is
**process-wide**. It is reset in :func:`configure_downloader`, which ``run_pipeline``
calls at startup. Concurrent ``run_pipeline`` invocations in one process can
interleave counts; use one pipeline at a time per process for meaningful
``metrics.json`` export of ``http_urllib3_retry_events``.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
from typing import cast, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from requests.structures import CaseInsensitiveDict
from requests.utils import requote_uri
from urllib3.util.retry import Retry

from ..utils import progress
from ..utils.log_redaction import format_exception_for_log, redact_for_log
from ..utils.progress import ProgressReporter
from . import http_policy

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
RSS_FEED_HTTP_RETRY_TOTAL = 10
RSS_FEED_HTTP_BACKOFF_FACTOR = 2.0
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
_SESSION_REGISTRY: List[requests.Session] = []
_SESSION_REGISTRY_LOCK = threading.Lock()


def should_log_download_summary() -> bool:
    """Return True when explicit download summaries should be emitted."""
    try:
        return not sys.stderr.isatty()
    except AttributeError:  # pragma: no cover - very old Python
        return True


def normalize_url(url: str) -> str:
    """Normalize URLs while preserving already-encoded segments."""
    normalized = requote_uri(url)
    if normalized != url:
        logger.debug("Normalized URL %s -> %s", url, normalized)
    else:
        logger.debug("URL %s did not require normalization", url)
    return cast(str, normalized)


def _create_logging_retry(
    total: int,
    backoff_factor: float,
    status_forcelist: Tuple[int, ...],
) -> Retry:
    """Build urllib3 Retry with WARNING logs on each urllib3 retry attempt."""

    class LoggingRetry(Retry):
        def increment(self, method=None, url=None, *args, **kwargs):  # type: ignore[override]
            new_retry = super().increment(method=method, url=url, *args, **kwargs)
            _increment_http_retry_events()
            attempt = len(new_retry.history) + 1
            reason = kwargs.get("error") or kwargs.get("response")
            logger.warning(
                f"Retrying HTTP request (attempt {attempt}/{new_retry.total}) "
                f"{method or ''} {url or ''} due to {reason}"
            )
            resp_obj = kwargs.get("response")
            if resp_obj is not None:
                try:
                    http_policy.note_retry_after_from_response(resp_obj, request_url=url)
                except Exception:  # pragma: no cover - defensive
                    logger.debug("Retry-After note failed", exc_info=True)
            return new_retry

    return LoggingRetry(
        total=total,
        read=total,
        connect=total,
        status=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=HTTP_RETRY_ALLOWED_METHODS,
        raise_on_status=False,
        respect_retry_after_header=True,
    )


def _configure_http_session(session: requests.Session) -> None:
    """Attach retry-enabled HTTP adapters (default policy for media/transcripts)."""
    retry = _create_logging_retry(
        _effective_http_retry_total(),
        _effective_http_backoff_factor(),
        HTTP_RETRY_STATUS_CODES,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    logger.debug("Configured HTTP session %s with retry-enabled adapters", hex(id(session)))


def _configure_rss_feed_http_session(session: requests.Session) -> None:
    """Attach retry/backoff policy tuned for RSS feed XML fetches."""
    retry = _create_logging_retry(
        _effective_rss_retry_total(),
        _effective_rss_backoff_factor(),
        HTTP_RETRY_STATUS_CODES,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    logger.debug(
        "Configured RSS feed HTTP session %s with retry-enabled adapters", hex(id(session))
    )


def _get_thread_request_session() -> requests.Session:
    # Suppress urllib3 debug logs on first use
    _suppress_urllib3_debug_logs()

    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _configure_http_session(session)
        setattr(_THREAD_LOCAL, "session", session)
        with _SESSION_REGISTRY_LOCK:
            _SESSION_REGISTRY.append(session)
        logger.debug("Created new thread-local HTTP session %s", hex(id(session)))
    else:
        logger.debug("Reusing thread-local HTTP session %s", hex(id(session)))
    return session


def _get_thread_feed_request_session() -> requests.Session:
    """Thread-local session for RSS feed XML (stronger retries/backoff than generic fetch)."""
    _suppress_urllib3_debug_logs()

    session = getattr(_THREAD_LOCAL, "feed_session", None)
    if session is None:
        session = requests.Session()
        _configure_rss_feed_http_session(session)
        setattr(_THREAD_LOCAL, "feed_session", session)
        with _SESSION_REGISTRY_LOCK:
            _SESSION_REGISTRY.append(session)
        logger.debug("Created new thread-local RSS feed HTTP session %s", hex(id(session)))
    else:
        logger.debug("Reusing thread-local RSS feed HTTP session %s", hex(id(session)))
    return session


def _close_all_sessions() -> None:
    with _SESSION_REGISTRY_LOCK:
        for session in _SESSION_REGISTRY:
            try:
                session.close()
            # Best-effort cleanup; ignore shutdown errors
            except Exception:  # pragma: no cover  # nosec B110
                pass
        _SESSION_REGISTRY.clear()
    for attr in ("session", "feed_session"):
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


def _synthetic_rss_response(url: str, body: bytes) -> requests.Response:
    """Build a 200 OK response with in-memory RSS body (after server 304 + cache)."""
    r = requests.Response()
    r.status_code = 200
    r._content = body
    r.url = url
    r.headers = CaseInsensitiveDict({"Content-Type": "application/rss+xml; charset=utf-8"})
    return r


def _open_http_request(
    url: str,
    user_agent: str,
    timeout: int,
    *,
    stream: bool = False,
    session: Optional[requests.Session] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    accept_not_modified: bool = False,
) -> Optional[requests.Response]:
    """Execute an HTTP GET request and return the response if successful."""
    normalized_url = normalize_url(url)
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
        sess = session if session is not None else _get_thread_request_session()
        logger.debug(
            "Opening HTTP connection to %s (timeout=%s, stream=%s) via session %s",
            normalized_url,
            timeout,
            stream,
            hex(id(sess)),
        )
        with http_policy.gated_http_request(normalized_url):
            resp = sess.get(normalized_url, headers=headers, timeout=timeout, stream=stream)

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
    except requests.RequestException as exc:
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


def http_head(url: str, user_agent: str, timeout: int) -> Optional[requests.Response]:
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
    headers = {"User-Agent": user_agent}
    br = http_policy._STATE.circuit
    if br is not None:
        try:
            br.check_allow(normalized_url)
        except http_policy.CircuitOpenError:
            logger.debug("HEAD skipped (circuit open): %s", redact_for_log(normalized_url))
            return None
    try:
        session = _get_thread_request_session()
        logger.debug(
            "Checking file size via HEAD request to %s (timeout=%s) via session %s",
            normalized_url,
            timeout,
            hex(id(session)),
        )
        with http_policy.gated_http_request(normalized_url):
            resp = session.head(normalized_url, headers=headers, timeout=timeout)
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
    except requests.RequestException as exc:
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
) -> Optional[requests.Response]:
    """HTTP GET with default retry/backoff (transcripts, episode media, generic fetches)."""

    return _open_http_request(url, user_agent, timeout, stream=stream)


def fetch_rss_feed_url(
    url: str, user_agent: str, timeout: int, *, stream: bool = False
) -> Optional[requests.Response]:
    """HTTP GET for RSS feed XML with RSS-tuned urllib3 retries and exponential backoff."""

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
        session=_get_thread_feed_request_session(),
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
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                body_parts.append(chunk)
                cast(ProgressReporter, reporter).update(len(chunk))

        return b"".join(body_parts), ctype
    except (requests.RequestException, OSError) as exc:
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
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                chunk_size = len(chunk)
                total_bytes += chunk_size
                cast(ProgressReporter, reporter).update(chunk_size)
        logger.debug("Finished downloading %s (%s bytes written)", url, total_bytes)
        return True, total_bytes
    except (requests.RequestException, OSError) as exc:
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
