"""HTTP session management and download helpers for podcast_scraper."""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
from typing import cast, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from requests.utils import requote_uri
from urllib3.util.retry import Retry

from . import progress
from .progress import ProgressReporter

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
DEFAULT_HTTP_BACKOFF_FACTOR = 0.5
DEFAULT_HTTP_RETRY_TOTAL = 5
DOWNLOAD_CHUNK_SIZE = 1024 * 256
HTTP_RETRY_ALLOWED_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
HTTP_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)

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


def _configure_http_session(session: requests.Session) -> None:
    """Attach retry-enabled HTTP adapters to a session."""

    class LoggingRetry(Retry):
        def increment(self, method=None, url=None, *args, **kwargs):  # type: ignore[override]
            new_retry = super().increment(method=method, url=url, *args, **kwargs)
            attempt = len(new_retry.history) + 1
            reason = kwargs.get("error") or kwargs.get("response")
            logger.warning(
                f"Retrying HTTP request (attempt {attempt}/{new_retry.total}) "
                f"{method or ''} {url or ''} due to {reason}"
            )
            return new_retry

    retry = LoggingRetry(
        total=DEFAULT_HTTP_RETRY_TOTAL,
        read=DEFAULT_HTTP_RETRY_TOTAL,
        connect=DEFAULT_HTTP_RETRY_TOTAL,
        status=DEFAULT_HTTP_RETRY_TOTAL,
        backoff_factor=DEFAULT_HTTP_BACKOFF_FACTOR,
        status_forcelist=HTTP_RETRY_STATUS_CODES,
        allowed_methods=HTTP_RETRY_ALLOWED_METHODS,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    logger.debug("Configured HTTP session %s with retry-enabled adapters", hex(id(session)))


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


def _close_all_sessions() -> None:
    with _SESSION_REGISTRY_LOCK:
        for session in _SESSION_REGISTRY:
            try:
                session.close()
            # Best-effort cleanup; ignore shutdown errors
            except Exception:  # pragma: no cover  # nosec B110
                pass
        _SESSION_REGISTRY.clear()


atexit.register(_close_all_sessions)


def _open_http_request(
    url: str, user_agent: str, timeout: int, *, stream: bool = False
) -> Optional[requests.Response]:
    """Execute an HTTP GET request and return the response if successful."""
    normalized_url = normalize_url(url)
    headers = {"User-Agent": user_agent}
    try:
        session = _get_thread_request_session()
        logger.debug(
            "Opening HTTP connection to %s (timeout=%s, stream=%s) via session %s",
            normalized_url,
            timeout,
            stream,
            hex(id(session)),
        )
        resp = session.get(normalized_url, headers=headers, timeout=timeout, stream=stream)
        resp.raise_for_status()
        logger.debug(
            "HTTP request to %s succeeded with status %s and Content-Length=%s",
            normalized_url,
            resp.status_code,
            resp.headers.get("Content-Length"),
        )
        return resp
    except requests.RequestException as exc:
        logger.warning(f"Failed to fetch {url}: {exc}")
        return None


def fetch_url(
    url: str, user_agent: str, timeout: int, *, stream: bool = False
) -> Optional[requests.Response]:
    """Public wrapper around the retry-enabled HTTP GET logic."""

    return _open_http_request(url, user_agent, timeout, stream=stream)


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
        logger.warning(f"Failed to read response from {url}: {exc}")
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
        logger.warning(f"Failed to download {url} to {out_path}: {exc}")
        return False, 0
    finally:
        resp.close()
