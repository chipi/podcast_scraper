"""HTTP session management and download helpers for podcast_scraper."""

from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from requests.utils import requote_uri
from urllib3.util.retry import Retry

from . import progress

logger = logging.getLogger(__name__)

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
        return not os.isatty(2)
    except AttributeError:  # pragma: no cover - very old Python
        return True


def normalize_url(url: str) -> str:
    """Normalize URLs while preserving already-encoded segments."""
    return requote_uri(url)


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


def _get_thread_request_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _configure_http_session(session)
        setattr(_THREAD_LOCAL, "session", session)
        with _SESSION_REGISTRY_LOCK:
            _SESSION_REGISTRY.append(session)
    return session


def _close_all_sessions() -> None:
    with _SESSION_REGISTRY_LOCK:
        for session in _SESSION_REGISTRY:
            try:
                session.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        _SESSION_REGISTRY.clear()


atexit.register(_close_all_sessions)


def _open_http_request(url: str, user_agent: str, timeout: int, *, stream: bool = False) -> Optional[requests.Response]:
    """Execute an HTTP GET request and return the response if successful."""
    normalized_url = normalize_url(url)
    headers = {"User-Agent": user_agent}
    try:
        session = _get_thread_request_session()
        resp = session.get(normalized_url, headers=headers, timeout=timeout, stream=stream)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.warning(f"Failed to fetch {url}: {exc}")
        return None


def fetch_url(url: str, user_agent: str, timeout: int, *, stream: bool = False) -> Optional[requests.Response]:
    """Public wrapper around the retry-enabled HTTP GET logic."""

    return _open_http_request(url, user_agent, timeout, stream=stream)


def http_get(url: str, user_agent: str, timeout: int) -> Tuple[Optional[bytes], Optional[str]]:
    """Fetch a URL and return its content and Content-Type header."""
    resp = fetch_url(url, user_agent, timeout, stream=True)
    if resp is None:
        return None, None
    try:
        ctype = resp.headers.get("Content-Type", "")
        content_length = resp.headers.get("Content-Length")
        try:
            total_size = int(content_length) if content_length else None
        except (TypeError, ValueError):
            total_size = None

        body_parts: List[bytes] = []
        with progress.progress_context(total_size, "Downloading") as reporter:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                body_parts.append(chunk)
                reporter.update(len(chunk))

        return b"".join(body_parts), ctype
    except (requests.RequestException, OSError) as exc:
        logger.warning(f"Failed to read response from {url}: {exc}")
        return None, None
    finally:
        resp.close()


def http_download_to_file(url: str, user_agent: str, timeout: int, out_path: str) -> Tuple[bool, int]:
    """Download content directly to a file path."""
    resp = fetch_url(url, user_agent, timeout, stream=True)
    if resp is None:
        return False, 0
    try:
        content_length = resp.headers.get("Content-Length")
        try:
            total_size = int(content_length) if content_length else None
        except (TypeError, ValueError):
            total_size = None

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        filename = os.path.basename(out_path) or os.path.basename(url)

        total_bytes = 0
        with open(out_path, "wb") as f, progress.progress_context(
            total_size,
            f"Downloading {filename}" if filename else "Downloading",
        ) as reporter:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                chunk_size = len(chunk)
                total_bytes += chunk_size
                reporter.update(chunk_size)
        return True, total_bytes
    except (requests.RequestException, OSError) as exc:
        logger.warning(f"Failed to download {url} to {out_path}: {exc}")
        return False, 0
    finally:
        resp.close()
