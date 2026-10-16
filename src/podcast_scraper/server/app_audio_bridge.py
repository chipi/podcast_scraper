"""Audio-bridge freshness/redirect resolution for the consumer player (#1070, RFC-100).

The default audio-source path returns the persisted origin URL for **direct** client
playback (bridge, never rehost). This module adds an *opt-in* freshness check: a HEAD that
follows redirects / tracking prefixes and reports the resolved final URL + content
type/length, so a client or operator can confirm an episode is playable. The no-store
pass-through proxy remains deferred until a host actually blocks direct play.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class AudioResolution:
    """Outcome of validating an origin enclosure URL."""

    verified: bool
    final_url: str
    content_type: str | None = None
    content_length: int | None = None


_HTTPS_TTL_S = 3600.0
_https_cache: dict[str, tuple[float, str | None]] = {}
_BBC_SELECTOR = re.compile(
    r"^https?://open\.live\.bbc\.co\.uk/mediaselector/\d+/redir/version/[\d.]+/mediaset/"
    r"[\w-]+/proto/https?/vpid/(\w+)"
)
_BBC_SELECT_JSON = (
    "https://open.live.bbc.co.uk/mediaselector/6/select/version/2.0/mediaset/"
    "audio-nondrm-download/vpid/{vpid}/format/json"
)


def _probe(url: str, timeout: float) -> tuple[int, str]:
    """GET the first byte of ``url`` following redirects; return ``(status, final_url)``.

    Isolated for testability — tests monkeypatch this to avoid real network I/O in CI.
    """
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url, headers={"Range": "bytes=0-0"})
        return resp.status_code, str(resp.url)


def _get_json(url: str, timeout: float) -> dict:
    """Isolated for testability, like ``_probe``."""
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        doc = client.get(url).json()
    return doc if isinstance(doc, dict) else {}


def _plays_over_https(url: str, timeout: float) -> bool:
    """A browser on an https page blocks a stream that is, or redirects to, http."""
    try:
        status, final = _probe(url, timeout)
    except httpx.HTTPError:
        return False
    return 200 <= status < 300 and final.startswith("https://")


def _bbc_https(url: str, timeout: float) -> str | None:
    """BBC's RSS enclosure redirects to an http-only ad server; its media selector lists the same
    programme as https files on BBC's own CDNs (links carry a token valid for hours)."""
    m = _BBC_SELECTOR.match(url)
    if not m:
        return None
    try:
        doc = _get_json(_BBC_SELECT_JSON.format(vpid=m.group(1)), timeout)
    except (httpx.HTTPError, ValueError):
        return None
    for media in doc.get("media") or []:
        for conn in media.get("connection") or []:
            href = conn.get("href")
            if conn.get("protocol") == "https" and isinstance(href, str):
                return href
    return None


def https_audio_url(url: str, *, timeout: float = 5.0) -> str | None:
    """An https URL that plays the same episode as ``url``, or ``None`` when there is none.

    Tried in order: ``url`` itself, its ``https://`` form, then a host's own resolver. Only http
    URLs are looked up (an https enclosure is returned unchanged, no network); results are cached
    for an hour, well inside the lifetime of a tokened link.
    """
    if url.startswith("https://"):
        return url
    now = time.monotonic()
    hit = _https_cache.get(url)
    if hit and now - hit[0] < _HTTPS_TTL_S:
        return hit[1]
    found: str | None = None
    upgraded = "https://" + url[len("http://") :] if url.startswith("http://") else None
    if upgraded and _plays_over_https(upgraded, timeout):
        found = upgraded
    if found is None:
        found = _bbc_https(url, timeout)
    _https_cache[url] = (now, found)
    return found


def _head_request(url: str, timeout: float) -> tuple[int, str, dict[str, str]]:
    """HEAD ``url`` following redirects; return ``(status, final_url, headers)``.

    Isolated for testability — tests monkeypatch this to avoid real network I/O in CI.
    """
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.head(url)
        return resp.status_code, str(resp.url), dict(resp.headers)


def resolve_audio(url: str, *, timeout: float = 5.0) -> AudioResolution:
    """Validate an origin enclosure URL via HEAD (following redirects).

    On any network error or non-2xx status, returns ``verified=False`` with the original
    URL so the client can still attempt direct playback.
    """
    try:
        status, final_url, headers = _head_request(url, timeout)
    except httpx.HTTPError:
        return AudioResolution(verified=False, final_url=url)
    if not 200 <= status < 300:
        return AudioResolution(verified=False, final_url=final_url or url)
    raw_len = headers.get("content-length")
    length = int(raw_len) if isinstance(raw_len, str) and raw_len.isdigit() else None
    return AudioResolution(
        verified=True,
        final_url=final_url or url,
        content_type=headers.get("content-type"),
        content_length=length,
    )
