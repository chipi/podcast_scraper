"""Audio-bridge freshness/redirect resolution for the consumer player (#1070, RFC-100).

The default audio-source path returns the persisted origin URL for **direct** client
playback (bridge, never rehost). This module adds an *opt-in* freshness check: a HEAD that
follows redirects / tracking prefixes and reports the resolved final URL + content
type/length, so a client or operator can confirm an episode is playable. The no-store
pass-through proxy remains deferred until a host actually blocks direct play.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class AudioResolution:
    """Outcome of validating an origin enclosure URL."""

    verified: bool
    final_url: str
    content_type: str | None = None
    content_length: int | None = None


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
