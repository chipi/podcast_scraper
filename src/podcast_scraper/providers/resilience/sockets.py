"""HTTP-client hardening + audio duration helpers used by the resilience layer.

The HTTP-client helper here was originally named ``dgx_http_client`` and lived
in ``tailnet_dgx/resilience.py``. It's not actually DGX-specific — it adds
TCP keepalive + ``Connection: close`` so a long-blocking HTTP POST to any
remote inference service doesn't hang forever if the underlying network path
dies mid-request. Renamed to :func:`hardened_http_client` to reflect what it
does, not where it happens to run today.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# TCP keepalive schedule: probe an idle socket after ~30s, then every 15s up to 4
# times, so a connection whose underlying path died mid-request (e.g. a Tailscale
# path switch) is declared dead in ~90s instead of the OS default (2h on macOS).
_KEEPALIVE_IDLE_SEC = 30
_KEEPALIVE_INTERVAL_SEC = 15
_KEEPALIVE_PROBES = 4


def probe_audio_duration_sec(audio_path: str) -> Optional[float]:
    """Best-effort audio duration (seconds) for timeout scaling; None on failure.

    Used to size the request timeout for audio-upload calls (whisper, pyannote).
    A miss is harmless — it just falls back to the flat base budget. Dependency-
    light (soundfile is already a transitive dep) and never raises.
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


def keepalive_socket_options(
    *,
    idle_sec: int = _KEEPALIVE_IDLE_SEC,
    interval_sec: int = _KEEPALIVE_INTERVAL_SEC,
    probes: int = _KEEPALIVE_PROBES,
) -> "list[tuple[int, int, int]]":
    """TCP keepalive ``setsockopt`` tuples, built defensively for the host platform.

    A long-blocking inference POST holds an idle socket open for the whole
    multi-minute compute run; if the path dies underneath it the socket stays
    ESTABLISHED for the OS default and the request hangs. Enabling
    ``SO_KEEPALIVE`` with a short probe schedule lets the kernel detect the
    dead path in ~90s and surface a connection error to fail over on.
    Each option is guarded with ``hasattr`` so a platform-absent constant is
    skipped, not raised (Linux exposes ``TCP_KEEPIDLE``; macOS exposes
    ``TCP_KEEPALIVE``).
    """
    import socket

    opts: "list[tuple[int, int, int]]" = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for name in ("TCP_KEEPIDLE", "TCP_KEEPALIVE"):  # Linux, then macOS
        if hasattr(socket, name):
            opts.append((socket.IPPROTO_TCP, getattr(socket, name), idle_sec))
            break
    if hasattr(socket, "TCP_KEEPINTVL"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval_sec))
    if hasattr(socket, "TCP_KEEPCNT"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, probes))
    return opts


def hardened_http_client(
    timeout: Any,
    *,
    headers: "Optional[dict[str, str]]" = None,
    subsystem: str = "dgx_inference",
) -> Any:
    """An ``httpx.Client`` hardened for long-blocking calls to a contended inference service.

    On top of the per-request fresh client + :func:`run_with_watchdog` deadline, this adds
    the two transport-level defences that actually fit a long-blocking upload:

    - **TCP keepalive** (:func:`keepalive_socket_options`) so a socket whose path died
      mid-request is reaped in ~90s instead of hanging on the OS default.
    - **``Connection: close``** so the server tears the socket down after the response and
      no half-dead connection lingers.

    A per-read timeout is deliberately NOT set: these POSTs stream zero bytes during the
    multi-minute GPU run, so any read deadline shorter than processing would false-abort a
    healthy call — the duration-scaled watchdog is the correct backstop.

    Delegates transport construction to
    :func:`podcast_scraper.net.outbound_http.create_client` so the DGX Whisper +
    diarize multipart POSTs (this helper's only production consumers) honor
    admin-configured proxy (#1129) and TLS trust (#1130) — including
    ``verify=False`` and mTLS (``client_cert``/``client_key``) which env-mirror
    doesn't cover. Before that wiring, those largest-egress requests silently
    bypassed the outbound registry.

    Was named ``dgx_http_client`` until 2026-06-15; renamed to reflect that the helper
    has nothing DGX-specific in it.
    """
    try:
        import httpx  # noqa: F401 - fail loud on missing dep before touching factory
    except ImportError as exc:  # friendly, actionable error
        raise RuntimeError("httpx required for hardened_http_client") from exc

    from podcast_scraper.net.outbound_http import create_client

    merged: dict[str, str] = {"Connection": "close"}
    if headers:
        merged.update(headers)
    return create_client(
        subsystem=subsystem,
        socket_options=keepalive_socket_options(),
        timeout=timeout,
        headers=merged,
    )


__all__ = [
    "effective_timeout_sec",
    "hardened_http_client",
    "keepalive_socket_options",
    "probe_audio_duration_sec",
]
