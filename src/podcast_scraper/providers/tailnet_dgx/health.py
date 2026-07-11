"""Health checks for tailnet DGX services (RFC-089 / #814).

Two services live on DGX behind the tailnet ACL:

- **Ollama** on ``:11434`` — LLM stages. Health endpoint: ``/api/tags``.
- **faster-whisper-server** on ``:8000`` — transcription stage (#814).
  Health endpoint: OpenAI-compatible ``/v1/models``.

This module exposes one helper per service. They share the same shape (return
True/False + log on failure) so callers can compose them.
"""

from __future__ import annotations

import logging
import socket
from enum import Enum
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_PATH = "/api/tags"
DEFAULT_TIMEOUT_SEC = 5.0
# TCP-connect liveness budget. A reachable host completes the handshake in well under a
# second even while the GPU is busy, so this only needs to cover network latency.
DEFAULT_CONNECT_TIMEOUT_SEC = 3.0
# Read budget for the optional busy-vs-ready refinement. An idle server answers /v1/models
# in ~10ms; anything slower than this means the response is queued behind an in-flight job.
DEFAULT_READY_READ_TIMEOUT_SEC = 4.0
FASTER_WHISPER_HEALTH_PATH = "/v1/models"


class DgxEndpointStatus(str, Enum):
    """Liveness/readiness of a single-flight DGX HTTP service (#956).

    Distinguishing these is the whole point: the faster-whisper / pyannote servers process
    one request at a time on the GPU, so an HTTP health GET *queues behind the in-flight
    transcription*. A flat-timeout check then reads a merely-BUSY box as DOWN and triggers a
    spurious fallback. We classify with a TCP connect (liveness) + a short read (readiness).
    """

    READY = "ready"  # listening and answered /v1/models promptly (idle / free)
    BUSY = "busy"  # listening (TCP accept) but the HTTP response is queued behind a job
    DOWN = "down"  # not listening / unreachable


def _bare_host(host: str) -> str:
    """Strip scheme/port/path so ``host`` can be used for a raw ``socket`` connect."""
    h = (host or "").strip()
    for prefix in ("http://", "https://"):
        if h.startswith(prefix):
            h = h[len(prefix) :]
    return h.rstrip("/").split("/", 1)[0].split(":", 1)[0]


def tcp_endpoint_listening(
    host: str, port: int, *, timeout_sec: float = DEFAULT_CONNECT_TIMEOUT_SEC
) -> bool:
    """Liveness via a raw TCP connect — UP (listening) vs DOWN (refused/unreachable).

    The kernel completes the TCP handshake as soon as the server ``accept()``s, regardless of
    whether the app is mid-transcription, and this enqueues **nothing** on the single-flight
    job queue. So a busy GPU reads as UP here — never a false ``down`` (#956). This is the
    primitive the health gate and per-request provider checks should use for the
    proceed-vs-fallback decision; ``BUSY`` boxes still get the request (it queues, and the
    request-level duration-scaled timeout handles excessive load).
    """
    target = _bare_host(host)
    try:
        with socket.create_connection((target, port), timeout=timeout_sec):
            return True
    except OSError as exc:
        logger.debug("DGX TCP connect %s:%s failed (down): %s", target, port, exc)
        return False


def probe_dgx_endpoint(
    host: str,
    port: int,
    *,
    path: str = FASTER_WHISPER_HEALTH_PATH,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SEC,
    read_timeout: float = DEFAULT_READY_READ_TIMEOUT_SEC,
) -> DgxEndpointStatus:
    """Classify a DGX endpoint as READY / BUSY / DOWN without false alarms (#956).

    1. TCP connect (non-queuing liveness): refused/unreachable → ``DOWN``.
    2. If listening, a short HTTP GET with a small *read* timeout: a prompt 200 → ``READY``;
       a read timeout (connected, but the response is stuck behind the GPU job) → ``BUSY``.
    """
    if not tcp_endpoint_listening(host, port, timeout_sec=connect_timeout):
        return DgxEndpointStatus.DOWN
    try:
        import httpx
    except ImportError:
        return DgxEndpointStatus.READY  # listening; can't refine without httpx
    base = dgx_whisper_base_url(host, port)
    url = f"{base}{path if path.startswith('/') else '/' + path}"
    try:
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
        return DgxEndpointStatus.READY if resp.status_code == 200 else DgxEndpointStatus.BUSY
    except Exception as exc:
        # TCP connected but no prompt HTTP response → up but queued behind a job → BUSY.
        logger.debug("DGX %s:%s connected but /v1/models slow (busy): %s", host, port, exc)
        return DgxEndpointStatus.BUSY


def dgx_ollama_base_url(host: str, port: int = 11434) -> str:
    """Build ``http://host:port`` base URL for Ollama on the tailnet."""
    h = (host or "").strip().rstrip("/")
    if h.startswith("http://") or h.startswith("https://"):
        return h
    return f"http://{h}:{port}"


def dgx_whisper_base_url(host: str, port: int = 8000) -> str:
    """Build ``http://host:port`` base URL for faster-whisper-server (#814)."""
    h = (host or "").strip().rstrip("/")
    if h.startswith("http://") or h.startswith("https://"):
        return h
    return f"http://{h}:{port}"


def dgx_diarize_base_url(host: str, port: int = 8001) -> str:
    """Build ``http://host:port`` base URL for the pyannote diarize service (#926)."""
    h = (host or "").strip().rstrip("/")
    if h.startswith("http://") or h.startswith("https://"):
        return h
    return f"http://{h}:{port}"


def check_ollama_health(
    host: str,
    *,
    port: int = 11434,
    path: str = DEFAULT_HEALTH_PATH,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    require_model_substring: Optional[str] = None,
) -> bool:
    """Return True when Ollama responds 200 and lists at least one model."""
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed; DGX health check skipped")
        return False

    base = dgx_ollama_base_url(host, port)
    url = f"{base}{path if path.startswith('/') else '/' + path}"
    try:
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.get(url)
        if resp.status_code != 200:
            return False
        data = resp.json()
        models = _extract_model_names(data)
        if not models:
            return False
        if require_model_substring:
            needle = require_model_substring.lower()
            return any(needle in m.lower() for m in models)
        return True
    except Exception as exc:
        logger.debug("DGX Ollama health check failed: %s", exc)
        return False


def _extract_model_names(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    names: List[str] = []
    for item in models:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
    return names


def _check_dgx_http_health(
    host: str,
    port: int,
    *,
    base_url: str,
    path: str,
    timeout_sec: float,
    require_model_substring: Optional[str],
    connect_timeout: float,
) -> bool:
    """Shared liveness + model-verify gate for the single-flight DGX HTTP services (#956).

    Returns True when the service should be *used* (UP), False only when it is genuinely DOWN:

    - Liveness is a **TCP connect** (:func:`tcp_endpoint_listening`) — it never queues on the
      GPU job queue, so a BUSY box reads as UP, not a false ``down`` → no spurious fallback.
    - When ``require_model_substring`` is set we additionally hit ``/v1/models`` to confirm the
      loaded model. If that response is queued behind an in-flight job (slow/non-200) the box
      is up-but-busy, so we trust the model is loaded rather than failing the gate.
    """
    if not tcp_endpoint_listening(host, port, timeout_sec=connect_timeout):
        return False  # genuinely DOWN — connection refused / unreachable
    if not require_model_substring:
        return True  # listening; skip the /v1/models ping (no queue) — liveness is enough
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed; DGX model verification skipped (host is up)")
        return True
    url = f"{base_url}{path if path.startswith('/') else '/' + path}"
    try:
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=timeout_sec,
            write=connect_timeout,
            pool=connect_timeout,
        )
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
        if resp.status_code != 200:
            return True  # up but not answering /v1/models cleanly → busy, not down
        data = resp.json()
        models = data.get("data") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return True  # unexpected envelope from a live server → don't fail the gate
        needle = require_model_substring.lower()
        return any(
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and needle in item["id"].lower()
            for item in models
        )
    except Exception as exc:
        logger.debug(
            "DGX %s:%s up but model probe slow (busy); assuming model loaded: %s",
            host,
            port,
            exc,
        )
        return True  # TCP up but /v1/models queued behind a job → busy, not down


def check_faster_whisper_health(
    host: str,
    *,
    port: int = 8000,
    path: str = FASTER_WHISPER_HEALTH_PATH,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    require_model_substring: Optional[str] = None,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SEC,
) -> bool:
    """Return True when faster-whisper-server is reachable (UP) — busy or idle, not DOWN.

    Liveness is a TCP connect so a GPU-busy server is **not** a false ``down`` (#956).
    ``require_model_substring`` (e.g. ``Systran/faster-whisper-large-v3``) additionally
    verifies the loaded model via ``/v1/models`` when the server answers promptly; a busy
    server is trusted to still have it loaded. See :func:`_check_dgx_http_health`.
    """
    return _check_dgx_http_health(
        host,
        port,
        base_url=dgx_whisper_base_url(host, port),
        path=path,
        timeout_sec=timeout_sec,
        require_model_substring=require_model_substring,
        connect_timeout=connect_timeout,
    )


def check_pyannote_diarize_health(
    host: str,
    *,
    port: int = 8001,
    path: str = FASTER_WHISPER_HEALTH_PATH,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    require_model_substring: Optional[str] = None,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SEC,
) -> bool:
    """Return True when the DGX pyannote service is reachable (UP) — busy or idle, not DOWN.

    Same single-flight semantics as :func:`check_faster_whisper_health`: liveness is a TCP
    connect so a GPU-busy diarize service is not a false ``down`` (#956). The diarize service
    publishes its loaded pipeline id under ``data[].id`` (e.g. the community-1 repo);
    ``require_model_substring`` matches against that pipeline id when the server answers.
    """
    return _check_dgx_http_health(
        host,
        port,
        base_url=dgx_diarize_base_url(host, port),
        path=path,
        timeout_sec=timeout_sec,
        require_model_substring=require_model_substring,
        connect_timeout=connect_timeout,
    )
