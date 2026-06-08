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
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_PATH = "/api/tags"
DEFAULT_TIMEOUT_SEC = 5.0
FASTER_WHISPER_HEALTH_PATH = "/v1/models"


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


def check_faster_whisper_health(
    host: str,
    *,
    port: int = 8000,
    path: str = FASTER_WHISPER_HEALTH_PATH,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    require_model_substring: Optional[str] = None,
) -> bool:
    """Return True when faster-whisper-server responds 200 on ``/v1/models``.

    faster-whisper-server's OpenAI-compatible ``/v1/models`` returns a list of
    available models in the standard OpenAI envelope:
    ``{"object": "list", "data": [{"id": "...", ...}, ...]}``.

    ``require_model_substring`` lets callers assert a specific model is loaded
    (e.g. ``Systran/faster-whisper-large-v3``) — handy as a pre-batch gate.
    """
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed; DGX Whisper health check skipped")
        return False

    base = dgx_whisper_base_url(host, port)
    url = f"{base}{path if path.startswith('/') else '/' + path}"
    try:
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.get(url)
        if resp.status_code != 200:
            return False
        if not require_model_substring:
            return True
        # faster-whisper-server / OpenAI envelope: {"data": [{"id": "..."}, ...]}
        data = resp.json()
        models = data.get("data") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return False
        needle = require_model_substring.lower()
        for item in models:
            if isinstance(item, dict):
                mid = item.get("id", "")
                if isinstance(mid, str) and needle in mid.lower():
                    return True
        return False
    except Exception as exc:
        logger.debug("DGX faster-whisper health check failed: %s", exc)
        return False
