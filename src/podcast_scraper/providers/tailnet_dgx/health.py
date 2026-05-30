"""Ollama health checks for tailnet DGX (RFC-089)."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_PATH = "/api/tags"
DEFAULT_TIMEOUT_SEC = 5.0


def dgx_ollama_base_url(host: str, port: int = 11434) -> str:
    """Build ``http://host:port`` base URL for Ollama on the tailnet."""
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
