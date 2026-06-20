"""Fetch + cache ``SYSTEM_PROMPT.txt`` from a HuggingFace model repo.

Some vendors (notably the Mistral family — Mistral-Small-3.2, Magistral,
Ministral) publish a vendor-specific system prompt at
``SYSTEM_PROMPT.txt`` in the model repo's root. Loading that as the
system prompt is a documented requirement for those models — using a
generic system prompt instead causes subtle text-quality drift and
strips the assistant priming the vendor expects (see
`autoresearch/MODEL_PLAYBOOK.md` § Mistral family).

This module provides a small fetch + cache helper. Callers decide
*when* to use the result (per-provider, per-config), so this is
strictly an I/O utility with no provider wiring.

Refs: task #109.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# Default cache layout — sibling to the HuggingFace transformers cache so
# ops can clean both with the same tool. Override with the ``cache_dir``
# argument or the ``HF_SYSTEM_PROMPT_CACHE`` env var (in that priority order).
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "podcast_scraper" / "hf_system_prompts"

# Files we treat as the vendor's system prompt. Order matters: the first
# hit wins. ``SYSTEM_PROMPT.txt`` is the documented Mistral convention.
_SYSTEM_PROMPT_FILE_CANDIDATES: tuple[str, ...] = ("SYSTEM_PROMPT.txt",)


def _resolve_cache_dir(cache_dir: Optional[Path]) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    env_dir = os.environ.get("HF_SYSTEM_PROMPT_CACHE")
    if env_dir:
        return Path(env_dir)
    return _DEFAULT_CACHE_DIR


def _safe_cache_key(model_id: str, filename: str) -> str:
    """Turn ``mistralai/Mistral-Small-3.2`` + ``SYSTEM_PROMPT.txt`` into a
    filesystem-safe cache filename. Avoid path traversal by replacing
    every separator with ``__``."""
    safe_id = model_id.replace("/", "__").replace("\\", "__").replace("..", "_")
    return f"{safe_id}__{filename}"


def _hf_resolve_url(model_id: str, filename: str, revision: str = "main") -> str:
    """Return the public HuggingFace resolve URL for a file in a model repo."""
    return f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"


def load_hf_system_prompt(
    model_id: str,
    *,
    cache_dir: Optional[Path] = None,
    revision: str = "main",
    timeout: float = 10.0,
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """Return the contents of the vendor's ``SYSTEM_PROMPT.txt`` for ``model_id``,
    or ``None`` if the model doesn't publish one.

    Caches the file to ``cache_dir`` (or
    ``$HF_SYSTEM_PROMPT_CACHE`` / ``~/.cache/podcast_scraper/hf_system_prompts``)
    so subsequent calls for the same model don't re-fetch.

    Args:
        model_id: The HuggingFace repo id (e.g. ``mistralai/Mistral-Small-3.2-24B-Instruct-2506``).
        cache_dir: Override the default cache location.
        revision: HF git revision to fetch from. Default ``main``.
        timeout: HTTP timeout in seconds.
        hf_token: Optional bearer token for private/gated repos. If unset,
            falls back to the ``HF_TOKEN`` env var.

    Returns:
        The file contents as a string, or None when the model has no
        published system prompt (404 on every candidate filename) or when
        the network fetch fails. Errors are logged at WARNING; callers
        should treat None as "use the default prompt path".

    Cache behaviour:
        - Empty model_id raises ValueError (programmer error).
        - Cached file present → returned without an HTTP call.
        - Cache miss → fetches all candidate filenames in order, caches
          the first one that returns 200, returns its contents.
        - Cache miss + 404 on every candidate → caches a sentinel marker
          file so subsequent calls don't re-fetch; returns None.
        - Cache miss + network error → returns None WITHOUT caching the
          sentinel, so the next call retries.
    """
    if not model_id:
        raise ValueError("model_id must be non-empty")

    cache_root = _resolve_cache_dir(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Sentinel marker for models that have been confirmed to NOT publish
    # a system prompt. Avoids re-fetching on every call.
    sentinel = cache_root / f"{_safe_cache_key(model_id, '__NONE__')}.sentinel"
    if sentinel.exists():
        return None

    for filename in _SYSTEM_PROMPT_FILE_CANDIDATES:
        cache_path = cache_root / _safe_cache_key(model_id, filename)
        if cache_path.exists():
            try:
                return cache_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "load_hf_system_prompt: cache read failed for %s: %s",
                    cache_path,
                    exc,
                )
                # Fall through to a fresh fetch.

        url = _hf_resolve_url(model_id, filename, revision=revision)
        token = hf_token or os.environ.get("HF_TOKEN")
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = requests.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            logger.warning(
                "load_hf_system_prompt: network error fetching %s: %s",
                url,
                exc,
            )
            return None

        if response.status_code == 200:
            content = response.text
            try:
                cache_path.write_text(content, encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "load_hf_system_prompt: cache write failed for %s: %s",
                    cache_path,
                    exc,
                )
            return content
        if response.status_code == 404:
            # File not found at this candidate name; try the next.
            continue
        logger.warning(
            "load_hf_system_prompt: HTTP %d fetching %s",
            response.status_code,
            url,
        )
        return None

    # Every candidate filename came back 404 → vendor hasn't published a
    # system prompt for this model. Cache the sentinel so we don't retry.
    try:
        sentinel.write_text("", encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "load_hf_system_prompt: sentinel write failed: %s",
            exc,
        )
    return None


__all__ = ["load_hf_system_prompt"]
