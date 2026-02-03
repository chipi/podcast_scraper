"""Utility functions for ML model cache directory management.

This module provides functions to get cache directories for ML models,
preferring a local cache directory in the project root if it exists,
falling back to the standard user cache directories.
"""

import os
from pathlib import Path
from typing import Optional

# Cache the project root to avoid repeated filesystem operations
_project_root: Optional[Path] = None


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to the project root (directory containing pyproject.toml)
    """
    global _project_root
    if _project_root is not None:
        return _project_root

    # Start from this file and walk up to find pyproject.toml
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / "pyproject.toml").exists():
            _project_root = current
            return _project_root
        current = current.parent
    # Fallback: assume we're in src/podcast_scraper/ and go up 2 levels
    _project_root = Path(__file__).resolve().parent.parent.parent
    return _project_root


def get_whisper_cache_dir() -> Path:
    """Get Whisper model cache directory.

    Priority order:
    1. WHISPER_CACHE_DIR environment variable (highest priority)
    2. CACHE_DIR environment variable (if WHISPER_CACHE_DIR not set)
    3. Local cache in project root (.cache/whisper/) if exists
    4. Standard user cache (~/.cache/whisper/)

    Returns:
        Path to Whisper cache directory
    """
    # 1. Check WHISPER_CACHE_DIR env var FIRST (highest priority)
    whisper_cache = os.environ.get("WHISPER_CACHE_DIR")
    if whisper_cache:
        return Path(whisper_cache)

    # 2. Check CACHE_DIR env var (general cache directory)
    cache_dir = os.environ.get("CACHE_DIR")
    if cache_dir:
        return Path(cache_dir) / "whisper"

    # 3. Check local cache (development convenience)
    project_root = get_project_root()
    local_cache = project_root / ".cache" / "whisper"
    if local_cache.exists():
        return local_cache

    # 4. Standard user cache as final fallback
    return Path.home() / ".cache" / "whisper"


def get_transformers_cache_dir() -> Path:
    """Get Transformers model cache directory.

    Priority order:
    1. HF_HUB_CACHE environment variable (CI sets this explicitly - highest priority)
    2. CACHE_DIR/huggingface/hub (if CACHE_DIR is set)
    3. Local cache in project root (.cache/huggingface/hub/) if exists
    4. huggingface_hub.constants.HF_HUB_CACHE (respects HF_HOME env var)
    5. Standard user cache (~/.cache/huggingface/hub/)

    The env var takes priority because CI explicitly sets it to ensure consistent
    cache paths across all workers and steps. This is critical for supply chain
    security - we want to use exactly the cache that CI validated.

    Returns:
        Path to Transformers cache directory
    """
    # 1. Check HF_HUB_CACHE env var FIRST (CI sets this explicitly)
    # This takes priority because CI explicitly sets it to ensure consistent paths
    # and we want to use exactly the cache that CI validated
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache)

    # 2. Check CACHE_DIR (base cache directory) and derive path from it
    cache_dir = os.environ.get("CACHE_DIR")
    if cache_dir:
        return Path(cache_dir) / "huggingface" / "hub"

    # 3. Check local cache (development convenience)
    project_root = get_project_root()
    local_cache = project_root / ".cache" / "huggingface" / "hub"
    if local_cache.exists():
        return local_cache

    # 3. Fall back to huggingface_hub constants
    try:
        # Try modern huggingface_hub API first (transformers 4.20+)
        from huggingface_hub import constants

        if constants.HF_HUB_CACHE:
            return Path(constants.HF_HUB_CACHE)
    except (ImportError, AttributeError, TypeError):
        pass

    try:
        # Fallback to transformers file_utils (older versions)
        from transformers import file_utils

        if hasattr(file_utils, "default_cache_path") and file_utils.default_cache_path:
            return Path(file_utils.default_cache_path)
    except (ImportError, AttributeError, TypeError):
        pass

    # 4. Standard user cache as final fallback
    return Path.home() / ".cache" / "huggingface" / "hub"


def get_spacy_cache_dir() -> Optional[Path]:
    """Get spaCy model cache directory.

    Note: spaCy models are typically installed as Python packages, so this
    may return None if no user cache exists.

    Returns:
        Path to spaCy cache directory, or None if not available
    """
    project_root = get_project_root()
    local_cache = project_root / ".cache" / "spacy"
    if local_cache.exists():
        return local_cache
    # Fallback to standard user cache
    user_cache = Path.home() / ".local" / "share" / "spacy"
    if user_cache.exists():
        return user_cache
    return None
