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

    Prefers local cache in project root (.cache/whisper/), falls back to ~/.cache/whisper/

    Returns:
        Path to Whisper cache directory
    """
    project_root = get_project_root()
    local_cache = project_root / ".cache" / "whisper"
    if local_cache.exists():
        return local_cache
    # Fallback to standard user cache
    return Path.home() / ".cache" / "whisper"


def get_transformers_cache_dir() -> Path:
    """Get Transformers model cache directory.

    Priority order:
    1. Local cache in project root (.cache/huggingface/hub/) if exists
    2. HF_HUB_CACHE environment variable (CI sets this explicitly)
    3. huggingface_hub.constants.HF_HUB_CACHE (respects HF_HOME env var)
    4. Standard user cache (~/.cache/huggingface/hub/)

    Returns:
        Path to Transformers cache directory
    """
    # 1. Check local cache first (development preference)
    project_root = get_project_root()
    local_cache = project_root / ".cache" / "huggingface" / "hub"
    if local_cache.exists():
        return local_cache

    # 2. Check HF_HUB_CACHE env var (CI sets this explicitly)
    # This is checked BEFORE importing huggingface_hub because the constants
    # are evaluated at import time and may not reflect runtime env changes
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache)

    # 3. Fall back to huggingface_hub constants
    try:
        # Try modern huggingface_hub API first (transformers 4.20+)
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except ImportError:
        try:
            # Fallback to transformers file_utils (older versions)
            from transformers import file_utils

            return Path(file_utils.default_cache_path)
        except (ImportError, AttributeError):
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
