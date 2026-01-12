"""Caching utilities for preprocessed audio files."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

PREPROCESSING_CACHE_DIR = ".cache/preprocessing"


def get_cached_audio_path(
    cache_key: str,
    cache_dir: str = PREPROCESSING_CACHE_DIR,
) -> Optional[str]:
    """Check if preprocessed audio exists in cache.

    Args:
        cache_key: Content-based cache key
        cache_dir: Cache directory path

    Returns:
        Path to cached audio if exists, None otherwise
    """
    cache_path = os.path.join(cache_dir, f"{cache_key}.opus")
    if os.path.exists(cache_path):
        logger.debug("Cache hit for preprocessed audio: %s", cache_key)
        return cache_path
    logger.debug("Cache miss for preprocessed audio: %s", cache_key)
    return None


def save_to_cache(
    source_path: str,
    cache_key: str,
    cache_dir: str = PREPROCESSING_CACHE_DIR,
) -> str:
    """Save preprocessed audio to cache.

    Args:
        source_path: Path to preprocessed audio
        cache_key: Content-based cache key
        cache_dir: Cache directory path

    Returns:
        Path to cached audio
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.opus")

    # Copy to cache
    import shutil

    shutil.copy2(source_path, cache_path)
    logger.debug("Saved preprocessed audio to cache: %s", cache_path)
    return cache_path
