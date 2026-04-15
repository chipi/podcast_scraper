"""Caching utilities for transcript files.

This module provides transcript caching by audio hash to enable fast
multi-provider experimentation without re-transcribing the same audio.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from podcast_scraper.utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)

# Use constant for consistency with other cache directories
TRANSCRIPT_CACHE_DIR = ".cache/transcripts"


def get_audio_hash(audio_path: str) -> str:
    """Generate hash of audio file (first 1MB for speed).

    Args:
        audio_path: Path to audio file

    Returns:
        SHA256 hash (first 16 hex chars for cache key)
    """
    hasher = hashlib.sha256()
    try:
        with open(audio_path, "rb") as f:
            # Hash first 1MB for performance (same as preprocessing cache)
            hasher.update(f.read(1024 * 1024))
    except OSError as exc:
        logger.warning(
            "Failed to hash audio file for transcript cache: %s",
            format_exception_for_log(exc),
        )
        # Use file path as fallback (not ideal, but better than failing)
        hasher.update(audio_path.encode("utf-8"))
    return hasher.hexdigest()[:16]  # 16 hex chars (64 bits)


def _parse_segments_from_cache(
    cache_data: Dict[str, Any],
    cache_path: str,
) -> Optional[List[Dict[str, Any]]]:
    """Return validated segment list from cache payload, or None."""
    if "segments" not in cache_data:
        return None
    raw: Any = cache_data["segments"]
    if raw is None:
        return None
    if not isinstance(raw, list):
        logger.warning(
            "Cached transcript has invalid 'segments' (expected list of objects): %s",
            cache_path,
        )
        return None
    if not raw:
        return None
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            logger.warning(
                "Cached transcript 'segments' must be a list of objects; ignoring: %s",
                cache_path,
            )
            return None
        out.append(item)
    return out


def get_cached_transcript_entry(
    audio_hash: str,
    cache_dir: str = TRANSCRIPT_CACHE_DIR,
) -> Optional[Tuple[str, Optional[List[Dict[str, Any]]]]]:
    """Load transcript and optional segments from cache.

    Args:
        audio_hash: Hash of audio file (from get_audio_hash())
        cache_dir: Cache directory path

    Returns:
        ``(transcript, segments)`` on hit (segments may be None), else None.
    """
    cache_path = os.path.join(cache_dir, f"{audio_hash}.json")
    if not os.path.exists(cache_path):
        logger.debug("Transcript cache miss: %s", audio_hash)
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data: Dict[str, Any] = json.load(f)
        transcript: Any = cache_data.get("transcript")
        if not transcript or not isinstance(transcript, str):
            logger.warning("Cached transcript file missing 'transcript' field: %s", cache_path)
            return None
        segments = _parse_segments_from_cache(cache_data, cache_path)
        logger.debug("Transcript cache hit: %s", audio_hash)
        return (str(transcript), segments)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        logger.warning(
            "Failed to read cached transcript: %s",
            format_exception_for_log(exc),
        )
        return None


def get_cached_transcript(
    audio_hash: str,
    cache_dir: str = TRANSCRIPT_CACHE_DIR,
) -> Optional[str]:
    """Check if transcript exists in cache.

    Args:
        audio_hash: Hash of audio file (from get_audio_hash())
        cache_dir: Cache directory path

    Returns:
        Transcript text if cache hit, None otherwise
    """
    entry = get_cached_transcript_entry(audio_hash, cache_dir)
    if entry is None:
        return None
    return entry[0]


def save_transcript_to_cache(
    audio_hash: str,
    transcript: str,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    cache_dir: str = TRANSCRIPT_CACHE_DIR,
    segments: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Save transcript to cache.

    Args:
        audio_hash: Hash of audio file (from get_audio_hash())
        transcript: Transcript text to cache
        provider_name: Optional provider name (e.g., "whisper", "openai")
        model: Optional model identifier (e.g., "large-v3", "whisper-1")
        cache_dir: Cache directory path
        segments: Optional Whisper-style segments for GI timestamp mapping
            (``start``/``end`` seconds, ``text`` per item); omitted from JSON when empty

    Returns:
        Path to cached transcript file
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{audio_hash}.json")

    cache_data: Dict[str, Any] = {
        "transcript": transcript,
        "cached_at": datetime.utcnow().isoformat(),
    }
    if provider_name:
        cache_data["provider"] = provider_name
    if model:
        # Ensure model is a string (defensive check for non-serializable objects)
        if isinstance(model, str):
            cache_data["model"] = model
        else:
            # Convert to string representation if not already a string
            # This handles cases where a model object might be passed instead of a string
            cache_data["model"] = str(model)
            logger.warning(
                "Model passed to cache was not a string, converted to string: %s",
                type(model).__name__,
            )
    if segments:
        cache_data["segments"] = segments

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                cache_data,
                f,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        logger.debug("Saved transcript to cache: %s", cache_path)
        return cache_path
    except OSError as exc:
        logger.warning(
            "Failed to save transcript to cache: %s",
            format_exception_for_log(exc),
        )
        raise
    except (TypeError, ValueError) as exc:
        logger.warning(
            "Failed to serialize transcript cache (segments not JSON-serializable?): %s",
            format_exception_for_log(exc),
        )
        raise
