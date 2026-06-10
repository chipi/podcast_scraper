"""Durable raw-audio cache for reprocessing (#947).

Episode audio is downloaded, transcribed, then discarded — so re-diarization (#876)
and any reprocess must re-fetch from the live RSS feed, which fails once an episode
rolls off the feed. This module persists the **raw** downloaded audio keyed by the
episode **GUID**, so a later reprocess reads it locally instead of re-fetching.

Unlike the image cache (``corpus_artwork``), which is content-addressed by the bytes
it just fetched, this cache is **GUID-addressed**: we must be able to look up an
episode's audio *before* downloading it. Files live, sharded, under::

    <root>/sha256/<aa>/<bb>/<sha256(guid)><ext>

``<root>`` is external to the corpus by default (lean backups); set
``audio_cache_in_corpus`` to keep it under ``<corpus>/.podcast_scraper/audio-cache``
for a self-contained snapshot.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from .. import config_constants
from .corpus_media import AUDIO_MEDIA_EXTENSIONS
from .path_validation import safe_relpath_under_corpus_root

logger = logging.getLogger(__name__)

# In-corpus location (POSIX, relative to the corpus/output root).
IN_CORPUS_AUDIO_CACHE_REL = ".podcast_scraper/audio-cache"

# Extensions probed on lookup (known audio + a generic fallback).
_LOOKUP_EXTENSIONS = tuple(AUDIO_MEDIA_EXTENSIONS) + (".bin",)


def resolve_cache_root(cfg, effective_output_dir: Optional[str]) -> Optional[Path]:
    """Resolve the audio-cache root for this run, or None when disabled.

    - ``audio_cache_enabled=False`` -> None (caching off).
    - ``audio_cache_in_corpus=True`` -> ``<corpus>/.podcast_scraper/audio-cache``.
    - explicit ``audio_cache_dir`` -> that path (``~`` expanded).
    - otherwise the external default (``DEFAULT_AUDIO_CACHE_DIR``).
    """
    if not getattr(cfg, "audio_cache_enabled", True):
        return None
    if getattr(cfg, "audio_cache_in_corpus", False):
        if not effective_output_dir:
            logger.debug("audio cache: in-corpus requested but no output_dir; disabling")
            return None
        dest = safe_relpath_under_corpus_root(Path(effective_output_dir), IN_CORPUS_AUDIO_CACHE_REL)
        return Path(dest) if dest else None
    custom = getattr(cfg, "audio_cache_dir", None)
    base = custom if custom else config_constants.DEFAULT_AUDIO_CACHE_DIR
    return Path(base).expanduser()


def _guid_digest(guid: Optional[str]) -> Optional[str]:
    if not guid or not str(guid).strip():
        return None
    return hashlib.sha256(str(guid).strip().encode("utf-8")).hexdigest()


def cache_path_for_guid(root: Path, guid: str, ext: str) -> Optional[Path]:
    """Sharded destination path for a GUID + extension (no I/O)."""
    digest = _guid_digest(guid)
    if digest is None:
        return None
    ext = ext if ext.startswith(".") else f".{ext}"
    return root / "sha256" / digest[:2] / digest[2:4] / f"{digest}{ext}"


def lookup_by_guid(root: Optional[Path], guid: Optional[str]) -> Optional[str]:
    """Return the cached audio path for a GUID (any known ext), or None on miss."""
    if root is None:
        return None
    digest = _guid_digest(guid)
    if digest is None:
        return None
    shard = root / "sha256" / digest[:2] / digest[2:4]
    for ext in _LOOKUP_EXTENSIONS:
        candidate = shard / f"{digest}{ext}"
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return str(candidate)
        except OSError:
            continue
    return None


def store(root: Optional[Path], guid: Optional[str], src_path: str) -> Optional[str]:
    """Copy ``src_path`` into the cache keyed by ``guid``; return cache path or None.

    Best-effort and idempotent: a non-empty existing entry is left untouched
    (dedupe-by-existence), and any failure logs + returns None without raising.
    The copy is staged to a ``.tmp`` sibling then atomically renamed so a partial
    or racing write never leaves a torn cache file.
    """
    if root is None:
        return None
    digest = _guid_digest(guid)
    if digest is None:
        logger.debug("audio cache: skipping store, episode has no GUID")
        return None
    if not src_path or not os.path.isfile(src_path):
        return None
    try:
        if os.path.getsize(src_path) <= 0:
            return None
    except OSError:
        return None
    ext = os.path.splitext(src_path)[1] or ".bin"
    dest = cache_path_for_guid(root, str(guid), ext)
    if dest is None:
        return None
    try:
        if dest.is_file() and dest.stat().st_size > 0:
            return str(dest)  # already cached
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(src_path, tmp)
        os.replace(tmp, dest)
    except OSError as exc:
        logger.warning("audio cache: failed to store %s: %s", src_path, exc)
        return None
    return str(dest)


def copy_into(cached_path: str, dest_path: str) -> bool:
    """Copy a cache hit into the run's temp media path so the pipeline is unchanged."""
    try:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        shutil.copy2(cached_path, dest_path)
        return True
    except OSError as exc:
        logger.warning("audio cache: failed to copy %s -> %s: %s", cached_path, dest_path, exc)
        return False
