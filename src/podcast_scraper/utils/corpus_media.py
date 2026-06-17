"""Corpus episode media paths (Wave 3 — local viewer playback)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from ..utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)

CORPUS_MEDIA_DIR = "media"

# Audio container extensions we may persist (source ext is preserved on copy).
# Ordered so the common podcast formats resolve first.
AUDIO_MEDIA_EXTENSIONS = (".mp3", ".m4a", ".aac", ".opus", ".ogg", ".wav", ".flac", ".mp4", ".webm")


def audio_relpath_for_transcript(transcript_relpath: str, media_ext: str = ".mp3") -> str:
    """Derive ``media/<stem><ext>`` from a ``transcripts/…/*.txt`` relpath.

    Media is flattened into a single ``media/`` directory keyed by the transcript
    stem (the persist step does the same), so subdirectories are intentionally
    dropped here.
    """
    norm = transcript_relpath.strip().replace("\\", "/").lstrip("/")
    stem = Path(norm).stem
    return f"{CORPUS_MEDIA_DIR}/{stem}{media_ext}"


def _same_filesystem(src: str, dest_abs: str) -> bool:
    """True when ``src`` and ``dest_abs``'s directory share a device (hardlink-able)."""
    try:
        return os.stat(src).st_dev == os.stat(os.path.dirname(dest_abs) or ".").st_dev
    except OSError:
        return False


def _link_episode_media(
    link_source: Optional[str],
    link_mode: str,
    dest_abs: str,
    corpus_root: str,
) -> bool:
    """Hard/sym-link ``dest_abs`` to ``link_source``; return True only on success (G6).

    Returns False — so the caller copies instead — for ``copy`` mode, a missing link
    source, a cross-filesystem hardlink, a symlink whose target resolves outside the
    corpus (the viewer media route's realpath guard would 404 it), or any OS error.
    """
    mode = (link_mode or "copy").lower()
    if mode not in ("hardlink", "symlink"):
        return False
    if not link_source or not os.path.isfile(link_source):
        return False
    try:
        if mode == "hardlink":
            if not _same_filesystem(link_source, dest_abs):
                return False
            target: str = link_source
        else:  # symlink — only safe when the target stays under the corpus root
            real_src = os.path.realpath(link_source)
            real_root = os.path.realpath(corpus_root)
            if not (real_src == real_root or real_src.startswith(real_root + os.sep)):
                return False
            # Relative target so an in-corpus snapshot stays self-contained when moved
            # or tar-backed-up (the whole point of audio_cache_in_corpus).
            target = os.path.relpath(os.path.abspath(link_source), os.path.dirname(dest_abs))
        if os.path.lexists(dest_abs):
            os.remove(dest_abs)
        if mode == "hardlink":
            os.link(target, dest_abs)
        else:
            os.symlink(target, dest_abs)
        logger.debug("Linked episode media (%s): %s -> %s", mode, dest_abs, link_source)
        return True
    except OSError as exc:
        logger.debug("Media %s failed (%s); falling back to copy", mode, exc)
        return False


def persist_episode_media(
    source_media_path: str,
    effective_output_dir: str,
    transcript_relpath: str,
    *,
    link_source: Optional[str] = None,
    link_mode: str = "copy",
) -> Optional[str]:
    """Persist downloaded audio into ``media/``; return corpus-relative path or None.

    By default the audio is copied. When ``link_mode`` is ``"hardlink"``/``"symlink"``
    and ``link_source`` (e.g. the retained #947 GUID audio-cache entry) is available,
    the corpus file is linked to it instead — halving the on-disk audio footprint when
    the cache is on the same filesystem (G6). Linking falls back to a copy on any
    failure, so persistence never breaks because of the optimisation.
    """
    if not source_media_path or not os.path.isfile(source_media_path):
        return None
    ext = os.path.splitext(source_media_path)[1] or ".mp3"
    dest_relpath = audio_relpath_for_transcript(transcript_relpath, media_ext=ext)
    dest_abs = os.path.join(effective_output_dir, dest_relpath)
    try:
        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
        if not _link_episode_media(link_source, link_mode, dest_abs, effective_output_dir):
            shutil.copy2(source_media_path, dest_abs)
        logger.debug("Persisted episode media: %s", dest_relpath)
        return dest_relpath
    except OSError as exc:
        logger.warning(
            "Failed to persist episode media to %s: %s",
            dest_relpath,
            format_exception_for_log(exc),
        )
        return None


def resolve_audio_relpath_for_metadata(
    effective_output_dir: str,
    transcript_file_path: Optional[str],
    explicit_relpath: Optional[str] = None,
) -> Optional[str]:
    """Return ``audio_relpath`` when the media file exists under the corpus root."""
    if explicit_relpath:
        candidate = explicit_relpath.strip().replace("\\", "/").lstrip("/")
        if os.path.isfile(os.path.join(effective_output_dir, candidate)):
            return candidate
    if not transcript_file_path:
        return None
    # The persisted file keeps the *source* extension (often .m4a), so probe every
    # known audio extension for the transcript stem rather than assuming .mp3.
    norm = transcript_file_path.strip().replace("\\", "/").lstrip("/")
    stem = Path(norm).stem
    for ext in AUDIO_MEDIA_EXTENSIONS:
        relpath = f"{CORPUS_MEDIA_DIR}/{stem}{ext}"
        if os.path.isfile(os.path.join(effective_output_dir, relpath)):
            return relpath
    return None
