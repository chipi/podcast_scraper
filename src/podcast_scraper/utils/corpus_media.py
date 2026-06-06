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


def audio_relpath_for_transcript(transcript_relpath: str, media_ext: str = ".mp3") -> str:
    """Derive ``media/<stem><ext>`` from a ``transcripts/…/*.txt`` relpath."""
    norm = transcript_relpath.strip().replace("\\", "/").lstrip("/")
    path = Path(norm)
    stem = path.stem
    if path.parent.name:
        return f"{CORPUS_MEDIA_DIR}/{stem}{media_ext}"
    return f"{CORPUS_MEDIA_DIR}/{stem}{media_ext}"


def persist_episode_media(
    source_media_path: str,
    effective_output_dir: str,
    transcript_relpath: str,
) -> Optional[str]:
    """Copy downloaded audio into ``media/``; return corpus-relative path or None."""
    if not source_media_path or not os.path.isfile(source_media_path):
        return None
    ext = os.path.splitext(source_media_path)[1] or ".mp3"
    dest_relpath = audio_relpath_for_transcript(transcript_relpath, media_ext=ext)
    dest_abs = os.path.join(effective_output_dir, dest_relpath)
    try:
        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
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
    inferred = audio_relpath_for_transcript(transcript_file_path)
    if os.path.isfile(os.path.join(effective_output_dir, inferred)):
        return inferred
    return None
