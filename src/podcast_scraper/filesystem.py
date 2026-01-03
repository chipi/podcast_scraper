"""Filesystem utilities for podcast_scraper."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from platformdirs import user_cache_dir, user_data_dir

from . import config

logger = logging.getLogger(__name__)

TEMP_DIR_NAME = ".tmp_media"
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
URL_HASH_LENGTH = 8
WHISPER_TITLE_MAX_CHARS = 32
EPISODE_NUMBER_FORMAT_WIDTH = 4
TRANSCRIPTS_SUBDIR = "transcripts"
METADATA_SUBDIR = "metadata"
_PLATFORMDIR_APP_NAMES = ("podcast_scraper", "podcast-scraper", "Podcast Scraper")


def _platformdirs_safe_roots() -> set[Path]:
    """Return resolved platformdirs locations considered safe for outputs."""

    roots: set[Path] = set()
    for getter in (user_data_dir, user_cache_dir):
        for app_name in _PLATFORMDIR_APP_NAMES:
            try:
                location = getter(app_name)
            # Fall back to next candidate on failure
            except Exception:  # nosec B112
                continue
            if not location:
                continue
            try:
                resolved = Path(location).expanduser().resolve()
            except (OSError, RuntimeError):
                continue
            roots.add(resolved)
    return roots


_PLATFORMDIR_SAFE_ROOTS = _platformdirs_safe_roots()


def sanitize_filename(name: str) -> str:
    """Sanitize strings for safe filename usage."""
    cleaned = name.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    cleaned = " ".join(cleaned.split())
    safe_chars = []
    for ch in cleaned:
        if ch.isalnum() or ch in {"_", "-", " ", "."}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip()
    # If result is empty or only underscores/spaces, return "untitled"
    if not safe or safe.replace("_", "").replace("-", "").replace(".", "").strip() == "":
        return "untitled"
    return safe


def write_file(path: str, data: bytes) -> None:
    """Persist arbitrary bytes to disk, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


def validate_and_normalize_output_dir(path: str) -> str:
    """Validate an output directory path and return an absolute, normalized version."""
    if not path or not path.strip():
        raise ValueError("Output directory path cannot be empty")

    path_obj = Path(path).expanduser()
    try:
        resolved = path_obj.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Invalid output directory path: {path} ({exc})")

    safe_roots = {Path.cwd().resolve(), Path.home().resolve(), *_PLATFORMDIR_SAFE_ROOTS}
    if any(resolved == root or resolved.is_relative_to(root) for root in safe_roots):
        return str(resolved)

    logger.warning(
        f"Output directory {resolved} is outside recommended locations (home or app data)."
    )
    return str(resolved)


def derive_output_dir(rss_url: str, override: Optional[str]) -> str:
    """Compute the default output directory for an RSS feed."""
    if override:
        return validate_and_normalize_output_dir(override)
    parsed = urlparse(rss_url)
    base = parsed.netloc or "feed"
    safe_base = sanitize_filename(base)
    # Deterministic hash for directory naming (not security sensitive)
    digest = hashlib.sha1(rss_url.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"output_rss_{safe_base}_{digest[:URL_HASH_LENGTH]}"


def setup_output_directory(cfg: config.Config) -> Tuple[str, Optional[str]]:
    """Derive the effective output directory and run suffix for a configuration.

    Creates the output directory structure with transcripts/ and metadata/ subdirectories.
    The structure is:
    - output_dir/run_<suffix>/transcripts/
    - output_dir/run_<suffix>/metadata/

    Or if no run_suffix:
    - output_dir/transcripts/
    - output_dir/metadata/
    """
    run_suffix: Optional[str] = None
    if cfg.run_id:
        run_suffix = (
            time.strftime(TIMESTAMP_FORMAT)
            if cfg.run_id.lower() == "auto"
            else sanitize_filename(cfg.run_id)
        )
        if cfg.transcribe_missing:
            model_part = sanitize_filename(cfg.whisper_model)
            run_suffix = (
                f"{run_suffix}_whisper_{model_part}" if run_suffix else f"whisper_{model_part}"
            )
    elif cfg.transcribe_missing:
        model_part = sanitize_filename(cfg.whisper_model)
        run_suffix = f"whisper_{model_part}"

    output_dir = cfg.output_dir
    if output_dir is None:
        raise ValueError("Configuration output_dir must be defined before running the pipeline")

    effective_output_dir = (
        os.path.join(output_dir, f"run_{run_suffix}") if run_suffix else output_dir
    )

    # Create transcripts/ and metadata/ subdirectories
    transcripts_dir = os.path.join(effective_output_dir, TRANSCRIPTS_SUBDIR)
    metadata_dir = os.path.join(effective_output_dir, METADATA_SUBDIR)
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    return effective_output_dir, run_suffix


def truncate_whisper_title(
    title: str, *, for_log: bool, max_len: int = WHISPER_TITLE_MAX_CHARS
) -> str:
    """Shorten episode titles so that Whisper filenames and logs remain manageable."""
    if len(title) <= max_len:
        return title
    if for_log and max_len > 1:
        return f"{title[: max_len - 1]}…"
    return title[:max_len]


def build_whisper_output_name(idx: int, ep_title_safe: str, run_suffix: Optional[str]) -> str:
    """Construct the filename for a Whisper transcript, including run suffix if present."""
    run_tag = f"_{run_suffix}" if run_suffix else ""
    safe_title = truncate_whisper_title(ep_title_safe, for_log=False)
    return f"{idx:0{EPISODE_NUMBER_FORMAT_WIDTH}d} - {safe_title}{run_tag}.txt"


def build_whisper_output_path(
    idx: int, ep_title_safe: str, run_suffix: Optional[str], output_dir: str
) -> str:
    """Return the full path where a Whisper transcript should be stored.

    Transcripts are stored in the transcripts/ subdirectory within the output directory.
    """
    transcripts_dir = os.path.join(output_dir, TRANSCRIPTS_SUBDIR)
    return os.path.join(transcripts_dir, build_whisper_output_name(idx, ep_title_safe, run_suffix))


__all__ = [
    "TEMP_DIR_NAME",
    "TIMESTAMP_FORMAT",
    "URL_HASH_LENGTH",
    "WHISPER_TITLE_MAX_CHARS",
    "EPISODE_NUMBER_FORMAT_WIDTH",
    "TRANSCRIPTS_SUBDIR",
    "METADATA_SUBDIR",
    "sanitize_filename",
    "write_file",
    "validate_and_normalize_output_dir",
    "derive_output_dir",
    "setup_output_directory",
    "truncate_whisper_title",
    "build_whisper_output_name",
    "build_whisper_output_path",
]
