"""Core utilities for podcast_scraper.

This module provides:
- Filesystem utilities (path building, sanitization, output directories)
- Progress reporting abstraction
"""

from .filesystem import (
    build_whisper_output_name,
    build_whisper_output_path,
    derive_output_dir,
    EPISODE_NUMBER_FORMAT_WIDTH,
    METADATA_SUBDIR,
    sanitize_filename,
    setup_output_directory,
    TEMP_DIR_NAME,
    TIMESTAMP_FORMAT,
    TRANSCRIPTS_SUBDIR,
    truncate_whisper_title,
    URL_HASH_LENGTH,
    validate_and_normalize_output_dir,
    WHISPER_TITLE_MAX_CHARS,
    write_file,
)
from .progress import (
    progress_context,
    ProgressFactory,
    ProgressReporter,
    set_progress_factory,
)

__all__ = [
    # Filesystem exports
    "EPISODE_NUMBER_FORMAT_WIDTH",
    "METADATA_SUBDIR",
    "TEMP_DIR_NAME",
    "TIMESTAMP_FORMAT",
    "TRANSCRIPTS_SUBDIR",
    "URL_HASH_LENGTH",
    "WHISPER_TITLE_MAX_CHARS",
    "build_whisper_output_name",
    "build_whisper_output_path",
    "derive_output_dir",
    "sanitize_filename",
    "setup_output_directory",
    "truncate_whisper_title",
    "validate_and_normalize_output_dir",
    "write_file",
    # Progress exports
    "ProgressFactory",
    "ProgressReporter",
    "progress_context",
    "set_progress_factory",
]
