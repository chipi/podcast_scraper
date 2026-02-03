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

from .. import config

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


def write_file(path: str, data: bytes, pipeline_metrics=None) -> None:
    """Persist arbitrary bytes to disk, creating parent directories as needed.

    This function is instrumented to log detailed I/O metrics for performance analysis.
    Each write operation logs: file path, bytes written, and elapsed time.

    Args:
        path: File path to write to
        data: Bytes to write
        pipeline_metrics: Optional metrics object to record write time
    """
    import logging
    import time

    logger = logging.getLogger(__name__)
    write_start = time.time()
    bytes_written = len(data)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)

    write_elapsed = time.time() - write_start
    # Log detailed I/O metrics for performance analysis
    # This helps identify if "writing_storage" metric is measuring actual I/O or waiting time
    logger.debug(
        "[STORAGE I/O] file=%s bytes=%d elapsed=%.3fs",
        path,
        bytes_written,
        write_elapsed,
    )

    # Record actual file write time in metrics (separate from io_and_waiting stage total)
    if pipeline_metrics is not None:
        pipeline_metrics.record_stage("writing_storage", write_elapsed)


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
    return f"output/rss_{safe_base}_{digest[:URL_HASH_LENGTH]}"


def _shorten_model_name(model_name: str) -> str:
    """Shorten model name for use in run suffix.

    Removes common prefixes like "facebook/", "google/", etc. and keeps
    only the essential model identifier.

    Args:
        model_name: Full model identifier (e.g., "facebook/bart-large-cnn")

    Returns:
        Shortened model name (e.g., "bart-large-cnn")
    """
    # Remove common prefixes
    for prefix in ["facebook/", "google/", "sshleifer/", "allenai/"]:
        if model_name.startswith(prefix):
            return model_name[len(prefix) :]
    return model_name


def _build_provider_model_suffix(cfg: config.Config) -> Optional[str]:
    """Build a compact run suffix that includes all providers and models.

    Creates a short identifier that includes:
    - Transcription provider + model (if transcription is used)
    - Summary provider + models (if summaries are generated)
    - Speaker detection provider + model (if speaker detection is used)

    Format: <transcription>_<summary>_<speaker>
    Example: "w_base.en_tf_bart-large-cnn_sp_en_core_web_sm"

    Args:
        cfg: Configuration object

    Returns:
        Compact run suffix string or None if no ML features are used
    """
    parts = []

    # Transcription provider + model
    if cfg.transcribe_missing:
        if cfg.transcription_provider == "whisper":
            model_short = _shorten_model_name(cfg.whisper_model)
            parts.append(f"w_{sanitize_filename(model_short)}")
        elif cfg.transcription_provider == "openai":
            model = getattr(cfg, "openai_transcription_model", "whisper-1")
            parts.append(f"oa_{sanitize_filename(model)}")

    # Summary provider + models
    if cfg.generate_summaries:
        if cfg.summary_provider in ("transformers", "local"):
            # Import here to avoid circular dependency
            from ..providers.ml import summarizer

            map_model = summarizer.select_summary_model(cfg)
            reduce_model = summarizer.select_reduce_model(cfg, map_model)

            map_short = _shorten_model_name(map_model)
            parts.append(f"tf_{sanitize_filename(map_short)}")

            # Only include reduce model if different from map model
            if reduce_model != map_model:
                reduce_short = _shorten_model_name(reduce_model)
                parts.append(f"r_{sanitize_filename(reduce_short)}")
        elif cfg.summary_provider == "openai":
            model = getattr(cfg, "openai_summary_model", "gpt-4o-mini")
            parts.append(f"oa_{sanitize_filename(model)}")

    # Speaker detection provider + model
    if cfg.screenplay or cfg.auto_speakers:
        if cfg.speaker_detector_provider in ("spacy", "ner"):
            ner_model = cfg.ner_model
            if not ner_model:
                # Use default NER model when not specified
                ner_model = config.DEFAULT_NER_MODEL
            # Shorten common spaCy model names
            if ner_model.startswith("en_core_web_"):
                model_short = ner_model.replace("en_core_web_", "spacy_")
            else:
                model_short = ner_model
            parts.append(f"sp_{sanitize_filename(model_short)}")
        elif cfg.speaker_detector_provider == "openai":
            model = getattr(cfg, "openai_speaker_model", "gpt-4o-mini")
            parts.append(f"oa_{sanitize_filename(model)}")

    if not parts:
        return None

    return "_".join(parts)


def setup_output_directory(cfg: config.Config) -> Tuple[str, Optional[str]]:
    """Derive the effective output directory and run suffix for a configuration.

    Creates the output directory structure with transcripts/ and metadata/ subdirectories.
    The structure is:
    - output_dir/run_<suffix>/transcripts/
    - output_dir/run_<suffix>/metadata/

    Or if no run_suffix:
    - output_dir/transcripts/
    - output_dir/metadata/

    The run_suffix includes all providers and models used:
    - Transcription: provider + model (e.g., "w_base.en" for whisper, "oa_whisper-1" for openai)
    - Summarization: provider + models (e.g., "tf_bart-large-cnn" for transformers)
    - Speaker detection: provider + model (e.g., "sp_en_core_web_sm" for spacy)

    If a directory with the same name already exists and clean_output is False,
    a counter is appended to make it unique (e.g., run_<suffix>_1, run_<suffix>_2).
    """
    run_suffix: Optional[str] = None

    # Start with run_id if provided
    if cfg.run_id:
        run_suffix = (
            time.strftime(TIMESTAMP_FORMAT)
            if cfg.run_id.lower() == "auto"
            else sanitize_filename(cfg.run_id)
        )

    # Build provider/model suffix
    provider_suffix = _build_provider_model_suffix(cfg)

    if provider_suffix:
        if run_suffix:
            run_suffix = f"{run_suffix}_{provider_suffix}"
        else:
            run_suffix = provider_suffix

    output_dir = cfg.output_dir
    if output_dir is None:
        raise ValueError("Configuration output_dir must be defined before running the pipeline")

    # Base effective output directory
    base_effective_output_dir = (
        os.path.join(output_dir, f"run_{run_suffix}") if run_suffix else output_dir
    )

    # If clean_output is False and directory exists, append counter to make it unique
    effective_output_dir = base_effective_output_dir
    if not cfg.clean_output and run_suffix and os.path.exists(base_effective_output_dir):
        counter = 1
        while True:
            effective_output_dir = f"{base_effective_output_dir}_{counter}"
            if not os.path.exists(effective_output_dir):
                # Update run_suffix to include counter
                run_suffix = f"{run_suffix}_{counter}"
                logger.info(
                    "Output directory already exists, using unique name: %s",
                    effective_output_dir,
                )
                break
            counter += 1
            # Safety limit to prevent infinite loops
            if counter > 10000:
                raise RuntimeError(
                    f"Too many existing directories with similar names: {base_effective_output_dir}"
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
