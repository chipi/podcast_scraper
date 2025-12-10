"""Command-line interface helpers for podcast_scraper."""

from __future__ import annotations

import argparse
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)
from urllib.parse import urlparse

from pydantic import ValidationError

from . import __version__, config, filesystem, progress, workflow

if TYPE_CHECKING:  # pragma: no cover - typing only
    import tqdm

_LOGGER = logging.getLogger(__name__)

# Progress bar constants
TQDM_NCOLS = 80
TQDM_MIN_INTERVAL = 0.5
TQDM_MIN_ITERS = 1
BYTES_PER_KB = 1024


class _TqdmProgress:
    """Simple adapter that exposes tqdm's update interface."""

    def __init__(self, bar: "tqdm.tqdm") -> None:
        self._bar = bar

    def update(self, advance: int) -> None:
        self._bar.update(advance)


@contextmanager
def _tqdm_progress(total: Optional[int], description: str) -> Iterator[_TqdmProgress]:
    """Create a tqdm progress context matching the shared progress API."""
    from tqdm import tqdm

    kwargs: Dict[str, Any] = {"desc": description}
    if total is None:
        kwargs.update(
            total=None,
            unit="",
            leave=False,
            miniters=TQDM_MIN_ITERS,
            mininterval=TQDM_MIN_INTERVAL,
            bar_format="{desc}: {elapsed}",
            ncols=TQDM_NCOLS,
            dynamic_ncols=False,
        )
    else:
        kwargs.update(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=BYTES_PER_KB,
            leave=True,
        )

    with tqdm(**kwargs) as bar:
        yield _TqdmProgress(bar)


def _validate_rss_url(rss_value: str, errors: List[str]) -> None:
    """Validate RSS URL format.

    Args:
        rss_value: RSS URL string
        errors: List to append validation errors to
    """
    if not rss_value:
        errors.append("RSS URL is required")
        return

    parsed_obj = urlparse(rss_value)
    if parsed_obj.scheme not in ("http", "https"):
        errors.append(f"RSS URL must be http or https: {rss_value}")
    if not parsed_obj.netloc:
        errors.append(f"RSS URL must have a valid hostname: {rss_value}")


def _validate_whisper_config(args: argparse.Namespace, errors: List[str]) -> None:
    """Validate Whisper-related configuration.

    Args:
        args: Parsed arguments
        errors: List to append validation errors to
    """
    valid_models = (
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v2",
        "large-v3",
        "tiny.en",
        "base.en",
        "small.en",
        "medium.en",
        "large.en",
    )
    if args.transcribe_missing and args.whisper_model not in valid_models:
        errors.append(f"--whisper-model must be one of {valid_models}, got: {args.whisper_model}")


def _validate_speaker_config(args: argparse.Namespace, errors: List[str]) -> None:
    """Validate speaker-related configuration.

    Args:
        args: Parsed arguments
        errors: List to append validation errors to
    """
    if args.screenplay and args.num_speakers < config.MIN_NUM_SPEAKERS:
        errors.append(
            f"--num-speakers must be at least {config.MIN_NUM_SPEAKERS}, got: {args.num_speakers}"
        )

    if args.speaker_names:
        names = [n.strip() for n in args.speaker_names.split(",") if n.strip()]
        if len(names) < config.MIN_NUM_SPEAKERS:
            errors.append("At least two speaker names required when specifying --speaker-names")


def _validate_workers_config(args: argparse.Namespace, errors: List[str]) -> None:
    """Validate workers configuration.

    Args:
        args: Parsed arguments
        errors: List to append validation errors to
    """
    if args.workers < 1:
        errors.append("--workers must be at least 1")


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed CLI arguments and raise ValueError when invalid."""
    errors: List[str] = []

    # Validate RSS URL
    rss_value = (args.rss or "").strip()
    _validate_rss_url(rss_value, errors)

    # Validate numeric arguments
    if args.max_episodes is not None and args.max_episodes <= 0:
        errors.append(f"--max-episodes must be positive, got: {args.max_episodes}")

    if args.timeout <= 0:
        errors.append(f"--timeout must be positive, got: {args.timeout}")

    if args.delay_ms < 0:
        errors.append(f"--delay-ms must be non-negative, got: {args.delay_ms}")

    # Validate feature-specific configs
    _validate_whisper_config(args, errors)
    _validate_speaker_config(args, errors)
    _validate_workers_config(args, errors)

    # Validate output directory
    if args.output_dir:
        try:
            filesystem.validate_and_normalize_output_dir(args.output_dir)
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        raise ValueError("Invalid input parameters:\n  " + "\n  ".join(errors))


def _add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    """Add cache management arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    cache_group = parser.add_argument_group("Cache Management")
    cache_group.add_argument(
        "--cache-info",
        action="store_true",
        help="Show Hugging Face model cache information and exit",
    )
    cache_group.add_argument(
        "--prune-cache",
        action="store_true",
        help="Remove all cached transformer models to free disk space",
    )
    cache_group.add_argument(
        "--cache-dir",
        default=None,
        help="Custom cache directory for transformer models (default: ~/.cache/huggingface/hub)",
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument("--config", default=None, help="Path to configuration file (JSON or YAML)")
    parser.add_argument("rss", nargs="?", default=None, help="Podcast RSS feed URL")
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: output_rss_<host>_<hash>)"
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="Maximum number of episodes to process"
    )
    parser.add_argument(
        "--prefer-type",
        action="append",
        default=[],
        help="Preferred transcript types or extensions (repeatable)",
    )
    parser.add_argument("--user-agent", default=config.DEFAULT_USER_AGENT, help="User-Agent header")
    parser.add_argument(
        "--timeout",
        type=int,
        default=config.DEFAULT_TIMEOUT_SECONDS,
        help="Request timeout in seconds",
    )
    parser.add_argument("--delay-ms", type=int, default=0, help="Delay between requests (ms)")
    parser.add_argument(
        "--run-id", default=None, help="Optional run identifier; use 'auto' for timestamp"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip episodes whose output already exists"
    )
    parser.add_argument(
        "--reuse-media",
        action="store_true",
        help="Reuse existing media files instead of re-downloading (for faster testing)",
    )
    parser.add_argument(
        "--clean-output", action="store_true", help="Remove the output directory before processing"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show planned work without saving files"
    )
    parser.add_argument("--version", action="store_true", help="Show program version and exit")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (logs will be written to both console and file)",
    )
    parser.add_argument(
        "--log-level",
        default=config.DEFAULT_LOG_LEVEL,
        type=str.upper,
        help="Logging level (e.g., DEBUG, INFO)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.DEFAULT_WORKERS,
        help="Number of concurrent download workers",
    )


def _add_transcription_arguments(parser: argparse.ArgumentParser) -> None:
    """Add transcription-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--transcribe-missing",
        action="store_true",
        help="Use Whisper when no transcript is provided",
    )
    parser.add_argument("--whisper-model", default="base", help="Whisper model to use")
    parser.add_argument(
        "--screenplay", action="store_true", help="Format Whisper transcript as screenplay"
    )
    parser.add_argument(
        "--screenplay-gap",
        type=float,
        default=config.DEFAULT_SCREENPLAY_GAP_SECONDS,
        help="Gap (seconds) to trigger speaker change",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=config.DEFAULT_NUM_SPEAKERS,
        help="Number of speakers to alternate between",
    )
    parser.add_argument(
        "--speaker-names",
        default="",
        help="Comma-separated speaker names to use instead of SPEAKER 1..N",
    )


def _add_metadata_arguments(parser: argparse.ArgumentParser) -> None:
    """Add metadata-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--generate-metadata",
        action="store_true",
        help="Generate metadata documents alongside transcripts",
    )
    parser.add_argument(
        "--metadata-format",
        choices=["json", "yaml"],
        default="json",
        help="Format for metadata files (default: json)",
    )
    parser.add_argument(
        "--metadata-subdirectory",
        default=None,
        help="Store metadata files in subdirectory (default: same as transcripts)",
    )


def _add_speaker_detection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add speaker detection-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--language",
        default=config.DEFAULT_LANGUAGE,
        help="Language for transcription and NER (default: en)",
    )
    parser.add_argument(
        "--ner-model",
        default=None,
        help="spaCy NER model to use (default: derived from language)",
    )
    parser.add_argument(
        "--auto-speakers",
        action="store_true",
        default=True,
        help="Enable automatic speaker name detection (default: True)",
    )
    parser.add_argument(
        "--no-auto-speakers",
        dest="auto_speakers",
        action="store_false",
        help="Disable automatic speaker name detection",
    )
    parser.add_argument(
        "--cache-detected-hosts",
        action="store_true",
        default=True,
        help="Cache detected hosts across episodes (default: True)",
    )
    parser.add_argument(
        "--no-cache-detected-hosts",
        dest="cache_detected_hosts",
        action="store_false",
        help="Disable caching of detected hosts",
    )


def _add_summarization_arguments(parser: argparse.ArgumentParser) -> None:
    """Add summarization-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--generate-summaries",
        action="store_true",
        help="Generate summaries for episodes",
    )
    parser.add_argument(
        "--summary-provider",
        choices=["local", "openai", "anthropic"],
        default="local",
        help="Summary provider to use (default: local)",
    )
    parser.add_argument(
        "--summary-model",
        default=None,
        help="Model identifier for local summarization (e.g., facebook/bart-large-cnn)",
    )
    parser.add_argument(
        "--summary-max-length",
        type=int,
        default=config.DEFAULT_SUMMARY_MAX_LENGTH,
        help=f"Maximum summary length in tokens (default: {config.DEFAULT_SUMMARY_MAX_LENGTH})",
    )
    parser.add_argument(
        "--summary-min-length",
        type=int,
        default=config.DEFAULT_SUMMARY_MIN_LENGTH,
        help=f"Minimum summary length in tokens (default: {config.DEFAULT_SUMMARY_MIN_LENGTH})",
    )
    parser.add_argument(
        "--summary-device",
        choices=["cuda", "mps", "cpu", "auto"],
        default=None,
        help="Device for summarization (cuda/mps/cpu/auto, default: auto-detect)",
    )
    parser.add_argument(
        "--summary-chunk-size",
        type=int,
        default=None,
        help="Chunk size for long transcripts in tokens (default: model max length)",
    )
    parser.add_argument(
        "--summary-prompt",
        type=str,
        default=None,
        help="Custom prompt/instruction to guide summarization (default: built-in prompt)",
    )
    parser.add_argument(
        "--save-cleaned-transcript",
        action="store_true",
        default=True,
        help="Save cleaned transcript to separate file (default: True)",
    )
    parser.add_argument(
        "--no-save-cleaned-transcript",
        dest="save_cleaned_transcript",
        action="store_false",
        help="Don't save cleaned transcript to separate file",
    )


def _load_and_merge_config(
    parser: argparse.ArgumentParser, config_path: str, argv: Optional[Sequence[str]]
) -> argparse.Namespace:
    """Load configuration file and merge with CLI arguments.

    Args:
        parser: Argument parser
        config_path: Path to configuration file
        argv: Command-line arguments

    Returns:
        Parsed arguments with config merged

    Raises:
        ValueError: If config is invalid or RSS URL is missing
    """
    config_data = config.load_config_file(config_path)
    valid_dests = {action.dest for action in parser._actions if action.dest}
    unknown_keys = [key for key in config_data.keys() if key not in valid_dests]
    if unknown_keys:
        raise ValueError("Unknown config option(s): " + ", ".join(sorted(unknown_keys)))

    try:
        config_model = config.Config.model_validate(config_data)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc

    defaults_updates: Dict[str, Any] = config_model.model_dump(
        exclude_none=True,
        by_alias=True,
    )

    speaker_list = defaults_updates.get("speaker_names")
    if isinstance(speaker_list, list):
        defaults_updates["speaker_names"] = ",".join(speaker_list)

    parser.set_defaults(**defaults_updates)
    args = parser.parse_args(argv)
    if not args.rss:
        raise ValueError("RSS URL is required (provide in config as 'rss' or via CLI)")
    return args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments, optionally merging configuration file defaults."""
    parser = argparse.ArgumentParser(
        description="Download podcast episode transcripts from an RSS feed."
    )

    # Add argument groups
    _add_common_arguments(parser)
    _add_transcription_arguments(parser)
    _add_metadata_arguments(parser)
    _add_speaker_detection_arguments(parser)
    _add_summarization_arguments(parser)
    _add_cache_arguments(parser)

    initial_args, _ = parser.parse_known_args(argv)

    if initial_args.version:
        print(f"podcast_scraper {__version__}")
        raise SystemExit(0)

    if initial_args.config:
        args = _load_and_merge_config(parser, initial_args.config, argv)
    else:
        args = parser.parse_args(argv)

    validate_args(args)
    return args


def _build_config(args: argparse.Namespace) -> config.Config:
    """Materialize a Config object from already-validated CLI arguments."""
    speaker_names_list = [s.strip() for s in (args.speaker_names or "").split(",") if s.strip()]
    payload: Dict[str, Any] = {
        "rss_url": args.rss,
        "output_dir": filesystem.derive_output_dir(args.rss, args.output_dir),
        "max_episodes": args.max_episodes,
        "user_agent": args.user_agent,
        "timeout": args.timeout,
        "delay_ms": args.delay_ms,
        "prefer_types": args.prefer_type,
        "transcribe_missing": args.transcribe_missing,
        "whisper_model": args.whisper_model,
        "screenplay": args.screenplay,
        "screenplay_gap_s": args.screenplay_gap,
        "screenplay_num_speakers": args.num_speakers,
        "screenplay_speaker_names": speaker_names_list,
        "run_id": args.run_id,
        "log_level": args.log_level,
        "log_file": args.log_file,
        "workers": args.workers,
        "skip_existing": args.skip_existing,
        "reuse_media": args.reuse_media,
        "clean_output": args.clean_output,
        "dry_run": args.dry_run,
        "language": args.language,
        "ner_model": args.ner_model,
        "auto_speakers": args.auto_speakers,
        "cache_detected_hosts": args.cache_detected_hosts,
        "generate_metadata": args.generate_metadata,
        "metadata_format": args.metadata_format,
        "metadata_subdirectory": args.metadata_subdirectory,
        "generate_summaries": args.generate_summaries,
        "summary_provider": args.summary_provider,
        "summary_model": args.summary_model,
        "summary_max_length": args.summary_max_length,
        "summary_min_length": args.summary_min_length,
        "summary_device": args.summary_device,
        "summary_batch_size": config.DEFAULT_SUMMARY_BATCH_SIZE,  # Not exposed in CLI yet
        "summary_chunk_size": args.summary_chunk_size,
        "summary_cache_dir": None,  # Not exposed in CLI yet
        "summary_prompt": args.summary_prompt,
        "save_cleaned_transcript": args.save_cleaned_transcript,
    }
    # Pydantic's model_validate returns the correct type, but mypy needs help
    return cast(config.Config, config.Config.model_validate(payload))


def _log_configuration(cfg: config.Config, logger: logging.Logger) -> None:
    """Log all configuration values in a structured format.

    Args:
        cfg: Configuration object
        logger: Logger instance to use
    """
    logger.info("=" * 80)
    logger.info("Configuration")
    logger.info("=" * 80)

    # Core settings
    logger.info("Core Settings:")
    logger.info(f"  RSS URL: {cfg.rss_url}")
    logger.info(f"  Output Directory: {cfg.output_dir}")
    logger.info(f"  Max Episodes: {cfg.max_episodes or 'all'}")
    logger.info(f"  Workers: {cfg.workers}")
    logger.info(f"  Log Level: {cfg.log_level}")
    logger.info(f"  Log File: {cfg.log_file or 'console only'}")
    logger.info(f"  Run ID: {cfg.run_id or 'none'}")

    # HTTP settings
    logger.info("HTTP Settings:")
    logger.info(f"  Timeout: {cfg.timeout}s")
    logger.info(f"  Delay: {cfg.delay_ms}ms")
    logger.info(
        f"  User-Agent: {cfg.user_agent[:50]}..."
        if len(cfg.user_agent) > 50
        else f"  User-Agent: {cfg.user_agent}"
    )
    logger.info(f"  Prefer Types: {cfg.prefer_types if cfg.prefer_types else 'none'}")

    # Transcription settings
    logger.info("Transcription Settings:")
    logger.info(f"  Transcribe Missing: {cfg.transcribe_missing}")
    if cfg.transcribe_missing:
        logger.info(f"  Whisper Model: {cfg.whisper_model}")
        logger.info(f"  Screenplay Format: {cfg.screenplay}")
        if cfg.screenplay:
            logger.info(f"  Screenplay Gap: {cfg.screenplay_gap_s}s")
            logger.info(f"  Number of Speakers: {cfg.screenplay_num_speakers}")
            if cfg.screenplay_speaker_names:
                logger.info(f"  Speaker Names: {', '.join(cfg.screenplay_speaker_names)}")

    # Speaker detection settings
    logger.info("Speaker Detection Settings:")
    logger.info(f"  Auto Speakers: {cfg.auto_speakers}")
    logger.info(f"  Language: {cfg.language}")
    if cfg.ner_model:
        logger.info(f"  NER Model: {cfg.ner_model}")
    logger.info(f"  Cache Detected Hosts: {cfg.cache_detected_hosts}")

    # Metadata settings
    logger.info("Metadata Settings:")
    logger.info(f"  Generate Metadata: {cfg.generate_metadata}")
    if cfg.generate_metadata:
        logger.info(f"  Metadata Format: {cfg.metadata_format}")
        if cfg.metadata_subdirectory:
            logger.info(f"  Metadata Subdirectory: {cfg.metadata_subdirectory}")
        else:
            logger.info("  Metadata Subdirectory: same as transcripts")

    # Summarization settings
    logger.info("Summarization Settings:")
    logger.info(f"  Generate Summaries: {cfg.generate_summaries}")
    if cfg.generate_summaries:
        logger.info(f"  Summary Provider: {cfg.summary_provider}")
        if cfg.summary_provider == "local":
            if cfg.summary_model:
                logger.info(f"  Summary Model: {cfg.summary_model}")
            else:
                logger.info("  Summary Model: auto-selected")
            logger.info(f"  Summary Device: {cfg.summary_device or 'auto-detect'}")
            if cfg.summary_chunk_size:
                logger.info(f"  Summary Chunk Size: {cfg.summary_chunk_size} tokens")
        logger.info(f"  Summary Max Length: {cfg.summary_max_length} tokens")
        logger.info(f"  Summary Min Length: {cfg.summary_min_length} tokens")
        if cfg.summary_prompt:
            logger.info(f"  Summary Prompt: {cfg.summary_prompt[:80]}...")

    # Processing options
    logger.info("Processing Options:")
    logger.info(f"  Skip Existing: {cfg.skip_existing}")
    logger.info(f"  Reuse Media: {cfg.reuse_media}")
    logger.info(f"  Clean Output: {cfg.clean_output}")
    logger.info(f"  Dry Run: {cfg.dry_run}")

    logger.info("=" * 80)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    apply_log_level_fn: Optional[Callable[[str, Optional[str]], None]] = None,
    run_pipeline_fn: Optional[Callable[[config.Config], Tuple[int, str]]] = None,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Entry point for the CLI; returns an exit status code."""
    progress.set_progress_factory(_tqdm_progress)
    log = logger or _LOGGER
    if apply_log_level_fn is None:
        apply_log_level_fn = workflow.apply_log_level
    if run_pipeline_fn is None:
        run_pipeline_fn = workflow.run_pipeline

    try:
        args = parse_args(argv)
    except ValueError as exc:
        log.error(f"Error: {exc}")
        return 1

    # Handle cache management commands
    if args.cache_info or args.prune_cache:
        try:
            from . import summarizer

            cache_dir = args.cache_dir or None
            if args.cache_info:
                cache_size = summarizer.get_cache_size(cache_dir)
                size_str = summarizer.format_cache_size(cache_size)
                cache_path = (
                    Path(cache_dir)
                    if cache_dir
                    else (
                        summarizer.HF_CACHE_DIR
                        if summarizer.HF_CACHE_DIR.exists()
                        else summarizer.HF_CACHE_DIR_LEGACY
                    )
                )
                log.info("Hugging Face Model Cache Information:")
                log.info(f"  Cache directory: {cache_path}")
                log.info(f"  Total size: {size_str}")
                if cache_size == 0:
                    log.info("  Cache is empty")
                return 0
            elif args.prune_cache:
                deleted = summarizer.prune_cache(cache_dir, dry_run=False)
                log.info(f"Pruned {deleted} files from cache")
                return 0
        except ImportError:
            log.error("Cache management requires transformers library")
            return 1
        except Exception as exc:
            log.error(f"Cache operation failed: {exc}")
            return 1

    try:
        cfg = _build_config(args)
    except ValidationError as exc:
        log.error(f"Invalid configuration: {exc}")
        return 1

    apply_log_level_fn(cfg.log_level, cfg.log_file)

    log.info("Starting podcast transcript scrape")
    _log_configuration(cfg, log)

    try:
        _, summary = run_pipeline_fn(cfg)
    except Exception as exc:  # pragma: no cover - defensive
        log.error(f"Unexpected failure: {exc}")
        return 1

    log.info(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())
