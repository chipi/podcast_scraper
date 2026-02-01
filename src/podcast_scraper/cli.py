"""Command-line interface helpers for podcast_scraper."""

from __future__ import annotations

import argparse
import logging
import os
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

from . import __version__, config
from .utils import filesystem, progress
from .workflow import orchestration as workflow
from .workflow.stages import setup

if TYPE_CHECKING:
    import rich.progress

_LOGGER = logging.getLogger(__name__)

BYTES_PER_KB = 1024


class _RichProgress:
    """Simple adapter that exposes rich Progress update interface with validation."""

    def __init__(
        self,
        progress: "rich.progress.Progress",
        task_id: "rich.progress.TaskID",
        description: str,
        total: Optional[int],
    ) -> None:
        self._progress = progress
        self._task_id = task_id
        self._description = description
        self._total = total
        self._completed = 0
        self._has_updated = False

    def update(self, advance: int) -> None:
        """Update progress with validation to prevent showing 0% incorrectly.

        Args:
            advance: Number of units to advance (must be >= 0)
        """
        if advance < 0:
            # Don't allow negative progress
            return

        self._has_updated = True
        self._completed += advance

        if self._total is not None:
            # For determinate progress, ensure we don't exceed total
            completed = min(self._completed, self._total)
            self._progress.update(self._task_id, completed=completed)
        else:
            # For indeterminate progress, just advance - Rich shows elapsed time
            # advance parameter makes the bar animate
            self._progress.update(self._task_id, advance=advance)


# Module-level shared Progress instance to prevent duplicate bars and conflicts
_shared_progress: Optional["rich.progress.Progress"] = None
_shared_progress_lock = None


def _get_shared_progress() -> "rich.progress.Progress":
    """Get or create shared Progress instance for all progress bars.

    Using a single Progress instance prevents duplicate bars and conflicts
    when multiple progress contexts are active simultaneously.
    """
    global _shared_progress, _shared_progress_lock

    if _shared_progress is None:
        import threading

        from rich.console import Console
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

        _shared_progress_lock = threading.Lock()

        # Create console for progress bars
        console = Console(
            force_terminal=os.getenv("TERM") != "dumb",
            stderr=True,  # Progress to stderr
            width=None,  # Auto-detect terminal width
        )

        # Compact progress bar columns
        columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ]

        _shared_progress = Progress(
            *columns,
            console=console,
            transient=False,  # Keep bars visible until removed
            expand=False,  # Compact mode
            refresh_per_second=10,  # Higher refresh for smoother updates
        )
        _shared_progress.start()

    return _shared_progress


@contextmanager
def _rich_progress(total: Optional[int], description: str) -> Iterator[_RichProgress]:
    """Create a rich progress context matching the shared progress API.

    Uses a shared Progress instance to prevent duplicate bars and conflicts.
    Each progress context gets its own task in the shared Progress.
    """
    # Validate total parameter
    if total is not None and total <= 0:
        total = None

    # Get shared Progress instance
    progress_bar = _get_shared_progress()

    # Thread-safe task management
    if _shared_progress_lock is not None:
        _shared_progress_lock.acquire()

    task_id = None
    try:
        # Add new task to shared progress
        # Each call gets its own task, allowing multiple concurrent operations
        task_id = progress_bar.add_task(
            description,
            total=total,
        )

        # For determinate progress, initialize at 0
        if total is not None and total > 0:
            progress_bar.update(task_id, completed=0)
    finally:
        if _shared_progress_lock is not None:
            _shared_progress_lock.release()

    try:
        yield _RichProgress(progress_bar, task_id, description, total)
    finally:
        # Remove task when done (thread-safe)
        if _shared_progress_lock is not None:
            _shared_progress_lock.acquire()
        try:
            progress_bar.remove_task(task_id)
        finally:
            if _shared_progress_lock is not None:
                _shared_progress_lock.release()


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
        "--output-dir", default=None, help="Output directory (default: output/rss_<host>_<hash>)"
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


def _add_openai_arguments(parser: argparse.ArgumentParser) -> None:
    """Add OpenAI API-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--openai-api-base",
        default=None,
        help="OpenAI API base URL (for E2E testing or custom endpoints)",
    )
    parser.add_argument(
        "--openai-transcription-model",
        default=None,
        help="OpenAI model for transcription (default: whisper-1)",
    )
    parser.add_argument(
        "--openai-speaker-model",
        default=None,
        help="OpenAI model for speaker detection (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--openai-summary-model",
        default=None,
        help="OpenAI model for summarization (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--openai-temperature",
        type=float,
        default=None,
        help="Temperature for OpenAI generation (0.0-2.0, default: 0.3)",
    )


def _add_transcription_arguments(parser: argparse.ArgumentParser) -> None:
    """Add transcription-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--transcribe-missing",
        action="store_true",
        dest="transcribe_missing",
        help="Use transcription provider when no transcript is provided (default: True)",
    )
    parser.add_argument(
        "--no-transcribe-missing",
        action="store_false",
        dest="transcribe_missing",
        help="Do not transcribe episodes when transcripts are missing",
    )
    parser.add_argument(
        "--transcription-provider",
        choices=["whisper", "openai"],
        default="whisper",
        help="Transcription provider to use (default: whisper)",
    )
    parser.add_argument("--whisper-model", default="base.en", help="Whisper model to use")
    parser.add_argument(
        "--whisper-device",
        choices=["cuda", "mps", "cpu", "auto"],
        default=None,
        help="Device for Whisper transcription (cuda/mps/cpu/auto, default: auto-detect)",
    )
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
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Path to save pipeline metrics JSON file. "
        "If not specified, defaults to {effective_output_dir}/metrics.json "
        "(same level as transcripts/ and metadata/ subdirectories). "
        "Set to empty string to disable metrics export.",
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
        "--speaker-detector-provider",
        choices=["spacy", "openai"],
        default="spacy",
        help="Speaker detection provider to use (default: spacy)",
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


def _add_preprocessing_arguments(parser: argparse.ArgumentParser) -> None:
    """Add audio preprocessing arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    preprocessing_group = parser.add_argument_group("Audio Preprocessing")
    preprocessing_group.add_argument(
        "--enable-preprocessing",
        action="store_true",
        dest="preprocessing_enabled",
        help="Enable audio preprocessing before transcription (experimental)",
    )
    preprocessing_group.add_argument(
        "--preprocessing-cache-dir",
        default=None,
        help="Custom cache directory for preprocessed audio (default: .cache/preprocessing)",
    )
    preprocessing_group.add_argument(
        "--preprocessing-sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for preprocessing in Hz (default: 16000)",
    )
    preprocessing_group.add_argument(
        "--preprocessing-silence-threshold",
        type=str,
        default="-50dB",
        help="Silence detection threshold (default: -50dB)",
    )
    preprocessing_group.add_argument(
        "--preprocessing-silence-duration",
        type=float,
        default=2.0,
        help="Minimum silence duration to remove in seconds (default: 2.0)",
    )
    preprocessing_group.add_argument(
        "--preprocessing-target-loudness",
        type=int,
        default=-16,
        help="Target loudness in LUFS for normalization (default: -16)",
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
        choices=["transformers", "openai"],
        default="transformers",
        help="Summary provider to use (default: transformers)",
    )
    parser.add_argument(
        "--summary-model",
        default=None,
        help="Model identifier for local summarization (e.g., facebook/bart-large-cnn)",
    )
    parser.add_argument(
        "--summary-reduce-model",
        default=None,
        help="Model identifier for reduce phase of map-reduce summarization "
        "(e.g., allenai/led-base-16384). Defaults to LED-large if not specified.",
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


def _parse_cache_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse cache subcommand arguments.

    Args:
        argv: Command-line arguments (without 'cache' prefix)

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog="podcast-scraper cache",
        description="Manage ML model caches (Whisper, Transformers, spaCy).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status and disk usage for all ML model caches",
    )
    parser.add_argument(
        "--clean",
        nargs="?",
        const="all",
        choices=["all", "whisper", "transformers", "spacy"],
        help="Clean cache. Specify type (whisper, transformers, spacy) or use 'all' (default)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt when cleaning cache",
    )

    args = parser.parse_args(argv)

    # Require either --status or --clean
    if not args.status and not args.clean:
        parser.error("Either --status or --clean must be specified")

    return args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments, optionally merging configuration file defaults."""
    # Check if first argument is "cache" subcommand
    if argv and len(argv) > 0 and argv[0] == "cache":
        # Handle cache subcommand
        cache_argv = list(argv[1:]) if len(argv) > 1 else []
        args = _parse_cache_args(cache_argv)
        args.command = "cache"  # Mark as cache command
        return args

    # Normal parsing for main command
    parser = argparse.ArgumentParser(
        description="Download podcast episode transcripts from an RSS feed."
    )

    # Add argument groups
    _add_common_arguments(parser)
    _add_transcription_arguments(parser)
    _add_preprocessing_arguments(parser)
    _add_metadata_arguments(parser)
    _add_speaker_detection_arguments(parser)
    _add_summarization_arguments(parser)
    _add_openai_arguments(parser)
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
        "transcription_provider": args.transcription_provider,
        "whisper_model": args.whisper_model,
        "whisper_device": args.whisper_device,
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
        "speaker_detector_provider": args.speaker_detector_provider,
        "auto_speakers": args.auto_speakers,
        "cache_detected_hosts": args.cache_detected_hosts,
        "generate_metadata": args.generate_metadata,
        "metadata_format": args.metadata_format,
        "metadata_subdirectory": args.metadata_subdirectory,
        "generate_summaries": args.generate_summaries,
        "metrics_output": args.metrics_output,
        "summary_provider": args.summary_provider,
        "summary_model": args.summary_model,
        "summary_reduce_model": args.summary_reduce_model,
        "summary_device": args.summary_device,
        "summary_batch_size": config.DEFAULT_SUMMARY_BATCH_SIZE,  # Not exposed in CLI yet
        "summary_chunk_size": args.summary_chunk_size,
        "summary_cache_dir": None,  # Not exposed in CLI yet
        "summary_prompt": args.summary_prompt,
        "save_cleaned_transcript": args.save_cleaned_transcript,
        "preprocessing_enabled": getattr(args, "preprocessing_enabled", False),
        "preprocessing_cache_dir": getattr(args, "preprocessing_cache_dir", None),
        "preprocessing_sample_rate": getattr(args, "preprocessing_sample_rate", 16000),
        "preprocessing_silence_threshold": getattr(
            args, "preprocessing_silence_threshold", "-50dB"
        ),
        "preprocessing_silence_duration": getattr(args, "preprocessing_silence_duration", 2.0),
        "preprocessing_target_loudness": getattr(args, "preprocessing_target_loudness", -16),
        "openai_api_base": args.openai_api_base,
    }
    # Add OpenAI model args only if provided (fields have non-Optional types with defaults)
    if args.openai_transcription_model is not None:
        payload["openai_transcription_model"] = args.openai_transcription_model
    if args.openai_speaker_model is not None:
        payload["openai_speaker_model"] = args.openai_speaker_model
    if args.openai_summary_model is not None:
        payload["openai_summary_model"] = args.openai_summary_model
    if args.openai_temperature is not None:
        payload["openai_temperature"] = args.openai_temperature
    # Explicitly include openai_api_key=None to trigger field validator
    # The field validator will load it from OPENAI_API_KEY env var if available
    payload["openai_api_key"] = None
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
        if cfg.summary_provider == "transformers":
            if cfg.summary_model:
                logger.info(f"  Summary Model: {cfg.summary_model}")
            else:
                logger.info("  Summary Model: auto-selected")
            logger.info(f"  Summary Device: {cfg.summary_device or 'auto-detect'}")
            if cfg.summary_chunk_size:
                logger.info(f"  Summary Chunk Size: {cfg.summary_chunk_size} tokens")
        logger.info(
            f"  Summary Map: max_new_tokens={cfg.summary_map_params.get('max_new_tokens')}, "
            f"min_new_tokens={cfg.summary_map_params.get('min_new_tokens')}"
        )
        logger.info(
            f"  Summary Reduce: max_new_tokens={cfg.summary_reduce_params.get('max_new_tokens')}, "
            f"min_new_tokens={cfg.summary_reduce_params.get('min_new_tokens')}"
        )
        if cfg.summary_map_params and cfg.summary_reduce_params and cfg.summary_tokenize:
            logger.info(
                "  Using explicit ML parameters from config (map_params/reduce_params/tokenize)"
            )
            logger.info(
                f"    Map: max_new_tokens={cfg.summary_map_params.get('max_new_tokens')}, "
                f"num_beams={cfg.summary_map_params.get('num_beams')}"
            )
            logger.info(
                f"    Reduce: max_new_tokens={cfg.summary_reduce_params.get('max_new_tokens')}, "
                f"num_beams={cfg.summary_reduce_params.get('num_beams')}"
            )
        if cfg.summary_prompt:
            logger.info(f"  Summary Prompt: {cfg.summary_prompt[:80]}...")

    # Processing options
    logger.info("Processing Options:")
    logger.info(f"  Skip Existing: {cfg.skip_existing}")
    logger.info(f"  Reuse Media: {cfg.reuse_media}")
    logger.info(f"  Clean Output: {cfg.clean_output}")
    logger.info(f"  Dry Run: {cfg.dry_run}")

    logger.info("=" * 80)


def main(  # noqa: C901 - main function handles multiple command paths
    argv: Optional[Sequence[str]] = None,
    *,
    apply_log_level_fn: Optional[Callable[[str, Optional[str]], None]] = None,
    run_pipeline_fn: Optional[Callable[[config.Config], Tuple[int, str]]] = None,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Entry point for the CLI; returns an exit status code."""
    # Initialize ML environment variables early (before any ML imports)
    setup.initialize_ml_environment()

    progress.set_progress_factory(_rich_progress)
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
    except SystemExit as exc:
        # argparse may call sys.exit() for --help, etc.
        return exc.code if isinstance(exc.code, int) else 0

    # Handle cache subcommand
    if hasattr(args, "command") and args.command == "cache":
        try:
            from .cache import manager as cache_manager

            if args.status:
                cache_info = cache_manager.get_all_cache_info()
                print("\nML Model Cache Status")
                print("=" * 50)

                # Whisper
                whisper_info = cache_info["whisper"]
                whisper_size = cache_manager.format_size(whisper_info["size"])
                whisper_count = whisper_info["count"]
                print(f"\nWhisper models: {whisper_size} ({whisper_count} models)")
                print(f"  Cache directory: {whisper_info['dir']}")
                if whisper_info["models"]:
                    for model in whisper_info["models"]:
                        print(f"  - {model['name']:30s} {cache_manager.format_size(model['size'])}")

                # Transformers
                transformers_info = cache_info["transformers"]
                transformers_size = cache_manager.format_size(transformers_info["size"])
                transformers_count = transformers_info["count"]
                print(f"\nTransformers: {transformers_size} ({transformers_count} models)")
                print(f"  Cache directory: {transformers_info['dir']}")
                if transformers_info["dir"].exists():
                    print(
                        "  ⚠️  Warning: This cache may be shared with other applications "
                        "using Hugging Face models."
                    )
                if transformers_info["models"]:
                    for model in transformers_info["models"]:
                        print(f"  - {model['name']:30s} {cache_manager.format_size(model['size'])}")

                # spaCy
                spacy_info = cache_info["spacy"]
                if spacy_info["dir"]:
                    spacy_size = cache_manager.format_size(spacy_info["size"])
                    spacy_count = spacy_info["count"]
                    print(f"\nspaCy: {spacy_size} ({spacy_count} models)")
                    print(f"  Cache directory: {spacy_info['dir']}")
                    print("  Note: spaCy models are typically installed as Python packages.")
                    if spacy_info["models"]:
                        for model in spacy_info["models"]:
                            model_size = cache_manager.format_size(model["size"])
                            print(f"  - {model['name']:30s} {model_size}")
                else:
                    print("\nspaCy: No cache directory found")
                    print("  Note: spaCy models are typically installed as Python packages.")

                print(f"\nTotal: {cache_manager.format_size(cache_info['total_size'])}")
                print()
                return 0

            elif args.clean:
                confirm = not args.yes
                cache_type = args.clean

                if cache_type == "all":
                    results = cache_manager.clean_all_caches(confirm=confirm)
                    total_deleted = sum(count for count, _ in results.values())
                    total_freed = sum(bytes_freed for _, bytes_freed in results.values())
                    freed_str = cache_manager.format_size(total_freed)
                    print(
                        f"\nCleaned all caches: {total_deleted} model(s) removed, "
                        f"{freed_str} freed"
                    )
                    return 0
                elif cache_type == "whisper":
                    deleted, freed = cache_manager.clean_whisper_cache(confirm=confirm)
                    freed_str = cache_manager.format_size(freed)
                    print(
                        f"\nCleaned Whisper cache: {deleted} model(s) removed, "
                        f"{freed_str} freed"
                    )
                    return 0
                elif cache_type == "transformers":
                    deleted, freed = cache_manager.clean_transformers_cache(confirm=confirm)
                    freed_str = cache_manager.format_size(freed)
                    print(
                        f"\nCleaned Transformers cache: {deleted} model(s) removed, "
                        f"{freed_str} freed"
                    )
                    return 0
                elif cache_type == "spacy":
                    deleted, freed = cache_manager.clean_spacy_cache(confirm=confirm)
                    freed_str = cache_manager.format_size(freed)
                    print(
                        f"\nCleaned spaCy cache: {deleted} model(s) removed, " f"{freed_str} freed"
                    )
                    return 0

        except ImportError as exc:
            log.error(f"Cache management requires cache_manager module: {exc}")
            return 1
        except Exception as exc:
            log.error(f"Cache operation failed: {exc}")
            return 1

    # Handle cache management commands
    if (
        hasattr(args, "cache_info")
        and args.cache_info
        or hasattr(args, "prune_cache")
        and args.prune_cache
    ):
        try:
            from .providers.ml import summarizer

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
                        else summarizer.HF_CACHE_DIR
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
