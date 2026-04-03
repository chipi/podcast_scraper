"""Command-line interface helpers for podcast_scraper.

Low MI (radon): see docs/ci/CODE_QUALITY_TRENDS.md § Low-MI modules.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from urllib.parse import urlparse

from pydantic import ValidationError

from . import __version__, config, config_constants
from .utils import filesystem, progress
from .utils.log_redaction import format_exception_for_log
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


def _get_shared_progress() -> Optional["rich.progress.Progress"]:
    """Get or create shared Progress instance for all progress bars.

    Using a single Progress instance prevents duplicate bars and conflicts
    when multiple progress contexts are active simultaneously.

    Returns None if output is redirected (not a TTY), in which case progress
    bars should be disabled to keep log files clean.
    """
    global _shared_progress, _shared_progress_lock

    # Check if stderr is a TTY - only show progress bars in interactive terminals
    # This prevents ANSI escape sequences and progress bar noise in log files
    try:
        is_tty = sys.stderr.isatty()
    except (AttributeError, OSError):
        # Fallback for very old Python or if stderr is closed
        is_tty = False

    # Also respect TERM=dumb (used in tests and some CI environments)
    if os.getenv("TERM") == "dumb":
        is_tty = False

    # Don't create progress bars if output is redirected
    if not is_tty:
        return None

    if _shared_progress is None:
        import threading

        from rich.console import Console
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

        _shared_progress_lock = threading.Lock()

        # Create console for progress bars
        # force_terminal=False allows Rich to auto-detect terminal capabilities
        # but we've already checked isatty() above, so this is just for safety
        console = Console(
            force_terminal=False,  # Auto-detect, don't force terminal mode
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
def _rich_progress(total: Optional[int], description: str) -> Iterator[Union[_RichProgress, Any]]:
    """Create a rich progress context matching the shared progress API.

    Uses a shared Progress instance to prevent duplicate bars and conflicts.
    Each progress context gets its own task in the shared Progress.

    If output is redirected (not a TTY), returns a no-op progress reporter
    to keep log files clean without ANSI escape sequences.
    """
    # Validate total parameter
    if total is not None and total <= 0:
        total = None

    # Get shared Progress instance (returns None if not a TTY)
    progress_bar = _get_shared_progress()

    # If not a TTY, return no-op progress to avoid polluting log files
    if progress_bar is None:
        from podcast_scraper.utils.progress import _NoopProgress

        yield _NoopProgress()
        return

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
    """Add common arguments to parser (Issue #379).

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
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for reproducibility (Issue #429); sets torch/numpy/transformers seeds",
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
        "--json-logs",
        action="store_true",
        dest="json_logs",
        help="Output structured JSON logs for monitoring/alerting (Issue #379)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.DEFAULT_WORKERS,
        help="Number of concurrent download workers",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        dest="fail_fast",
        help="Stop on first episode failure (Issue #379)",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=None,
        dest="max_failures",
        metavar="N",
        help="Stop after N episode failures (Issue #379)",
    )


def _add_openai_arguments(parser: argparse.ArgumentParser) -> None:
    """Add OpenAI API-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
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
        "--openai-insight-model",
        default=None,
        help=(
            "OpenAI model for GIL generate_insights only when gi_insight_source=provider "
            "(default: same as --openai-summary-model)"
        ),
    )
    parser.add_argument(
        "--openai-temperature",
        type=float,
        default=None,
        help="Temperature for OpenAI generation (0.0-2.0, default: 0.3)",
    )
    parser.add_argument(
        "--openai-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for OpenAI responses (default: model-specific)",
    )
    parser.add_argument(
        "--openai-cleaning-model",
        default=None,
        help=(
            "OpenAI model for transcript cleaning "
            "(default: gpt-4o-mini, cheaper than summary model)"
        ),
    )
    parser.add_argument(
        "--openai-cleaning-temperature",
        type=float,
        default=None,
        help="Temperature for OpenAI cleaning (0.0-2.0, default: 0.2, lower = more deterministic)",
    )


def _add_gemini_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Gemini API-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--gemini-api-base",
        type=str,
        default=None,
        help="Gemini API base URL (for E2E testing, or set GEMINI_API_BASE env var)",
    )
    parser.add_argument(
        "--gemini-transcription-model",
        type=str,
        default=None,
        help="Gemini transcription model (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--gemini-speaker-model",
        type=str,
        default=None,
        help="Gemini speaker detection model (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--gemini-summary-model",
        type=str,
        default=None,
        help="Gemini summarization model (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--gemini-temperature",
        type=float,
        default=None,
        help="Temperature for Gemini models (0.0-2.0, default: 0.3)",
    )
    parser.add_argument(
        "--gemini-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for Gemini responses (default: model-specific)",
    )
    parser.add_argument(
        "--gemini-cleaning-model",
        type=str,
        default=None,
        help=(
            "Gemini model for transcript cleaning "
            "(default: gemini-2.0-flash; legacy gemini-1.5-flash is remapped)"
        ),
    )
    parser.add_argument(
        "--gemini-cleaning-temperature",
        type=float,
        default=None,
        help="Temperature for Gemini cleaning (0.0-2.0, default: 0.2, lower = more deterministic)",
    )


def _add_anthropic_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Anthropic API-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--anthropic-api-base",
        type=str,
        default=None,
        help="Anthropic API base URL (for E2E testing, or set ANTHROPIC_API_BASE env var)",
    )
    parser.add_argument(
        "--anthropic-speaker-model",
        type=str,
        default=None,
        help="Anthropic model for speaker detection (default: claude-haiku-4-5)",
    )
    parser.add_argument(
        "--anthropic-summary-model",
        type=str,
        default=None,
        help="Anthropic model for summarization (default: claude-haiku-4-5)",
    )
    parser.add_argument(
        "--anthropic-temperature",
        type=float,
        default=None,
        help="Temperature for Anthropic models (0.0-1.0, default: 0.3)",
    )
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for Anthropic responses (default: model-specific)",
    )
    parser.add_argument(
        "--anthropic-cleaning-model",
        type=str,
        default=None,
        help=(
            "Anthropic model for transcript cleaning "
            "(default: claude-haiku-4-5, cheaper than summary model)"
        ),
    )
    parser.add_argument(
        "--anthropic-cleaning-temperature",
        type=float,
        default=None,
        help=(
            "Temperature for Anthropic cleaning "
            "(0.0-1.0, default: 0.2, lower = more deterministic)"
        ),
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
        choices=["whisper", "openai", "gemini", "mistral"],
        default="whisper",
        help=(
            "Transcription provider to use (default: whisper). "
            "Note: deepseek, grok, and ollama do NOT support transcription."
        ),
    )
    parser.add_argument("--whisper-model", default="base.en", help="Whisper model to use")
    parser.add_argument(
        "--whisper-device",
        choices=["cuda", "mps", "cpu", "auto"],
        default=None,
        help="Device for Whisper transcription (cuda/mps/cpu/auto, default: auto-detect)",
    )
    parser.add_argument(
        "--transcription-device",
        choices=["cuda", "mps", "cpu", "auto"],
        default=None,
        help="Device for transcription stage (overrides whisper_device, Issue #387). "
        "Allows CPU/GPU mix to regain overlap (e.g., transcription on CPU, summarization on MPS).",
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
    parser.add_argument(
        "--generate-gi",
        action="store_true",
        dest="generate_gi",
        help="Generate Grounded Insight Layer (GIL) artifacts (gi.json) per episode. "
        "Requires generate_metadata. See docs/guides/GROUNDED_INSIGHTS_GUIDE.md.",
    )
    parser.add_argument(
        "--gi-insight-source",
        choices=["provider", "summary_bullets", "stub"],
        default=None,
        dest="gi_insight_source",
        help="Source of insight texts: provider (LLM), summary_bullets, or stub (default: stub). "
        "See docs/guides/GROUNDED_INSIGHTS_GUIDE.md.",
    )
    parser.add_argument(
        "--gi-max-insights",
        type=int,
        default=None,
        metavar="N",
        dest="gi_max_insights",
        help="Max number of insights when using provider or summary_bullets (default: 5).",
    )
    parser.add_argument(
        "--generate-kg",
        action="store_true",
        dest="generate_kg",
        help="Generate Knowledge Graph (KG) artifacts (kg.json) per episode. "
        "Requires generate_metadata. Separate from GIL; see docs/guides/KNOWLEDGE_GRAPH_GUIDE.md.",
    )
    parser.add_argument(
        "--kg-extraction-source",
        choices=["stub", "summary_bullets", "provider"],
        default=None,
        dest="kg_extraction_source",
        help="KG extraction source: provider (LLM JSON), summary_bullets, or stub. "
        "Default: summary_bullets. See KNOWLEDGE_GRAPH_GUIDE.md.",
    )
    parser.add_argument(
        "--kg-max-topics",
        type=int,
        default=None,
        metavar="N",
        dest="kg_max_topics",
        help="Max topic nodes (summary bullets or provider extraction). Default: 5.",
    )
    parser.add_argument(
        "--kg-max-entities",
        type=int,
        default=None,
        metavar="N",
        dest="kg_max_entities",
        help="Max entity nodes from provider KG extraction. Default: 15.",
    )
    parser.add_argument(
        "--kg-extraction-model",
        default=None,
        dest="kg_extraction_model",
        help="Optional model override for KG LLM extraction (uses summary model if omitted).",
    )
    parser.add_argument(
        "--kg-extraction-provider",
        choices=[
            "transformers",
            "hybrid_ml",
            "openai",
            "gemini",
            "grok",
            "mistral",
            "deepseek",
            "anthropic",
            "ollama",
        ],
        default=None,
        dest="kg_extraction_provider",
        help="When kg_extraction_source is provider, which backend runs extract_kg_graph "
        "(default: same as summary_provider).",
    )
    parser.add_argument(
        "--no-kg-merge-pipeline-entities",
        action="store_false",
        dest="kg_merge_pipeline_entities",
        help="Do not merge detected hosts/guests after provider KG extraction.",
    )
    parser.set_defaults(kg_merge_pipeline_entities=True)
    parser.add_argument(
        "--quote-extraction-provider",
        choices=[
            "transformers",
            "hybrid_ml",
            "openai",
            "gemini",
            "grok",
            "mistral",
            "deepseek",
            "anthropic",
            "ollama",
        ],
        default=None,
        dest="quote_extraction_provider",
        help="GIL quote extraction (QA) provider; same as summary (default: transformers).",
    )
    parser.add_argument(
        "--entailment-provider",
        choices=[
            "transformers",
            "hybrid_ml",
            "openai",
            "gemini",
            "grok",
            "mistral",
            "deepseek",
            "anthropic",
            "ollama",
        ],
        default=None,
        dest="entailment_provider",
        help="Provider for GIL entailment (NLI). Same backends as summary (default: transformers).",
    )
    parser.add_argument(
        "--vector-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable corpus vector indexing after pipeline finalize (RFC-061 / #484).",
    )
    parser.add_argument(
        "--vector-backend",
        choices=["faiss", "qdrant"],
        default=None,
        help="Vector index backend (default: faiss). qdrant is not implemented in Phase 1.",
    )
    parser.add_argument(
        "--vector-index-path",
        default=None,
        dest="vector_index_path",
        help="Directory for FAISS index (relative paths resolve under output_dir).",
    )
    parser.add_argument(
        "--vector-embedding-model",
        default=None,
        dest="vector_embedding_model",
        help="Embedding model id or registry alias for corpus vectors (default: config).",
    )
    parser.add_argument(
        "--vector-faiss-index-mode",
        choices=["auto", "flat", "ivf_flat", "ivfpq"],
        default=None,
        dest="vector_faiss_index_mode",
        help="FAISS structure: auto thresholds per #484, or force flat/ivf_flat/ivfpq.",
    )
    parser.add_argument(
        "--vector-index-types",
        default=None,
        dest="vector_index_types",
        metavar="TYPES",
        help=(
            "Comma-separated doc types: insight,quote,summary,transcript,"
            "kg_topic,kg_entity (default: all)."
        ),
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
        choices=["spacy", "openai", "gemini", "anthropic", "mistral", "grok", "deepseek", "ollama"],
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
        choices=[
            "transformers",
            "hybrid_ml",
            "openai",
            "gemini",
            "anthropic",
            "mistral",
            "grok",
            "deepseek",
            "ollama",
        ],
        default="transformers",
        help="Summary provider to use (default: transformers)",
    )
    parser.add_argument(
        "--hybrid-map-model",
        default=None,
        help="Hybrid MAP model (classic). Example: longt5-base, long-fast, pegasus-cnn.",
    )
    parser.add_argument(
        "--hybrid-reduce-model",
        default=None,
        help="Hybrid REDUCE model (instruction-tuned). Example: google/flan-t5-base.",
    )
    parser.add_argument(
        "--hybrid-reduce-backend",
        choices=["transformers", "ollama", "llama_cpp"],
        default=None,
        help="Hybrid REDUCE backend (default: transformers).",
    )
    parser.add_argument(
        "--hybrid-map-device",
        choices=["cuda", "mps", "cpu", "auto"],
        default=None,
        help="Device for hybrid MAP model (default: summary_device/auto).",
    )
    parser.add_argument(
        "--hybrid-reduce-device",
        choices=["cuda", "mps", "cpu", "auto"],
        default=None,
        help="Device for hybrid REDUCE model (default: summary_device/auto).",
    )
    parser.add_argument(
        "--hybrid-quantization",
        default=None,
        help="Quantization for hybrid models (e.g., llama_cpp). Optional.",
    )
    parser.add_argument(
        "--summary-mode-id",
        default=None,
        help="RFC-044 summarization mode ID (e.g., ml_prod_authority_v1). "
        "When set, providers can use promoted baseline defaults from the Model Registry.",
    )
    parser.add_argument(
        "--summary-mode-precedence",
        choices=["mode", "config"],
        default=None,
        help="When --summary-mode-id is set, controls precedence: "
        "'mode' = mode overrides config; 'config' = config overrides mode.",
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
        "--summarization-device",
        choices=["cuda", "mps", "cpu", "auto"],
        default=None,
        help="Device for summarization stage (overrides summary_device, Issue #387). "
        "Allows CPU/GPU mix to regain overlap (e.g., transcription on CPU, summarization on MPS).",
    )
    parser.add_argument(
        "--mps-exclusive",
        action="store_true",
        default=True,
        dest="mps_exclusive",
        help="Serialize GPU work on MPS to prevent memory contention (default: enabled)",
    )
    parser.add_argument(
        "--no-mps-exclusive",
        action="store_false",
        dest="mps_exclusive",
        help="Allow concurrent GPU operations on MPS (for systems with sufficient GPU memory)",
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
    parser.add_argument(
        "--transcript-cleaning-strategy",
        choices=["pattern", "llm", "hybrid"],
        default=None,
        help="Strategy for cleaning transcripts before summarization. "
        "'pattern': uses regex-based cleaning (default). "
        "'llm': uses LLM-based semantic cleaning. "
        "'hybrid': uses pattern-based, then conditionally LLM-based if needed (default: hybrid). "
        "Only applies when using LLM providers for summarization.",
    )


def _add_mistral_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Mistral API-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--mistral-api-key",
        type=str,
        default=None,
        help="Mistral API key (or set MISTRAL_API_KEY env var)",
    )
    parser.add_argument(
        "--mistral-api-base",
        type=str,
        default=None,
        help="Mistral API base URL (for E2E testing, or set MISTRAL_API_BASE env var)",
    )
    parser.add_argument(
        "--mistral-transcription-model",
        type=str,
        default=None,
        help="Mistral transcription model (default: voxtral-mini-latest)",
    )
    parser.add_argument(
        "--mistral-speaker-model",
        type=str,
        default=None,
        help="Mistral speaker detection model (default: mistral-small-latest)",
    )
    parser.add_argument(
        "--mistral-summary-model",
        type=str,
        default=None,
        help="Mistral summarization model (default: mistral-small-latest)",
    )
    parser.add_argument(
        "--mistral-temperature",
        type=float,
        default=None,
        help="Temperature for Mistral models (0.0-1.0, default: 0.3)",  # noqa: E501
    )
    parser.add_argument(
        "--mistral-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for Mistral responses (default: model-specific)",
    )
    parser.add_argument(
        "--mistral-cleaning-model",
        type=str,
        default=None,
        help=(
            "Mistral model for transcript cleaning "
            "(default: mistral-small-latest, cheaper than summary model)"
        ),
    )
    parser.add_argument(
        "--mistral-cleaning-temperature",
        type=float,
        default=None,
        help="Temperature for Mistral cleaning (0.0-1.0, default: 0.2, lower = more deterministic)",
    )


def _add_deepseek_arguments(parser: argparse.ArgumentParser) -> None:
    """Add DeepSeek API-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--deepseek-api-key",
        type=str,
        default=None,
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
    )
    parser.add_argument(
        "--deepseek-api-base",
        type=str,
        default=None,
        help="DeepSeek API base URL (for E2E testing, or set DEEPSEEK_API_BASE env var)",
    )
    parser.add_argument(
        "--deepseek-speaker-model",
        type=str,
        default=None,
        help="DeepSeek speaker detection model (default: deepseek-chat)",
    )
    parser.add_argument(
        "--deepseek-summary-model",
        type=str,
        default=None,
        help="DeepSeek summarization model (default: deepseek-chat)",
    )
    parser.add_argument(
        "--deepseek-temperature",
        type=float,
        default=None,
        help="Temperature for DeepSeek models (0.0-2.0, default: 0.3)",  # noqa: E501
    )
    parser.add_argument(
        "--deepseek-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for DeepSeek responses (default: model-specific)",  # noqa: E501
    )
    parser.add_argument(
        "--deepseek-cleaning-model",
        type=str,
        default=None,
        help=(
            "DeepSeek model for transcript cleaning "
            "(default: deepseek-chat, cheaper than summary model)"
        ),
    )
    parser.add_argument(
        "--deepseek-cleaning-temperature",
        type=float,
        default=None,
        help=(
            "Temperature for DeepSeek cleaning "
            "(0.0-2.0, default: 0.2, lower = more deterministic)"
        ),
    )


def _add_grok_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Grok API-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--grok-api-key",
        type=str,
        default=None,
        help="Grok API key (or set GROK_API_KEY env var)",
    )
    parser.add_argument(
        "--grok-api-base",
        type=str,
        default=None,
        help="Grok API base URL (for E2E testing, or set GROK_API_BASE env var)",
    )
    parser.add_argument(
        "--grok-speaker-model",
        type=str,
        default=None,
        help="Grok speaker detection model (default: grok-2)",
    )
    parser.add_argument(
        "--grok-summary-model",
        type=str,
        default=None,
        help="Grok summarization model (default: grok-2)",
    )
    parser.add_argument(
        "--grok-temperature",
        type=float,
        default=None,
        help="Temperature for Grok models (0.0-2.0, default: 0.3)",  # noqa: E501
    )
    parser.add_argument(
        "--grok-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for Grok responses (default: model-specific)",
    )
    parser.add_argument(
        "--grok-cleaning-model",
        type=str,
        default=None,
        help=(
            "Grok model for transcript cleaning "
            "(default: grok-3-mini, cheaper than summary model)"
        ),
    )
    parser.add_argument(
        "--grok-cleaning-temperature",
        type=float,
        default=None,
        help=(
            "Temperature for Grok cleaning " "(0.0-2.0, default: 0.2, lower = more deterministic)"
        ),
    )


def _add_ollama_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Ollama API-related arguments to parser.

    Args:
        parser: Argument parser to add arguments to
    """
    parser.add_argument(
        "--ollama-api-base",
        type=str,
        default=None,
        help=(
            "Ollama API base URL (for E2E testing, or set OLLAMA_API_BASE env var, "
            "default: http://localhost:11434/v1)"
        ),
    )
    parser.add_argument(
        "--ollama-speaker-model",
        type=str,
        default=None,
        help=("Ollama speaker detection model " "(default: llama3.1:8b for both test and prod)"),
    )
    parser.add_argument(
        "--ollama-summary-model",
        type=str,
        default=None,
        help=("Ollama summarization model " "(default: llama3.1:8b for both test and prod)"),
    )
    parser.add_argument(
        "--ollama-temperature",
        type=float,
        default=None,
        help="Temperature for Ollama models (0.0-2.0, default: 0.3)",
    )
    parser.add_argument(
        "--ollama-max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for Ollama responses (default: model-specific)",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=None,
        help="Timeout in seconds for Ollama API calls (default: 120, local inference can be slow)",
    )
    parser.add_argument(
        "--ollama-cleaning-model",
        type=str,
        default=None,
        help=(
            "Ollama model for transcript cleaning " "(default: llama3.1:8b, same as summary model)"
        ),
    )
    parser.add_argument(
        "--ollama-cleaning-temperature",
        type=float,
        default=None,
        help="Temperature for Ollama cleaning (0.0-2.0, default: 0.2, lower = more deterministic)",
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
    # Debug: Log grok model values from config file
    if "grok_speaker_model" in config_data:
        import logging

        logging.getLogger(__name__).debug(
            "Config file grok_speaker_model: %s", config_data["grok_speaker_model"]
        )
    if "grok_summary_model" in config_data:
        import logging

        logging.getLogger(__name__).debug(
            "Config file grok_summary_model: %s", config_data["grok_summary_model"]
        )
    valid_dests = {action.dest for action in parser._actions if action.dest}
    # Also check against Config model field aliases (some fields are config-only, not CLI args)
    config_field_aliases = {
        field.alias or field_name for field_name, field in config.Config.model_fields.items()
    }
    valid_keys = valid_dests | config_field_aliases
    unknown_keys = [key for key in config_data.keys() if key not in valid_keys]
    if unknown_keys:
        raise ValueError("Unknown config option(s): " + ", ".join(sorted(unknown_keys)))

    try:
        config_model = config.Config.model_validate(config_data)
        # Debug: Log grok model values after validation
        import logging

        logging.getLogger(__name__).debug(
            "Config model grok_speaker_model: %s", config_model.grok_speaker_model
        )
        logging.getLogger(__name__).debug(
            "Config model grok_summary_model: %s", config_model.grok_summary_model
        )
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


def _parse_doctor_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse arguments for doctor subcommand (Issue #379).

    Args:
        argv: Command-line arguments (excluding 'doctor')

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run diagnostic checks on podcast_scraper environment",
        prog="podcast_scraper doctor",
    )
    parser.add_argument(
        "--check-network",
        action="store_true",
        help="Also check network connectivity (optional)",
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Try loading default Whisper and summarizer models once (slow, optional)",
    )
    args = parser.parse_args(argv)
    return args


def _parse_pricing_assumptions_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse arguments for ``pricing-assumptions`` subcommand."""
    parser = argparse.ArgumentParser(
        description="Show LLM pricing assumptions YAML status and staleness hints.",
        prog="podcast_scraper pricing-assumptions",
    )
    parser.add_argument(
        "--file",
        dest="assumptions_file",
        type=str,
        default="config/pricing_assumptions.yaml",
        help="Path to pricing assumptions YAML (default: config/pricing_assumptions.yaml)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when last_reviewed is older than stale_review_after_days in metadata",
    )
    return parser.parse_args(argv or [])


def _run_pricing_assumptions(args: argparse.Namespace) -> int:
    """Print pricing assumptions report; optional strict exit on stale metadata."""
    from . import pricing_assumptions

    path_cfg = str(getattr(args, "assumptions_file", "") or "").strip()
    report = pricing_assumptions.format_status_report(path_cfg)
    print(report, end="")
    if getattr(args, "strict", False):
        payload, _resolved = pricing_assumptions.get_loaded_table(path_cfg)
        if payload:
            stale, _msgs = pricing_assumptions.check_staleness(payload)
            if stale:
                return 1
    return 0


def _doctor_check_python() -> bool:
    """Check Python version; print result. Return True if OK."""
    import sys

    print("✓ Checking Python version...")
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print(f"  ✗ Python {v.major}.{v.minor} is too old")
        print("    Required: Python 3.10 or higher")
        return False
    print(f"  ✓ Python {v.major}.{v.minor}.{v.micro}")
    return True


def _doctor_check_ffmpeg() -> bool:
    """Check ffmpeg; print result. Return True if OK."""
    import shutil
    import subprocess

    print("\n✓ Checking ffmpeg...")
    path = shutil.which("ffmpeg")
    if not path:
        print("  ✗ ffmpeg not found in PATH")
        print("    Install: https://ffmpeg.org/download.html")
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"  ✓ ffmpeg found: {path}")
            print(f"    {result.stdout.split(chr(10))[0]}")
            return True
        print(f"  ✗ ffmpeg found but failed to run: {path}")
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"  ✗ ffmpeg check failed: {e}")
        return False


def _doctor_check_write_permissions() -> bool:
    """Check write permissions; print result. Return True if OK."""
    print("\n✓ Checking write permissions...")
    try:
        test_dir = Path.home() / ".podcast_scraper_test"
        test_dir.mkdir(exist_ok=True)
        (test_dir / "test_write.txt").write_text("test")
        (test_dir / "test_write.txt").unlink()
        test_dir.rmdir()
        print("  ✓ Write permissions OK")
        return True
    except Exception as e:
        print(f"  ✗ Write permission check failed: {e}")
        return False


def _doctor_check_cache_dir() -> bool:
    """Check model cache dir; print result. Return True if OK (or optional)."""
    print("\n✓ Checking model cache directory...")
    try:
        from .cache import manager as cache_manager

        cache_dir = cache_manager.get_all_cache_info().get("whisper", {}).get("dir")
        if cache_dir and cache_dir.exists():
            test_file = cache_dir / ".podcast_scraper_test_write"
            try:
                test_file.write_text("test")
                test_file.unlink()
                print(f"  ✓ Model cache directory writable: {cache_dir}")
                return True
            except Exception as e:
                print(f"  ✗ Model cache directory not writable: {e}")
                return False
        print("  ⚠ Model cache directory not found (will be created on first use)")
        return True
    except ImportError:
        print("  ⚠ Cache manager not available (optional)")
        return True


def _doctor_check_ml_deps() -> bool:
    """Check ML deps (optional); print result. Always returns True."""
    print("\n✓ Checking ML dependencies...")
    for label, mod in [
        ("PyTorch", "torch"),
        ("Transformers", "transformers"),
        ("Whisper", "whisper"),
        ("spaCy", "spacy"),
    ]:
        try:
            m = __import__(mod)
            print(f"  ✓ {label}: {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"  ⚠ {label} not installed (required for some features)")
    return True


def _doctor_check_models_load() -> bool:
    """Check loading default Whisper and summarizer; print result."""
    print("\n✓ Checking model load (default Whisper and summarizer)...")
    ok = True
    try:
        import whisper

        model = getattr(config, "TEST_DEFAULT_WHISPER_MODEL", "base.en")
        whisper.load_model(model, device="cpu", download_root=None)
        print(f"  ✓ Whisper model loaded: {model}")
    except Exception as e:
        print(f"  ✗ Whisper model load failed: {e}")
        ok = False
    try:
        from podcast_scraper.providers.ml.summarizer import SummaryModel

        model = getattr(config, "TEST_DEFAULT_SUMMARY_MODEL", "facebook/bart-base")
        SummaryModel(model, device="cpu")
        print(f"  ✓ Summarizer model loaded: {model}")
    except Exception as e:
        print(f"  ✗ Summarizer model load failed: {e}")
        ok = False
    return ok


def _doctor_check_network() -> bool:
    """Check network connectivity; print result. Return True if OK."""
    print("\n✓ Checking network connectivity...")
    try:
        import urllib.request

        urllib.request.urlopen("https://www.google.com", timeout=5)
        print("  ✓ Network connectivity OK")
        return True
    except Exception as e:
        print(f"  ✗ Network connectivity check failed: {e}")
        return False


def _run_doctor_checks(check_network: bool = False, check_models: bool = False) -> int:
    """Run diagnostic checks (Issue #379, #429).

    Args:
        check_network: Whether to check network connectivity
        check_models: Whether to try loading default ML models once

    Returns:
        Exit code (0 = all checks passed, 1 = some checks failed)
    """
    print("\n" + "=" * 60)
    print("podcast_scraper Doctor - Diagnostic Checks")
    print("=" * 60 + "\n")
    all_passed = _doctor_check_python()
    all_passed = _doctor_check_ffmpeg() and all_passed
    all_passed = _doctor_check_write_permissions() and all_passed
    all_passed = _doctor_check_cache_dir() and all_passed
    _doctor_check_ml_deps()
    if check_models:
        all_passed = _doctor_check_models_load() and all_passed
    if check_network:
        all_passed = _doctor_check_network() and all_passed
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed!")
        return 0
    print("✗ Some checks failed. Please fix the issues above.")
    return 1


def _run_gi(args: argparse.Namespace, log: Optional[logging.Logger] = None) -> int:
    """Dispatch gi subcommand (validate, inspect, show-insight, explore, query)."""
    logger = log or _LOGGER
    sub = getattr(args, "gi_subcommand", None)
    if sub == "validate":
        return _run_gi_validate(args, logger=logger)
    if sub == "inspect":
        return _run_gi_inspect(args, logger=logger)
    if sub == "show-insight":
        return _run_gi_show_insight(args, logger=logger)
    if sub == "explore":
        return _run_gi_explore(args, logger=logger)
    if sub == "query":
        return _run_gi_query(args, logger=logger)
    if sub == "export":
        return _run_gi_export(args, logger=logger)
    logger.error("Unknown gi subcommand: %s", sub)
    return 1


def _run_gi_export(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Export all gi.json under a run as NDJSON or merged JSON (symmetric with ``kg export``)."""
    import json
    import sys

    from .gi.contracts import build_gi_corpus_bundle_output
    from .gi.corpus import export_merged_json, export_ndjson, load_gi_artifacts
    from .gi.explore import EXIT_INVALID_ARGS, EXIT_NO_ARTIFACTS, EXIT_SUCCESS, scan_artifact_paths

    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("--output-dir is required")
        return EXIT_INVALID_ARGS
    out_root = Path(output_dir)
    if not out_root.is_dir():
        logger.error("Output directory does not exist: %s", output_dir)
        return EXIT_NO_ARTIFACTS
    paths = scan_artifact_paths(out_root)
    if not paths:
        logger.error("No .gi.json artifacts found under %s", output_dir)
        return EXIT_NO_ARTIFACTS
    strict = getattr(args, "strict", False)
    try:
        loaded = load_gi_artifacts(paths, validate=True, strict=strict)
    except Exception as e:
        logger.error("Validation failed: %s", format_exception_for_log(e))
        return 1
    if not loaded:
        logger.error("No valid artifacts loaded")
        return EXIT_NO_ARTIFACTS
    fmt = getattr(args, "format", "ndjson")
    out_file = getattr(args, "out", None)
    if fmt == "ndjson":
        if out_file:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:

                def _write_ndj(s: str) -> None:
                    f.write(s)

                export_ndjson(loaded, output_dir=out_root, stream_write=_write_ndj)
            logger.info("Wrote NDJSON to %s", out_file)
        else:

            def _write_stdout(s: str) -> None:
                sys.stdout.write(s)

            export_ndjson(loaded, output_dir=out_root, stream_write=_write_stdout)
        return EXIT_SUCCESS
    bundle = export_merged_json(loaded, output_dir=out_root)
    validated = build_gi_corpus_bundle_output(bundle)
    text = json.dumps(validated.model_dump(mode="json"), indent=2, ensure_ascii=False)
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(text, encoding="utf-8")
        logger.info("Wrote merged JSON to %s", out_file)
    else:
        print(text)
    return EXIT_SUCCESS


def _run_gi_validate(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Validate .gi.json files (parity with ``kg validate``)."""
    from .gi.explore import EXIT_INVALID_ARGS, EXIT_NO_ARTIFACTS, EXIT_SUCCESS
    from .gi.io import collect_gi_paths_from_inputs, read_artifact

    paths_arg = list(getattr(args, "paths", None) or [])
    if not paths_arg:
        logger.error("Provide one or more paths to .gi.json files or directories")
        return EXIT_INVALID_ARGS
    try:
        paths = collect_gi_paths_from_inputs([Path(p) for p in paths_arg])
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", format_exception_for_log(e))
        return EXIT_INVALID_ARGS
    if not paths:
        logger.error("No .gi.json files found")
        return EXIT_NO_ARTIFACTS
    strict = getattr(args, "strict", False)
    quiet = getattr(args, "quiet", False)
    failed = 0
    for path in paths:
        try:
            read_artifact(path, validate=True, strict=strict)
            if not quiet:
                print(f"OK {path}")
        except Exception as e:
            failed += 1
            logger.error("FAIL %s: %s", path, format_exception_for_log(e))
    if failed:
        logger.error("%s of %s file(s) failed validation", failed, len(paths))
        return 1
    if not quiet:
        print(f"All {len(paths)} file(s) passed validation.")
    return EXIT_SUCCESS


def _resolve_artifact_path_gi(args: argparse.Namespace) -> Optional[Path]:
    """Resolve path to .gi.json from --episode-path or --output-dir + --episode-id."""
    ep_path = getattr(args, "episode_path", None)
    output_dir = getattr(args, "output_dir", None)
    episode_id = getattr(args, "episode_id", None)

    if ep_path:
        p = Path(ep_path)
        if p.is_file() and p.suffix == ".json" and ".gi.json" in p.name:
            return p
        if p.is_dir():
            for f in p.glob("*.gi.json"):
                return f
        if p.is_file():
            return p
        return None

    if output_dir and episode_id:
        from .gi import find_artifact_by_episode_id

        return find_artifact_by_episode_id(Path(output_dir), episode_id)
    return None


def _run_gi_inspect(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run gi inspect: load artifact, build InspectOutput, print."""
    from .gi import build_inspect_output, load_artifact_and_transcript

    artifact_path = _resolve_artifact_path_gi(args)
    if not artifact_path:
        episode_id = getattr(args, "episode_id", None)
        episode_path = getattr(args, "episode_path", None)
        output_dir = getattr(args, "output_dir", None)
        if not episode_id and not episode_path:
            logger.error("Provide --episode-id with --output-dir, or --episode-path")
        elif output_dir and not episode_id:
            logger.error("Provide --episode-id when using --output-dir")
        else:
            logger.error("Artifact not found for the given episode")
        return 1

    try:
        artifact, transcript_text, _ = load_artifact_and_transcript(
            artifact_path,
            validate=True,
            strict=getattr(args, "strict", False),
            load_transcript=True,
        )
    except FileNotFoundError as e:
        logger.error("%s", format_exception_for_log(e))
        return 1
    except ValueError as e:
        logger.error("Validation failed: %s", format_exception_for_log(e))
        return 1

    out = build_inspect_output(artifact, transcript_text)
    fmt = getattr(args, "format", "pretty")
    show = getattr(args, "show", False)
    stats = getattr(args, "stats", True)

    if fmt == "json":
        print(out.model_dump_json(indent=2))
        return 0

    # Pretty
    lines = [
        f"Episode: {out.episode_id}",
        f"Schema: {out.schema_version}  Model: {out.model_version}",
    ]
    if stats and out.stats:
        s = out.stats
        lines.append(
            f"Insights: {s.get('insight_count', 0)}  "
            f"Grounded: {s.get('grounded_count', 0)}  "
            f"Ungrounded: {s.get('ungrounded_count', 0)}  "
            f"Quotes: {s.get('quote_count', 0)}"
        )
    for ins in out.insights:
        lines.append(f"  [{ins.insight_id}] grounded={ins.grounded}")
        if show or ins.supporting_quotes:
            lines.append(f"    {ins.text[:200]}{'...' if len(ins.text) > 200 else ''}")
            for q in ins.supporting_quotes:
                lines.append(
                    f"    → {q.quote_id}: {q.text[:100]}{'...' if len(q.text) > 100 else ''}"
                )
    print("\n".join(lines))
    return 0


def _run_gi_show_insight(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run gi show-insight: find insight by id, print with quotes and evidence."""
    from .gi import (
        build_inspect_output,
        find_artifact_by_insight_id,
        load_artifact_and_transcript,
    )

    insight_id = getattr(args, "id", None)
    if not insight_id:
        logger.error("--id INSIGHT_ID is required")
        return 1

    artifact_path = _resolve_artifact_path_gi(args)
    if not artifact_path:
        output_dir = getattr(args, "output_dir", None)
        if output_dir:
            artifact_path = find_artifact_by_insight_id(Path(output_dir), insight_id)
        if not artifact_path:
            logger.error(
                "Provide --episode-path to the .gi.json file, or --output-dir to scan for "
                "artifact containing insight %s",
                insight_id,
            )
            return 1

    try:
        artifact, transcript_text, _ = load_artifact_and_transcript(
            artifact_path,
            validate=True,
            strict=False,
            load_transcript=True,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", format_exception_for_log(e))
        return 1

    out = build_inspect_output(artifact, transcript_text)
    insight = next((i for i in out.insights if i.insight_id == insight_id), None)
    if not insight:
        logger.error("Insight %s not found in artifact", insight_id)
        return 1

    fmt = getattr(args, "format", "pretty")
    context_chars = getattr(args, "context_chars", 80)

    if fmt == "json":
        print(insight.model_dump_json(indent=2))
        return 0

    lines = [
        f"Insight: {insight.insight_id}",
        f"  grounded={insight.grounded}  episode_id={insight.episode_id}",
        f"  {insight.text}",
        "Supporting quotes:",
    ]
    for q in insight.supporting_quotes:
        lines.append(f"  [{q.quote_id}] {q.text}")
        if q.evidence.excerpt:
            ev = q.evidence
            lines.append("    evidence: " + ev.transcript_ref)
            lines.append(f"      span: [{ev.char_start}:{ev.char_end}]")
            lines.append(f"    excerpt: {q.evidence.excerpt}")
            if transcript_text and context_chars > 0:
                start = max(0, q.evidence.char_start - context_chars)
                end = min(len(transcript_text), q.evidence.char_end + context_chars)
                lines.append(f"    context: ...{transcript_text[start:end]}...")
    print("\n".join(lines))
    return 0


def _run_gi_explore(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run gi explore: scan output_dir, filter by topic/speaker, print insights with quotes."""
    import json

    from .gi.explore import (
        aggregate_topic_entries_for_insights,
        build_explore_output,
        EXIT_INVALID_ARGS,
        EXIT_NO_ARTIFACTS,
        EXIT_NO_RESULTS,
        EXIT_STRICT_VALIDATION_FAILED,
        EXIT_SUCCESS,
        explore_output_to_rfc_dict,
        explore_resolve_insights_and_loaded,
        ExploreValidationError,
        scan_artifact_paths,
    )

    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("--output-dir is required for gi explore")
        return EXIT_INVALID_ARGS
    out_path = Path(output_dir)
    if not out_path.is_dir():
        logger.error("Output directory does not exist: %s", output_dir)
        return EXIT_NO_ARTIFACTS

    paths = scan_artifact_paths(out_path)
    if not paths:
        logger.error("No .gi.json artifacts found under %s", output_dir)
        return EXIT_NO_ARTIFACTS

    strict = getattr(args, "strict", False)
    topic = getattr(args, "topic", None)
    speaker = getattr(args, "speaker", None)
    grounded_only = getattr(args, "grounded_only", False)
    min_confidence = getattr(args, "min_confidence", None)
    limit = getattr(args, "limit", 50)
    sort_by = getattr(args, "explore_sort", "confidence")

    try:
        insights, _semantic_ranked, loaded, ep_searched = explore_resolve_insights_and_loaded(
            out_path,
            paths,
            topic=topic,
            speaker=speaker,
            grounded_only=grounded_only,
            min_confidence=min_confidence,
            sort_by=cast(Literal["confidence", "time"], sort_by),
            strict=strict,
        )
    except ExploreValidationError as e:
        logger.error(
            "Strict validation failed for %s: %s",
            e.path,
            format_exception_for_log(e),
        )
        return EXIT_STRICT_VALIDATION_FAILED

    if limit and limit > 0:
        insights = insights[:limit]

    topic_rows = aggregate_topic_entries_for_insights(loaded, insights)
    explore_out = build_explore_output(
        insights,
        episodes_searched=ep_searched,
        topic=topic,
        speaker_filter=speaker,
        topics=topic_rows,
    )

    out_format = getattr(args, "format", "pretty")
    out_file = getattr(args, "out", None)
    if out_format == "json":
        payload = json.dumps(explore_output_to_rfc_dict(explore_out), indent=2)
    else:
        lines = [
            f"Topic: {explore_out.topic or '(all)'}",
            f"Speaker filter: {explore_out.speaker_filter or '(none)'}",
            f"Episodes searched: {explore_out.episodes_searched}",
            f"Insights: {explore_out.summary.get('insight_count', 0)} "
            f"(grounded: {explore_out.summary.get('grounded_insight_count', 0)})",
            f"Quotes: {explore_out.summary.get('quote_count', 0)}",
            f"Distinct speakers (quotes): {explore_out.summary.get('speaker_count', 0)}",
            "",
        ]
        if explore_out.top_speakers:
            lines.append("Top speakers (by quote count):")
            for ts in explore_out.top_speakers[:10]:
                sid = str(ts.speaker_id)
                nm = (ts.name or "").strip()
                if nm and nm.casefold() != sid.casefold():
                    label = f"{sid} ({nm})"
                else:
                    label = sid
                lines.append(f"  {label}: quotes={ts.quote_count}, insights={ts.insight_count}")
            lines.append("")
        for ins in explore_out.insights:
            ep_title = ins.episode_title or ""
            lines.append(
                f"[{ins.insight_id}] {ins.episode_id}"
                f"{(' — ' + ep_title) if ep_title else ''} grounded={ins.grounded}"
            )
            lines.append(f"  {ins.text}")
            for q in ins.supporting_quotes:
                lines.append(
                    f"    quote: {q.text[:80]}..." if len(q.text) > 80 else f"    quote: {q.text}"
                )
        payload = "\n".join(lines)

    if out_file:
        Path(out_file).write_text(payload, encoding="utf-8")
        logger.info("Wrote output to %s", out_file)
    else:
        print(payload)

    if not insights and (topic or speaker):
        return EXIT_NO_RESULTS
    return EXIT_SUCCESS


def _run_gi_query(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run gi query: UC4 natural-language patterns → explore-style RFC JSON answer."""
    import json

    from .gi.explore import (
        EXIT_INVALID_ARGS,
        EXIT_NO_ARTIFACTS,
        EXIT_SUCCESS,
        run_uc4_semantic_qa,
        scan_artifact_paths,
    )

    output_dir = getattr(args, "output_dir", None)
    question = getattr(args, "question", None)
    if not output_dir or not question or not str(question).strip():
        logger.error("--output-dir and --question are required for gi query")
        return EXIT_INVALID_ARGS
    out_path = Path(output_dir)
    if not out_path.is_dir():
        logger.error("Output directory does not exist: %s", output_dir)
        return EXIT_NO_ARTIFACTS
    if not scan_artifact_paths(out_path):
        logger.error("No .gi.json artifacts found under %s", output_dir)
        return EXIT_NO_ARTIFACTS

    limit = getattr(args, "query_limit", 20)
    strict = getattr(args, "strict", False)
    result = run_uc4_semantic_qa(out_path, question.strip(), limit=limit, strict=strict)
    if result is None:
        logger.error(
            "Question did not match a supported pattern (RFC-050 UC4). Examples: "
            "'What insights about X?', 'What insights are there about X?', "
            "'What did Y say?', 'What did Y say about X?', "
            "'Which topics have the most insights?', 'Top topics'."
        )
        return EXIT_INVALID_ARGS

    fmt = getattr(args, "format", "json")
    if fmt == "json":
        print(json.dumps(result, indent=2))
    else:
        print(result.get("explanation", ""))
        ans = result.get("answer")
        if isinstance(ans, dict):
            summ = ans.get("summary") or {}
            print(
                f"Insights: {summ.get('insight_count', 0)}  "
                f"Episodes searched: {ans.get('episodes_searched', 0)}"
            )
    return EXIT_SUCCESS


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
        "--delete-model",
        type=str,
        default=None,
        metavar="MODEL_NAME",
        help=(
            "Delete cache for a specific Transformers model "
            "(e.g., 'google/pegasus-cnn_dailymail')"
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt when cleaning cache or deleting model",
    )

    args = parser.parse_args(argv)

    # Require either --status or --clean
    if not args.status and not args.clean:
        parser.error("Either --status or --clean must be specified")

    return args


def _parse_gi_args(gi_argv: Sequence[str]) -> argparse.Namespace:
    """Parse 'gi' subcommand arguments (validate, inspect, show-insight, explore, query)."""
    parser = argparse.ArgumentParser(
        prog="podcast_scraper gi",
        description="Inspect Grounded Insight Layer (GIL) artifacts (gi.json).",
    )
    subparsers = parser.add_subparsers(dest="gi_subcommand", required=True, help="Command")

    val = subparsers.add_parser(
        "validate",
        help="Validate .gi.json files against schema (use --strict for full JSON Schema)",
    )
    val.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="Files or directories containing .gi.json",
    )
    val.add_argument(
        "--strict",
        action="store_true",
        help="Full JSON Schema validation (docs/gi/gi.schema.json)",
    )
    val.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print failures",
    )

    # gi inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect one episode's GIL artifact")
    inspect_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory containing metadata/*.gi.json",
    )
    inspect_parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Episode ID (artifact episode_id field); required if not using --episode-path",
    )
    inspect_parser.add_argument(
        "--episode-path",
        type=str,
        default=None,
        help="Path to .gi.json file (or directory containing one .gi.json)",
    )
    inspect_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if artifact does not pass schema validation",
    )
    inspect_parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format (default: pretty)",
    )
    inspect_parser.add_argument(
        "--show",
        action="store_true",
        help="Show full insight text and supporting quotes",
    )
    inspect_parser.add_argument(
        "--stats",
        action="store_true",
        default=True,
        help="Show summary stats (default: True)",
    )
    inspect_parser.add_argument("--no-stats", dest="stats", action="store_false")

    # gi show-insight
    show_parser = subparsers.add_parser(
        "show-insight",
        help="Show one insight by ID with quotes and evidence",
    )
    show_parser.add_argument(
        "--id",
        type=str,
        required=True,
        metavar="INSIGHT_ID",
        help="Insight node id from gi.json (e.g. insight:a1b2c3d4e5f67890)",
    )
    show_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory containing metadata/*.gi.json",
    )
    show_parser.add_argument(
        "--episode-path",
        type=str,
        default=None,
        help="Path to .gi.json file for the episode",
    )
    show_parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format (default: pretty)",
    )
    show_parser.add_argument(
        "--context-chars",
        type=int,
        default=80,
        metavar="N",
        help="Characters of context around evidence span (default: 80)",
    )

    # gi explore
    explore_parser = subparsers.add_parser(
        "explore",
        help="Cross-episode query: insights about a topic with supporting quotes",
    )
    explore_parser.add_argument(
        "--topic",
        type=str,
        default=None,
        metavar="LABEL",
        help="Topic filter (match Topic label or substring in insight text)",
    )
    explore_parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        metavar="SUBSTRING",
        help="Speaker filter: substring match on quote speaker_id or graph speaker name (UC2)",
    )
    explore_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Output directory containing metadata/*.gi.json",
    )
    explore_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Max number of insights to return (default: 50)",
    )
    explore_parser.add_argument(
        "--grounded-only",
        action="store_true",
        help="Only return grounded insights",
    )
    explore_parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        metavar="0..1",
        help="Minimum insight confidence (0.0-1.0)",
    )
    explore_parser.add_argument(
        "--sort",
        choices=("confidence", "time"),
        default="confidence",
        dest="explore_sort",
        help="Sort insights by confidence (desc) or episode publish_date (desc; RFC-050)",
    )
    explore_parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format (default: pretty)",
    )
    explore_parser.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="PATH",
        help="Write output to file (optional)",
    )
    explore_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first artifact that fails schema validation (exit 5)",
    )

    # gi query (UC4: tiny pattern map → explore RFC answer)
    query_parser = subparsers.add_parser(
        "query",
        help="Natural-language question → matched explore result (RFC-050 UC4)",
    )
    query_parser.add_argument(
        "--question",
        type=str,
        required=True,
        metavar="TEXT",
        help="Question (e.g. 'What insights about inflation?' or 'What did Sam say?')",
    )
    query_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Output directory containing metadata/*.gi.json",
    )
    query_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        metavar="N",
        dest="query_limit",
        help="Max insights in the answer (default: 20)",
    )
    query_parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="json",
        help="Output format (default: json)",
    )
    query_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first artifact that fails schema validation (exit 5)",
    )

    gi_exp = subparsers.add_parser(
        "export",
        help="Export all gi.json under a run as NDJSON or merged JSON bundle (RFC-050 / kg parity)",
    )
    gi_exp.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Pipeline output directory to scan for .gi.json",
    )
    gi_exp.add_argument(
        "--format",
        choices=("ndjson", "merged"),
        default="ndjson",
        help="ndjson: one artifact per line; merged: single JSON document (default: ndjson)",
    )
    gi_exp.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="PATH",
        help="Write to file (default: stdout)",
    )
    gi_exp.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any artifact fails strict schema validation",
    )

    args = parser.parse_args(gi_argv)
    args.command = "gi"
    return args


def _parse_kg_args(kg_argv: Sequence[str]) -> argparse.Namespace:
    """Parse 'kg' subcommand arguments (validate, inspect, export, entities, topics)."""
    parser = argparse.ArgumentParser(
        prog="podcast_scraper kg",
        description="Knowledge Graph (KG) tools for per-episode kg.json (RFC-056).",
    )
    subparsers = parser.add_subparsers(dest="kg_subcommand", required=True, help="Command")

    val = subparsers.add_parser("validate", help="Validate .kg.json files against schema")
    val.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="Files or directories containing .kg.json",
    )
    val.add_argument(
        "--strict",
        action="store_true",
        help="Full JSON Schema validation (requires docs/kg/kg.schema.json)",
    )
    val.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print failures",
    )

    ins = subparsers.add_parser("inspect", help="Summarize one episode KG artifact")
    ins.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory containing metadata/*.kg.json",
    )
    ins.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Episode ID (artifact episode_id field); required if not using --episode-path",
    )
    ins.add_argument(
        "--episode-path",
        type=str,
        default=None,
        help="Path to .kg.json (or directory containing one .kg.json)",
    )
    ins.add_argument(
        "--strict",
        action="store_true",
        help="Fail if artifact does not pass strict schema validation",
    )
    ins.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format (default: pretty)",
    )

    exp = subparsers.add_parser(
        "export",
        help="Export all kg.json under a run as NDJSON or merged JSON bundle",
    )
    exp.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Pipeline output directory to scan for .kg.json",
    )
    exp.add_argument(
        "--format",
        choices=("ndjson", "merged"),
        default="ndjson",
        help="ndjson: one artifact per line; merged: single JSON document (default: ndjson)",
    )
    exp.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="PATH",
        help="Write to file (default: stdout)",
    )
    exp.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any artifact fails strict schema validation",
    )

    ent = subparsers.add_parser(
        "entities",
        help="Cross-episode roll-up of Entity nodes (episode counts, mentions)",
    )
    ent.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Pipeline output directory to scan",
    )
    ent.add_argument(
        "--min-episodes",
        type=int,
        default=1,
        metavar="N",
        help="Minimum distinct episodes for an entity to appear (default: 1)",
    )
    ent.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format (default: pretty)",
    )
    ent.add_argument(
        "--strict",
        action="store_true",
        help="Skip artifacts that fail strict schema validation",
    )

    top = subparsers.add_parser(
        "topics",
        help="Topic pair co-occurrence within the same episode",
    )
    top.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Pipeline output directory to scan",
    )
    top.add_argument(
        "--min-support",
        type=int,
        default=1,
        metavar="N",
        help="Minimum episodes in which the pair appears (default: 1)",
    )
    top.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format (default: pretty)",
    )
    top.add_argument(
        "--strict",
        action="store_true",
        help="Skip artifacts that fail strict schema validation",
    )

    args = parser.parse_args(kg_argv)
    args.command = "kg"
    return args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments, optionally merging configuration file defaults."""
    # Check if first argument is "gi" subcommand (#438)
    if argv and len(argv) > 0 and argv[0] == "gi":
        gi_argv = list(argv[1:]) if len(argv) > 1 else []
        return _parse_gi_args(gi_argv)

    if argv and len(argv) > 0 and argv[0] == "kg":
        kg_argv = list(argv[1:]) if len(argv) > 1 else []
        return _parse_kg_args(kg_argv)

    if argv and len(argv) > 0 and argv[0] == "search":
        from .search.cli_handlers import parse_search_argv

        search_argv = list(argv[1:]) if len(argv) > 1 else []
        return parse_search_argv(search_argv)

    if argv and len(argv) > 0 and argv[0] == "index":
        from .search.cli_handlers import parse_index_argv

        index_argv = list(argv[1:]) if len(argv) > 1 else []
        return parse_index_argv(index_argv)

    # Check if first argument is "cache" subcommand
    if argv and len(argv) > 0 and argv[0] == "cache":
        # Handle cache subcommand
        cache_argv = list(argv[1:]) if len(argv) > 1 else []
        args = _parse_cache_args(cache_argv)
        args.command = "cache"  # Mark as cache command
        return args

    # Check if first argument is "doctor" subcommand (Issue #379)
    if argv and len(argv) > 0 and argv[0] == "doctor":
        # Handle doctor subcommand
        doctor_argv = list(argv[1:]) if len(argv) > 1 else []
        args = _parse_doctor_args(doctor_argv)
        args.command = "doctor"  # Mark as doctor command
        return args

    if argv and len(argv) > 0 and argv[0] == "pricing-assumptions":
        pa_argv = list(argv[1:]) if len(argv) > 1 else []
        args = _parse_pricing_assumptions_args(pa_argv)
        args.command = "pricing-assumptions"
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
    _add_gemini_arguments(parser)
    _add_anthropic_arguments(parser)
    _add_mistral_arguments(parser)
    _add_deepseek_arguments(parser)
    _add_grok_arguments(parser)
    _add_ollama_arguments(parser)
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


def _build_config(args: argparse.Namespace) -> config.Config:  # noqa: C901
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
        "seed": getattr(args, "seed", None),
        "log_level": args.log_level,
        "log_file": args.log_file,
        "json_logs": getattr(args, "json_logs", False),
        "workers": args.workers,
        "fail_fast": getattr(args, "fail_fast", False),
        "max_failures": getattr(args, "max_failures", None),
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
        "generate_gi": getattr(args, "generate_gi", False),
        "generate_kg": getattr(args, "generate_kg", False),
        "kg_extraction_source": getattr(args, "kg_extraction_source", None) or "summary_bullets",
        "kg_max_topics": (
            config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX
            if getattr(args, "kg_max_topics", None) is None
            else args.kg_max_topics
        ),
        "kg_max_entities": (
            15 if getattr(args, "kg_max_entities", None) is None else args.kg_max_entities
        ),
        "kg_extraction_model": getattr(args, "kg_extraction_model", None),
        "kg_extraction_provider": getattr(args, "kg_extraction_provider", None),
        "kg_merge_pipeline_entities": getattr(args, "kg_merge_pipeline_entities", True),
        "gi_insight_source": getattr(args, "gi_insight_source", None) or "stub",
        "gi_max_insights": (
            config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX
            if getattr(args, "gi_max_insights", None) is None
            else args.gi_max_insights
        ),
        "quote_extraction_provider": getattr(args, "quote_extraction_provider", None),
        "entailment_provider": getattr(args, "entailment_provider", None),
        "generate_summaries": args.generate_summaries,
        "metrics_output": args.metrics_output,
        "summary_provider": args.summary_provider,
        "summary_mode_id": getattr(args, "summary_mode_id", None),
        "summary_mode_precedence": getattr(args, "summary_mode_precedence", None),
        "summary_model": args.summary_model,
        "summary_reduce_model": args.summary_reduce_model,
        "summary_device": args.summary_device,
        "mps_exclusive": getattr(args, "mps_exclusive", True),
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
    # Hybrid ML provider args: only set when provided so Config defaults apply.
    if getattr(args, "hybrid_map_model", None) is not None:
        payload["hybrid_map_model"] = args.hybrid_map_model
    if getattr(args, "hybrid_reduce_model", None) is not None:
        payload["hybrid_reduce_model"] = args.hybrid_reduce_model
    if getattr(args, "hybrid_reduce_backend", None) is not None:
        payload["hybrid_reduce_backend"] = args.hybrid_reduce_backend
    if getattr(args, "hybrid_map_device", None) is not None:
        payload["hybrid_map_device"] = args.hybrid_map_device
    if getattr(args, "hybrid_reduce_device", None) is not None:
        payload["hybrid_reduce_device"] = args.hybrid_reduce_device
    if getattr(args, "hybrid_quantization", None) is not None:
        payload["hybrid_quantization"] = args.hybrid_quantization
    # Add OpenAI model args only if provided (fields have non-Optional types with defaults)
    if args.openai_transcription_model is not None:
        payload["openai_transcription_model"] = args.openai_transcription_model
    if args.openai_speaker_model is not None:
        payload["openai_speaker_model"] = args.openai_speaker_model
    if args.openai_summary_model is not None:
        payload["openai_summary_model"] = args.openai_summary_model
    if hasattr(args, "openai_insight_model") and args.openai_insight_model is not None:
        payload["openai_insight_model"] = args.openai_insight_model
    if args.openai_temperature is not None:
        payload["openai_temperature"] = args.openai_temperature
    if hasattr(args, "openai_max_tokens") and args.openai_max_tokens is not None:
        payload["openai_max_tokens"] = args.openai_max_tokens
    if hasattr(args, "openai_cleaning_model") and args.openai_cleaning_model is not None:
        payload["openai_cleaning_model"] = args.openai_cleaning_model
    if (
        hasattr(args, "openai_cleaning_temperature")
        and args.openai_cleaning_temperature is not None
    ):
        payload["openai_cleaning_temperature"] = args.openai_cleaning_temperature
    # Explicitly include openai_api_key=None to trigger field validator
    # The field validator will load it from OPENAI_API_KEY env var if available
    # But allow CLI override if provided
    payload["openai_api_key"] = getattr(args, "openai_api_key", None)
    # Add Gemini API configuration
    payload["gemini_api_base"] = getattr(args, "gemini_api_base", None)
    gemini_transcription_model = getattr(args, "gemini_transcription_model", None)
    if gemini_transcription_model is not None:
        payload["gemini_transcription_model"] = gemini_transcription_model
    gemini_speaker_model = getattr(args, "gemini_speaker_model", None)
    if gemini_speaker_model is not None:
        payload["gemini_speaker_model"] = gemini_speaker_model
    gemini_summary_model = getattr(args, "gemini_summary_model", None)
    if gemini_summary_model is not None:
        payload["gemini_summary_model"] = gemini_summary_model
    gemini_temperature = getattr(args, "gemini_temperature", None)
    if gemini_temperature is not None:
        payload["gemini_temperature"] = gemini_temperature
    gemini_max_tokens = getattr(args, "gemini_max_tokens", None)
    if gemini_max_tokens is not None:
        payload["gemini_max_tokens"] = gemini_max_tokens
    gemini_cleaning_model = getattr(args, "gemini_cleaning_model", None)
    if gemini_cleaning_model is not None:
        payload["gemini_cleaning_model"] = gemini_cleaning_model
    gemini_cleaning_temperature = getattr(args, "gemini_cleaning_temperature", None)
    if gemini_cleaning_temperature is not None:
        payload["gemini_cleaning_temperature"] = gemini_cleaning_temperature
    # Explicitly include gemini_api_key=None to trigger field validator
    # The field validator will load it from GEMINI_API_KEY env var if available
    # But allow CLI override if provided
    payload["gemini_api_key"] = getattr(args, "gemini_api_key", None)
    # Add Anthropic API configuration
    payload["anthropic_api_base"] = getattr(args, "anthropic_api_base", None)
    if hasattr(args, "anthropic_speaker_model") and args.anthropic_speaker_model is not None:
        payload["anthropic_speaker_model"] = args.anthropic_speaker_model
    if hasattr(args, "anthropic_summary_model") and args.anthropic_summary_model is not None:
        payload["anthropic_summary_model"] = args.anthropic_summary_model
    if hasattr(args, "anthropic_temperature") and args.anthropic_temperature is not None:
        payload["anthropic_temperature"] = args.anthropic_temperature
    if hasattr(args, "anthropic_max_tokens") and args.anthropic_max_tokens is not None:
        payload["anthropic_max_tokens"] = args.anthropic_max_tokens
    if hasattr(args, "anthropic_cleaning_model") and args.anthropic_cleaning_model is not None:
        payload["anthropic_cleaning_model"] = args.anthropic_cleaning_model
    if (
        hasattr(args, "anthropic_cleaning_temperature")
        and args.anthropic_cleaning_temperature is not None
    ):
        payload["anthropic_cleaning_temperature"] = args.anthropic_cleaning_temperature
    # Explicitly include anthropic_api_key=None to trigger field validator
    # The field validator will load it from ANTHROPIC_API_KEY env var if available
    # But allow CLI override if provided
    payload["anthropic_api_key"] = getattr(args, "anthropic_api_key", None)
    # Add Grok API configuration
    payload["grok_api_base"] = getattr(args, "grok_api_base", None)
    if hasattr(args, "grok_speaker_model") and args.grok_speaker_model is not None:
        payload["grok_speaker_model"] = args.grok_speaker_model
    if hasattr(args, "grok_summary_model") and args.grok_summary_model is not None:
        payload["grok_summary_model"] = args.grok_summary_model
    if hasattr(args, "grok_temperature") and args.grok_temperature is not None:
        payload["grok_temperature"] = args.grok_temperature
    if hasattr(args, "grok_max_tokens") and args.grok_max_tokens is not None:
        payload["grok_max_tokens"] = args.grok_max_tokens
    if hasattr(args, "grok_cleaning_model") and args.grok_cleaning_model is not None:
        payload["grok_cleaning_model"] = args.grok_cleaning_model
    if hasattr(args, "grok_cleaning_temperature") and args.grok_cleaning_temperature is not None:
        payload["grok_cleaning_temperature"] = args.grok_cleaning_temperature
    # Explicitly include grok_api_key=None to trigger field validator
    # The field validator will load it from GROK_API_KEY env var if available
    # But allow CLI override if provided
    payload["grok_api_key"] = getattr(args, "grok_api_key", None)
    # Add Mistral API configuration
    payload["mistral_api_base"] = getattr(args, "mistral_api_base", None)
    if (
        hasattr(args, "mistral_transcription_model")
        and args.mistral_transcription_model is not None
    ):
        payload["mistral_transcription_model"] = args.mistral_transcription_model
    if hasattr(args, "mistral_speaker_model") and args.mistral_speaker_model is not None:
        payload["mistral_speaker_model"] = args.mistral_speaker_model
    if hasattr(args, "mistral_summary_model") and args.mistral_summary_model is not None:
        payload["mistral_summary_model"] = args.mistral_summary_model
    if hasattr(args, "mistral_temperature") and args.mistral_temperature is not None:
        payload["mistral_temperature"] = args.mistral_temperature
    if hasattr(args, "mistral_max_tokens") and args.mistral_max_tokens is not None:
        payload["mistral_max_tokens"] = args.mistral_max_tokens
    if hasattr(args, "mistral_cleaning_model") and args.mistral_cleaning_model is not None:
        payload["mistral_cleaning_model"] = args.mistral_cleaning_model
    if (
        hasattr(args, "mistral_cleaning_temperature")
        and args.mistral_cleaning_temperature is not None
    ):
        payload["mistral_cleaning_temperature"] = args.mistral_cleaning_temperature
    # Explicitly include mistral_api_key=None to trigger field validator
    # The field validator will load it from MISTRAL_API_KEY env var if available
    # But allow CLI override if provided
    payload["mistral_api_key"] = getattr(args, "mistral_api_key", None)
    # Add DeepSeek API configuration
    payload["deepseek_api_base"] = getattr(args, "deepseek_api_base", None)
    if hasattr(args, "deepseek_speaker_model") and args.deepseek_speaker_model is not None:
        payload["deepseek_speaker_model"] = args.deepseek_speaker_model
    if hasattr(args, "deepseek_summary_model") and args.deepseek_summary_model is not None:
        payload["deepseek_summary_model"] = args.deepseek_summary_model
    if hasattr(args, "deepseek_temperature") and args.deepseek_temperature is not None:
        payload["deepseek_temperature"] = args.deepseek_temperature
    if hasattr(args, "deepseek_max_tokens") and args.deepseek_max_tokens is not None:
        payload["deepseek_max_tokens"] = args.deepseek_max_tokens
    if hasattr(args, "deepseek_cleaning_model") and args.deepseek_cleaning_model is not None:
        payload["deepseek_cleaning_model"] = args.deepseek_cleaning_model
    if (
        hasattr(args, "deepseek_cleaning_temperature")
        and args.deepseek_cleaning_temperature is not None
    ):
        payload["deepseek_cleaning_temperature"] = args.deepseek_cleaning_temperature
    # Explicitly include deepseek_api_key=None to trigger field validator
    # The field validator will load it from DEEPSEEK_API_KEY env var if available
    # But allow CLI override if provided
    payload["deepseek_api_key"] = getattr(args, "deepseek_api_key", None)
    # Add Ollama API configuration
    payload["ollama_api_base"] = getattr(args, "ollama_api_base", None)
    if hasattr(args, "ollama_speaker_model") and args.ollama_speaker_model is not None:
        payload["ollama_speaker_model"] = args.ollama_speaker_model
    if hasattr(args, "ollama_summary_model") and args.ollama_summary_model is not None:
        payload["ollama_summary_model"] = args.ollama_summary_model
    if hasattr(args, "ollama_temperature") and args.ollama_temperature is not None:
        payload["ollama_temperature"] = args.ollama_temperature
    if hasattr(args, "ollama_max_tokens") and args.ollama_max_tokens is not None:
        payload["ollama_max_tokens"] = args.ollama_max_tokens
    if hasattr(args, "ollama_timeout") and args.ollama_timeout is not None:
        payload["ollama_timeout"] = args.ollama_timeout
    if hasattr(args, "ollama_cleaning_model") and args.ollama_cleaning_model is not None:
        payload["ollama_cleaning_model"] = args.ollama_cleaning_model
    if (
        hasattr(args, "ollama_cleaning_temperature")
        and args.ollama_cleaning_temperature is not None
    ):
        payload["ollama_cleaning_temperature"] = args.ollama_cleaning_temperature
    # Add transcript_cleaning_strategy if provided
    if (
        hasattr(args, "transcript_cleaning_strategy")
        and args.transcript_cleaning_strategy is not None
    ):
        payload["transcript_cleaning_strategy"] = args.transcript_cleaning_strategy
    # GIL / evidence tuning from config file: _load_and_merge_config puts these on args via
    # set_defaults(model_dump); they must be copied here or CLI runs ignore YAML (Issue: gi_qa
    # thresholds appeared to "do nothing").
    _gil_tuning_keys = (
        "gi_require_grounding",
        "gi_fail_on_missing_grounding",
        "gi_evidence_extract_retries",
        "gi_qa_score_min",
        "gi_nli_entailment_min",
        "gi_qa_window_chars",
        "gi_qa_window_overlap_chars",
        "gi_qa_model",
        "gi_nli_model",
        "gi_embedding_model",
        "extractive_qa_device",
        "nli_device",
    )
    for _gil_key in _gil_tuning_keys:
        if hasattr(args, _gil_key):
            payload[_gil_key] = getattr(args, _gil_key)
    # Vector / semantic corpus (#484): CLI + YAML via set_defaults must reach Config.
    _vector_optional_keys = (
        "vector_index_path",
        "vector_embedding_model",
        "vector_chunk_size_tokens",
        "vector_chunk_overlap_tokens",
        "vector_backend",
        "vector_faiss_index_mode",
    )
    for _vk in _vector_optional_keys:
        if hasattr(args, _vk):
            _vv = getattr(args, _vk)
            if _vv is not None:
                payload[_vk] = _vv
    _vs = getattr(args, "vector_search", None)
    if _vs is not None:
        payload["vector_search"] = _vs
    if hasattr(args, "vector_index_types"):
        _vit = getattr(args, "vector_index_types")
        if isinstance(_vit, str):
            _parsed = [x.strip() for x in _vit.split(",") if x.strip()]
            payload["vector_index_types"] = _parsed if _parsed else None
        elif isinstance(_vit, list):
            payload["vector_index_types"] = _vit
    # Pydantic's model_validate returns the correct type, but mypy needs help
    return cast(config.Config, config.Config.model_validate(payload))


def _log_configuration_summary(cfg: config.Config, logger: logging.Logger) -> None:
    """Log a compact two-line config summary at INFO."""
    ep = cfg.max_episodes if cfg.max_episodes is not None else "all"
    if cfg.transcribe_missing:
        parts = [f"on:{cfg.transcription_provider}", f"screenplay={cfg.screenplay}"]
        if cfg.transcription_provider == "whisper":
            parts.insert(1, f"whisper_model={cfg.whisper_model}")
        transcribe = ",".join(parts)
    else:
        transcribe = "off"
    summ = f"on:{cfg.summary_provider}" if cfg.generate_summaries else "off"
    meta = f"on:{cfg.metadata_format}" if cfg.generate_metadata else "off"
    gi = "on" if cfg.generate_gi else "off"
    kg = "on" if getattr(cfg, "generate_kg", False) else "off"
    flag_parts = []
    if cfg.skip_existing:
        flag_parts.append("skip_existing")
    if cfg.reuse_media:
        flag_parts.append("reuse_media")
    if cfg.clean_output:
        flag_parts.append("clean_output")
    if cfg.dry_run:
        flag_parts.append("dry_run")
    flags_s = ",".join(flag_parts) if flag_parts else "none"
    logger.info(
        "config: rss=%s | out=%s | episodes=%s | workers=%s | log=%s | log_file=%s | run_id=%s",
        cfg.rss_url,
        cfg.output_dir,
        ep,
        cfg.workers,
        cfg.log_level,
        cfg.log_file or "console",
        cfg.run_id or "-",
    )
    logger.info(
        "config: http timeout=%ss delay=%sms | transcribe=%s | speakers=%s lang=%s ner=%s | "
        "summary=%s | metadata=%s | gi=%s | kg=%s | flags=%s",
        cfg.timeout,
        cfg.delay_ms,
        transcribe,
        "on" if cfg.auto_speakers else "off",
        cfg.language,
        cfg.ner_model or "-",
        summ,
        meta,
        gi,
        kg,
        flags_s,
    )
    if cfg.generate_gi:
        logger.info(
            "config: gi evidence: qa_min=%s nli_min=%s | quote=%s entail=%s | gi_embedding=%s",
            cfg.gi_qa_score_min,
            cfg.gi_nli_entailment_min,
            getattr(cfg, "quote_extraction_provider", "transformers"),
            getattr(cfg, "entailment_provider", "transformers"),
            cfg.gi_embedding_model,
        )


def _log_configuration_runtime_warnings(cfg: config.Config, logger: logging.Logger) -> None:
    """Surface important misconfigurations at WARNING (always, not only in DEBUG detail)."""
    if (
        cfg.generate_gi
        and getattr(cfg, "gi_insight_source", "stub") == "stub"
        and not config._is_test_environment()
    ):
        logger.warning(
            "GIL: gi_insight_source is 'stub' — insight text is a placeholder. "
            "For real wording use gi_insight_source: summary_bullets (with "
            "generate_summaries and summary bullets) or provider with an LLM "
            "summary_provider. ML providers (transformers, hybrid_ml) do not "
            "implement generate_insights. See docs/guides/GROUNDED_INSIGHTS_GUIDE.md."
        )
    _local_gil = frozenset({"transformers", "hybrid_ml"})
    _sp = getattr(cfg, "summary_provider", "transformers")
    _qe = getattr(cfg, "quote_extraction_provider", "transformers")
    _en = getattr(cfg, "entailment_provider", "transformers")
    if (
        cfg.generate_gi
        and getattr(cfg, "gi_require_grounding", True)
        and _sp in config.GIL_EVIDENCE_ALIGN_SUMMARY_PROVIDERS
        and (_qe in _local_gil or _en in _local_gil)
        and not config._is_test_environment()
    ):
        logger.warning(
            "GIL: summary_provider=%s uses an API or hybrid stack, but "
            "quote_extraction_provider=%s and/or entailment_provider=%s still use "
            "local Hugging Face QA/NLI (requires .[ml] / sentence-transformers for NLI). "
            "For API-only grounding, set both evidence providers to match "
            "summary_provider, or leave defaults with gil_evidence_match_summary_provider: "
            "true (default). See GROUNDED_INSIGHTS_GUIDE — GIL evidence provider matrix.",
            _sp,
            _qe,
            _en,
        )
    _kg_eff = getattr(cfg, "kg_extraction_provider", None) or getattr(cfg, "summary_provider", "")
    if (
        getattr(cfg, "generate_kg", False)
        and getattr(cfg, "kg_extraction_source", "summary_bullets") == "provider"
        and _kg_eff in ("transformers", "hybrid_ml")
        and not config._is_test_environment()
    ):
        logger.warning(
            "KG: kg_extraction_source is 'provider' but the effective KG backend "
            "(kg_extraction_provider or summary_provider) is ML — "
            "extract_kg_graph is a no-op; pipeline falls back to summary bullets "
            "when available, else episode + hosts/guests only."
        )


def _log_configuration_detail(cfg: config.Config, logger: logging.Logger) -> None:
    """Full config breakdown (DEBUG only)."""
    d = logger.debug
    d("=" * 80)
    d("Configuration (detail)")
    d("=" * 80)

    # Core settings
    d("Core Settings:")
    d(f"  RSS URL: {cfg.rss_url}")
    d(f"  Output Directory: {cfg.output_dir}")
    d(f"  Max Episodes: {cfg.max_episodes or 'all'}")
    d(f"  Workers: {cfg.workers}")
    d(f"  Log Level: {cfg.log_level}")
    d(f"  Log File: {cfg.log_file or 'console only'}")
    d(f"  Run ID: {cfg.run_id or 'none'}")

    # HTTP settings
    d("HTTP Settings:")
    d(f"  Timeout: {cfg.timeout}s")
    d(f"  Delay: {cfg.delay_ms}ms")
    d(
        f"  User-Agent: {cfg.user_agent[:50]}..."
        if len(cfg.user_agent) > 50
        else f"  User-Agent: {cfg.user_agent}"
    )
    d(f"  Prefer Types: {cfg.prefer_types if cfg.prefer_types else 'none'}")

    # Transcription settings
    d("Transcription Settings:")
    d(f"  Transcribe Missing: {cfg.transcribe_missing}")
    if cfg.transcribe_missing:
        d(f"  Whisper Model: {cfg.whisper_model}")
        d(f"  Screenplay Format: {cfg.screenplay}")
        if cfg.screenplay:
            d(f"  Screenplay Gap: {cfg.screenplay_gap_s}s")
            d(f"  Number of Speakers: {cfg.screenplay_num_speakers}")
            if cfg.screenplay_speaker_names:
                d(f"  Speaker Names: {', '.join(cfg.screenplay_speaker_names)}")

    # Speaker detection settings
    d("Speaker Detection Settings:")
    d(f"  Auto Speakers: {cfg.auto_speakers}")
    d(f"  Language: {cfg.language}")
    if cfg.ner_model:
        d(f"  NER Model: {cfg.ner_model}")
    d(f"  Cache Detected Hosts: {cfg.cache_detected_hosts}")

    # Metadata settings
    d("Metadata Settings:")
    d(f"  Generate Metadata: {cfg.generate_metadata}")
    if cfg.generate_metadata:
        d(f"  Metadata Format: {cfg.metadata_format}")
        if cfg.metadata_subdirectory:
            d(f"  Metadata Subdirectory: {cfg.metadata_subdirectory}")
        else:
            d("  Metadata Subdirectory: same as transcripts")

    # Summarization settings
    d("Summarization Settings:")
    d(f"  Generate Summaries: {cfg.generate_summaries}")
    if cfg.generate_summaries:
        d(f"  Summary Provider: {cfg.summary_provider}")
        if cfg.summary_provider == "transformers":
            if cfg.summary_model:
                d(f"  Summary Model: {cfg.summary_model}")
            else:
                d("  Summary Model: auto-selected")
            d(f"  Summary Device: {cfg.summary_device or 'auto-detect'}")
            if cfg.summary_chunk_size:
                d(f"  Summary Chunk Size: {cfg.summary_chunk_size} tokens")
        elif cfg.summary_provider == "hybrid_ml":
            d(
                f"  Hybrid MAP: {getattr(cfg, 'hybrid_map_model', 'longt5-base')}, "
                f"REDUCE: {getattr(cfg, 'hybrid_reduce_model', 'google/flan-t5-base')}"
            )
            d(f"  Hybrid REDUCE backend: {getattr(cfg, 'hybrid_reduce_backend', 'transformers')}")
            d(
                f"  Hybrid devices: MAP={getattr(cfg, 'hybrid_map_device', None) or 'default'}, "
                f"REDUCE={getattr(cfg, 'hybrid_reduce_device', None) or 'default'}"
            )
            if cfg.summary_chunk_size:
                d(f"  Summary Chunk Size: {cfg.summary_chunk_size} tokens")
        d(
            "  Summary map/reduce tokens: map max=%s min=%s | reduce max=%s min=%s",
            cfg.summary_map_params.get("max_new_tokens"),
            cfg.summary_map_params.get("min_new_tokens"),
            cfg.summary_reduce_params.get("max_new_tokens"),
            cfg.summary_reduce_params.get("min_new_tokens"),
        )
        if cfg.summary_map_params and cfg.summary_reduce_params and cfg.summary_tokenize:
            d(
                "  Summary beams (explicit map/reduce/tokenize): map=%s reduce=%s",
                cfg.summary_map_params.get("num_beams"),
                cfg.summary_reduce_params.get("num_beams"),
            )
        if cfg.summary_prompt:
            d(f"  Summary Prompt: {cfg.summary_prompt[:80]}...")

    # Grounded Insights (GIL) — warnings logged separately at WARNING
    d("Grounded Insights (GIL):")
    d(f"  Generate GI: {cfg.generate_gi}")
    if cfg.generate_gi:
        d(f"  GI require grounding: {getattr(cfg, 'gi_require_grounding', True)}")
        d(
            f"  GI fail on missing grounding: "
            f"{getattr(cfg, 'gi_fail_on_missing_grounding', False)}"
        )
        d(f"  GI insight source: {getattr(cfg, 'gi_insight_source', 'stub')}")
        _gi_max = getattr(
            cfg,
            "gi_max_insights",
            config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX,
        )
        d(f"  GI max insights: {_gi_max}")
        d(
            "  Quote extraction provider: %s",
            getattr(cfg, "quote_extraction_provider", "transformers"),
        )
        d(
            "  Entailment provider: %s",
            getattr(cfg, "entailment_provider", "transformers"),
        )
        d("  GI QA score min: %s", cfg.gi_qa_score_min)
        d("  GI NLI entailment min: %s", cfg.gi_nli_entailment_min)
        d("  GI embedding model: %s", getattr(cfg, "gi_embedding_model", ""))
        d("  GI QA model: %s", getattr(cfg, "gi_qa_model", ""))
        d("  GI NLI model: %s", getattr(cfg, "gi_nli_model", ""))

    d("Knowledge Graph (KG):")
    d(f"  Generate KG: {getattr(cfg, 'generate_kg', False)}")
    if getattr(cfg, "generate_kg", False):
        d(
            "  KG extraction source: %s",
            getattr(cfg, "kg_extraction_source", "summary_bullets"),
        )
        _kep = getattr(cfg, "kg_extraction_provider", None)
        if _kep:
            d("  KG extraction provider: %s", _kep)
        elif getattr(cfg, "kg_extraction_source", "summary_bullets") == "provider":
            d("  KG extraction provider: (same as summary_provider)")
        d(
            "  KG max topics: %s",
            getattr(cfg, "kg_max_topics", config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX),
        )
        d("  KG max entities: %s", getattr(cfg, "kg_max_entities", 15))
        d(
            "  KG merge pipeline entities: %s",
            getattr(cfg, "kg_merge_pipeline_entities", True),
        )
        km = getattr(cfg, "kg_extraction_model", None)
        if km:
            d("  KG extraction model override: %s", km)

    d(
        "Processing Options: skip_existing=%s reuse_media=%s clean_output=%s dry_run=%s",
        cfg.skip_existing,
        cfg.reuse_media,
        cfg.clean_output,
        cfg.dry_run,
    )

    d("=" * 80)


def _log_configuration(cfg: config.Config, logger: logging.Logger) -> None:
    """Log config: compact INFO summary, runtime warnings, optional DEBUG detail."""
    _log_configuration_summary(cfg, logger)
    _log_configuration_runtime_warnings(cfg, logger)
    if logger.isEnabledFor(logging.DEBUG):
        _log_configuration_detail(cfg, logger)


def _validate_python_version() -> None:
    """Validate Python version at startup (Issue #379).

    Raises:
        SystemExit: If Python version is too old
    """
    if sys.version_info < (3, 10):
        print(
            f"Error: Python {sys.version_info.major}.{sys.version_info.minor} is not supported.\n"
            "podcast_scraper requires Python 3.10 or higher.\n"
            f"Current version: {sys.version}",
            file=sys.stderr,
        )
        sys.exit(1)


def _validate_ffmpeg() -> None:
    """Validate ffmpeg is available at startup (Issue #379).

    Raises:
        SystemExit: If ffmpeg is not found
    """
    import shutil

    if not shutil.which("ffmpeg"):
        print(
            "Error: ffmpeg is not installed or not in PATH.\n"
            "ffmpeg is required for audio processing.\n"
            "Install: https://ffmpeg.org/download.html",
            file=sys.stderr,
        )
        sys.exit(1)


def main(  # noqa: C901 - main function handles multiple command paths
    argv: Optional[Sequence[str]] = None,
    *,
    apply_log_level_fn: Optional[Callable[[str, Optional[str], bool], None]] = None,
    run_pipeline_fn: Optional[Callable[[config.Config], Tuple[int, str]]] = None,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Entry point for the CLI; returns an exit status code."""
    if argv is None:
        argv = sys.argv[1:]
    # Validate Python version and dependencies at startup (Issue #379)
    _validate_python_version()
    # Only validate ffmpeg for main pipeline command, not for cache/doctor/gi/kg subcommands
    if (
        argv
        and len(argv) > 0
        and argv[0]
        in (
            "cache",
            "doctor",
            "gi",
            "index",
            "kg",
            "pricing-assumptions",
            "search",
        )
    ):
        pass  # Skip ffmpeg check for subcommands
    else:
        _validate_ffmpeg()

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

    # Handle doctor subcommand (Issue #379, #429)
    if hasattr(args, "command") and args.command == "doctor":
        return _run_doctor_checks(
            check_network=getattr(args, "check_network", False),
            check_models=getattr(args, "check_models", False),
        )

    if hasattr(args, "command") and args.command == "pricing-assumptions":
        return _run_pricing_assumptions(args)

    # Handle gi subcommand (#438)
    if hasattr(args, "command") and args.command == "gi":
        return _run_gi(args, log=log)

    # Handle kg subcommand (RFC-056)
    if hasattr(args, "command") and args.command == "kg":
        from .kg.cli_handlers import run_kg

        return run_kg(args, log)

    # Semantic corpus search / index (RFC-061 / #484)
    if hasattr(args, "command") and args.command == "search":
        from .search.cli_handlers import run_search_cli

        return run_search_cli(args, log)

    if hasattr(args, "command") and args.command == "index":
        from .search.cli_handlers import run_index_cli

        return run_index_cli(args, log)

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

            # Handle --delete-model option
            if args.delete_model:
                success, freed = cache_manager.delete_transformers_model_cache(
                    args.delete_model, confirm=confirm, force=args.yes
                )
                if success:
                    freed_str = cache_manager.format_size(freed)
                    print(f"\nDeleted cache for model '{args.delete_model}': {freed_str} freed")
                    print("The model will be re-downloaded on next use.")
                    return 0
                else:
                    print(f"\nFailed to delete cache for model '{args.delete_model}'")
                    return 1

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

    apply_log_level_fn(cfg.log_level, cfg.log_file, cfg.json_logs)

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
    import warnings

    # Silence thinc/spaCy FutureWarning about torch.cuda.amp.autocast (version compatibility issue)
    # This warning is from thinc/shims/pytorch.py and is not actionable by users.
    # Tracked in Issue #416; see docs/guides/DEPENDENCIES_GUIDE.md for details.
    # TODO(#416): Remove this suppression when thinc fixes the deprecation (monitor thinc releases).
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*torch\.cuda\.amp\.autocast.*deprecated.*",
        module="thinc.*",
    )

    # Enable faulthandler for crash diagnostics (segfault debugging)
    # This provides native backtraces when crashes occur, especially useful for
    # debugging segfaults from native extensions (PyTorch MPS, Transformers, spaCy)
    import faulthandler

    # Enable faulthandler if not already enabled via PYTHONFAULTHANDLER env var
    # This provides native backtraces for segfaults (actual crashes), but we don't
    # use dump_traceback_later() because it causes false alarms during normal long-running
    # operations like Whisper transcription (which can take minutes).
    if os.getenv("PYTHONFAULTHANDLER") != "1":
        faulthandler.enable(all_threads=True)
        # Note: We intentionally do NOT use dump_traceback_later() here because:
        # 1. It triggers false alarms during normal long-running operations (transcription)
        # 2. The timeout (even if increased) is arbitrary and doesn't reflect actual hangs
        # 3. faulthandler.enable() still provides backtraces for actual segfaults/crashes
        # For debugging actual hangs, use: PYTHONFAULTHANDLER=1 python -m podcast_scraper.cli ...

    raise SystemExit(main())
