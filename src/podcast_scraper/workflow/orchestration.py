"""Core workflow orchestration: main pipeline execution.

This module handles the main podcast scraping pipeline workflow.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Set, Tuple

from .. import (
    config,
    models,
)
from ..rss import (
    extract_episode_description as _extract_episode_description_rss,
)
from .episode_processor import (
    process_episode_download as _process_episode_download_original,
    transcribe_media_to_text as _transcribe_media_to_text_original,
)


def extract_episode_description(item):  # noqa: F811
    """Extract episode description, using re-exported version if available (for testability)."""
    import sys
    from unittest.mock import Mock

    _workflow_pkg = sys.modules.get("podcast_scraper.workflow")
    if _workflow_pkg and hasattr(_workflow_pkg, "extract_episode_description"):
        func = getattr(_workflow_pkg, "extract_episode_description")
        if isinstance(func, Mock) or func is not extract_episode_description:
            return func(item)
    return _extract_episode_description_rss(item)


from ..speaker_detectors.factory import create_speaker_detector as _create_speaker_detector_factory
from ..summarization.factory import (
    create_summarization_provider as _create_summarization_provider_factory,
)
from ..transcription.factory import (
    create_transcription_provider as _create_transcription_provider_factory,
)

# Re-export factory functions for testability
create_summarization_provider = _create_summarization_provider_factory
create_speaker_detector = _create_speaker_detector_factory
create_transcription_provider = _create_transcription_provider_factory
from . import helpers as wf_helpers, stages as wf_stages


def _log_effective_parallelism(cfg: config.Config, summary_provider: Optional[Any]) -> None:
    """Log effective parallelism configuration for all pipeline stages (Issue #380).

    Args:
        cfg: Configuration object
        summary_provider: Optional summary provider instance (for device detection)
    """
    logger.info("=" * 60)
    logger.info("Parallelism Configuration:")
    logger.info(f"  Download workers: {cfg.workers}")
    transcription_configured = cfg.transcription_parallelism
    transcription_effective = (
        1 if cfg.transcription_provider == "whisper" else transcription_configured
    )
    if transcription_effective != transcription_configured:
        logger.info(
            f"  Transcription workers: configured={transcription_configured}, "
            f"effective={transcription_effective} "
            f"({cfg.transcription_provider} provider limitation)"
        )
    else:
        logger.info(f"  Transcription workers: {transcription_effective}")
    processing_configured = cfg.processing_parallelism
    processing_effective = processing_configured
    logger.info(f"  Processing workers: {processing_effective}")
    if cfg.generate_summaries:
        # Get actual device from summary provider
        model_device = "cpu"  # Default
        if summary_provider:
            # Try to get device from actual model instances
            if hasattr(summary_provider, "_map_model") and summary_provider._map_model:
                if hasattr(summary_provider._map_model, "device"):
                    model_device = summary_provider._map_model.device
            elif hasattr(summary_provider, "_reduce_model") and summary_provider._reduce_model:
                if hasattr(summary_provider._reduce_model, "device"):
                    model_device = summary_provider._reduce_model.device
            # Fallback to config if model not loaded yet
            elif cfg.summary_device:
                model_device = cfg.summary_device
            # Fallback to provider attribute
            elif hasattr(summary_provider, "device"):
                model_device = summary_provider.device

        # Determine serialization status
        serialization_reasons = []
        if model_device in ("mps", "cuda"):
            if cfg.mps_exclusive:
                serialization_reasons.append("mps_exclusive")
            # Check if tokenizer serialization is enabled (lock exists)
            if summary_provider and hasattr(summary_provider, "_map_model"):
                if hasattr(summary_provider._map_model, "_summarize_lock"):
                    serialization_reasons.append("tokenizer_lock")
        serialization_status = (
            f", serialized ({', '.join(serialization_reasons)})" if serialization_reasons else ""
        )

        if model_device == "cpu":
            max_workers_limit = (
                cfg.summary_max_workers_cpu if cfg.summary_max_workers_cpu is not None else 4
            )
            estimated_workers = min(cfg.summary_batch_size or 1, max_workers_limit)
        elif model_device in ("mps", "cuda"):
            max_workers_limit = (
                cfg.summary_max_workers_gpu if cfg.summary_max_workers_gpu is not None else 2
            )
            estimated_workers = min(max_workers_limit, cfg.summary_batch_size or 1)
        else:
            estimated_workers = 1
        logger.info(
            f"  Summarization workers: {estimated_workers} (device={model_device}"
            f"{serialization_status})"
        )
    else:
        logger.info("  Summarization workers: N/A (summarization disabled)")
    logger.info("=" * 60)


def create_speaker_detector(cfg: config.Config):  # type: ignore[no-redef]  # noqa: F811
    """Create speaker detector, using re-exported version if available (for testability).

    This wrapper allows tests to patch podcast_scraper.workflow.create_speaker_detector
    and have it work even though workflow.py imports the function directly.
    """
    import sys
    from unittest.mock import Mock

    # Check if package has re-exported (patched) version
    # Use explicit package name since workflow.py is loaded dynamically
    # and __package__ might not be set correctly
    _workflow_pkg = sys.modules.get("podcast_scraper.workflow")
    if _workflow_pkg and hasattr(_workflow_pkg, "create_speaker_detector"):
        func = getattr(_workflow_pkg, "create_speaker_detector")
        # If it's a Mock (patched), use it
        if isinstance(func, Mock):
            return func(cfg)
        # If it's different from factory (but not this wrapper), use it
        # (handles case where re-export points to a different function)
        if func is not _create_speaker_detector_factory and func is not create_speaker_detector:
            return func(cfg)
    # Otherwise use factory directly
    return _create_speaker_detector_factory(cfg)


def create_transcription_provider(cfg: config.Config):  # type: ignore[no-redef]  # noqa: F811
    """Create transcription provider, using re-exported version if available (for testability).

    This wrapper allows tests to patch podcast_scraper.workflow.create_transcription_provider
    and have it work even though workflow.py imports the function directly.
    """
    import sys
    from unittest.mock import Mock

    _workflow_pkg = sys.modules.get("podcast_scraper.workflow")
    if _workflow_pkg and hasattr(_workflow_pkg, "create_transcription_provider"):
        func = getattr(_workflow_pkg, "create_transcription_provider")
        if isinstance(func, Mock):
            return func(cfg)
        if (
            func is not _create_transcription_provider_factory
            and func is not create_transcription_provider
        ):
            return func(cfg)
    return _create_transcription_provider_factory(cfg)


def transcribe_media_to_text(*args, **kwargs):  # noqa: F811
    """Transcribe media to text, using re-exported version if available (for testability).

    This wrapper allows tests to patch podcast_scraper.workflow.transcribe_media_to_text
    and have it work even though workflow.py imports the function directly.
    """
    import sys
    from unittest.mock import Mock

    _workflow_pkg = sys.modules.get("podcast_scraper.workflow")
    if _workflow_pkg and hasattr(_workflow_pkg, "transcribe_media_to_text"):
        func = getattr(_workflow_pkg, "transcribe_media_to_text")
        if isinstance(func, Mock):
            return func(*args, **kwargs)
        if func is not _transcribe_media_to_text_original and func is not transcribe_media_to_text:
            return func(*args, **kwargs)
    return _transcribe_media_to_text_original(*args, **kwargs)


def process_episode_download(*args, **kwargs):  # noqa: F811
    """Process episode download, using re-exported version if available (for testability).

    This wrapper allows tests to patch podcast_scraper.workflow.process_episode_download
    and have it work even though workflow.py imports the function directly.
    """
    import sys
    from unittest.mock import Mock

    _workflow_pkg = sys.modules.get("podcast_scraper.workflow")
    if _workflow_pkg and hasattr(_workflow_pkg, "process_episode_download"):
        func = getattr(_workflow_pkg, "process_episode_download")
        if isinstance(func, Mock):
            return func(*args, **kwargs)
        if func is not _process_episode_download_original and func is not process_episode_download:
            return func(*args, **kwargs)
    return _process_episode_download_original(*args, **kwargs)


logger = logging.getLogger(__name__)

# Module-level registry for preloaded MLProvider instance
# This allows factories to reuse the same instance across capabilities
_preloaded_ml_provider: Optional[Any] = None


def _both_providers_use_mps(
    cfg: config.Config, transcription_provider: Any, summary_provider: Any
) -> bool:
    """Check if both Whisper transcription and summarization use MPS.

    Args:
        cfg: Configuration object
        transcription_provider: Transcription provider instance
        summary_provider: Summarization provider instance (can be None)

    Returns:
        True if both providers use MPS, False otherwise
    """
    # Check transcription provider device
    transcription_uses_mps = False
    if transcription_provider is not None:
        # Check if it's MLProvider (Whisper)
        provider_type = type(transcription_provider).__name__
        if provider_type == "MLProvider":
            # MLProvider uses _detect_whisper_device() method
            try:
                whisper_device = transcription_provider._detect_whisper_device()
                transcription_uses_mps = whisper_device == "mps"
            except (AttributeError, Exception):
                # If method doesn't exist or fails, assume not MPS
                transcription_uses_mps = False
        # OpenAI provider doesn't use MPS (API-based)
        elif provider_type == "OpenAIProvider":
            transcription_uses_mps = False

    # Check summarization provider device
    summarization_uses_mps = False
    if summary_provider is not None and cfg.generate_summaries:
        provider_type = type(summary_provider).__name__
        if provider_type == "MLProvider":
            # MLProvider uses SummaryModel which has device attribute
            try:
                # First check if models are initialized and have device attribute
                if hasattr(summary_provider, "_map_model") and summary_provider._map_model:
                    if hasattr(summary_provider._map_model, "device"):
                        map_device = summary_provider._map_model.device
                        summarization_uses_mps = map_device == "mps"
                # Also check reduce_model if map_model not available
                elif hasattr(summary_provider, "_reduce_model") and summary_provider._reduce_model:
                    if hasattr(summary_provider._reduce_model, "device"):
                        reduce_device = summary_provider._reduce_model.device
                        summarization_uses_mps = reduce_device == "mps"
                # Fallback: check summary_device from config
                if not summarization_uses_mps and cfg.summary_device:
                    summarization_uses_mps = cfg.summary_device == "mps"
                # Final fallback: auto-detect if MPS is available (when device not explicitly set)
                if not summarization_uses_mps and not cfg.summary_device:
                    try:
                        import torch

                        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                            summarization_uses_mps = True
                    except ImportError:
                        summarization_uses_mps = False
            except (AttributeError, Exception):
                # If attributes don't exist or fail, check config
                if cfg.summary_device:
                    summarization_uses_mps = cfg.summary_device == "mps"
                # Final fallback: auto-detect if MPS is available
                elif not cfg.summary_device:
                    try:
                        import torch

                        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                            summarization_uses_mps = True
                    except ImportError:
                        summarization_uses_mps = False

    return transcription_uses_mps and summarization_uses_mps


class _FeedMetadata(NamedTuple):
    """Feed metadata for metadata generation."""

    description: Optional[str]
    image_url: Optional[str]
    last_updated: Optional[datetime]


class _HostDetectionResult(NamedTuple):
    """Result of host detection and pattern analysis."""

    cached_hosts: Set[str]
    heuristics: Optional[Dict[str, Any]]
    speaker_detector: Any = None  # Stage 3: Optional SpeakerDetector instance


class _TranscriptionResources(NamedTuple):
    """Resources needed for transcription."""

    transcription_provider: Any  # Stage 2: TranscriptionProvider instance
    temp_dir: Optional[str]
    transcription_jobs: List[models.TranscriptionJob]
    transcription_jobs_lock: Optional[threading.Lock]
    saved_counter_lock: Optional[threading.Lock]


class _ProcessingJob(NamedTuple):
    """Job for processing (metadata/summarization) stage."""

    episode: models.Episode
    transcript_path: str
    transcript_source: Literal["direct_download", "whisper_transcription"]
    detected_names: Optional[List[str]]
    whisper_model: Optional[str]


class _ProcessingResources(NamedTuple):
    """Resources needed for processing stage."""

    processing_jobs: List[_ProcessingJob]
    processing_jobs_lock: Optional[threading.Lock]
    processing_complete_event: Optional[threading.Event]


def _preload_ml_models_if_needed(cfg: config.Config) -> None:  # noqa: F811
    """Preload ML models early in the pipeline if configured to use them.

    Uses re-exported version if available (for testability).

    This wrapper allows tests to patch podcast_scraper.workflow._preload_ml_models_if_needed
    and have it work even though workflow.py imports the function directly.

    Args:
        cfg: Configuration object

    Raises:
        RuntimeError: If required model cannot be loaded
        ImportError: If ML dependencies are not installed
    """
    import sys
    from unittest.mock import Mock

    _workflow_pkg = sys.modules.get("podcast_scraper.workflow")
    if _workflow_pkg and hasattr(_workflow_pkg, "_preload_ml_models_if_needed"):
        func = getattr(_workflow_pkg, "_preload_ml_models_if_needed")
        if isinstance(func, Mock):
            func(cfg)  # type: ignore[no-any-return]
            return
        if func is not _preload_ml_models_if_needed:
            func(cfg)  # type: ignore[no-any-return]
            return
    wf_stages.setup.preload_ml_models_if_needed(cfg)


def apply_log_level(level: str, log_file: Optional[str] = None, json_logs: bool = False) -> None:
    """Apply logging level to root logger and configure handlers.

    Args:
        level: Log level string (e.g., 'DEBUG', 'INFO', 'WARNING')
        log_file: Optional path to log file. If provided, logs will be written to both
                  console and file.
        json_logs: If True, use JSON formatter for structured logging (Issue #379)

    Raises:
        ValueError: If log level is invalid
        OSError: If log file cannot be created or written to
    """
    numeric_level = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    root_logger = logging.getLogger()

    # Choose formatter based on json_logs flag (Issue #379)
    if json_logs:
        from ..utils.json_logging import JSONFormatter

        formatter = JSONFormatter()
        log_format = None  # Not used with JSON formatter
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    # Remove existing handlers if we're setting up fresh
    if not root_logger.handlers:
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(numeric_level)
    else:
        # Update existing handlers
        root_logger.setLevel(numeric_level)
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)
            # Update formatter if json_logs changed
            handler.setFormatter(formatter)

    # Add file handler if log_file is specified
    if log_file:
        # Check if file handler already exists
        file_handler_exists = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)
            for h in root_logger.handlers
        )

        if not file_handler_exists:
            # Create directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Set up file handler
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")

    logger.setLevel(numeric_level)


def run_pipeline(cfg: config.Config) -> Tuple[int, str]:
    """Execute the main podcast scraping pipeline.

    This is the primary entry point for programmatic use of podcast_scraper. It orchestrates
    the complete workflow from RSS feed fetching to transcript generation and optional
    metadata/summarization.

    The pipeline executes the following stages:

    1. Setup output directory (with optional run ID subdirectory)
    2. Fetch and parse RSS feed
    3. Detect speakers (if auto-detection enabled)
    4. Process episodes concurrently:
       - Download published transcripts
       - Or queue media for Whisper transcription
    5. Transcribe queued media files sequentially (if Whisper enabled)
    6. Generate metadata documents (if enabled)
    7. Generate episode summaries (if enabled)
    8. Clean up temporary files

    Args:
        cfg: Configuration object with all pipeline settings. See `Config` for available options.

    Returns:
        Tuple[int, str]: A tuple containing:

            - count (int): Number of episodes processed (transcripts saved or planned)
            - summary (str): Human-readable summary message describing the run

    Raises:
        RuntimeError: If output directory cleanup fails when `clean_output=True`
        ValueError: If RSS URL is invalid or feed cannot be parsed
        FileNotFoundError: If configuration file references missing files
        OSError: If file system operations fail

    Example:
        >>> from podcast_scraper import Config, run_pipeline
        >>>
        >>> cfg = Config(
        ...     rss="https://example.com/feed.xml",
        ...     output_dir="./transcripts",
        ...     max_episodes=10
        ... )
        >>> count, summary = run_pipeline(cfg)
        >>> print(f"Downloaded {count} transcripts: {summary}")
        Downloaded 10 transcripts: Processed 10/50 episodes

    Example with Whisper transcription:
        >>> cfg = Config(
        ...     rss="https://example.com/feed.xml",
        ...     transcribe_missing=True,
        ...     whisper_model="base",
        ...     screenplay=True,
        ...     num_speakers=2
        ... )
        >>> count, summary = run_pipeline(cfg)

    Example with metadata and summaries:
        >>> cfg = Config(
        ...     rss="https://example.com/feed.xml",
        ...     generate_metadata=True,
        ...     generate_summaries=True
        ... )
        >>> count, summary = run_pipeline(cfg)

    Note:
        For non-interactive use (daemons, services), consider using the `service.run()`
        function instead, which provides structured error handling and return values.

    See Also:
        - `Config`: Configuration model with all available options
        - `service.run()`: Service API with structured error handling
        - `load_config_file()`: Load configuration from JSON/YAML file
    """
    # Initialize ML environment variables (suppress progress bars, etc.)
    wf_stages.setup.initialize_ml_environment()

    # Initialize metrics collector
    from . import metrics

    pipeline_metrics = metrics.Metrics()

    # Step 1: Setup pipeline environment
    effective_output_dir, run_suffix, full_config_string = (
        wf_stages.setup.setup_pipeline_environment(cfg)
    )

    # Step 1.5: Preload ML models if configured
    wf_stages.setup.preload_ml_models_if_needed(cfg)

    # Step 1.6: Create all providers once (singleton pattern per run)
    # Providers are created here and passed to stages to avoid redundant initialization
    transcription_provider = None
    speaker_detector = None
    summary_provider = None

    # Create transcription provider if needed
    if cfg.transcribe_missing and not cfg.dry_run:
        try:
            import sys
            from unittest.mock import Mock

            _workflow_pkg = sys.modules.get("podcast_scraper.workflow")
            if _workflow_pkg and hasattr(_workflow_pkg, "create_transcription_provider"):
                func = getattr(_workflow_pkg, "create_transcription_provider")
                if isinstance(func, Mock) or func is not _create_transcription_provider_factory:
                    transcription_provider = func(cfg)
                else:
                    transcription_provider = _create_transcription_provider_factory(cfg)
            else:
                transcription_provider = _create_transcription_provider_factory(cfg)
            transcription_provider.initialize()
            logger.debug(
                "Transcription provider initialized: %s",
                type(transcription_provider).__name__,
            )
        except Exception as exc:
            logger.error("Failed to initialize transcription provider: %s", exc)
            # Fail fast - provider initialization should succeed
            transcription_provider = None
            raise

    # Create speaker detector if needed
    if cfg.auto_speakers and not cfg.dry_run:
        try:
            import sys
            from unittest.mock import Mock

            _workflow_pkg = sys.modules.get("podcast_scraper.workflow")
            if _workflow_pkg and hasattr(_workflow_pkg, "create_speaker_detector"):
                func = getattr(_workflow_pkg, "create_speaker_detector")
                if isinstance(func, Mock) or func is not _create_speaker_detector_factory:
                    speaker_detector = func(cfg)
                else:
                    speaker_detector = _create_speaker_detector_factory(cfg)
            else:
                speaker_detector = _create_speaker_detector_factory(cfg)
            speaker_detector.initialize()
            logger.debug(
                "Speaker detector initialized: %s",
                type(speaker_detector).__name__,
            )
        except Exception as exc:
            logger.error("Failed to initialize speaker detector: %s", exc)
            # Fail fast - provider initialization should succeed
            speaker_detector = None
            raise

    # Create summarization provider if needed
    if cfg.generate_summaries and not cfg.dry_run:
        try:
            import sys
            from unittest.mock import Mock

            _workflow_pkg = sys.modules.get("podcast_scraper.workflow")
            if _workflow_pkg and hasattr(_workflow_pkg, "create_summarization_provider"):
                func = getattr(_workflow_pkg, "create_summarization_provider")
                if isinstance(func, Mock) or func is not _create_summarization_provider_factory:
                    summary_provider = func(cfg)
                else:
                    summary_provider = _create_summarization_provider_factory(cfg)
            else:
                summary_provider = _create_summarization_provider_factory(cfg)
            summary_provider.initialize()
            logger.debug(
                "Summarization provider initialized: %s",
                type(summary_provider).__name__,
            )
        except ImportError as e:
            # Fail fast when generate_summaries=True - dependencies must be available
            error_msg = (
                f"Summarization dependencies not available but generate_summaries=True: {e}. "
                "Install ML dependencies or set generate_summaries=False."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            # Fail fast - provider initialization must succeed when generate_summaries=True
            error_msg = (
                f"Failed to initialize summarization provider (generate_summaries=True): {e}. "
                "Cannot proceed with summarization."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    # Step 1.7: Log effective parallelism configuration (Issue #380)
    _log_effective_parallelism(cfg, summary_provider)

    # Step 1.5: Create run manifest (Issue #379)
    run_manifest = None
    if not cfg.dry_run:
        try:
            from .run_manifest import create_run_manifest

            run_manifest = create_run_manifest(
                cfg=cfg,
                output_dir=effective_output_dir,
                run_id=cfg.run_id,
            )
            manifest_path = os.path.join(effective_output_dir, "run_manifest.json")
            run_manifest.save_to_file(manifest_path)
            logger.info(f"Run manifest saved to: {manifest_path}")
        except Exception as e:
            logger.warning(f"Failed to generate run manifest: {e}")

    # Step 2: Fetch and parse RSS feed (scraping stage)
    scraping_start = time.time()
    feed, rss_bytes = wf_stages.scraping.fetch_and_parse_feed(cfg)
    pipeline_metrics.record_stage("scraping", time.time() - scraping_start)

    # Step 3: Extract feed metadata (if metadata generation enabled)
    # Reuse RSS bytes from initial fetch to avoid duplicate network request
    feed_metadata = wf_stages.scraping.extract_feed_metadata_for_generation(cfg, feed, rss_bytes)

    # Step 4: Prepare episodes from RSS items (parsing stage)
    parsing_start = time.time()
    episodes = wf_stages.scraping.prepare_episodes_from_feed(feed, cfg)
    pipeline_metrics.episodes_scraped_total = len(episodes)
    pipeline_metrics.record_stage("parsing", time.time() - parsing_start)

    # Step 5: Detect hosts and analyze patterns (if auto_speakers enabled)
    # This is part of normalizing stage
    normalizing_start = time.time()
    host_detection_result = wf_stages.processing.detect_feed_hosts_and_patterns(
        cfg, feed, episodes, pipeline_metrics, speaker_detector=speaker_detector
    )

    # Step 6: Setup transcription resources (Whisper model, temp dir)
    transcription_resources = wf_stages.transcription.setup_transcription_resources(
        cfg, effective_output_dir, transcription_provider=transcription_provider
    )

    # Step 6.5: Setup processing resources (metadata/summarization queue)
    processing_resources = wf_stages.processing.setup_processing_resources(cfg)

    # Wrap all processing in try-finally to ensure cleanup always happens
    # This prevents memory leaks if exceptions occur during processing
    try:
        # Step 7: Prepare episode processing arguments with speaker detection
        download_args = wf_stages.processing.prepare_episode_download_args(
            episodes,
            cfg,
            effective_output_dir,
            run_suffix,
            transcription_resources,
            host_detection_result,
            pipeline_metrics,
        )
        pipeline_metrics.record_stage("normalizing", time.time() - normalizing_start)

        # Step 8: Process episodes (download transcripts or queue transcription jobs)
        # Start processing stage concurrently (metadata/summarization)
        # NOTE: writing_start measures the entire "processing" stage, not just file I/O
        # This includes: downloads, transcription waiting, thread synchronization, AND actual I/O
        # For accurate I/O metrics, check [STORAGE I/O] debug logs
        writing_start = time.time()
        transcription_complete_event = threading.Event()
        downloads_complete_event = threading.Event()  # Signal when all downloads are complete
        transcription_saved = [0]  # Use list to allow modification from thread

        # Determine if we should serialize MPS work to prevent memory contention
        should_serialize_mps = False
        if cfg.mps_exclusive:
            should_serialize_mps = _both_providers_use_mps(
                cfg, transcription_provider, summary_provider
            )
            if should_serialize_mps:
                logger.info(
                    "MPS exclusive mode enabled: Serializing GPU work "
                    "(transcription completes before summarization starts)"
                )

        # Start processing thread if metadata generation is enabled
        processing_thread = None
        if cfg.generate_metadata:
            processing_thread = threading.Thread(
                target=wf_stages.processing.process_processing_jobs_concurrent,
                args=(
                    processing_resources,
                    feed,
                    cfg,
                    effective_output_dir,
                    run_suffix,
                    feed_metadata,
                    host_detection_result,
                    pipeline_metrics,
                    summary_provider,
                    transcription_complete_event,
                    should_serialize_mps,
                ),
                daemon=False,
                name="ProcessingProcessor",
            )
            processing_thread.start()
            logger.debug(
                "Started concurrent processing thread (parallelism=%d)", cfg.processing_parallelism
            )

        # Start transcription processing concurrently if transcription is enabled
        if cfg.transcribe_missing and not cfg.dry_run:
            # Start transcription processing in background thread
            transcription_thread = threading.Thread(
                target=wf_stages.transcription.process_transcription_jobs_concurrent,
                args=(
                    transcription_resources,
                    download_args,
                    episodes,
                    feed,
                    cfg,
                    effective_output_dir,
                    run_suffix,
                    feed_metadata,
                    host_detection_result,
                    processing_resources,
                    pipeline_metrics,
                    summary_provider,
                    # Pass downloads_complete_event, not transcription_complete_event
                    downloads_complete_event,
                    transcription_saved,
                ),
                daemon=False,
                name="TranscriptionProcessor",
            )
            transcription_thread.start()
            logger.debug("Started concurrent transcription processing thread")

        saved = wf_stages.processing.process_episodes(
            download_args,
            episodes,
            feed,
            cfg,
            effective_output_dir,
            run_suffix,
            feed_metadata,
            host_detection_result,
            transcription_resources,
            processing_resources,
            pipeline_metrics,
            summary_provider,
        )

        # Signal that downloads are complete (so transcription thread can exit when queue is empty)
        if cfg.transcribe_missing and not cfg.dry_run:
            downloads_complete_event.set()

        # Step 9: Wait for transcription to complete (if started)
        if cfg.transcribe_missing and not cfg.dry_run:
            # Wait for transcription thread to finish processing remaining jobs
            transcription_thread.join()
            saved += transcription_saved[0]
            logger.debug("Concurrent transcription processing completed")
        elif cfg.transcribe_missing:
            # Dry-run mode: process transcription jobs sequentially after downloads
            saved += wf_stages.transcription.process_transcription_jobs(
                transcription_resources,
                download_args,
                episodes,
                feed,
                cfg,
                effective_output_dir,
                run_suffix,
                feed_metadata,
                host_detection_result,
                pipeline_metrics,
                summary_provider,
            )
        # Record "io_and_waiting" stage time (renamed from "writing_storage" for clarity)
        # This metric measures the entire "processing" stage:
        # - Episode downloads
        # - Transcription job processing (including waiting for jobs)
        # - Thread synchronization (join() calls)
        # - Actual file I/O operations
        # For accurate I/O metrics, check [STORAGE I/O] debug logs which show per-file:
        # - File path
        # - Bytes written
        # - Elapsed time per write
        # This will reveal if the bottleneck is actual I/O or waiting time
        io_and_waiting_time = time.time() - writing_start
        pipeline_metrics.record_stage("io_and_waiting", io_and_waiting_time)
        logger.debug(
            "[IO METRIC] io_and_waiting stage total: %.2fs "
            "(includes downloads, transcription waiting, thread sync, and actual I/O)",
            io_and_waiting_time,
        )

        # Step 9.5: Wait for processing to complete (if started)
        if processing_thread is not None:
            # Signal that transcription is complete (processing waits for this)
            transcription_complete_event.set()
            # Wait for processing thread to finish
            processing_thread.join()
            logger.debug("Concurrent processing completed")

        # Step 10: Parallel episode summarization (if enabled and multiple episodes)
        # Process episodes that need summarization in parallel for better performance
        # This runs after all episodes are processed and checks for episodes that might
        # have been skipped during inline processing or need summary regeneration
        # Note: Parallel summarization uses direct model loading for thread safety
        # (each worker needs its own model instance). This is intentional and not a fallback.
        if (
            cfg.generate_summaries
            and cfg.generate_metadata
            and summary_provider is not None
            and not cfg.dry_run
            and len(episodes) > 1
        ):
            # Only run parallel summarization if we have multiple episodes to process
            # It will skip episodes that already have summaries
            wf_stages.summarization_stage.parallel_episode_summarization(
                episodes=episodes,
                feed=feed,
                cfg=cfg,
                effective_output_dir=effective_output_dir,
                run_suffix=run_suffix,
                feed_metadata=feed_metadata,
                host_detection_result=host_detection_result,
                summary_provider=summary_provider,
                download_args=download_args,
                pipeline_metrics=pipeline_metrics,
            )

    finally:
        # Step 9.5: Unload models to free memory
        # This runs even if exceptions occur above, preventing memory leaks
        # Stage 2: Cleanup transcription provider (which handles model unloading)
        if (
            transcription_resources is not None
            and transcription_resources.transcription_provider is not None
        ):
            try:
                provider = transcription_resources.transcription_provider
                provider.cleanup()
                logger.debug("Cleaned up transcription provider")
            except Exception as e:
                logger.warning("Failed to cleanup transcription provider: %s", e)

        # Stage 4: Cleanup provider (which handles model unloading)
        # Cleanup preloaded MLProvider instance
        global _preloaded_ml_provider
        if _preloaded_ml_provider is not None:
            try:
                _preloaded_ml_provider.cleanup()
            except Exception as e:
                logger.warning("Error cleaning up preloaded MLProvider: %s", e)
            finally:
                _preloaded_ml_provider = None

        if summary_provider is not None:
            try:
                summary_provider.cleanup()
                logger.debug("Cleaned up summarization provider")
            except Exception as e:
                logger.warning("Failed to cleanup summarization provider: %s", e)

        # Note: spaCy model cache was removed. Models are managed by providers
        # and cleaned up via provider.cleanup() method above.

    # Step 10: Cleanup temporary files
    wf_helpers.cleanup_pipeline(temp_dir=transcription_resources.temp_dir)

    # Step 11: Generate summary and log metrics
    pipeline_metrics.log_metrics()

    # Step 12: Save metrics to file if configured
    if cfg.metrics_output is not None:
        # Explicit path provided
        if cfg.metrics_output:
            metrics_path = cfg.metrics_output
        else:
            # Empty string means disabled
            metrics_path = None
    else:
        # Default: save to output directory
        metrics_path = os.path.join(effective_output_dir, "metrics.json")

    if metrics_path:
        try:
            pipeline_metrics.save_to_file(metrics_path)
        except Exception as e:
            logger.warning(f"Failed to save metrics to {metrics_path}: {e}")

    # Step 13: Generate run index (Issue #379)
    if not cfg.dry_run:
        try:
            from .run_index import create_run_index

            # Pass episode_statuses from metrics if available
            episode_statuses = (
                pipeline_metrics.episode_statuses
                if hasattr(pipeline_metrics, "episode_statuses")
                else None
            )

            run_index = create_run_index(
                run_id=cfg.run_id or datetime.utcnow().isoformat() + "Z",
                feed_url=cfg.rss_url,
                episodes=episodes,
                effective_output_dir=effective_output_dir,
                episode_statuses=episode_statuses,
            )
            index_path = os.path.join(effective_output_dir, "index.json")
            run_index.save_to_file(index_path)
            logger.info(f"Run index saved to: {index_path}")
        except Exception as e:
            logger.warning(f"Failed to generate run index: {e}")

    # Step 14: Generate run summary (Issue #379)
    if not cfg.dry_run:
        try:
            from .run_summary import create_run_summary, save_run_summary

            run_summary = create_run_summary(
                run_manifest=run_manifest,
                pipeline_metrics=pipeline_metrics,
                output_dir=effective_output_dir,
                run_id=cfg.run_id,
            )
            save_run_summary(run_summary, effective_output_dir)
        except Exception as e:
            logger.warning(f"Failed to generate run summary: {e}")

    return wf_helpers.generate_pipeline_summary(
        cfg, saved, transcription_resources, effective_output_dir, pipeline_metrics
    )
