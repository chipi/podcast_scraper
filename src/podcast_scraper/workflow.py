"""Core workflow orchestration: main pipeline execution.

This module handles the main podcast scraping pipeline workflow.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, cast, Dict, List, Literal, NamedTuple, Optional, Set, Tuple

from . import (
    config,
    filesystem,
    metadata,
    metrics,
    models,
    progress,
)
from .episode_processor import (
    process_episode_download,
    transcribe_media_to_text,
)
from .rss_parser import (
    create_episode_from_item,
    extract_episode_description,
    extract_episode_metadata,
    extract_episode_published_date,
    extract_feed_metadata,
)
from .speaker_detectors.factory import create_speaker_detector
from .summarization.factory import create_summarization_provider
from .transcription.factory import create_transcription_provider

logger = logging.getLogger(__name__)

# Module-level registry for preloaded MLProvider instance
# This allows factories to reuse the same instance across capabilities
_preloaded_ml_provider: Optional[Any] = None


def _initialize_ml_environment() -> None:
    """Initialize environment variables for ML libraries.

    This function sets default environment variables that improve the behavior of ML libraries
    (PyTorch, Transformers, Hugging Face Hub) in production. These defaults can be overridden
    by setting the environment variables in `.env` file or system environment.

    Environment Variables:
    - HF_HUB_DISABLE_PROGRESS_BARS: Disables progress bars to avoid misleading
      "Downloading" messages when loading models from cache. Default: "1" (disabled)
      Can be overridden in `.env` file or system environment.
    - OMP_NUM_THREADS, MKL_NUM_THREADS, TORCH_NUM_THREADS: Not set here by default
      (allows full CPU utilization). Users can set these in `.env` file or system
      environment to limit thread usage if needed.

    This is called automatically when the pipeline starts, but can also be called
    manually if needed.

    Note:
        - These settings are different from test environment settings, which limit
          threads to reduce resource usage. In production, we want full performance.
        - User-set values (from `.env` or system environment) take precedence over
          these defaults (respects `override=False` in dotenv loading).
        - See `examples/.env.example` for documentation of all environment variables.

    See Also:
        - `examples/.env.example`: Template with all environment variables
        - `docs/api/CONFIGURATION.md`: Complete environment variable documentation
    """
    # Disable Hugging Face Hub progress bars to avoid misleading "Downloading" messages
    # when loading models from cache. This is especially important when models are
    # already cached and no actual download is happening.
    # Only set if not already set by user (respects .env file or system environment)
    if "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        logger.debug("Set HF_HUB_DISABLE_PROGRESS_BARS=1 to suppress misleading progress bars")

    # Note: We don't set OMP_NUM_THREADS, MKL_NUM_THREADS, or TORCH_NUM_THREADS here
    # because in production we want to use all available CPU cores for best performance.
    # Users can set these environment variables in `.env` file or system environment
    # if they want to limit threads (e.g., in Docker containers with limited resources).
    # In tests, these are set to "1" to reduce resource usage (see tests/conftest.py).
    # See `examples/.env.example` for documentation.


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


def _should_preload_ml_models(cfg: config.Config) -> bool:
    """Check if ML models should be preloaded based on configuration.

    Args:
        cfg: Configuration object

    Returns:
        True if any ML models need to be preloaded, False otherwise
    """
    # Check if any ML models are needed
    needs_whisper = cfg.transcribe_missing and cfg.transcription_provider == "whisper"
    needs_transformers = cfg.generate_summaries and cfg.summary_provider == "transformers"
    needs_spacy = cfg.auto_speakers and cfg.speaker_detector_provider == "spacy"

    return needs_whisper or needs_transformers or needs_spacy


def _ensure_ml_models_cached(cfg: config.Config) -> None:
    """Ensure required ML models are cached, downloading them if needed.

    This function checks if required models are cached, and if not, downloads them
    using the centralized preload script logic. This is the ONLY place where models
    can be downloaded - libraries are never allowed to download on their own.

    Args:
        cfg: Configuration object

    Note:
        This only downloads models if they're not already cached. It uses the same
        preload logic as the preload script, ensuring all downloads go through our
        centralized mechanism. Models are then loaded with local_files_only=True
        to prevent libraries from attempting their own downloads.
    """
    # Skip if preloading is disabled or in dry run
    if not cfg.preload_models or cfg.dry_run:
        return

    # Skip if no ML models are needed
    if not _should_preload_ml_models(cfg):
        return

    # Skip in test environments (tests should use pre-cached models)
    if config._is_test_environment():
        return

    try:
        from .cache_utils import (
            get_transformers_cache_dir,
            get_whisper_cache_dir,
        )

        models_to_download = []

        # Check Whisper models
        if cfg.transcribe_missing and cfg.transcription_provider == "whisper":
            whisper_cache = get_whisper_cache_dir()
            model_file = whisper_cache / f"{cfg.whisper_model}.pt"
            if not model_file.exists():
                models_to_download.append(("whisper", cfg.whisper_model))
                logger.info(
                    f"Whisper model {cfg.whisper_model} not cached, will download on first use"
                )

        # Check Transformers models
        if cfg.generate_summaries and cfg.summary_provider == "transformers":
            from . import summarizer

            transformers_cache = get_transformers_cache_dir()
            # Get MAP model
            map_model = summarizer.select_summary_model(cfg)
            model_cache_name = map_model.replace("/", "--")
            model_cache_path = transformers_cache / f"models--{model_cache_name}"
            if not model_cache_path.exists():
                models_to_download.append(("transformers", map_model))
                logger.info(
                    f"Transformers model {map_model} not cached, will download on first use"
                )

            # Get REDUCE model (might be different)
            reduce_model = summarizer.select_reduce_model(cfg, map_model)
            if reduce_model != map_model:
                reduce_cache_name = reduce_model.replace("/", "--")
                reduce_cache_path = transformers_cache / f"models--{reduce_cache_name}"
                if not reduce_cache_path.exists():
                    models_to_download.append(("transformers", reduce_model))
                    logger.info(
                        f"Transformers model {reduce_model} not cached, will download on first use"
                    )

        # Download missing models using centralized preload functions (internal package code)
        if models_to_download:
            logger.info("Downloading missing ML models (this may take a few minutes)...")
            try:
                # Import preload functions from internal package module
                # This is the ONLY place where models can be downloaded
                from .model_loader import (
                    preload_transformers_models,
                    preload_whisper_models,
                )

                # Group models by type
                whisper_models = [m[1] for m in models_to_download if m[0] == "whisper"]
                transformers_models = [m[1] for m in models_to_download if m[0] == "transformers"]

                if whisper_models:
                    preload_whisper_models(whisper_models)
                if transformers_models:
                    preload_transformers_models(transformers_models)

                logger.info("Missing models downloaded and cached successfully")
            except ImportError:
                logger.warning(
                    "Model loader module not available. "
                    "You may need to run 'make preload-ml-models' manually."
                )
            except Exception as e:
                logger.warning(
                    f"Could not automatically download models: {e}. "
                    "You may need to run 'make preload-ml-models' manually."
                )
                # Don't fail - let the normal loading process handle the error

    except ImportError:
        # Preload script not available - that's okay, we'll try to load anyway
        pass
    except Exception as e:
        # Don't fail on cache check errors - let normal loading handle it
        logger.debug(f"Error checking model cache: {e}")


def _preload_ml_models_if_needed(cfg: config.Config) -> None:
    """Preload ML models early in the pipeline if configured to use them.

    This function creates an MLProvider instance and calls preload() on it,
    storing the instance in a module-level registry for reuse by factories.

    Args:
        cfg: Configuration object

    Raises:
        RuntimeError: If required model cannot be loaded
        ImportError: If ML dependencies are not installed
    """
    global _preloaded_ml_provider

    # Skip if preloading is disabled or in dry run
    if not cfg.preload_models or cfg.dry_run:
        return

    # Skip if no ML models are needed
    if not _should_preload_ml_models(cfg):
        return

    # Ensure models are cached (download if needed, but only in production)
    _ensure_ml_models_cached(cfg)

    # Create MLProvider instance and preload models
    try:
        from .ml.ml_provider import MLProvider

        _preloaded_ml_provider = MLProvider(cfg)
        _preloaded_ml_provider.preload()
        logger.debug("ML models preloaded successfully, instance stored for reuse")
    except ImportError as e:
        logger.warning("ML dependencies not available, skipping model preloading: %s", e)
        _preloaded_ml_provider = None
    except Exception as e:
        logger.error("Failed to preload ML models: %s", e)
        _preloaded_ml_provider = None
        # Re-raise to fail fast for required models
        raise


def _update_metric_safely(
    pipeline_metrics: metrics.Metrics,
    metric_name: str,
    value: int,
    lock: Optional[threading.Lock] = None,
) -> None:
    """Update a metric value in a thread-safe manner.

    Args:
        pipeline_metrics: Metrics object to update
        metric_name: Name of the metric attribute to update
        value: Value to add to the metric
        lock: Optional lock for thread safety
    """
    if lock:
        with lock:
            setattr(pipeline_metrics, metric_name, getattr(pipeline_metrics, metric_name) + value)
    else:
        setattr(pipeline_metrics, metric_name, getattr(pipeline_metrics, metric_name) + value)


def _call_generate_metadata(
    episode: models.Episode,
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    transcript_path: Optional[str],
    transcript_source: Optional[Literal["direct_download", "whisper_transcription"]],
    whisper_model: Optional[str],
    feed_metadata: _FeedMetadata,
    host_detection_result: _HostDetectionResult,
    detected_names: Optional[List[str]],
    summary_provider=None,  # SummarizationProvider instance (required)
    pipeline_metrics: Optional[metrics.Metrics] = None,
) -> None:
    """Call _generate_episode_metadata with common parameters.

    This helper reduces code duplication by centralizing the metadata generation call.

    Args:
        episode: Episode object
        feed: RssFeed object
        cfg: Configuration object
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix
        transcript_path: Path to transcript file
        transcript_source: Source of transcript (direct_download or whisper_transcription)
        whisper_model: Whisper model name if used (for metadata, not for transcription)
        feed_metadata: Feed metadata tuple
        host_detection_result: Host detection result
        detected_names: Detected guest names
        summary_provider: SummarizationProvider instance (required)
        pipeline_metrics: Metrics object
    """
    _generate_episode_metadata(
        feed=feed,
        episode=episode,
        feed_url=cfg.rss_url or "",
        cfg=cfg,
        output_dir=effective_output_dir,
        run_suffix=run_suffix,
        transcript_file_path=transcript_path,
        transcript_source=transcript_source,
        whisper_model=whisper_model,
        detected_hosts=(
            list(host_detection_result.cached_hosts) if host_detection_result.cached_hosts else None
        ),
        detected_guests=detected_names if detected_names else None,
        feed_description=feed_metadata.description,
        feed_image_url=feed_metadata.image_url,
        feed_last_updated=feed_metadata.last_updated,
        summary_provider=summary_provider,
        pipeline_metrics=pipeline_metrics,
    )


def apply_log_level(level: str, log_file: Optional[str] = None) -> None:
    """Apply logging level to root logger and configure handlers.

    Args:
        level: Log level string (e.g., 'DEBUG', 'INFO', 'WARNING')
        log_file: Optional path to log file. If provided, logs will be written to both
                  console and file.

    Raises:
        ValueError: If log level is invalid
        OSError: If log file cannot be created or written to
    """
    numeric_level = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    root_logger = logging.getLogger()
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    # Remove existing handlers if we're setting up fresh
    if not root_logger.handlers:
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(console_handler)
        root_logger.setLevel(numeric_level)
    else:
        # Update existing handlers
        root_logger.setLevel(numeric_level)
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)

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
            file_handler.setFormatter(logging.Formatter(log_format))
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
    _initialize_ml_environment()

    # Initialize metrics collector
    pipeline_metrics = metrics.Metrics()

    # Step 1: Setup pipeline environment
    effective_output_dir, run_suffix = _setup_pipeline_environment(cfg)

    # Step 1.5: Preload ML models if configured
    _preload_ml_models_if_needed(cfg)

    # Step 2: Fetch and parse RSS feed (scraping stage)
    scraping_start = time.time()
    feed, rss_bytes = _fetch_and_parse_feed(cfg)
    pipeline_metrics.record_stage("scraping", time.time() - scraping_start)

    # Step 3: Extract feed metadata (if metadata generation enabled)
    # Reuse RSS bytes from initial fetch to avoid duplicate network request
    feed_metadata = _extract_feed_metadata_for_generation(cfg, feed, rss_bytes)

    # Step 4: Prepare episodes from RSS items (parsing stage)
    parsing_start = time.time()
    episodes = _prepare_episodes_from_feed(feed, cfg)
    pipeline_metrics.episodes_scraped_total = len(episodes)
    pipeline_metrics.record_stage("parsing", time.time() - parsing_start)

    # Step 5: Detect hosts and analyze patterns (if auto_speakers enabled)
    # This is part of normalizing stage
    normalizing_start = time.time()
    host_detection_result = _detect_feed_hosts_and_patterns(cfg, feed, episodes)

    # Step 6: Setup transcription resources (Whisper model, temp dir)
    transcription_resources = _setup_transcription_resources(cfg, effective_output_dir)

    # Step 6.5: Setup processing resources (metadata/summarization queue)
    processing_resources = _setup_processing_resources(cfg)

    # Step 6.6: Setup summary provider if summarization is enabled
    # Stage 4: Use provider pattern for summarization
    summary_provider = None

    if cfg.generate_summaries and not cfg.dry_run:
        try:
            # Stage 4: Create and initialize summarization provider
            summary_provider = create_summarization_provider(cfg)
            summary_provider.initialize()
            logger.debug(
                "Summarization provider initialized: %s",
                type(summary_provider).__name__,
            )
        except ImportError:
            logger.info("Summarization dependencies not available, skipping summary generation")
        except Exception as e:
            logger.error("Failed to initialize summarization provider: %s", e)
            # Fail fast - provider initialization should succeed
            # If provider creation fails, we cannot proceed with summarization
            summary_provider = None

    # Wrap all processing in try-finally to ensure cleanup always happens
    # This prevents memory leaks if exceptions occur during processing
    try:
        # Step 7: Prepare episode processing arguments with speaker detection
        download_args = _prepare_episode_download_args(
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
        writing_start = time.time()
        transcription_complete_event = threading.Event()
        downloads_complete_event = threading.Event()  # Signal when all downloads are complete
        transcription_saved = [0]  # Use list to allow modification from thread

        # Start processing thread if metadata generation is enabled
        processing_thread = None
        if cfg.generate_metadata:
            processing_thread = threading.Thread(
                target=_process_processing_jobs_concurrent,
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
                target=_process_transcription_jobs_concurrent,
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

        saved = _process_episodes(
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
            saved += _process_transcription_jobs(
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
        pipeline_metrics.record_stage("writing_storage", time.time() - writing_start)

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
            _parallel_episode_summarization(
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
    _cleanup_pipeline(temp_dir=transcription_resources.temp_dir)

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

    return _generate_pipeline_summary(
        cfg, saved, transcription_resources, effective_output_dir, pipeline_metrics
    )


def _setup_pipeline_environment(cfg: config.Config) -> Tuple[str, Optional[str]]:
    """Setup output directory and handle cleanup if needed.

    Creates the output directory structure based on configuration, optionally
    adding a run ID subdirectory. If clean_output is enabled, removes any
    existing output directory before creating it.

    Args:
        cfg: Configuration object with output_dir, run_id, clean_output,
            and dry_run settings

    Returns:
        Tuple[str, Optional[str]]: A tuple containing:
            - effective_output_dir (str): Full path to output directory
              (may include run_id subdirectory)
            - run_suffix (Optional[str]): Run ID suffix if run_id was provided,
              None otherwise

    Raises:
        RuntimeError: If output directory cleanup fails when clean_output=True
        OSError: If directory creation fails

    Example:
        >>> cfg = Config(rss_url="...", output_dir="./out", run_id="test_run")
        >>> output_dir, run_suffix = _setup_pipeline_environment(cfg)
        >>> print(output_dir)  # "./out/test_run"
        >>> print(run_suffix)  # "test_run"
    """
    effective_output_dir, run_suffix = filesystem.setup_output_directory(cfg)
    logger.debug("Effective output dir=%s (run_suffix=%s)", effective_output_dir, run_suffix)

    if cfg.clean_output and cfg.dry_run:
        if os.path.exists(effective_output_dir):
            logger.info(
                "Dry-run: would remove existing output directory (--clean-output): %s",
                effective_output_dir,
            )
    elif cfg.clean_output:
        try:
            if os.path.exists(effective_output_dir):
                shutil.rmtree(effective_output_dir)
                logger.info(
                    "Removed existing output directory (--clean-output): %s",
                    effective_output_dir,
                )
        except OSError as exc:
            raise RuntimeError(
                f"Failed to clean output directory {effective_output_dir}: {exc}"
            ) from exc

    if cfg.dry_run:
        logger.info(f"Dry-run: not creating output directory {effective_output_dir}")
    else:
        os.makedirs(effective_output_dir, exist_ok=True)
        # Recreate subdirectories after cleanup (if clean_output was used)
        # or ensure they exist (setup_output_directory creates them, but they may have been removed)
        transcripts_dir = os.path.join(effective_output_dir, filesystem.TRANSCRIPTS_SUBDIR)
        metadata_dir = os.path.join(effective_output_dir, filesystem.METADATA_SUBDIR)
        os.makedirs(transcripts_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

    return effective_output_dir, run_suffix


def _fetch_and_parse_feed(cfg: config.Config) -> tuple[models.RssFeed, bytes]:
    """Fetch and parse RSS feed.

    Fetches RSS feed once and returns both the parsed feed and raw XML bytes
    to avoid duplicate network requests.

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (Parsed RssFeed object, RSS XML bytes)
    """
    from . import downloader
    from .rss_parser import parse_rss_items

    if cfg.rss_url is None:
        raise ValueError("RSS URL is required")

    # Fetch RSS feed once
    resp = downloader.fetch_url(cfg.rss_url, cfg.user_agent, cfg.timeout, stream=False)
    if resp is None:
        raise ValueError("Failed to fetch RSS feed.")

    try:
        rss_bytes = resp.content
        feed_base_url = resp.url or cfg.rss_url
    finally:
        resp.close()

    # Parse RSS feed
    try:
        feed_title, feed_authors, items = parse_rss_items(rss_bytes)
    except Exception as exc:
        raise ValueError(f"Failed to parse RSS XML: {exc}") from exc

    feed = models.RssFeed(
        title=feed_title, authors=feed_authors, items=items, base_url=feed_base_url
    )
    logger.debug("Fetched RSS feed title=%s (%s items)", feed.title, len(feed.items))

    return feed, rss_bytes


def _extract_feed_metadata_for_generation(
    cfg: config.Config, feed: models.RssFeed, rss_bytes: bytes
) -> _FeedMetadata:
    """Extract feed metadata for metadata generation.

    Args:
        cfg: Configuration object
        feed: Parsed RssFeed object
        rss_bytes: Raw RSS XML bytes (reused from initial fetch to avoid duplicate request)

    Returns:
        _FeedMetadata tuple
    """
    if not cfg.generate_metadata or not rss_bytes:
        return _FeedMetadata(None, None, None)

    try:
        feed_description, feed_image_url, feed_last_updated = extract_feed_metadata(
            rss_bytes, feed.base_url
        )
        return _FeedMetadata(feed_description, feed_image_url, feed_last_updated)
    except Exception as exc:
        logger.debug("Failed to extract feed metadata: %s", exc)
        return _FeedMetadata(None, None, None)


def _prepare_episodes_from_feed(feed: models.RssFeed, cfg: config.Config) -> List[models.Episode]:
    """Create Episode objects from RSS items.

    Args:
        feed: Parsed RssFeed object
        cfg: Configuration object

    Returns:
        List of Episode objects
    """
    items = feed.items
    total_items = len(items)
    if cfg.max_episodes is not None:
        items = items[: cfg.max_episodes]

    logger.info(f"Episodes to process: {len(items)} of {total_items}")

    episodes = [
        create_episode_from_item(item, idx, feed.base_url)
        for idx, item in enumerate(items, start=1)
    ]
    logger.debug("Materialized %s episode objects", len(episodes))
    return episodes


def _detect_feed_hosts_and_patterns(
    cfg: config.Config, feed: models.RssFeed, episodes: List[models.Episode]
) -> _HostDetectionResult:
    """Detect hosts from feed metadata and analyze episode patterns.

    Args:
        cfg: Configuration object
        feed: Parsed RssFeed object
        episodes: List of Episode objects

    Returns:
        _HostDetectionResult with cached_hosts and heuristics
    """
    cached_hosts: set[str] = set()
    heuristics: Optional[Dict[str, Any]] = None

    if not cfg.auto_speakers or not cfg.cache_detected_hosts:
        return _HostDetectionResult(cached_hosts, heuristics, None)

    # Stage 3: Use provider pattern for speaker detection
    try:
        speaker_detector = create_speaker_detector(cfg)
        # Initialize provider (loads spaCy model)
        speaker_detector.initialize()

        # Detect hosts: prefer RSS author tags, fall back to NER
        feed_hosts = speaker_detector.detect_hosts(
            feed_title=feed.title,
            feed_description=None,  # TODO: Extract from feed XML if needed
            feed_authors=feed.authors if feed.authors else None,
        )

        # Validate hosts with first episode: hosts should appear in first episode too
        # Skip validation if hosts came from author tags (they're already reliable)
        if feed_hosts and episodes and not feed.authors:
            # Only validate if we used NER (not author tags)
            first_episode = episodes[0]
            first_episode_description = extract_episode_description(first_episode.item)
            # Validate hosts by checking if they appear in first episode
            # Use provider's detect_speakers to extract persons from first episode
            first_episode_speakers, _, _ = speaker_detector.detect_speakers(
                episode_title=first_episode.title,
                episode_description=first_episode_description,
                known_hosts=set(),
            )
            first_episode_persons = set(first_episode_speakers)
            # Only keep hosts that also appear in first episode (validation)
            validated_hosts = feed_hosts & first_episode_persons
            if validated_hosts != feed_hosts:
                logger.debug(
                    "Host validation: %d hosts from feed, %d validated with first episode",
                    len(feed_hosts),
                    len(validated_hosts),
                )
                if validated_hosts:
                    logger.debug(
                        "Validated hosts (appear in feed and first episode): %s",
                        list(validated_hosts),
                    )
                if feed_hosts - validated_hosts:
                    logger.debug(
                        "Hosts from feed not found in first episode (discarded): %s",
                        list(feed_hosts - validated_hosts),
                    )
            cached_hosts = validated_hosts if validated_hosts else feed_hosts
        else:
            # If hosts came from author tags, use them directly (no validation needed)
            cached_hosts = feed_hosts

        if cached_hosts:
            source = "RSS author tags" if feed.authors else "feed metadata (NER)"
            logger.info("=" * 60)
            logger.info("DETECTED HOSTS (from %s): %s", source, ", ".join(sorted(cached_hosts)))
            logger.info("=" * 60)
        elif cfg.auto_speakers:
            logger.debug("No hosts detected from feed metadata")

        # Analyze patterns from first few episodes to extract heuristics
        if cfg.auto_speakers and episodes:
            heuristics_dict = speaker_detector.analyze_patterns(
                episodes=episodes, known_hosts=cached_hosts
            )
            if heuristics_dict:
                heuristics = heuristics_dict
                if heuristics.get("title_position_preference"):
                    logger.debug(
                        "Pattern analysis: guest names typically appear at %s of title",
                        heuristics["title_position_preference"],
                    )

        # Return result with provider instance
        return _HostDetectionResult(cached_hosts, heuristics, speaker_detector)
    except Exception as exc:
        logger.error("Failed to initialize speaker detector provider: %s", exc)
        # Fail fast - provider initialization should succeed
        # Return empty result without provider
        return _HostDetectionResult(set(), None, None)


def _setup_transcription_resources(
    cfg: config.Config, effective_output_dir: str
) -> _TranscriptionResources:
    """Setup transcription provider and temp directory for transcription.

    Args:
        cfg: Configuration object
        effective_output_dir: Output directory path

    Returns:
        _TranscriptionResources object
    """
    transcription_provider = None

    if cfg.transcribe_missing and not cfg.dry_run:
        # Stage 2: Use provider pattern
        try:
            transcription_provider = create_transcription_provider(cfg)
            transcription_provider.initialize()
            logger.debug(
                "Transcription provider initialized: %s",
                type(transcription_provider).__name__,
            )
        except Exception as exc:
            logger.error("Failed to initialize transcription provider: %s", exc)
            # Fail fast - provider initialization should succeed
            # If provider creation fails, we cannot proceed with transcription
            transcription_provider = None

    temp_dir = None
    if cfg.transcribe_missing:
        temp_dir = os.path.join(effective_output_dir, filesystem.TEMP_DIR_NAME)
        if not cfg.dry_run:
            os.makedirs(temp_dir, exist_ok=True)
        logger.debug("Temp directory for media downloads: %s", temp_dir)

    transcription_jobs: List[models.TranscriptionJob] = []
    transcription_jobs_lock = threading.Lock() if cfg.workers > 1 else None
    saved_counter_lock = threading.Lock() if cfg.workers > 1 else None

    return _TranscriptionResources(
        transcription_provider,
        temp_dir,
        transcription_jobs,
        transcription_jobs_lock,
        saved_counter_lock,
    )


def _setup_processing_resources(cfg: config.Config) -> _ProcessingResources:
    """Set up resources for processing stage (metadata/summarization).

    Args:
        cfg: Configuration object

    Returns:
        _ProcessingResources with processing queue and locks
    """
    processing_jobs: List[_ProcessingJob] = []
    processing_jobs_lock = (
        threading.Lock()
        if (cfg.workers > 1 or cfg.transcription_parallelism > 1 or cfg.processing_parallelism > 1)
        else None
    )
    processing_complete_event = threading.Event()

    return _ProcessingResources(
        processing_jobs,
        processing_jobs_lock,
        processing_complete_event,
    )


def _prepare_episode_download_args(
    episodes: List[models.Episode],
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    transcription_resources: _TranscriptionResources,
    host_detection_result: _HostDetectionResult,
    pipeline_metrics: metrics.Metrics,
) -> List[Tuple]:
    """Prepare download arguments for each episode with speaker detection.

    Performs speaker detection (if enabled) for each episode and packages all
    necessary information into tuples for parallel processing. Speaker detection
    includes host detection from feed metadata and guest detection from episode
    titles and descriptions using NER.

    Args:
        episodes: List of Episode objects to process
        cfg: Configuration object with auto_speakers, cache_detected_hosts settings
        effective_output_dir: Full path to output directory
        run_suffix: Optional run ID suffix for file naming
        transcription_resources: Transcription resources (Whisper model, temp dir, job queue)
        host_detection_result: Previously detected hosts and heuristics from feed metadata
        pipeline_metrics: Metrics collector for tracking speaker extraction timing

    Returns:
        List[Tuple]: List of argument tuples, each containing:
            (episode, cfg, temp_dir, effective_output_dir, run_suffix,
             whisper_model, transcription_jobs, detected_speaker_names,
             transcription_jobs_lock, saved_counter_lock, pipeline_metrics)
    """
    download_args = []
    for episode in episodes:
        detected_speaker_names = None
        # Detect guests for all episodes when auto_speakers is enabled
        # (not just when transcribing, so we can log guests even for transcript downloads)
        # Note: Guest detection works in dry-run mode (no media download/transcription needed)
        if cfg.auto_speakers:
            # Always log episode info
            logger.info("Episode %d: %s", episode.idx, episode.title)

            # Extract episode description for NER (limited to first 20 chars)
            episode_description = extract_episode_description(episode.item)
            # Detect speaker names from episode title and first 20 chars of description
            # Guests are detected from episode title and description snippet
            extract_names_start = time.time()

            # Stage 3: Use provider for speaker detection
            speaker_detector = host_detection_result.speaker_detector
            if speaker_detector:
                cached_hosts_for_detection = (
                    host_detection_result.cached_hosts if cfg.cache_detected_hosts else set()
                )
                detected_speakers, detected_hosts_set, detection_succeeded = (
                    speaker_detector.detect_speakers(
                        episode_title=episode.title,
                        episode_description=episode_description,
                        known_hosts=cached_hosts_for_detection,
                    )
                )
            else:
                # Fallback: No provider available (should not happen in normal flow)
                logger.warning("Speaker detector not available, skipping speaker detection")
                detected_speakers, detected_hosts_set, detection_succeeded = [], set(), False
            extract_names_elapsed = time.time() - extract_names_start
            # Record speaker detection time if metrics available
            if pipeline_metrics is not None:
                pipeline_metrics.record_extract_names_time(extract_names_elapsed)

            # Use manual guest name as fallback ONLY if detection failed
            # Manual names: first item = host, second item = guest
            if (
                not detection_succeeded
                and cfg.screenplay_speaker_names
                and len(cfg.screenplay_speaker_names) >= 2
            ):
                # Keep detected hosts, only use manual guest fallback
                manual_host = cfg.screenplay_speaker_names[0]
                manual_guest = cfg.screenplay_speaker_names[1]

                # Use detected hosts if available, otherwise use manual host
                if detected_hosts_set:
                    fallback_names = list(detected_hosts_set) + [manual_guest]
                    logger.debug(
                        "  → Guest detection failed, using manual guest fallback: %s (hosts: %s)",
                        manual_guest,
                        ", ".join(detected_hosts_set),
                    )
                else:
                    # No hosts detected either, use both manual names
                    fallback_names = [manual_host, manual_guest]
                    logger.debug(
                        "  → Detection failed, using manual fallback: %s, %s",
                        manual_host,
                        manual_guest,
                    )
                detected_speaker_names = fallback_names
            elif detection_succeeded:
                detected_speaker_names = detected_speakers
            # Note: Guest logging happens inside detect_speaker_names()
            # Note: We don't update cached_hosts here because hosts are only
            # detected from feed metadata, not from episodes
        elif cfg.screenplay_speaker_names:
            # If auto_speakers is disabled, use manual names directly
            detected_speaker_names = cfg.screenplay_speaker_names

        download_args.append(
            (
                episode,
                cfg,
                transcription_resources.temp_dir,
                effective_output_dir,
                run_suffix,
                transcription_resources.transcription_jobs,
                transcription_resources.transcription_jobs_lock,
                detected_speaker_names,
            )
        )

    return download_args


def _process_episodes(
    download_args: List[Tuple],
    episodes: List[models.Episode],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: _FeedMetadata,
    host_detection_result: _HostDetectionResult,
    transcription_resources: _TranscriptionResources,
    processing_resources: _ProcessingResources,
    pipeline_metrics: metrics.Metrics,
    summary_provider=None,  # SummarizationProvider instance (required)
) -> int:
    """Process episodes: download transcripts or queue transcription jobs.

    Args:
        download_args: List of download argument tuples
        episodes: List of Episode objects
        feed: Parsed RssFeed object
        cfg: Configuration object
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix
        feed_metadata: Feed metadata tuple
        host_detection_result: Host detection result

    Returns:
        Number of transcripts saved
    """
    if not download_args:
        return 0

    saved = 0
    if cfg.workers <= 1 or len(download_args) == 1:
        # Sequential processing
        for args in download_args:
            (
                episode,
                cfg_arg,
                temp_dir_arg,
                output_dir_arg,
                run_suffix_arg,
                jobs_arg,
                lock_arg,
                detected_names,
            ) = args
            try:
                success, transcript_path, transcript_source, bytes_downloaded = (
                    process_episode_download(
                        episode,
                        cfg_arg,
                        temp_dir_arg,
                        output_dir_arg,
                        run_suffix_arg,
                        jobs_arg,
                        lock_arg,
                        detected_names,
                        pipeline_metrics=pipeline_metrics,
                    )
                )
                if bytes_downloaded:
                    _update_metric_safely(
                        pipeline_metrics, "bytes_downloaded_total", bytes_downloaded
                    )
                if success:
                    saved += 1
                    # Track transcript source
                    if transcript_source == "direct_download":
                        _update_metric_safely(pipeline_metrics, "transcripts_downloaded", 1)
                    logger.debug("Episode %s yielded transcript (saved=%s)", episode.idx, saved)

                    # Queue processing job if metadata generation enabled and transcript available
                    # Skip if transcript_source is None (Whisper pending) - queued after
                    if cfg.generate_metadata and transcript_source is not None:
                        transcript_source_typed = cast(
                            Literal["direct_download", "whisper_transcription"],
                            transcript_source,
                        )
                        processing_job = _ProcessingJob(
                            episode=episode,
                            transcript_path=transcript_path or "",
                            transcript_source=transcript_source_typed,
                            detected_names=detected_names,
                            whisper_model=None,  # Direct downloads don't use Whisper
                        )
                        # Queue processing job (processing thread will pick it up)
                        if processing_resources.processing_jobs_lock:
                            with processing_resources.processing_jobs_lock:
                                processing_resources.processing_jobs.append(processing_job)
                        else:
                            processing_resources.processing_jobs.append(processing_job)
                        logger.debug(
                            "Queued processing job for episode %s (transcript_source=%s)",
                            episode.idx,
                            transcript_source_typed,
                        )
                elif transcript_path is None and transcript_source is None:
                    # Episode was skipped (skip_existing)
                    _update_metric_safely(pipeline_metrics, "episodes_skipped_total", 1)
            except Exception as exc:  # pragma: no cover
                _update_metric_safely(pipeline_metrics, "errors_total", 1)
                logger.error(
                    f"[{episode.idx}] episode processing raised an unexpected error: {exc}"
                )
    else:
        # Concurrent processing
        saved_counter_lock = transcription_resources.saved_counter_lock
        # Note: processing_resources is accessed via closure
        with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            future_map = {
                executor.submit(
                    process_episode_download,
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    args[6],
                    args[7],
                    pipeline_metrics=pipeline_metrics,
                ): args[0].idx
                for args in download_args
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    success, transcript_path, transcript_source, bytes_downloaded = future.result()
                    if bytes_downloaded:
                        _update_metric_safely(
                            pipeline_metrics,
                            "bytes_downloaded_total",
                            bytes_downloaded,
                            saved_counter_lock,
                        )
                    if success:
                        if saved_counter_lock:
                            with saved_counter_lock:
                                saved += 1
                        else:
                            saved += 1
                        if transcript_source == "direct_download":
                            _update_metric_safely(
                                pipeline_metrics, "transcripts_downloaded", 1, saved_counter_lock
                            )
                        logger.debug("Episode %s yielded transcript (saved=%s)", idx, saved)
                    elif transcript_path is None and transcript_source is None:
                        # Episode was skipped (skip_existing)
                        _update_metric_safely(
                            pipeline_metrics, "episodes_skipped_total", 1, saved_counter_lock
                        )

                    # Queue processing job if metadata generation enabled and transcript available
                    # Skip if transcript_source is None (Whisper pending) - queued after
                    if cfg.generate_metadata and transcript_source is not None:
                        episode_obj = next((ep for ep in episodes if ep.idx == idx), None)
                        if episode_obj:
                            # Find detected names for this episode
                            detected_names_for_ep = None
                            for args in download_args:
                                if args[0].idx == idx:
                                    detected_names_for_ep = args[7]
                                    break
                            transcript_source_typed = cast(
                                Literal["direct_download", "whisper_transcription"],
                                transcript_source,
                            )
                            processing_job = _ProcessingJob(
                                episode=episode_obj,
                                transcript_path=transcript_path or "",
                                transcript_source=transcript_source_typed,
                                detected_names=detected_names_for_ep,
                                whisper_model=None,  # Direct downloads don't use Whisper
                            )
                            # Queue processing job (processing thread will pick it up)
                            if processing_resources.processing_jobs_lock:
                                with processing_resources.processing_jobs_lock:
                                    processing_resources.processing_jobs.append(processing_job)
                            else:
                                processing_resources.processing_jobs.append(processing_job)
                            logger.debug(
                                "Queued processing job for episode %s (transcript_source=%s)",
                                episode_obj.idx,
                                transcript_source_typed,
                            )
                except Exception as exc:  # pragma: no cover
                    _update_metric_safely(pipeline_metrics, "errors_total", 1, saved_counter_lock)
                    logger.error(f"[{idx}] episode processing raised an unexpected error: {exc}")

    return saved


def _process_transcription_jobs(
    transcription_resources: _TranscriptionResources,
    download_args: List[Tuple],
    episodes: List[models.Episode],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: _FeedMetadata,
    host_detection_result: _HostDetectionResult,
    pipeline_metrics: metrics.Metrics,
    summary_provider=None,  # SummarizationProvider instance (required)
) -> int:
    """Process Whisper transcription jobs sequentially.

    Args:
        transcription_resources: Transcription resources
        download_args: List of download argument tuples
        episodes: List of Episode objects
        feed: Parsed RssFeed object
        cfg: Configuration object
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix
        feed_metadata: Feed metadata tuple
        host_detection_result: Host detection result

    Returns:
        Number of transcripts saved from transcription
    """
    if not transcription_resources.transcription_jobs or not cfg.transcribe_missing:
        return 0

    saved = 0
    total_jobs = len(transcription_resources.transcription_jobs)
    if cfg.dry_run:
        logger.info(f"Dry-run: would transcribe {total_jobs} episodes with Whisper")
    else:
        logger.info(f"Starting Whisper transcription for {total_jobs} episodes")

    with progress.progress_context(total_jobs, "Whisper transcription") as reporter:
        jobs_processed = 0
        for job in transcription_resources.transcription_jobs:
            try:
                # Stage 2: Use provider if available, otherwise fall back to direct model
                # For backward compatibility, we pass both provider and model
                # transcribe_media_to_text will use provider if available
                success, transcript_path, bytes_downloaded = transcribe_media_to_text(
                    job,
                    cfg,
                    None,  # whisper_model no longer needed (use provider instead)
                    run_suffix,
                    effective_output_dir,
                    transcription_provider=transcription_resources.transcription_provider,
                    pipeline_metrics=pipeline_metrics,
                )
                if bytes_downloaded:
                    _update_metric_safely(
                        pipeline_metrics, "bytes_downloaded_total", bytes_downloaded
                    )
                if success:
                    saved += 1
                    _update_metric_safely(pipeline_metrics, "transcripts_transcribed", 1)

                    # Generate metadata if enabled
                    if cfg.generate_metadata:
                        episode_obj = next((ep for ep in episodes if ep.idx == job.idx), None)
                        if episode_obj:
                            # Find detected names for this episode
                            detected_names_for_ep = None
                            for args in download_args:
                                if args[0].idx == job.idx:
                                    detected_names_for_ep = args[7]
                                    break
                            _call_generate_metadata(
                                episode=episode_obj,
                                feed=feed,
                                cfg=cfg,
                                effective_output_dir=effective_output_dir,
                                run_suffix=run_suffix,
                                transcript_path=transcript_path,
                                transcript_source="whisper_transcription",
                                whisper_model=None,  # No longer needed (use provider instead)
                                feed_metadata=feed_metadata,
                                host_detection_result=host_detection_result,
                                detected_names=detected_names_for_ep,
                                summary_provider=summary_provider,
                                pipeline_metrics=pipeline_metrics,
                            )
            except Exception as exc:  # pragma: no cover
                _update_metric_safely(pipeline_metrics, "errors_total", 1)
                logger.error(f"[{job.idx}] transcription raised an unexpected error: {exc}")

            reporter.update(1)
            jobs_processed += 1
            logger.debug(
                "Processed transcription job idx=%s (saved=%s, processed=%s/%s)",
                job.idx,
                saved,
                jobs_processed,
                total_jobs,
            )

    return saved


def _process_transcription_jobs_concurrent(
    transcription_resources: _TranscriptionResources,
    download_args: List[Tuple],
    episodes: List[models.Episode],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: _FeedMetadata,
    host_detection_result: _HostDetectionResult,
    processing_resources: _ProcessingResources,
    pipeline_metrics: metrics.Metrics,
    summary_provider=None,  # SummarizationProvider instance (required)
    downloads_complete_event: Optional[threading.Event] = None,
    saved_counter: Optional[List[int]] = None,
) -> None:
    """Process transcription jobs concurrently as they become available.

    This function runs in a separate thread and processes transcription jobs
    from the queue as downloads complete, rather than waiting for all downloads
    to finish before starting transcription.

    Uses transcription_parallelism config to control episode-level parallelism:
    - Whisper provider: Always sequential (parallelism = 1, ignores config > 1)
    - OpenAI provider: Parallel with rate limiting (uses parallelism config)

    Args:
        transcription_resources: Transcription resources
        download_args: List of download argument tuples
        episodes: List of Episode objects
        feed: Parsed RssFeed object
        cfg: Configuration object (uses transcription_parallelism)
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix
        feed_metadata: Feed metadata tuple
        host_detection_result: Host detection result
        pipeline_metrics: Metrics collector
        summary_model: Optional summary model
        reduce_model: Optional reduce model
        downloads_complete_event: Event to signal when downloads are complete
        saved_counter: List to store count of saved transcripts (for thread communication)
    """
    if saved_counter is None:
        saved_counter = [0]

    # Get parallelism from config (provider-specific behavior)
    # Whisper: Always sequential (parallelism = 1)
    # OpenAI: Uses parallelism config (default: 5)
    max_workers = cfg.transcription_parallelism
    if cfg.transcription_provider == "whisper":
        # Whisper is memory/CPU bound - always sequential
        max_workers = 1
        logger.debug(
            "Whisper provider: Using sequential processing (parallelism=%d ignored)",
            cfg.transcription_parallelism,
        )
    else:
        # Other providers (OpenAI) can use parallelism
        logger.debug(
            "Transcription provider '%s': Using parallelism=%d",
            cfg.transcription_provider,
            max_workers,
        )

    saved = 0
    jobs_processed = 0
    processed_job_indices = set()  # Track which jobs we've processed
    processed_job_indices_lock = threading.Lock()  # Lock for thread-safe access

    logger.debug("Concurrent transcription processor started (max_workers=%d)", max_workers)

    def _process_single_job(job: models.TranscriptionJob) -> tuple[bool, Optional[str], int]:
        """Process a single transcription job.

        Returns:
            Tuple of (success, transcript_path, bytes_downloaded)
        """
        try:
            success, transcript_path, bytes_downloaded = transcribe_media_to_text(
                job,
                cfg,
                None,  # whisper_model no longer needed (use provider instead)
                run_suffix,
                effective_output_dir,
                transcription_provider=transcription_resources.transcription_provider,
                pipeline_metrics=pipeline_metrics,
            )
            if bytes_downloaded:
                _update_metric_safely(pipeline_metrics, "bytes_downloaded_total", bytes_downloaded)
            if success:
                _update_metric_safely(pipeline_metrics, "transcripts_transcribed", 1)

                # Queue processing job if metadata generation is enabled
                if cfg.generate_metadata:
                    episode_obj = next((ep for ep in episodes if ep.idx == job.idx), None)
                    if episode_obj:
                        # Find detected names for this episode
                        detected_names_for_ep = None
                        for args in download_args:
                            if args[0].idx == job.idx:
                                detected_names_for_ep = args[7]
                                break
                        processing_job = _ProcessingJob(
                            episode=episode_obj,
                            transcript_path=transcript_path or "",
                            transcript_source="whisper_transcription",
                            detected_names=detected_names_for_ep,
                            whisper_model=cfg.whisper_model,
                        )
                        # Queue processing job (processing thread will pick it up)
                        if processing_resources.processing_jobs_lock:
                            with processing_resources.processing_jobs_lock:
                                processing_resources.processing_jobs.append(processing_job)
                        else:
                            processing_resources.processing_jobs.append(processing_job)
                        logger.debug(
                            "Queued processing job for episode %s (whisper_transcription)",
                            episode_obj.idx,
                        )
            return success, transcript_path, bytes_downloaded
        except Exception as exc:  # pragma: no cover
            _update_metric_safely(pipeline_metrics, "errors_total", 1)
            logger.error(f"[{job.idx}] transcription raised an unexpected error: {exc}")
            return False, None, 0

    # Process jobs as they become available
    # Continue until downloads are complete AND queue is empty
    if max_workers <= 1:
        # Sequential processing (Whisper default)
        while True:
            # Check for new jobs in the queue
            job_found = False
            current_job = None

            if transcription_resources.transcription_jobs_lock:
                with transcription_resources.transcription_jobs_lock:
                    # Find first unprocessed job
                    with processed_job_indices_lock:
                        for job in transcription_resources.transcription_jobs:
                            if job.idx not in processed_job_indices:
                                job_found = True
                                processed_job_indices.add(job.idx)
                                current_job = job
                                break
            else:
                # Single-threaded mode
                with processed_job_indices_lock:
                    for job in transcription_resources.transcription_jobs:
                        if job.idx not in processed_job_indices:
                            job_found = True
                            processed_job_indices.add(job.idx)
                            current_job = job
                            break

            if job_found and current_job:
                success, transcript_path, bytes_downloaded = _process_single_job(current_job)
                if success:
                    saved += 1
                jobs_processed += 1
                logger.debug(
                    "Processed transcription job idx=%s (saved=%s, processed=%s)",
                    current_job.idx,
                    saved,
                    jobs_processed,
                )
                continue

            # No job found - check if we should continue waiting
            if downloads_complete_event and downloads_complete_event.is_set():
                # Downloads are complete, check if queue is empty
                with processed_job_indices_lock:
                    if transcription_resources.transcription_jobs_lock:
                        with transcription_resources.transcription_jobs_lock:
                            total_jobs = len(transcription_resources.transcription_jobs)
                    else:
                        total_jobs = len(transcription_resources.transcription_jobs)
                    queue_empty = total_jobs == len(processed_job_indices)

                if queue_empty:
                    # All jobs processed, exit
                    break

            # Wait a bit before checking again (avoid busy-waiting)
            if not (downloads_complete_event and downloads_complete_event.is_set()):
                time.sleep(0.1)
            else:
                # Downloads complete but might have more jobs - check more frequently
                time.sleep(0.05)
    else:
        # Parallel processing (OpenAI provider)
        from concurrent.futures import as_completed, ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            while True:
                # Submit new jobs as they become available
                if transcription_resources.transcription_jobs_lock:
                    with transcription_resources.transcription_jobs_lock:
                        with processed_job_indices_lock:
                            for job in transcription_resources.transcription_jobs:
                                if job.idx not in processed_job_indices and job.idx not in futures:
                                    processed_job_indices.add(job.idx)
                                    future = executor.submit(_process_single_job, job)
                                    futures[future] = job.idx
                else:
                    with processed_job_indices_lock:
                        for job in transcription_resources.transcription_jobs:
                            if job.idx not in processed_job_indices and job.idx not in futures:
                                processed_job_indices.add(job.idx)
                                future = executor.submit(_process_single_job, job)
                                futures[future] = job.idx

                # Process completed futures
                # Use longer timeout to avoid TimeoutError when processing jobs take time
                for future in as_completed(list(futures.keys()), timeout=1.0):
                    job_idx = futures.pop(future)
                    try:
                        success, transcript_path, bytes_downloaded = future.result()
                        if success:
                            saved += 1
                        jobs_processed += 1
                        logger.debug(
                            "Processed transcription job idx=%s (saved=%s, processed=%s)",
                            job_idx,
                            saved,
                            jobs_processed,
                        )
                    except Exception as exc:  # pragma: no cover
                        logger.error(f"[{job_idx}] transcription future raised error: {exc}")

                # Check if we should continue
                if downloads_complete_event and downloads_complete_event.is_set():
                    # Downloads complete - check if all jobs are submitted and completed
                    with processed_job_indices_lock:
                        if transcription_resources.transcription_jobs_lock:
                            with transcription_resources.transcription_jobs_lock:
                                total_jobs = len(transcription_resources.transcription_jobs)
                        else:
                            total_jobs = len(transcription_resources.transcription_jobs)
                        all_submitted = total_jobs == len(processed_job_indices)
                    if all_submitted and len(futures) == 0:
                        # All jobs submitted and completed
                        break

                # Wait a bit before checking again
                if not (downloads_complete_event and downloads_complete_event.is_set()):
                    time.sleep(0.1)
                else:
                    time.sleep(0.05)

    # Update saved counter
    saved_counter[0] = saved
    total_jobs = (
        len(transcription_resources.transcription_jobs)
        if transcription_resources.transcription_jobs
        else 0
    )
    logger.info(
        f"Concurrent transcription processing completed: {saved}/{total_jobs} "
        f"transcripts saved (parallelism={max_workers})"
    )


def _process_processing_jobs_concurrent(
    processing_resources: _ProcessingResources,
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: _FeedMetadata,
    host_detection_result: _HostDetectionResult,
    pipeline_metrics: metrics.Metrics,
    summary_provider=None,  # SummarizationProvider instance (required)
    transcription_complete_event: Optional[threading.Event] = None,
) -> None:
    """Process metadata/summarization jobs concurrently as they become available.

    This function runs in a separate thread and processes jobs from the processing
    queue as transcripts become available from downloads or transcription.

    Args:
        processing_resources: Processing resources with queue and locks
        feed: Parsed RssFeed object
        cfg: Configuration object (uses processing_parallelism)
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix
        feed_metadata: Feed metadata tuple
        host_detection_result: Host detection result
        pipeline_metrics: Metrics collector
        summary_model: Optional summary model
        reduce_model: Optional reduce model
        transcription_complete_event: Event to signal when transcription is complete
    """
    max_workers = cfg.processing_parallelism
    logger.debug("Concurrent processing processor started (max_workers=%d)", max_workers)

    jobs_processed = 0
    processed_job_indices = set()  # Track which jobs we've processed
    processed_job_indices_lock = threading.Lock()  # Lock for thread-safe access

    def _wait_for_transcript_file(
        transcript_path: str, episode_idx: int, max_wait: float = 5.0
    ) -> bool:
        """Wait for transcript file to exist before processing.

        This prevents race conditions where metadata generation starts before
        the transcript file is fully written to disk.

        Args:
            transcript_path: Path to transcript file (relative or absolute)
            episode_idx: Episode index for logging
            max_wait: Maximum time to wait in seconds (default: 5.0)

        Returns:
            True if file exists, False if timeout exceeded
        """
        if not transcript_path:
            return False

        # Build full path if relative
        if not os.path.isabs(transcript_path):
            full_path = os.path.join(effective_output_dir, transcript_path)
        else:
            full_path = transcript_path

        # Check if file already exists
        if os.path.exists(full_path):
            return True

        # Wait for file to appear (with timeout)
        wait_interval = 0.1  # Check every 100ms
        waited = 0.0
        while waited < max_wait:
            if os.path.exists(full_path):
                logger.debug(
                    "[%s] Transcript file appeared after %.2fs: %s",
                    episode_idx,
                    waited,
                    full_path,
                )
                return True
            time.sleep(wait_interval)
            waited += wait_interval

        # Timeout exceeded
        logger.warning(
            "[%s] Transcript file not found after %.2fs: %s",
            episode_idx,
            max_wait,
            full_path,
        )
        return False

    def _process_single_processing_job(job: _ProcessingJob) -> None:
        """Process a single processing job (metadata/summarization)."""
        try:
            # Wait for transcript file to exist if transcript_path is provided
            # This is a defensive measure to prevent potential race conditions where
            # metadata generation starts before the transcript file is fully written to disk.
            # Note: Testing (30 runs) suggests the race condition may have been fixed during
            # refactoring (filesystem.write_file uses context manager ensuring file is written
            # before returning), but we keep this check as a safety measure for edge cases
            # or different filesystem behaviors.
            if job.transcript_path and job.transcript_source == "whisper_transcription":
                if not _wait_for_transcript_file(job.transcript_path, job.episode.idx):
                    logger.warning(
                        "[%s] Skipping metadata generation: transcript file not found: %s",
                        job.episode.idx,
                        job.transcript_path,
                    )
                    return

            _call_generate_metadata(
                episode=job.episode,
                feed=feed,
                cfg=cfg,
                effective_output_dir=effective_output_dir,
                run_suffix=run_suffix,
                transcript_path=job.transcript_path,
                transcript_source=job.transcript_source,
                whisper_model=None,  # No longer needed (use provider instead)
                feed_metadata=feed_metadata,
                host_detection_result=host_detection_result,
                detected_names=job.detected_names,
                summary_provider=summary_provider,
                pipeline_metrics=pipeline_metrics,
            )
        except Exception as exc:  # pragma: no cover
            _update_metric_safely(pipeline_metrics, "errors_total", 1)
            logger.error(f"[{job.episode.idx}] processing raised an unexpected error: {exc}")

    # Process jobs as they become available
    if max_workers <= 1:
        # Sequential processing
        while True:
            # Check for new jobs in the queue
            job_found = False
            current_job = None

            if processing_resources.processing_jobs_lock:
                with processing_resources.processing_jobs_lock:
                    with processed_job_indices_lock:
                        for job in processing_resources.processing_jobs:
                            if job.episode.idx not in processed_job_indices:
                                job_found = True
                                processed_job_indices.add(job.episode.idx)
                                current_job = job
                                break
            else:
                with processed_job_indices_lock:
                    for job in processing_resources.processing_jobs:
                        if job.episode.idx not in processed_job_indices:
                            job_found = True
                            processed_job_indices.add(job.episode.idx)
                            current_job = job
                            break

            if job_found and current_job:
                _process_single_processing_job(current_job)
                jobs_processed += 1
                logger.debug(
                    "Processed processing job idx=%s (processed=%s)",
                    current_job.episode.idx,
                    jobs_processed,
                )
                continue

            # No job found - check if we should continue waiting
            if transcription_complete_event and transcription_complete_event.is_set():
                # Transcription complete - check if queue is empty
                with processed_job_indices_lock:
                    if processing_resources.processing_jobs_lock:
                        with processing_resources.processing_jobs_lock:
                            total_jobs = len(processing_resources.processing_jobs)
                    else:
                        total_jobs = len(processing_resources.processing_jobs)
                    queue_empty = total_jobs == len(processed_job_indices)

                if queue_empty:
                    # All jobs processed, exit
                    break

            # Wait a bit before checking again
            if not (transcription_complete_event and transcription_complete_event.is_set()):
                time.sleep(0.1)
            else:
                time.sleep(0.05)
    else:
        # Parallel processing
        from concurrent.futures import as_completed, ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            while True:
                # Submit new jobs as they become available
                if processing_resources.processing_jobs_lock:
                    with processing_resources.processing_jobs_lock:
                        with processed_job_indices_lock:
                            for job in processing_resources.processing_jobs:
                                if (
                                    job.episode.idx not in processed_job_indices
                                    and job.episode.idx not in futures
                                ):
                                    processed_job_indices.add(job.episode.idx)
                                    future = executor.submit(_process_single_processing_job, job)
                                    futures[future] = job.episode.idx
                else:
                    with processed_job_indices_lock:
                        for job in processing_resources.processing_jobs:
                            if (
                                job.episode.idx not in processed_job_indices
                                and job.episode.idx not in futures
                            ):
                                processed_job_indices.add(job.episode.idx)
                                future = executor.submit(_process_single_processing_job, job)
                                futures[future] = job.episode.idx

                # Process completed futures
                # Use timeout with exception handling to avoid TimeoutError when
                # processing jobs take time
                try:
                    for future in as_completed(list(futures.keys()), timeout=1.0):
                        episode_idx = futures.pop(future)
                        try:
                            future.result()
                            jobs_processed += 1
                            logger.debug(
                                "Processed processing job idx=%s (processed=%s)",
                                episode_idx,
                                jobs_processed,
                            )
                        except Exception as exc:  # pragma: no cover
                            logger.error(f"[{episode_idx}] processing future raised error: {exc}")
                except TimeoutError:
                    # Some futures are still pending - continue loop to check again
                    # This is expected when jobs are still being processed
                    pass

                # Check if we should continue
                if transcription_complete_event and transcription_complete_event.is_set():
                    # Transcription complete - check if all jobs are submitted and completed
                    with processed_job_indices_lock:
                        if processing_resources.processing_jobs_lock:
                            with processing_resources.processing_jobs_lock:
                                total_jobs = len(processing_resources.processing_jobs)
                        else:
                            total_jobs = len(processing_resources.processing_jobs)
                        all_submitted = total_jobs == len(processed_job_indices)
                    if all_submitted and len(futures) == 0:
                        # All jobs submitted and completed
                        break

                # Wait a bit before checking again
                if not (transcription_complete_event and transcription_complete_event.is_set()):
                    time.sleep(0.1)
                else:
                    time.sleep(0.05)

    total_jobs = (
        len(processing_resources.processing_jobs) if processing_resources.processing_jobs else 0
    )
    logger.info(
        f"Concurrent processing completed: {jobs_processed}/{total_jobs} "
        f"jobs processed (parallelism={max_workers})"
    )


def _cleanup_pipeline(temp_dir: Optional[str]) -> None:
    """Cleanup temporary files and directories.

    Args:
        temp_dir: Path to temporary directory (if any)
    """
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temp directory: {temp_dir}")
        except OSError as exc:
            logger.debug(f"Failed to remove temp directory {temp_dir}: {exc}")


def _generate_pipeline_summary(
    cfg: config.Config,
    saved: int,
    transcription_resources: _TranscriptionResources,
    effective_output_dir: str,
    pipeline_metrics: metrics.Metrics,
) -> Tuple[int, str]:
    """Generate pipeline summary message with detailed statistics.

    Creates a human-readable summary of pipeline execution, including counts
    of transcripts processed, performance metrics (download/transcription times),
    and output location. In dry-run mode, reports planned operations instead
    of actual results.

    Args:
        cfg: Configuration object (checks dry_run, transcribe_missing flags)
        saved: Number of transcripts successfully saved or planned
        transcription_resources: Transcription resources containing the job queue
        effective_output_dir: Full path to output directory for display
        pipeline_metrics: Metrics collector with timing data for operations

    Returns:
        Tuple[int, str]: A tuple containing:
            - count (int): Total episodes processed (saved + transcribed)
            - summary (str): Multi-line summary message with statistics and metrics

    Example output (normal mode):
        >>> count, summary = _generate_pipeline_summary(...)
        >>> print(summary)
        Processed 10 episodes
          - Direct downloads: 8
          - Whisper transcriptions: 2
          - Average download time: 2.3s/episode
          - Average transcription time: 45.2s/episode
          - Output directory: ./transcripts
    """
    if cfg.dry_run:
        planned_downloads = saved
        planned_transcriptions = (
            len(transcription_resources.transcription_jobs) if cfg.transcribe_missing else 0
        )
        planned_total = planned_downloads + planned_transcriptions
        logger.debug(
            "Dry-run summary: planned_downloads=%s planned_transcriptions=%s",
            planned_downloads,
            planned_transcriptions,
        )
        # Print each metric on its own line for better readability
        summary_lines = [f"Dry run complete. transcripts_planned={planned_total}"]
        summary_lines.append(f"  - Direct downloads planned: {planned_downloads}")
        summary_lines.append(f"  - Whisper transcriptions planned: {planned_transcriptions}")
        summary_lines.append(f"  - Output directory: {effective_output_dir}")
        summary = "\n".join(summary_lines)
        # Don't log here - caller (cli.py) will log the summary
        return planned_total, summary
    else:
        # Build detailed summary with statistics
        # Print each metric on its own line for better readability
        summary_lines = [f"Done. transcripts_saved={saved}"]

        # Add breakdown by source
        if pipeline_metrics.transcripts_downloaded > 0:
            summary_lines.append(
                f"  - Transcripts downloaded: {pipeline_metrics.transcripts_downloaded}"
            )
        if pipeline_metrics.transcripts_transcribed > 0:
            summary_lines.append(
                f"  - Episodes transcribed: {pipeline_metrics.transcripts_transcribed}"
            )

        # Add metadata and summary statistics
        if cfg.generate_metadata and pipeline_metrics.metadata_files_generated > 0:
            summary_lines.append(
                f"  - Metadata files generated: {pipeline_metrics.metadata_files_generated}"
            )
        if cfg.generate_summaries and pipeline_metrics.episodes_summarized > 0:
            summary_lines.append(f"  - Episodes summarized: {pipeline_metrics.episodes_summarized}")

        # Add error count if any
        if pipeline_metrics.errors_total > 0:
            summary_lines.append(f"  - Errors: {pipeline_metrics.errors_total}")

        # Add skipped count if any
        if pipeline_metrics.episodes_skipped_total > 0:
            summary_lines.append(f"  - Episodes skipped: {pipeline_metrics.episodes_skipped_total}")

        # Add performance metrics (averages per episode) - one per line
        metrics_dict = pipeline_metrics.finish()
        if metrics_dict.get("avg_download_media_seconds", 0) > 0:
            avg_download = metrics_dict["avg_download_media_seconds"]
            summary_lines.append(f"  - Average download time: {avg_download:.1f}s/episode")
        if metrics_dict.get("avg_transcribe_seconds", 0) > 0:
            avg_transcribe = metrics_dict["avg_transcribe_seconds"]
            summary_lines.append(f"  - Average transcription time: {avg_transcribe:.1f}s/episode")
        if metrics_dict.get("avg_extract_names_seconds", 0) > 0:
            avg_extract = metrics_dict["avg_extract_names_seconds"]
            summary_lines.append(f"  - Average name extraction time: {avg_extract:.1f}s/episode")
        if metrics_dict.get("avg_summarize_seconds", 0) > 0:
            avg_summarize = metrics_dict["avg_summarize_seconds"]
            summary_lines.append(f"  - Average summary time: {avg_summarize:.1f}s/episode")

        summary_lines.append(f"  - Output directory: {effective_output_dir}")

        # Join all lines
        summary = "\n".join(summary_lines)
        # Don't log here - caller (cli.py) will log the summary
        return saved, summary


def _generate_episode_metadata(
    feed: models.RssFeed,
    episode: models.Episode,
    feed_url: str,
    cfg: config.Config,
    output_dir: str,
    run_suffix: Optional[str],
    transcript_file_path: Optional[str],
    transcript_source: Optional[Literal["direct_download", "whisper_transcription"]],
    whisper_model: Optional[str],
    detected_hosts: Optional[List[str]],
    detected_guests: Optional[List[str]],
    feed_description: Optional[str],
    feed_image_url: Optional[str],
    feed_last_updated: Optional[datetime],
    summary_provider=None,  # SummarizationProvider instance (preferred)
    summary_model=None,  # Backward compatibility - deprecated
    reduce_model=None,  # Backward compatibility - deprecated
    pipeline_metrics=None,
) -> None:
    """Generate and save episode metadata document.

    Creates a comprehensive metadata document for the episode containing feed info,
    episode details, transcript information, detected speakers, and optionally an
    AI-generated summary. The metadata is saved in JSON or YAML format based on
    configuration.

    Args:
        feed: RssFeed object with feed title and authors
        episode: Episode object with title, index, and RSS item data
        feed_url: RSS feed URL for reference
        cfg: Configuration object (metadata_format, metadata_subdirectory, generate_summaries)
        output_dir: Full path to output directory
        run_suffix: Optional run ID suffix for file naming
        transcript_file_path: Relative path to transcript file (from output_dir)
        transcript_source: How transcript was obtained
            ("direct_download" or "whisper_transcription")
        whisper_model: Name of Whisper model used for transcription (if applicable)
        detected_hosts: List of detected podcast host names from NER
        detected_guests: List of detected guest names from episode title/description
        feed_description: Podcast feed description text
        feed_image_url: URL to podcast artwork/cover image
        feed_last_updated: Last update timestamp from feed metadata
        summary_model: Optional loaded summary model (MAP model) for generating episode summary
        reduce_model: Optional loaded REDUCE model for final combine (reused across episodes)
        pipeline_metrics: Optional metrics collector for tracking summary generation time

    Raises:
        OSError: If metadata file cannot be written
        ValueError: If metadata generation fails

    Note:
        Metadata file is saved as either .json or .yaml based on cfg.metadata_format.
        If generate_summaries is enabled and summary_model is provided, an AI-generated
        summary will be included in the metadata document.
    """
    if not cfg.generate_metadata:
        return

    # Extract episode metadata
    (
        episode_description,
        episode_guid,
        episode_link,
        episode_duration_seconds,
        episode_number,
        episode_image_url,
    ) = extract_episode_metadata(episode.item, feed.base_url)
    episode_published_date = extract_episode_published_date(episode.item)

    # Generate metadata document
    metadata.generate_episode_metadata(
        feed=feed,
        episode=episode,
        feed_url=feed_url,
        cfg=cfg,
        output_dir=output_dir,
        run_suffix=run_suffix,
        transcript_file_path=transcript_file_path,
        transcript_source=transcript_source,
        whisper_model=whisper_model,
        summary_provider=summary_provider,
        detected_hosts=detected_hosts,
        detected_guests=detected_guests,
        feed_description=feed_description,
        feed_image_url=feed_image_url,
        feed_last_updated=feed_last_updated,
        episode_description=episode_description,
        episode_published_date=episode_published_date,
        episode_guid=episode_guid,
        episode_link=episode_link,
        episode_duration_seconds=episode_duration_seconds,
        episode_number=episode_number,
        episode_image_url=episode_image_url,
        pipeline_metrics=pipeline_metrics,
    )


def _parallel_episode_summarization(
    episodes: List[models.Episode],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: _FeedMetadata,
    host_detection_result: _HostDetectionResult,
    summary_provider=None,  # SummarizationProvider instance
    download_args: Optional[List[Tuple]] = None,
    pipeline_metrics: Optional[metrics.Metrics] = None,
) -> None:
    """Process episode summarization in parallel for episodes with existing transcripts.

    This function identifies episodes that have transcripts but may not have summaries yet,
    and processes them in parallel for better performance.

    For local transformers providers, each worker thread gets its own model instance to ensure
    thread safety, as HuggingFace pipelines/models are not thread-safe and cannot be shared
    across threads. For API providers (e.g., OpenAI), a single provider instance is used.

    Args:
        episodes: List of Episode objects
        feed: Parsed RssFeed object
        cfg: Configuration object
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix
        feed_metadata: Feed metadata tuple
        host_detection_result: Host detection result
        summary_provider: SummarizationProvider instance (required for parallel processing)
    """
    import os

    # Collect episodes that need summarization
    # Build a map of episode idx to detected names for guest detection
    episode_to_detected_names = {}
    if download_args is None:
        download_args = []
    for args in download_args:
        episode_obj, _, _, _, _, _, _, detected_names = args
        episode_to_detected_names[episode_obj.idx] = detected_names

    episodes_to_summarize = []
    for episode in episodes:
        # Check if transcript file exists - try common transcript file patterns
        # First try Whisper output path (most common)
        transcript_path = filesystem.build_whisper_output_path(
            episode.idx, episode.title_safe, run_suffix, effective_output_dir
        )
        if not os.path.exists(transcript_path):
            # Try other possible extensions
            for ext in [".txt", ".vtt", ".srt"]:
                base_name = filesystem.build_whisper_output_name(
                    episode.idx, episode.title_safe, run_suffix
                ).replace(".txt", ext)
                candidate_path = os.path.join(effective_output_dir, base_name)
                if os.path.exists(candidate_path):
                    transcript_path = candidate_path
                    break
        if os.path.exists(transcript_path):
            # Check if metadata file exists and if it needs summarization
            metadata_path = metadata._determine_metadata_path(
                episode, effective_output_dir, run_suffix, cfg
            )
            needs_summary = True
            if os.path.exists(metadata_path):
                # Check if summary already exists in metadata
                try:
                    import json

                    import yaml

                    with open(metadata_path, "r", encoding="utf-8") as f:
                        if cfg.metadata_format == "yaml":
                            metadata_content = yaml.safe_load(f)
                        else:
                            metadata_content = json.load(f)
                        if metadata_content and metadata_content.get("summary"):
                            needs_summary = False
                # Graceful fallback: assume needs summarization
                except Exception:  # nosec B110
                    pass

            if needs_summary:
                detected_names = episode_to_detected_names.get(episode.idx)
                episodes_to_summarize.append(
                    (episode, transcript_path, metadata_path, detected_names)
                )

    if not episodes_to_summarize:
        logger.debug("No episodes need summarization (all already have summaries)")
        return

    if summary_provider is None:
        logger.warning("Summary provider not available, skipping parallel summarization")
        return

    logger.info(f"Processing summarization for {len(episodes_to_summarize)} episodes in parallel")

    # Check if provider requires separate instances for thread safety
    # For API providers (e.g., OpenAI), a single provider instance can be used
    # For local providers (e.g., Transformers), we need multiple model instances
    # Use provider attribute instead of isinstance check to maintain modularity
    requires_separate_instances = getattr(summary_provider, "_requires_separate_instances", False)

    if not requires_separate_instances:
        # API provider (e.g., OpenAI) - use single provider instance sequentially
        # API providers handle rate limiting internally, so parallelism isn't needed
        logger.debug("Using API provider for sequential summarization")
        for episode, transcript_path, metadata_path, detected_names in episodes_to_summarize:
            _summarize_single_episode(
                episode=episode,
                transcript_path=transcript_path,
                metadata_path=metadata_path,
                feed=feed,
                cfg=cfg,
                effective_output_dir=effective_output_dir,
                run_suffix=run_suffix,
                feed_metadata=feed_metadata,
                host_detection_result=host_detection_result,
                summary_provider=summary_provider,
                detected_names=detected_names,
                pipeline_metrics=pipeline_metrics,
            )
        return

    # Local provider - extract model configuration for parallel processing
    # Each worker will load its own model instance for thread safety
    summary_model = getattr(summary_provider, "map_model", None)
    reduce_model = getattr(summary_provider, "reduce_model", None)

    if summary_model is None:
        logger.error("Local provider has no map_model, cannot proceed with parallel summarization")
        return

    model_name = summary_model.model_name
    model_device = summary_model.device
    model_cache_dir = summary_model.cache_dir
    model_revision = getattr(summary_model, "revision", None)

    # Extract reduce model configuration (if different from MAP model)
    # If reduce_model is None or same as summary_model, workers will reuse MAP model
    reduce_model_name = None
    reduce_model_device = None
    reduce_model_cache_dir = None
    reduce_model_revision = None
    reduce_model_is_same_as_map = False
    if reduce_model is not None:
        if reduce_model is summary_model:
            # Reduce model is same as MAP model - workers will reuse their MAP model
            reduce_model_is_same_as_map = True
        else:
            # Extract reduce model configuration for per-worker loading
            reduce_model_name = reduce_model.model_name
            reduce_model_device = reduce_model.device
            reduce_model_cache_dir = reduce_model.cache_dir
            reduce_model_revision = getattr(reduce_model, "revision", None)

    # Determine number of workers based on device and configuration
    # Configuration values can be set explicitly, or will use environment-based defaults
    # Test environments use lower defaults to reduce memory usage
    max_workers = 1
    if model_device == "cpu":
        # Use configured max workers for CPU, or environment-based defaults
        if cfg.summary_max_workers_cpu is not None:
            max_workers_limit = cfg.summary_max_workers_cpu
        else:
            # Check environment to set appropriate default
            is_test_env = os.environ.get("PYTEST_CURRENT_TEST") is not None
            max_workers_limit = 1 if is_test_env else 4  # Test env: 1 (sequential), Production: 4
        max_workers = min(
            cfg.summary_batch_size or 1, max_workers_limit, len(episodes_to_summarize)
        )
    elif model_device in ("mps", "cuda"):
        # Use configured max workers for GPU, or environment-based defaults
        if cfg.summary_max_workers_gpu is not None:
            max_workers_limit = cfg.summary_max_workers_gpu
        else:
            # Check environment to set appropriate default
            is_test_env = os.environ.get("PYTEST_CURRENT_TEST") is not None
            max_workers_limit = 1 if is_test_env else 2
        max_workers = min(max_workers_limit, len(episodes_to_summarize))

    # Track if we need separate reduce models (for cleanup)
    has_separate_reduce_models = not reduce_model_is_same_as_map and reduce_model_name is not None

    if max_workers <= 1:
        # Sequential processing - use provider directly
        for episode, transcript_path, metadata_path, detected_names in episodes_to_summarize:
            _summarize_single_episode(
                episode=episode,
                transcript_path=transcript_path,
                metadata_path=metadata_path,
                feed=feed,
                cfg=cfg,
                effective_output_dir=effective_output_dir,
                run_suffix=run_suffix,
                feed_metadata=feed_metadata,
                host_detection_result=host_detection_result,
                summary_provider=summary_provider,
                detected_names=detected_names,
                pipeline_metrics=pipeline_metrics,
            )
    else:
        # Parallel processing - each worker gets its own model instance
        logger.debug(
            f"Using {max_workers} workers for parallel episode summarization "
            f"(pre-loading {max_workers} model instances for thread safety)"
        )

        # Pre-load model instances for all workers before starting parallel execution
        # This ensures all models are ready upfront and avoids lazy-loading overhead
        # Use provider's internal summarizer module for model creation
        from . import summarizer  # noqa: PLC0415

        worker_models: List[Any] = []
        worker_reduce_models: List[Any] = []  # type: ignore[assignment]
        model_kwargs = {
            "model_name": model_name,
            "device": model_device,
            "cache_dir": model_cache_dir,
        }
        if model_revision:
            model_kwargs["revision"] = model_revision

        # Prepare reduce model kwargs if needed
        reduce_model_kwargs = None
        if has_separate_reduce_models:
            reduce_model_kwargs = {
                "model_name": reduce_model_name,
                "device": reduce_model_device or model_device,
                "cache_dir": reduce_model_cache_dir or model_cache_dir,
            }
            if reduce_model_revision:
                reduce_model_kwargs["revision"] = reduce_model_revision

        logger.debug(f"Pre-loading {max_workers} MAP model instances...")
        if has_separate_reduce_models:
            logger.debug(f"Pre-loading {max_workers} REDUCE model instances...")
        elif reduce_model_is_same_as_map:
            logger.debug("REDUCE model same as MAP - workers will reuse MAP model")

        for i in range(max_workers):
            try:
                logger.debug(f"Loading MAP model instance {i+1}/{max_workers} for worker thread")
                worker_model = summarizer.SummaryModel(**model_kwargs)
                worker_models.append(worker_model)

                # Load reduce model if different from MAP
                if has_separate_reduce_models and reduce_model_kwargs:
                    logger.debug(
                        f"Loading REDUCE model instance {i+1}/{max_workers} for worker thread"
                    )
                    worker_reduce_model = summarizer.SummaryModel(**reduce_model_kwargs)
                    worker_reduce_models.append(worker_reduce_model)
                elif reduce_model_is_same_as_map:
                    # Reuse MAP model for REDUCE phase
                    worker_reduce_models.append(worker_model)
                else:
                    # No reduce model needed
                    worker_reduce_models.append(None)  # type: ignore[arg-type]
            except Exception as e:
                logger.error(f"Failed to load model instance {i+1}/{max_workers}: {e}")
                # If we can't load all models, fall back to sequential processing
                # First, unload any models that were successfully loaded to prevent memory leak
                logger.warning("Falling back to sequential processing due to model loading failure")
                if worker_models:
                    logger.debug(
                        f"Unloading {len(worker_models)} successfully loaded "
                        f"MAP model(s) before fallback"
                    )
                    for worker_model in worker_models:
                        try:
                            summarizer.unload_model(worker_model)
                        except Exception as unload_error:
                            logger.debug(
                                f"Error unloading worker MAP model during fallback: {unload_error}"
                            )
                if worker_reduce_models:
                    reduce_count = len(
                        [m for m in worker_reduce_models if m and m not in worker_models]
                    )
                    logger.debug(
                        f"Unloading {reduce_count} successfully loaded "
                        f"REDUCE model(s) before fallback"
                    )
                    for worker_reduce_model in worker_reduce_models:
                        if worker_reduce_model and worker_reduce_model not in worker_models:
                            try:
                                summarizer.unload_model(worker_reduce_model)
                            except Exception as unload_error:
                                logger.debug(
                                    f"Error unloading worker REDUCE model "
                                    f"during fallback: {unload_error}"
                                )
                # Now proceed with sequential processing using the original models
                # Pass reduce_model to maintain consistent behavior with parallel path
                for (
                    episode,
                    transcript_path,
                    metadata_path,
                    detected_names,
                ) in episodes_to_summarize:
                    _summarize_single_episode(
                        episode=episode,
                        transcript_path=transcript_path,
                        metadata_path=metadata_path,
                        feed=feed,
                        cfg=cfg,
                        effective_output_dir=effective_output_dir,
                        run_suffix=run_suffix,
                        feed_metadata=feed_metadata,
                        host_detection_result=host_detection_result,
                        summary_provider=summary_provider,
                        detected_names=detected_names,
                        pipeline_metrics=pipeline_metrics,
                    )
                return

        logger.debug(f"Successfully pre-loaded {len(worker_models)} MAP model instances")
        if has_separate_reduce_models:
            logger.debug(
                f"Successfully pre-loaded {len(worker_reduce_models)} REDUCE model instances"
            )

        # Use thread-local storage to assign pre-loaded models to worker threads
        # Each worker thread gets one model from the pre-loaded pool
        thread_local = threading.local()
        model_index = [0]  # Use list to allow modification in nested function
        model_index_lock = threading.Lock()

        def _get_worker_models():
            """Get pre-loaded model instances for current worker thread."""
            if not hasattr(thread_local, "map_model"):
                with model_index_lock:
                    if model_index[0] < len(worker_models):
                        idx = model_index[0]
                        thread_local.map_model = worker_models[idx]
                        thread_local.reduce_model = worker_reduce_models[idx]
                        model_index[0] += 1
                    else:
                        # Fallback: reuse last models (shouldn't happen with proper worker count)
                        thread_local.map_model = worker_models[-1]
                        thread_local.reduce_model = worker_reduce_models[-1]
            return thread_local.map_model, thread_local.reduce_model

        def _summarize_with_worker_model(args):
            """Wrapper to get worker-specific models and summarize episode."""
            episode, transcript_path, metadata_path, detected_names = args
            worker_map_model, worker_reduce_model = _get_worker_models()
            # Create a temporary provider-like wrapper for this worker's models
            # This allows _summarize_single_episode to use provider pattern
            # For parallel processing, we pass models directly (backward compatibility)
            _summarize_single_episode(
                episode=episode,
                transcript_path=transcript_path,
                metadata_path=metadata_path,
                feed=feed,
                cfg=cfg,
                effective_output_dir=effective_output_dir,
                run_suffix=run_suffix,
                feed_metadata=feed_metadata,
                host_detection_result=host_detection_result,
                summary_provider=None,  # Use models directly for parallel processing
                summary_model=worker_map_model,  # Worker-specific model
                reduce_model=worker_reduce_model,  # Worker-specific model
                detected_names=detected_names,
                pipeline_metrics=pipeline_metrics,
            )

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for episode_data in episodes_to_summarize:
                    future = executor.submit(_summarize_with_worker_model, episode_data)
                    futures.append(future)

                # Wait for all to complete and log progress
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    try:
                        future.result()
                        if completed % 5 == 0 or completed == len(futures):
                            logger.info(
                                f"Completed summarization for {completed}/{len(futures)} episodes"
                            )
                    except Exception as e:
                        logger.error(f"Error during parallel summarization: {e}")
        finally:
            # Cleanup: unload all worker models
            logger.debug("Unloading worker MAP model instances...")
            for worker_model in worker_models:
                try:
                    summarizer.unload_model(worker_model)
                except Exception as e:
                    logger.debug(f"Error unloading worker MAP model: {e}")
            # Unload REDUCE models (only if different from MAP)
            if has_separate_reduce_models:
                logger.debug("Unloading worker REDUCE model instances...")
                for worker_reduce_model in worker_reduce_models:
                    if worker_reduce_model and worker_reduce_model not in worker_models:
                        try:
                            summarizer.unload_model(worker_reduce_model)
                        except Exception as e:
                            logger.debug(f"Error unloading worker REDUCE model: {e}")
            # Clear worker model lists to help GC
            worker_models.clear()
            worker_reduce_models.clear()
            # Force garbage collection after unloading models to free memory more aggressively
            # This is especially important in test environments where multiple pipelines
            # run in parallel
            import gc

            gc.collect()


def _summarize_single_episode(
    episode: models.Episode,
    transcript_path: str,
    metadata_path: Optional[str],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: _FeedMetadata,
    host_detection_result: _HostDetectionResult,
    summary_provider=None,  # SummarizationProvider instance (preferred)
    summary_model=None,  # Backward compatibility for parallel processing
    reduce_model=None,  # Backward compatibility for parallel processing
    detected_names: Optional[List[str]] = None,
    pipeline_metrics: Optional[metrics.Metrics] = None,
) -> None:
    """Summarize a single episode (helper for parallel processing).

    Args:
        episode: Episode object
        transcript_path: Path to transcript file
        metadata_path: Path to metadata file (if exists)
        feed: Parsed RssFeed object
        cfg: Configuration object
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix
        feed_metadata: Feed metadata tuple
        host_detection_result: Host detection result
        summary_provider: SummarizationProvider instance (preferred)
        summary_model: Pre-loaded MAP summary model (backward compatibility for parallel processing)
        reduce_model: Optional REDUCE model (backward compatibility for parallel processing)
        detected_names: Detected guest names for this episode (optional)
    """
    import os

    # Use provided detected names or None
    detected_names_for_ep = detected_names

    # Extract episode metadata
    (
        episode_description,
        episode_guid,
        episode_link,
        episode_duration_seconds,
        episode_number,
        episode_image_url,
    ) = extract_episode_metadata(episode.item, feed.base_url)
    episode_published_date = extract_episode_published_date(episode.item)

    # Determine transcript source
    transcript_source: Literal["direct_download", "whisper_transcription"] = (
        "direct_download"  # Default, could be enhanced to detect Whisper
    )

    # Generate/update metadata with summary
    metadata.generate_episode_metadata(
        feed=feed,
        episode=episode,
        feed_url=cfg.rss_url or "",
        cfg=cfg,
        output_dir=effective_output_dir,
        run_suffix=run_suffix,
        transcript_file_path=os.path.relpath(transcript_path, effective_output_dir),
        transcript_source=transcript_source,
        whisper_model=None,
        summary_provider=summary_provider,  # Use provider when available
        summary_model=summary_model,  # Backward compatibility for parallel processing
        reduce_model=reduce_model,  # Backward compatibility for parallel processing
        detected_hosts=(
            list(host_detection_result.cached_hosts) if host_detection_result.cached_hosts else None
        ),
        detected_guests=detected_names_for_ep,
        feed_description=feed_metadata.description,
        feed_image_url=feed_metadata.image_url,
        feed_last_updated=feed_metadata.last_updated,
        episode_description=episode_description,
        episode_published_date=episode_published_date,
        episode_guid=episode_guid,
        episode_link=episode_link,
        episode_duration_seconds=episode_duration_seconds,
        episode_number=episode_number,
        episode_image_url=episode_image_url,
    )
