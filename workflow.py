"""Core workflow orchestration: main pipeline execution."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Set, Tuple, cast

from . import (
    config,
    filesystem,
    metadata,
    metrics,
    models,
    progress,
    speaker_detection,
    whisper_integration as whisper,
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
    fetch_and_parse_rss,
)

logger = logging.getLogger(__name__)


class _FeedMetadata(NamedTuple):
    """Feed metadata for metadata generation."""

    description: Optional[str]
    image_url: Optional[str]
    last_updated: Optional[datetime]


class _HostDetectionResult(NamedTuple):
    """Result of host detection and pattern analysis."""

    cached_hosts: Set[str]
    heuristics: Optional[Dict[str, Any]]


class _TranscriptionResources(NamedTuple):
    """Resources needed for transcription."""

    whisper_model: Any
    temp_dir: Optional[str]
    transcription_jobs: List[models.TranscriptionJob]
    transcription_jobs_lock: Optional[threading.Lock]
    saved_counter_lock: Optional[threading.Lock]


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
    # Initialize metrics collector
    pipeline_metrics = metrics.Metrics()

    # Step 1: Setup pipeline environment
    effective_output_dir, run_suffix = _setup_pipeline_environment(cfg)

    # Step 2: Fetch and parse RSS feed (scraping stage)
    scraping_start = time.time()
    feed = _fetch_and_parse_feed(cfg)
    pipeline_metrics.record_stage("scraping", time.time() - scraping_start)

    # Step 3: Extract feed metadata (if metadata generation enabled)
    feed_metadata = _extract_feed_metadata_for_generation(cfg, feed)

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

    # Step 6.5: Setup summary models if summarization is enabled
    # Load models once and reuse across all episodes to avoid memory leaks and redundant downloads
    summary_model = None
    reduce_model = None
    if cfg.generate_summaries and not cfg.dry_run:
        try:
            # Lazy import to avoid loading torch in dry-run mode
            from . import summarizer  # noqa: PLC0415

            # Load MAP model (for chunk summarization)
            model_name = summarizer.select_summary_model(cfg)
            summary_model = summarizer.SummaryModel(
                model_name=model_name,
                device=cfg.summary_device,
                cache_dir=cfg.summary_cache_dir,
            )
            logger.debug("Loaded MAP summary model for reuse across all episodes")

            # Load REDUCE model if different from MAP model (for final combine)
            reduce_model_name = summarizer.select_reduce_model(cfg, model_name)
            if reduce_model_name != model_name:
                reduce_model = summarizer.SummaryModel(
                    model_name=reduce_model_name,
                    device=cfg.summary_device,
                    cache_dir=cfg.summary_cache_dir,
                )
                logger.debug(
                    f"Loaded REDUCE summary model ({reduce_model_name}) "
                    f"for reuse across all episodes"
                )
            else:
                # Use MAP model for REDUCE phase if they're the same
                reduce_model = summary_model
                logger.debug("Using MAP model for REDUCE phase (same model)")
        except ImportError:
            logger.warning("Summarization dependencies not available, skipping summary generation")
        except Exception as e:
            logger.error(f"Failed to load summary model: {e}")

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
    writing_start = time.time()
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
        pipeline_metrics,
        summary_model,
        reduce_model,
    )

    # Step 9: Process transcription jobs sequentially
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
        summary_model,
        reduce_model,
    )
    pipeline_metrics.record_stage("writing_storage", time.time() - writing_start)

    # Step 9: Parallel episode summarization (if enabled and multiple episodes)
    # Process episodes that need summarization in parallel for better performance
    # This runs after all episodes are processed and checks for episodes that might
    # have been skipped during inline processing or need summary regeneration
    if (
        cfg.generate_summaries
        and cfg.generate_metadata
        and summary_model is not None
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
            summary_model=summary_model,
            reduce_model=reduce_model,
            download_args=download_args,
            pipeline_metrics=pipeline_metrics,
        )

    # Step 9.5: Unload models to free memory
    # This runs even if exceptions occur above, preventing memory leaks
    if summary_model is not None:
        try:
            from . import summarizer  # noqa: PLC0415

            summarizer.unload_model(summary_model)
            logger.debug("Unloaded MAP summary model to free memory")
        except Exception as e:
            logger.warning(f"Failed to unload MAP summary model: {e}")
    if reduce_model is not None and reduce_model != summary_model:
        try:
            from . import summarizer  # noqa: PLC0415

            summarizer.unload_model(reduce_model)
            logger.debug("Unloaded REDUCE summary model to free memory")
        except Exception as e:
            logger.warning(f"Failed to unload REDUCE summary model: {e}")

    # Clear spaCy model cache to free memory
    # Models are cached at module level, so we clear them after processing
    if cfg.auto_speakers:
        try:
            from . import speaker_detection  # noqa: PLC0415

            speaker_detection.clear_spacy_model_cache()
            logger.debug("Cleared spaCy model cache to free memory")
        except Exception as e:
            logger.warning(f"Failed to clear spaCy model cache: {e}")

    # Step 10: Cleanup temporary files
    _cleanup_pipeline(temp_dir=transcription_resources.temp_dir)

    # Step 11: Generate summary and log metrics
    pipeline_metrics.log_metrics()
    return _generate_pipeline_summary(
        cfg, saved, transcription_resources, effective_output_dir, pipeline_metrics
    )


def _setup_pipeline_environment(cfg: config.Config) -> Tuple[str, Optional[str]]:
    """Setup output directory and handle cleanup if needed.

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (effective_output_dir, run_suffix)

    Raises:
        RuntimeError: If output directory cleanup fails
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

    return effective_output_dir, run_suffix


def _fetch_and_parse_feed(cfg: config.Config) -> models.RssFeed:
    """Fetch and parse RSS feed.

    Args:
        cfg: Configuration object

    Returns:
        Parsed RssFeed object
    """
    feed = fetch_and_parse_rss(cfg)
    logger.debug("Fetched RSS feed title=%s (%s items)", feed.title, len(feed.items))
    return feed


def _extract_feed_metadata_for_generation(
    cfg: config.Config, feed: models.RssFeed
) -> _FeedMetadata:
    """Extract feed metadata for metadata generation.

    Args:
        cfg: Configuration object
        feed: Parsed RssFeed object

    Returns:
        _FeedMetadata tuple
    """
    if not cfg.generate_metadata or not cfg.rss_url:
        return _FeedMetadata(None, None, None)

    # Re-fetch RSS XML to extract feed metadata
    # TODO: Cache RSS XML to avoid re-fetching
    from . import downloader

    resp = downloader.fetch_url(cfg.rss_url, cfg.user_agent, cfg.timeout, stream=False)
    if not resp:
        return _FeedMetadata(None, None, None)

    try:
        rss_bytes = resp.content
        feed_description, feed_image_url, feed_last_updated = extract_feed_metadata(
            rss_bytes, feed.base_url
        )
        return _FeedMetadata(feed_description, feed_image_url, feed_last_updated)
    finally:
        resp.close()


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
        return _HostDetectionResult(cached_hosts, heuristics)

    # Detect hosts: prefer RSS author tags, fall back to NER
    nlp = speaker_detection.get_ner_model(cfg) if not feed.authors else None
    feed_hosts = speaker_detection.detect_hosts_from_feed(
        feed_title=feed.title,
        feed_description=None,  # TODO: Extract from feed XML if needed
        feed_authors=feed.authors if feed.authors else None,
        nlp=nlp,
    )

    # Validate hosts with first episode: hosts should appear in first episode too
    # Skip validation if hosts came from author tags (they're already reliable)
    if feed_hosts and episodes and not feed.authors:
        # Only validate if we used NER (not author tags)
        first_episode = episodes[0]
        first_episode_description = extract_episode_description(first_episode.item)
        if nlp:
            first_episode_persons: Set[str] = set()
            title_persons = speaker_detection.extract_person_entities(first_episode.title, nlp)
            first_episode_persons.update(name for name, _ in title_persons)
            if first_episode_description:
                desc_persons = speaker_detection.extract_person_entities(
                    first_episode_description, nlp
                )
                first_episode_persons.update(name for name, _ in desc_persons)
            # Only keep hosts that also appear in first episode (validation)
            validated_hosts = feed_hosts & first_episode_persons
            if validated_hosts != feed_hosts:
                logger.debug(
                    "Host validation: %d hosts from feed, %d validated with first episode",
                    len(feed_hosts),
                    len(validated_hosts),
                )
                if validated_hosts:
                    logger.info(
                        "Validated hosts (appear in feed and first episode): %s",
                        list(validated_hosts),
                    )
                if feed_hosts - validated_hosts:
                    logger.debug(
                        "Hosts from feed not found in first episode (discarded): %s",
                        list(feed_hosts - validated_hosts),
                    )
            cached_hosts = validated_hosts
        else:
            cached_hosts = feed_hosts
    else:
        # If hosts came from author tags, use them directly (no validation needed)
        cached_hosts = feed_hosts

    if cached_hosts:
        source = "RSS author tags" if feed.authors else "feed metadata (NER)"
        logger.info("=" * 60)
        logger.info("DETECTED HOSTS (from %s): %s", source, ", ".join(sorted(cached_hosts)))
        logger.info("=" * 60)
    elif cfg.auto_speakers:
        logger.info("No hosts detected from feed metadata")

    # Analyze patterns from first few episodes to extract heuristics
    if cfg.auto_speakers and episodes:
        nlp = speaker_detection.get_ner_model(cfg)
        if nlp:
            heuristics = speaker_detection.analyze_episode_patterns(
                episodes, nlp, cached_hosts, sample_size=5
            )
            if heuristics.get("title_position_preference"):
                logger.debug(
                    "Pattern analysis: guest names typically appear at %s of title",
                    heuristics["title_position_preference"],
                )

    return _HostDetectionResult(cached_hosts, heuristics)


def _setup_transcription_resources(
    cfg: config.Config, effective_output_dir: str
) -> _TranscriptionResources:
    """Setup Whisper model and temp directory for transcription.

    Args:
        cfg: Configuration object
        effective_output_dir: Output directory path

    Returns:
        _TranscriptionResources object
    """
    whisper_model = None
    if cfg.transcribe_missing and not cfg.dry_run:
        whisper_model = whisper.load_whisper_model(cfg)

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
        whisper_model, temp_dir, transcription_jobs, transcription_jobs_lock, saved_counter_lock
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

    Args:
        episodes: List of Episode objects
        cfg: Configuration object
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix
        transcription_resources: Transcription resources
        host_detection_result: Host detection result

    Returns:
        List of download argument tuples
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
            detected_speakers, detected_hosts_set, detection_succeeded = (
                speaker_detection.detect_speaker_names(
                    episode_title=episode.title,
                    episode_description=episode_description,
                    cfg=cfg,
                    cached_hosts=(
                        host_detection_result.cached_hosts if cfg.cache_detected_hosts else None
                    ),
                    heuristics=host_detection_result.heuristics,
                )
            )
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
                    logger.info(
                        "  → Guest detection failed, using manual guest fallback: %s (hosts: %s)",
                        manual_guest,
                        ", ".join(detected_hosts_set),
                    )
                else:
                    # No hosts detected either, use both manual names
                    fallback_names = [manual_host, manual_guest]
                    logger.info(
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
    pipeline_metrics: metrics.Metrics,
    summary_model=None,
    reduce_model=None,
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
                    pipeline_metrics.bytes_downloaded_total += bytes_downloaded
                if success:
                    saved += 1
                    # Track transcript source
                    if transcript_source == "direct_download":
                        pipeline_metrics.transcripts_downloaded += 1
                    logger.debug("Episode %s yielded transcript (saved=%s)", episode.idx, saved)
                elif transcript_path is None and transcript_source is None:
                    # Episode was skipped (skip_existing)
                    pipeline_metrics.episodes_skipped_total += 1
            except Exception as exc:  # pragma: no cover
                pipeline_metrics.errors_total += 1
                logger.error(
                    f"[{episode.idx}] episode processing raised an unexpected error: {exc}"
                )

            # Generate metadata if enabled and transcript_source is available
            # Skip if transcript_source is None (Whisper pending) - will be generated after
            # transcription
            if cfg.generate_metadata and transcript_source is not None:
                transcript_source_typed = cast(
                    Optional[Literal["direct_download", "whisper_transcription"]],
                    transcript_source,
                )
                _generate_episode_metadata(
                    feed=feed,
                    episode=episode,
                    feed_url=cfg.rss_url or "",
                    cfg=cfg,
                    output_dir=output_dir_arg,
                    run_suffix=run_suffix_arg,
                    transcript_file_path=transcript_path,
                    transcript_source=transcript_source_typed,
                    whisper_model=None,  # Will be updated after transcription
                    detected_hosts=(
                        list(host_detection_result.cached_hosts)
                        if host_detection_result.cached_hosts
                        else None
                    ),
                    detected_guests=detected_names if detected_names else None,
                    feed_description=feed_metadata.description,
                    feed_image_url=feed_metadata.image_url,
                    feed_last_updated=feed_metadata.last_updated,
                    summary_model=summary_model,
                    reduce_model=reduce_model,
                    pipeline_metrics=pipeline_metrics,
                )
    else:
        # Concurrent processing
        saved_counter_lock = transcription_resources.saved_counter_lock
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
                ): args[0].idx
                for args in download_args
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    success, transcript_path, transcript_source, bytes_downloaded = future.result()
                    if bytes_downloaded:
                        if saved_counter_lock:
                            with saved_counter_lock:
                                pipeline_metrics.bytes_downloaded_total += bytes_downloaded
                        else:
                            pipeline_metrics.bytes_downloaded_total += bytes_downloaded
                    if success:
                        if saved_counter_lock:
                            with saved_counter_lock:
                                saved += 1
                                if transcript_source == "direct_download":
                                    pipeline_metrics.transcripts_downloaded += 1
                        else:
                            saved += 1
                            if transcript_source == "direct_download":
                                pipeline_metrics.transcripts_downloaded += 1
                        logger.debug("Episode %s yielded transcript (saved=%s)", idx, saved)
                    elif transcript_path is None and transcript_source is None:
                        # Episode was skipped (skip_existing)
                        if saved_counter_lock:
                            with saved_counter_lock:
                                pipeline_metrics.episodes_skipped_total += 1
                        else:
                            pipeline_metrics.episodes_skipped_total += 1

                    # Generate metadata if enabled and transcript_source is available
                    # Skip if transcript_source is None (Whisper pending) - will be generated
                    # after transcription
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
                                Optional[Literal["direct_download", "whisper_transcription"]],
                                transcript_source,
                            )
                            _generate_episode_metadata(
                                feed=feed,
                                episode=episode_obj,
                                feed_url=cfg.rss_url or "",
                                cfg=cfg,
                                output_dir=effective_output_dir,
                                run_suffix=run_suffix,
                                transcript_file_path=transcript_path,
                                transcript_source=transcript_source_typed,
                                whisper_model=None,  # Will be updated after transcription
                                detected_hosts=(
                                    list(host_detection_result.cached_hosts)
                                    if host_detection_result.cached_hosts
                                    else None
                                ),
                                detected_guests=(
                                    detected_names_for_ep if detected_names_for_ep else None
                                ),
                                feed_description=feed_metadata.description,
                                feed_image_url=feed_metadata.image_url,
                                feed_last_updated=feed_metadata.last_updated,
                                summary_model=summary_model,
                                reduce_model=reduce_model,
                                pipeline_metrics=pipeline_metrics,
                            )
                except Exception as exc:  # pragma: no cover
                    pipeline_metrics.errors_total += 1
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
    summary_model=None,
    reduce_model=None,
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
                success, transcript_path, bytes_downloaded = transcribe_media_to_text(
                    job,
                    cfg,
                    transcription_resources.whisper_model,
                    run_suffix,
                    effective_output_dir,
                    pipeline_metrics=pipeline_metrics,
                )
                if bytes_downloaded:
                    pipeline_metrics.bytes_downloaded_total += bytes_downloaded
                if success:
                    saved += 1
                    pipeline_metrics.transcripts_transcribed += 1

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
                            _generate_episode_metadata(
                                feed=feed,
                                episode=episode_obj,
                                feed_url=cfg.rss_url or "",
                                cfg=cfg,
                                output_dir=effective_output_dir,
                                run_suffix=run_suffix,
                                transcript_file_path=transcript_path,
                                transcript_source="whisper_transcription",
                                whisper_model=cfg.whisper_model,
                                detected_hosts=(
                                    list(host_detection_result.cached_hosts)
                                    if host_detection_result.cached_hosts
                                    else None
                                ),
                                detected_guests=(
                                    detected_names_for_ep if detected_names_for_ep else None
                                ),
                                feed_description=feed_metadata.description,
                                feed_image_url=feed_metadata.image_url,
                                feed_last_updated=feed_metadata.last_updated,
                                summary_model=summary_model,
                            )
            except Exception as exc:  # pragma: no cover
                pipeline_metrics.errors_total += 1
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


def _cleanup_pipeline(temp_dir: Optional[str]) -> None:
    """Cleanup temporary files and directories.

    Args:
        temp_dir: Path to temporary directory (if any)
    """
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")
        except OSError as exc:
            logger.warning(f"Failed to remove temp directory {temp_dir}: {exc}")


def _generate_pipeline_summary(
    cfg: config.Config,
    saved: int,
    transcription_resources: _TranscriptionResources,
    effective_output_dir: str,
    pipeline_metrics: metrics.Metrics,
) -> Tuple[int, str]:
    """Generate pipeline summary message with detailed statistics.

    Args:
        cfg: Configuration object
        saved: Number of transcripts saved
        transcription_resources: Transcription resources
        effective_output_dir: Output directory path
        pipeline_metrics: Pipeline metrics collector

    Returns:
        Tuple of (count, summary_message)
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
    summary_model=None,
    reduce_model=None,
    pipeline_metrics=None,
) -> None:
    """Helper function to generate episode metadata.

    Args:
        feed: RssFeed object
        episode: Episode object
        feed_url: RSS feed URL
        cfg: Configuration object
        output_dir: Output directory path
        run_suffix: Optional run suffix
        transcript_file_path: Path to transcript file (relative to output_dir)
        transcript_source: Source of transcript ("direct_download" or "whisper_transcription")
        whisper_model: Whisper model used (if applicable)
        detected_hosts: List of detected host names
        detected_guests: List of detected guest names
        feed_description: Feed description
        feed_image_url: Feed image URL
        feed_last_updated: Feed last updated date
        summary_model: Pre-loaded summary model for reuse (optional)
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
        summary_model=summary_model,
        reduce_model=reduce_model,
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
    summary_model,
    download_args: List[Tuple],
    pipeline_metrics: metrics.Metrics,
    reduce_model=None,
) -> None:
    """Process episode summarization in parallel for episodes with existing transcripts.

    This function identifies episodes that have transcripts but may not have summaries yet,
    and processes them in parallel for better performance.

    Args:
        episodes: List of Episode objects
        feed: Parsed RssFeed object
        cfg: Configuration object
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix
        feed_metadata: Feed metadata tuple
        host_detection_result: Host detection result
        summary_model: Pre-loaded summary model
    """
    import os

    # Collect episodes that need summarization
    # Build a map of episode idx to detected names for guest detection
    episode_to_detected_names = {}
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

    logger.info(f"Processing summarization for {len(episodes_to_summarize)} episodes in parallel")

    # Determine number of workers based on device
    # GPU: Limited parallelism (2 workers max due to memory)
    # CPU: Can use more workers (up to 4)
    max_workers = 1
    if summary_model.device == "cpu":
        max_workers = min(cfg.summary_batch_size or 1, 4, len(episodes_to_summarize))
    elif summary_model.device in ("mps", "cuda"):
        # Very limited parallelism for GPU (2 max)
        max_workers = min(2, len(episodes_to_summarize))

    if max_workers <= 1:
        # Sequential processing
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
                summary_model=summary_model,
                reduce_model=reduce_model,
                detected_names=detected_names,
                pipeline_metrics=pipeline_metrics,
            )
    else:
        # Parallel processing
        logger.info(f"Using {max_workers} workers for parallel episode summarization")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for episode, transcript_path, metadata_path, detected_names in episodes_to_summarize:
                future = executor.submit(
                    _summarize_single_episode,
                    episode=episode,
                    transcript_path=transcript_path,
                    metadata_path=metadata_path,
                    feed=feed,
                    cfg=cfg,
                    effective_output_dir=effective_output_dir,
                    run_suffix=run_suffix,
                    feed_metadata=feed_metadata,
                    host_detection_result=host_detection_result,
                    summary_model=summary_model,
                    reduce_model=reduce_model,
                    detected_names=detected_names,
                    pipeline_metrics=pipeline_metrics,
                )
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
    summary_model,
    reduce_model=None,
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
        summary_model: Pre-loaded summary model
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
        summary_model=summary_model,
        reduce_model=reduce_model,
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
