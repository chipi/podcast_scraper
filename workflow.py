"""Core workflow orchestration: main pipeline execution."""

from __future__ import annotations

import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Set, Tuple, cast

from . import (
    config,
    filesystem,
    metadata,
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


def apply_log_level(level: str) -> None:
    """Apply logging level to root logger and all handlers.

    Args:
        level: Log level string (e.g., 'DEBUG', 'INFO', 'WARNING')

    Raises:
        ValueError: If log level is invalid
    """
    numeric_level = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        root_logger.setLevel(numeric_level)
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)
    logger.setLevel(numeric_level)


def run_pipeline(cfg: config.Config) -> Tuple[int, str]:
    """Execute the main podcast scraping pipeline.

    This orchestrates the entire workflow:
    1. Setup output directory
    2. Fetch and parse RSS feed
    3. Process episodes (download transcripts or media for transcription)
    4. Transcribe media files using Whisper if needed
    5. Clean up temporary files

    Args:
        cfg: Configuration object with all settings

    Returns:
        Tuple of (count, summary_message) where count is number of transcripts saved/planned

    Raises:
        RuntimeError: If output directory cleanup fails
        ValueError: If RSS fetch or parse fails
    """
    # Step 1: Setup pipeline environment
    effective_output_dir, run_suffix = _setup_pipeline_environment(cfg)

    # Step 2: Fetch and parse RSS feed
    feed = _fetch_and_parse_feed(cfg)

    # Step 3: Extract feed metadata (if metadata generation enabled)
    feed_metadata = _extract_feed_metadata_for_generation(cfg, feed)

    # Step 4: Prepare episodes from RSS items
    episodes = _prepare_episodes_from_feed(feed, cfg)

    # Step 5: Detect hosts and analyze patterns (if auto_speakers enabled)
    host_detection_result = _detect_feed_hosts_and_patterns(cfg, feed, episodes)

    # Step 6: Setup transcription resources (Whisper model, temp dir)
    transcription_resources = _setup_transcription_resources(cfg, effective_output_dir)

    # Step 7: Prepare episode processing arguments with speaker detection
    download_args = _prepare_episode_download_args(
        episodes,
        cfg,
        effective_output_dir,
        run_suffix,
        transcription_resources,
        host_detection_result,
    )

    # Step 8: Process episodes (download transcripts or queue transcription jobs)
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
    )

    # Step 10: Cleanup temporary files
    _cleanup_pipeline(temp_dir=transcription_resources.temp_dir)

    # Step 11: Generate summary
    return _generate_pipeline_summary(cfg, saved, transcription_resources, effective_output_dir)


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
            success, transcript_path, transcript_source = process_episode_download(
                episode,
                cfg_arg,
                temp_dir_arg,
                output_dir_arg,
                run_suffix_arg,
                jobs_arg,
                lock_arg,
                detected_names,
            )
            if success:
                saved += 1
                logger.debug("Episode %s yielded transcript (saved=%s)", episode.idx, saved)

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
                    success, transcript_path, transcript_source = future.result()
                    if success:
                        if saved_counter_lock:
                            with saved_counter_lock:
                                saved += 1
                        else:
                            saved += 1
                        logger.debug("Episode %s yielded transcript (saved=%s)", idx, saved)

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
                            )
                except Exception as exc:  # pragma: no cover
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
    if not transcription_resources.transcription_jobs or not cfg.transcribe_missing or cfg.dry_run:
        return 0

    saved = 0
    total_jobs = len(transcription_resources.transcription_jobs)
    logger.info(f"Starting Whisper transcription for {total_jobs} episodes")
    with progress.progress_context(total_jobs, "Whisper transcription") as reporter:
        jobs_processed = 0
        for job in transcription_resources.transcription_jobs:
            success, transcript_path = transcribe_media_to_text(
                job,
                cfg,
                transcription_resources.whisper_model,
                run_suffix,
                effective_output_dir,
            )
            if success:
                saved += 1

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
                        )

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
) -> Tuple[int, str]:
    """Generate pipeline summary message.

    Args:
        cfg: Configuration object
        saved: Number of transcripts saved
        transcription_resources: Transcription resources
        effective_output_dir: Output directory path

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
        summary = "Dry run complete. transcripts_planned=%s (direct=%s, whisper=%s) in %s" % (
            planned_total,
            planned_downloads,
            planned_transcriptions,
            effective_output_dir,
        )
        logger.info(summary)
        return planned_total, summary
    else:
        summary = f"Done. transcripts_saved={saved} in {effective_output_dir}"
        logger.info(summary)
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
    )
