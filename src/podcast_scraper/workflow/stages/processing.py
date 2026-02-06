"""Processing stage for episode download and preparation.

This module handles episode processing, download argument preparation,
and host detection.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Any, cast, Dict, List, Literal, Optional, Set, Tuple

from ... import config, models
from ...rss import BYTES_PER_MB, http_head, OPENAI_MAX_FILE_SIZE_BYTES
from .. import metrics
from ..episode_processor import process_episode_download as factory_process_episode_download


# Use wrapper function if available (for testability)
def process_episode_download(*args, **kwargs):
    import sys

    workflow_pkg = sys.modules.get("podcast_scraper.workflow")
    if workflow_pkg and hasattr(workflow_pkg, "process_episode_download"):
        func = getattr(workflow_pkg, "process_episode_download")
        from unittest.mock import Mock

        if isinstance(func, Mock):
            return func(*args, **kwargs)
    return factory_process_episode_download(*args, **kwargs)


from ...rss import extract_episode_description as rss_extract_episode_description


# Use wrapper function if available (for testability)
def extract_episode_description(item):
    import sys

    workflow_pkg = sys.modules.get("podcast_scraper.workflow")
    if workflow_pkg and hasattr(workflow_pkg, "extract_episode_description"):
        func = getattr(workflow_pkg, "extract_episode_description")
        from unittest.mock import Mock

        if isinstance(func, Mock):
            return func(item)
    return rss_extract_episode_description(item)


from ...speaker_detectors.factory import create_speaker_detector
from ..helpers import update_metric_safely
from ..types import (
    FeedMetadata,
    HostDetectionResult,
    ProcessingJob,
    ProcessingResources,
    TranscriptionResources,
)

# Import metadata stage for processing jobs
from . import metadata as metadata_stage

logger = logging.getLogger(__name__)


def detect_feed_hosts_and_patterns(
    cfg: config.Config,
    feed: models.RssFeed,
    episodes: List[models.Episode],
    pipeline_metrics: Optional[metrics.Metrics] = None,
    speaker_detector: Optional[Any] = None,
) -> HostDetectionResult:
    """Detect hosts from feed metadata and analyze episode patterns.

    Args:
        cfg: Configuration object
        feed: Parsed RssFeed object
        episodes: List of Episode objects
        pipeline_metrics: Optional metrics collector
        speaker_detector: Optional pre-initialized speaker detector instance.
            If None and auto_speakers=True, will create one (for backward compatibility).

    Returns:
        HostDetectionResult with cached_hosts and heuristics
    """
    cached_hosts: set[str] = set()
    heuristics: Optional[Dict[str, Any]] = None

    # If auto_speakers is disabled, skip speaker detection entirely
    if not cfg.auto_speakers:
        return HostDetectionResult(cached_hosts, heuristics, None)

    # In dry-run mode, still detect hosts from RSS author tags (no ML needed)
    # but skip NER-based detection and model initialization
    if cfg.dry_run:
        logger.info("(dry-run) would initialize speaker detector")
        # Still detect hosts from RSS author tags if available
        if feed.authors:
            cached_hosts = set(feed.authors)
            if cached_hosts:
                logger.info("=" * 60)
                logger.info(
                    "DETECTED HOSTS (from %s): %s",
                    "RSS author tags",
                    ", ".join(sorted(cached_hosts)),
                )
                logger.info("=" * 60)
        return HostDetectionResult(cached_hosts, heuristics, None)

    # Use provided speaker detector, or create one if not provided (backward compatibility)
    if speaker_detector is None:
        # Fallback: create speaker detector if not provided (for backward compatibility)
        # This should not happen in normal flow - providers should be created in orchestration
        logger.warning(
            "speaker_detector not provided to detect_feed_hosts_and_patterns, "
            "creating new instance (this should be created in orchestration)"
        )
        try:
            import sys

            workflow_pkg = sys.modules.get("podcast_scraper.workflow")
            if workflow_pkg and hasattr(workflow_pkg, "create_speaker_detector"):
                func = getattr(workflow_pkg, "create_speaker_detector")
                from unittest.mock import Mock

                if isinstance(func, Mock):
                    speaker_detector = func(cfg)
                else:
                    speaker_detector = create_speaker_detector(cfg)
            else:
                from ...speaker_detectors.factory import (
                    create_speaker_detector as factory_create_speaker_detector,
                )

                speaker_detector = factory_create_speaker_detector(cfg)
            # Initialize provider (loads spaCy model)
            speaker_detector.initialize()
        except Exception as exc:
            logger.error("Failed to initialize speaker detector: %s", exc)
            # Don't raise - allow pipeline to continue without speaker detection
            return HostDetectionResult(cached_hosts, heuristics, None)

    # Detect hosts: prefer RSS author tags, fall back to NER
    if speaker_detector is None:
        # No speaker detector available, return empty result
        return HostDetectionResult(cached_hosts, heuristics, None)

    feed_hosts = speaker_detector.detect_hosts(
        feed_title=feed.title,
        feed_description=None,  # TODO: Extract from feed XML if needed
        feed_authors=feed.authors if feed.authors else None,
    )

    # Priority: Use known_hosts from config if provided (show-level override)
    # This is useful when RSS metadata doesn't provide clean host names
    if cfg.known_hosts:
        known_hosts_set = set(cfg.known_hosts)
        logger.info(
            "Using known_hosts from config: %s",
            ", ".join(sorted(known_hosts_set)),
        )
        # Merge with feed_hosts (known_hosts takes precedence)
        cached_hosts = known_hosts_set | feed_hosts
        if cached_hosts:
            logger.info("=" * 60)
            logger.info(
                "DETECTED HOSTS (from config known_hosts + feed): %s",
                ", ".join(sorted(cached_hosts)),
            )
            logger.info("=" * 60)
            # Skip validation since known_hosts are trusted
            return HostDetectionResult(cached_hosts, heuristics, speaker_detector)

    # Validate hosts with first episode: hosts should appear in first episode too
    # Skip validation if hosts came from author tags (they're already reliable)
    if feed_hosts and episodes and not feed.authors:
        # Only validate if we used NER (not author tags)
        first_episode = episodes[0]
        first_episode_description = extract_episode_description(first_episode.item)
        # Validate hosts by checking if they appear in first episode
        # Use provider's detect_speakers to extract persons from first episode
        # Pass pipeline_metrics for LLM call tracking (if OpenAI provider)
        import inspect

        sig = inspect.signature(speaker_detector.detect_speakers)
        if "pipeline_metrics" in sig.parameters:
            first_episode_speakers, _, _ = (
                speaker_detector.detect_speakers(  # type: ignore[call-arg]
                    episode_title=first_episode.title,
                    episode_description=first_episode_description,
                    known_hosts=set(),
                    pipeline_metrics=pipeline_metrics,
                )
            )
        else:
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

    # Fallback to episode-level authors if no feed-level hosts found (Issue #380)
    # Initialize episode_authors before the if block to avoid UnboundLocalError
    episode_authors: Set[str] = set()
    if not cached_hosts and cfg.auto_speakers and episodes:
        from ...rss import parser as rss_parser

        # Check first 3 episodes for episode-level authors
        for episode in episodes[:3]:
            episode_author_list = rss_parser.extract_episode_authors(episode.item)
            for author in episode_author_list:
                # Filter out organization names (same logic as feed-level)
                # Organization names are typically all caps, short, and have no spaces
                author_stripped = author.strip()
                is_likely_org = (
                    len(author_stripped) <= 10
                    and author_stripped.isupper()
                    and " " not in author_stripped
                )
                if not is_likely_org:
                    episode_authors.add(author)

    if episode_authors:
        cached_hosts = episode_authors
        logger.info("=" * 60)
        logger.info(
            "DETECTED HOSTS (from episode-level authors): %s",
            ", ".join(sorted(cached_hosts)),
        )
        logger.info("=" * 60)

    # Fallback to known_hosts from config if no hosts detected (show-level override)
    # This is useful when RSS metadata doesn't provide clean host names
    # (e.g., "NPR" instead of actual hosts)
    if not cached_hosts and cfg.known_hosts:
        cached_hosts = set(cfg.known_hosts)
        logger.info("=" * 60)
        logger.info(
            "DETECTED HOSTS (from config known_hosts fallback): %s",
            ", ".join(sorted(cached_hosts)),
        )
        logger.info("=" * 60)

    if cached_hosts:
        # Determine source for logging
        if feed.authors:
            source = "RSS author tags"
        elif episode_authors and cached_hosts == episode_authors:
            source = "episode-level authors"
        elif cfg.known_hosts and cached_hosts == set(cfg.known_hosts):
            source = "config known_hosts (fallback)"
        else:
            source = "feed metadata (NER)"
        logger.info("=" * 60)
        logger.info("DETECTED HOSTS (from %s): %s", source, ", ".join(sorted(cached_hosts)))
        logger.info("=" * 60)
    elif cfg.auto_speakers:
        logger.debug(
            "No hosts detected from feed metadata, episode-level authors, or config known_hosts"
        )

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
    return HostDetectionResult(cached_hosts, heuristics, speaker_detector)


def setup_processing_resources(cfg: config.Config) -> ProcessingResources:
    """Set up resources for processing stage (metadata/summarization).

    Args:
        cfg: Configuration object

    Returns:
        ProcessingResources with processing queue and locks
    """
    processing_jobs: List[ProcessingJob] = []
    processing_jobs_lock = (
        threading.Lock()
        if (cfg.workers > 1 or cfg.transcription_parallelism > 1 or cfg.processing_parallelism > 1)
        else None
    )
    processing_complete_event = threading.Event()

    return ProcessingResources(
        processing_jobs,
        processing_jobs_lock,
        processing_complete_event,
    )


def prepare_episode_download_args(
    episodes: List[models.Episode],
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    transcription_resources: TranscriptionResources,
    host_detection_result: HostDetectionResult,
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
             transcription_jobs, transcription_jobs_lock, detected_speaker_names)
    """
    download_args = []
    for episode in episodes:
        detected_speaker_names = None
        # Initialize skip flags at the start of each episode to avoid UnboundLocalError
        skip_speaker_detection_due_to_size = False
        skip_episode_due_to_size = False

        # Detect guests for all episodes when auto_speakers is enabled
        # (not just when transcribing, so we can log guests even for transcript downloads)
        # Note: Guest detection works in dry-run mode (no media download/transcription needed)
        if cfg.auto_speakers:
            # Always log episode info
            logger.info("Episode %d: %s", episode.idx, episode.title)

            # Check file size before speaker detection if using API transcription providers
            # (OpenAI, Gemini) to avoid wasting API calls on episodes that will be skipped
            if (
                not cfg.dry_run
                and cfg.transcribe_missing
                and cfg.transcription_provider in ("openai", "gemini")
                and episode.media_url
            ):
                # Check file size using HTTP HEAD request
                resp = http_head(episode.media_url, cfg.user_agent, cfg.timeout)
                if resp:
                    content_length = resp.headers.get("Content-Length")
                    if content_length:
                        try:
                            file_size_bytes = int(content_length)
                            file_size_mb = file_size_bytes / BYTES_PER_MB
                            if file_size_bytes > OPENAI_MAX_FILE_SIZE_BYTES:
                                provider_name = (
                                    "OpenAI" if cfg.transcription_provider == "openai" else "Gemini"
                                )
                                # Only skip episode entirely if it has no transcript URLs
                                # (if it has transcript URLs, we can still download the transcript)
                                if not episode.transcript_urls:
                                    logger.info(
                                        "[%d] Skipping episode: Audio file size (%.1f MB) "
                                        "exceeds %s API limit (25 MB) and no transcript URLs "
                                        "available.",
                                        episode.idx,
                                        file_size_mb,
                                        provider_name,
                                    )
                                    skip_episode_due_to_size = True
                                else:
                                    logger.info(
                                        "[%d] Skipping speaker detection: Audio file size "
                                        "(%.1f MB) exceeds %s API limit (25 MB), but transcript "
                                        "URLs available.",
                                        episode.idx,
                                        file_size_mb,
                                        provider_name,
                                    )
                                    skip_speaker_detection_due_to_size = True
                            else:
                                provider_name = (
                                    "OpenAI" if cfg.transcription_provider == "openai" else "Gemini"
                                )
                                logger.debug(
                                    "[%d] File size check: %.1f MB (within %s limit)",
                                    episode.idx,
                                    file_size_mb,
                                    provider_name,
                                )
                        except (ValueError, TypeError):
                            # Content-Length header is invalid, proceed with speaker detection
                            logger.debug(
                                "[%d] Could not parse Content-Length header, "
                                "proceeding with speaker detection",
                                episode.idx,
                            )
                    else:
                        # No Content-Length header, proceed with speaker detection
                        logger.debug(
                            "[%d] No Content-Length header available, "
                            "proceeding with speaker detection",
                            episode.idx,
                        )
                else:
                    # HEAD request failed, proceed with speaker detection
                    logger.debug(
                        "[%d] HEAD request failed, proceeding with speaker detection",
                        episode.idx,
                    )

            # In dry-run mode, log what would be detected but skip actual detection
            if cfg.dry_run:
                episode_description = extract_episode_description(episode.item) or ""
                if len(episode_description) > 50:
                    desc_preview = episode_description[:50] + "..."
                else:
                    desc_preview = episode_description
                logger.info(
                    "(dry-run) would detect speakers from: %s | %s",
                    episode.title,
                    desc_preview,
                )
                detected_speakers: List[str] = []
                detected_hosts_set: set[str] = set()
                detection_succeeded = False
            elif skip_speaker_detection_due_to_size:
                # Skip speaker detection because file will be too large for OpenAI
                detected_speakers, detected_hosts_set, detection_succeeded = [], set(), False
            else:
                # Extract episode description for NER (limited to first 20 chars)
                episode_description = extract_episode_description(episode.item)
                # Detect speaker names from episode title and first 20 chars of description
                # Guests are detected from episode title and description snippet
                extract_names_start = time.time()

                # Stage 3: Use provider for speaker detection
                # Get speaker detector from host_detection_result (should be set in orchestration)
                speaker_detector = host_detection_result.speaker_detector
                if not speaker_detector:
                    # Fallback: create speaker detector if not in result
                    # (for backward compatibility)
                    # This should not happen in normal flow - providers should be
                    # created in orchestration
                    logger.warning(
                        "speaker_detector not found in host_detection_result, "
                        "creating new instance "
                        "(this should be created in orchestration)"
                    )
                    import sys

                    workflow_pkg = sys.modules.get("podcast_scraper.workflow")
                    if workflow_pkg and hasattr(workflow_pkg, "create_speaker_detector"):
                        func = getattr(workflow_pkg, "create_speaker_detector")
                        from unittest.mock import Mock

                        if isinstance(func, Mock):
                            speaker_detector = func(cfg)
                        else:
                            speaker_detector = create_speaker_detector(cfg)
                    else:
                        from ...speaker_detectors.factory import (
                            create_speaker_detector as factory_create_speaker_detector,
                        )

                        speaker_detector = factory_create_speaker_detector(cfg)
                    # Initialize the detector if it was just created
                    if speaker_detector:
                        speaker_detector.initialize()
                if speaker_detector:
                    # Combine cached_hosts with known_hosts from config
                    # (known_hosts takes precedence)
                    cached_hosts_for_detection = (
                        host_detection_result.cached_hosts if cfg.cache_detected_hosts else set()
                    )
                    # known_hosts from config takes precedence over cached_hosts
                    if cfg.known_hosts:
                        known_hosts_set = set(cfg.known_hosts)
                        # Merge: known_hosts + cached_hosts (known_hosts are trusted)
                        combined_hosts = known_hosts_set | cached_hosts_for_detection
                    else:
                        combined_hosts = cached_hosts_for_detection

                    # Pass pipeline_metrics for LLM call tracking
                    # (if OpenAI provider)
                    import inspect

                    sig = inspect.signature(speaker_detector.detect_speakers)
                    if "pipeline_metrics" in sig.parameters:
                        detected_speakers, detected_hosts_set, detection_succeeded = (
                            speaker_detector.detect_speakers(
                                episode_title=episode.title,
                                episode_description=episode_description,
                                known_hosts=combined_hosts,
                                pipeline_metrics=pipeline_metrics,
                            )
                        )
                    else:
                        detected_speakers, detected_hosts_set, detection_succeeded = (
                            speaker_detector.detect_speakers(
                                episode_title=episode.title,
                                episode_description=episode_description,
                                known_hosts=combined_hosts,
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
            # Manual names: first item = host, rest = guests
            if (
                not detection_succeeded
                and cfg.screenplay_speaker_names
                and len(cfg.screenplay_speaker_names) >= 2
            ):
                # Extract manual guests (all names except first, which is the host)
                manual_guests = cfg.screenplay_speaker_names[1:]

                # Log fallback info
                if detected_hosts_set:
                    logger.debug(
                        "  → Guest detection failed, using manual guest fallback: %s (hosts: %s)",
                        ", ".join(manual_guests),
                        ", ".join(detected_hosts_set),
                    )
                else:
                    logger.debug(
                        "  → Detection failed, using manual fallback guests: %s",
                        ", ".join(manual_guests),
                    )
                # Return only guests (hosts are already filtered out by taking [1:])
                detected_speaker_names = manual_guests
            elif detection_succeeded:
                # Filter out hosts from detected speakers (keep only guests)
                # CRITICAL: List comprehension already creates a new list, but be explicit
                # to prevent any shared mutable state issues
                detected_speaker_names = [
                    s for s in detected_speakers if s not in detected_hosts_set
                ]
            # Note: Guest logging happens inside detect_speaker_names()
            # Note: We don't update cached_hosts here because hosts are only
            # detected from feed metadata, not from episodes
        elif cfg.screenplay_speaker_names:
            # If auto_speakers is disabled, first name is host, rest are guests
            # Only pass guest names to the episode processing
            detected_speaker_names = (
                cfg.screenplay_speaker_names[1:] if len(cfg.screenplay_speaker_names) > 1 else []
            )

        # Skip episode entirely if file size exceeds limit and no transcript URLs available
        if skip_episode_due_to_size:
            if pipeline_metrics is not None:
                from ..helpers import update_metric_safely

                update_metric_safely(pipeline_metrics, "episodes_skipped_total", 1)
            continue

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


def process_episodes(  # noqa: C901
    download_args: List[Tuple],
    episodes: List[models.Episode],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: FeedMetadata,
    host_detection_result: HostDetectionResult,
    transcription_resources: TranscriptionResources,
    processing_resources: ProcessingResources,
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
        transcription_resources: Transcription resources
        processing_resources: Processing resources
        pipeline_metrics: Metrics collector
        summary_provider: SummarizationProvider instance

    Returns:
        Number of transcripts saved
    """
    from ..helpers import update_metric_safely

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
                    update_metric_safely(
                        pipeline_metrics, "bytes_downloaded_total", bytes_downloaded
                    )
                if success:
                    saved += 1
                    # Track transcript source
                    if transcript_source == "direct_download":
                        update_metric_safely(pipeline_metrics, "transcripts_downloaded", 1)
                    logger.debug("Episode %s yielded transcript (saved=%s)", episode.idx, saved)

                    # Update episode status: downloaded (Issue #391)
                    if pipeline_metrics is not None:
                        from ..helpers import get_episode_id_from_episode

                        episode_id, episode_number = get_episode_id_from_episode(
                            episode, cfg.rss_url or ""
                        )
                        pipeline_metrics.update_episode_status(
                            episode_id=episode_id, stage="downloaded"
                        )

                    # Queue processing job if metadata generation enabled and transcript available
                    # Skip if transcript_source is None (Whisper pending) - queued after
                    if cfg.generate_metadata and transcript_source is not None:
                        transcript_source_typed = cast(
                            Literal["direct_download", "whisper_transcription"],
                            transcript_source,
                        )
                        processing_job = ProcessingJob(
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
                    # Episode was skipped only if transcribe_missing is False
                    # If transcribe_missing is True, None/None means queued for transcription
                    if not cfg.transcribe_missing:
                        logger.debug(
                            "[%s] Episode skipped (no transcript, transcribe_missing=False)",
                            episode.idx,
                        )
                        update_metric_safely(pipeline_metrics, "episodes_skipped_total", 1)
                    else:
                        logger.debug(
                            "[%s] Episode queued for transcription "
                            "(not skipped, transcribe_missing=True)",
                            episode.idx,
                        )
            except Exception as exc:  # pragma: no cover
                update_metric_safely(pipeline_metrics, "errors_total", 1)
                logger.error(
                    f"[{episode.idx}] episode processing raised an unexpected error: {exc}",
                    exc_info=True,
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
                        update_metric_safely(
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
                            update_metric_safely(
                                pipeline_metrics, "transcripts_downloaded", 1, saved_counter_lock
                            )
                        logger.debug("Episode %s yielded transcript (saved=%s)", idx, saved)
                    elif transcript_path is None and transcript_source is None:
                        # Episode was skipped only if transcribe_missing is False
                        # If transcribe_missing is True, None/None means queued for transcription
                        if not cfg.transcribe_missing:
                            logger.debug(
                                "[%s] Episode skipped (no transcript, transcribe_missing=False)",
                                idx,
                            )
                            update_metric_safely(
                                pipeline_metrics, "episodes_skipped_total", 1, saved_counter_lock
                            )
                        else:
                            logger.debug(
                                "[%s] Episode queued for transcription "
                                "(not skipped, transcribe_missing=True)",
                                idx,
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
                            processing_job = ProcessingJob(
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
                    update_metric_safely(pipeline_metrics, "errors_total", 1, saved_counter_lock)
                    logger.error(f"[{idx}] episode processing raised an unexpected error: {exc}")

    return saved


# TODO: Reduce complexity - extract more helper functions for parallel processing logic
def process_processing_jobs_concurrent(  # noqa: C901
    processing_resources: ProcessingResources,
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: FeedMetadata,
    host_detection_result: HostDetectionResult,
    pipeline_metrics: metrics.Metrics,
    summary_provider=None,  # SummarizationProvider instance (required)
    transcription_complete_event: Optional[threading.Event] = None,
    should_serialize_mps: bool = False,
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
        summary_provider: SummarizationProvider instance (required)
        transcription_complete_event: Event to signal when transcription is complete
        should_serialize_mps: If True, wait for transcription before starting summarization
            (prevents MPS memory contention when both Whisper and summarization use MPS)
    """
    max_workers = cfg.processing_parallelism
    logger.info(
        "Processing workers: configured=%d, effective=%d",
        cfg.processing_parallelism,
        max_workers,
    )

    # If MPS exclusive mode is enabled, wait for transcription to complete before
    # starting any summarization work (prevents GPU memory contention)
    if should_serialize_mps and cfg.generate_summaries:
        if transcription_complete_event:
            logger.info(
                "MPS exclusive mode: Waiting for transcription to complete before "
                "starting summarization"
            )
            transcription_complete_event.wait()
            logger.info("Transcription complete, starting summarization")

    # Track successful vs failed jobs separately
    jobs_processed_ok = 0
    jobs_processed_failed = 0
    processed_job_indices = set()  # Track which jobs we've processed
    processed_job_indices_lock = threading.Lock()  # Lock for thread-safe access

    def _find_next_unprocessed_job() -> Optional[ProcessingJob]:
        """Find the next unprocessed job from the queue.

        Returns:
            ProcessingJob if found, None otherwise
        """
        if processing_resources.processing_jobs_lock:
            with processing_resources.processing_jobs_lock:
                with processed_job_indices_lock:
                    for job in processing_resources.processing_jobs:
                        if job.episode.idx not in processed_job_indices:
                            processed_job_indices.add(job.episode.idx)
                            return job
        else:
            with processed_job_indices_lock:
                for job in processing_resources.processing_jobs:
                    if job.episode.idx not in processed_job_indices:
                        processed_job_indices.add(job.episode.idx)
                        return job
        return None

    def _check_queue_empty() -> bool:
        """Check if processing queue is empty.

        Returns:
            True if queue is empty, False otherwise
        """
        with processed_job_indices_lock:
            if processing_resources.processing_jobs_lock:
                with processing_resources.processing_jobs_lock:
                    total_jobs = len(processing_resources.processing_jobs)
            else:
                total_jobs = len(processing_resources.processing_jobs)
            return total_jobs == len(processed_job_indices)

    def _run_parallel_processing_loop(
        processing_resources: ProcessingResources,
        processed_job_indices: set,
        processed_job_indices_lock: threading.Lock,
        process_job_func: Any,
        transcription_complete_event: Optional[threading.Event],
        max_workers: int,
    ) -> tuple[int, int]:
        """Run parallel processing loop with ThreadPoolExecutor.

        Returns:
            Tuple of (jobs_processed_ok, jobs_processed_failed)
        """
        jobs_processed_ok = [0]  # Use list for nonlocal access
        jobs_processed_failed = [0]  # Use list for nonlocal access

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            def _submit_new_jobs() -> None:
                """Submit new jobs as they become available."""
                if processing_resources.processing_jobs_lock:
                    with processing_resources.processing_jobs_lock:
                        with processed_job_indices_lock:
                            for job in processing_resources.processing_jobs:
                                if (
                                    job.episode.idx not in processed_job_indices
                                    and job.episode.idx not in futures
                                ):
                                    processed_job_indices.add(job.episode.idx)
                                    future = executor.submit(process_job_func, job)
                                    futures[future] = job.episode.idx
                else:
                    with processed_job_indices_lock:
                        for job in processing_resources.processing_jobs:
                            if (
                                job.episode.idx not in processed_job_indices
                                and job.episode.idx not in futures
                            ):
                                processed_job_indices.add(job.episode.idx)
                                future = executor.submit(process_job_func, job)
                                futures[future] = job.episode.idx

            def _process_completed_futures() -> None:
                """Process completed futures."""
                try:
                    for future in as_completed(list(futures.keys()), timeout=1.0):
                        episode_idx = futures.pop(future)
                        try:
                            success = future.result()
                            if success:
                                jobs_processed_ok[0] += 1
                            else:
                                jobs_processed_failed[0] += 1
                            logger.debug(
                                "Processed processing job idx=%s (ok=%s, failed=%s, total=%s)",
                                episode_idx,
                                jobs_processed_ok[0],
                                jobs_processed_failed[0],
                                jobs_processed_ok[0] + jobs_processed_failed[0],
                            )
                        except Exception as exc:  # pragma: no cover
                            jobs_processed_failed[0] += 1
                            logger.error(f"[{episode_idx}] processing future raised error: {exc}")
                except TimeoutError:
                    # Some futures are still pending - continue loop to check again
                    pass

            def _should_continue_processing() -> bool:
                """Check if processing should continue."""
                if transcription_complete_event and transcription_complete_event.is_set():
                    all_submitted = _check_queue_empty()
                    return not (all_submitted and len(futures) == 0)
                return True

            while True:
                _submit_new_jobs()
                _process_completed_futures()

                if not _should_continue_processing():
                    break

                # Wait a bit before checking again
                # Track queue wait time (Issue #387)
                queue_wait_start = time.time()
                if not (transcription_complete_event and transcription_complete_event.is_set()):
                    time.sleep(0.1)
                    queue_wait_duration = time.time() - queue_wait_start
                else:
                    time.sleep(0.05)
                    queue_wait_duration = time.time() - queue_wait_start
                if pipeline_metrics is not None:
                    pipeline_metrics.record_queue_wait_time(queue_wait_duration)

        return (jobs_processed_ok[0], jobs_processed_failed[0])

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

    def _process_single_processing_job(job: ProcessingJob) -> bool:
        """Process a single processing job (metadata/summarization).

        Returns:
            True if job succeeded, False if it failed
        """
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
                    return False

            # Extract spaCy model from summary_provider if available (Issue #387)
            nlp = None
            if summary_provider is not None:
                try:
                    # Check if provider has spaCy model (MLProvider pattern)
                    if (
                        hasattr(summary_provider, "_spacy_nlp")
                        and summary_provider._spacy_nlp is not None
                    ):
                        nlp = summary_provider._spacy_nlp
                except Exception:
                    pass  # Ignore errors when accessing provider attributes

            metadata_stage.call_generate_metadata(
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
                nlp=nlp,  # Pass spaCy model for reuse (Issue #387)
            )
            return True
        except Exception as exc:  # pragma: no cover
            update_metric_safely(pipeline_metrics, "errors_total", 1)
            logger.error(f"[{job.episode.idx}] processing raised an unexpected error: {exc}")
            return False

    # Process jobs as they become available
    if max_workers <= 1:
        # Sequential processing
        while True:
            current_job = _find_next_unprocessed_job()
            if current_job:
                success = _process_single_processing_job(current_job)
                if success:
                    jobs_processed_ok += 1
                else:
                    jobs_processed_failed += 1
                jobs_processed = jobs_processed_ok + jobs_processed_failed
                logger.debug(
                    "Processed processing job idx=%s (ok=%s, failed=%s, total=%s)",
                    current_job.episode.idx,
                    jobs_processed_ok,
                    jobs_processed_failed,
                    jobs_processed,
                )
                continue

            # No job found - check if we should continue waiting
            if transcription_complete_event and transcription_complete_event.is_set():
                if _check_queue_empty():
                    # All jobs processed, exit
                    break

            # Wait a bit before checking again
            if not (transcription_complete_event and transcription_complete_event.is_set()):
                time.sleep(0.1)
            else:
                time.sleep(0.05)
    else:
        # Parallel processing
        parallel_jobs_ok, parallel_jobs_failed = _run_parallel_processing_loop(
            processing_resources,
            processed_job_indices,
            processed_job_indices_lock,
            _process_single_processing_job,
            transcription_complete_event,
            max_workers,
        )
        jobs_processed_ok = parallel_jobs_ok
        jobs_processed_failed = parallel_jobs_failed
        jobs_processed = jobs_processed_ok + jobs_processed_failed

    total_jobs = (
        len(processing_resources.processing_jobs) if processing_resources.processing_jobs else 0
    )
    if jobs_processed_failed > 0:
        logger.info(
            f"Concurrent processing completed: {jobs_processed_ok} succeeded, "
            f"{jobs_processed_failed} failed ({jobs_processed}/{total_jobs} total, "
            f"parallelism={max_workers})"
        )
    else:
        logger.info(
            f"Concurrent processing completed: {jobs_processed_ok}/{total_jobs} "
            f"jobs processed (parallelism={max_workers})"
        )
