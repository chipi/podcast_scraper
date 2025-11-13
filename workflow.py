"""Core workflow orchestration: main pipeline execution."""

from __future__ import annotations

import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from . import (
    config,
    filesystem,
    models,
    progress,
    speaker_detection,
    whisper_integration as whisper,
)
from .episode_processor import (
    process_episode_download,
    transcribe_media_to_text,
)
from .rss_parser import create_episode_from_item, extract_episode_description, fetch_and_parse_rss

logger = logging.getLogger(__name__)


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


def run_pipeline(cfg: config.Config) -> Tuple[int, str]:  # noqa: C901 - orchestrates full pipeline
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

    feed = fetch_and_parse_rss(cfg)
    logger.debug("Fetched RSS feed title=%s (%s items)", feed.title, len(feed.items))

    items = feed.items
    total_items = len(items)
    if cfg.max_episodes is not None:
        items = items[: cfg.max_episodes]

    logger.info(f"Episodes to process: {len(items)} of {total_items}")

    # Create Episode objects from RSS items
    episodes = [
        create_episode_from_item(item, idx, feed.base_url)
        for idx, item in enumerate(items, start=1)
    ]
    logger.debug("Materialized %s episode objects", len(episodes))

    # Detect hosts from feed metadata if auto_speakers is enabled
    # Note: Host detection works in dry-run mode (no media download/transcription needed)
    cached_hosts: set[str] = set()
    heuristics: Optional[Dict[str, Any]] = None
    if cfg.auto_speakers and cfg.cache_detected_hosts:
        # Extract feed description if available
        feed_description = None  # TODO: Extract from feed XML if needed
        # Detect hosts: prefer RSS author tags, fall back to NER
        nlp = speaker_detection.get_ner_model(cfg) if not feed.authors else None
        feed_hosts = speaker_detection.detect_hosts_from_feed(
            feed_title=feed.title,
            feed_description=feed_description,
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

    saved = 0
    # Prepare download args with detected speaker names for each episode
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
                    cached_hosts=cached_hosts if cfg.cache_detected_hosts else None,
                    heuristics=heuristics,
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
                temp_dir,
                effective_output_dir,
                run_suffix,
                transcription_jobs,
                transcription_jobs_lock,
                detected_speaker_names,
            )
        )

    if not download_args:
        saved = 0
    elif cfg.workers <= 1 or len(download_args) == 1:
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
            # Update transcription job with detected names if created
            if process_episode_download(
                episode,
                cfg_arg,
                temp_dir_arg,
                output_dir_arg,
                run_suffix_arg,
                jobs_arg,
                lock_arg,
                detected_names,
            ):
                saved += 1
                logger.debug("Episode %s yielded transcript (saved=%s)", episode.idx, saved)
    else:
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
                    if future.result():
                        if saved_counter_lock:
                            with saved_counter_lock:
                                saved += 1
                        else:
                            saved += 1
                        logger.debug("Episode %s yielded transcript (saved=%s)", idx, saved)
                except Exception as exc:  # pragma: no cover
                    logger.error(f"[{idx}] episode processing raised an unexpected error: {exc}")

    if transcription_jobs and cfg.transcribe_missing and not cfg.dry_run:
        total_jobs = len(transcription_jobs)
        logger.info(f"Starting Whisper transcription for {total_jobs} episodes")
        with progress.progress_context(total_jobs, "Whisper transcription") as reporter:
            jobs_processed = 0
            for job in transcription_jobs:
                if transcribe_media_to_text(
                    job, cfg, whisper_model, run_suffix, effective_output_dir
                ):
                    saved += 1
                reporter.update(1)
                jobs_processed += 1
                logger.debug(
                    "Processed transcription job idx=%s (saved=%s, processed=%s/%s)",
                    job.idx,
                    saved,
                    jobs_processed,
                    total_jobs,
                )

    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")
        except OSError as exc:
            logger.warning(f"Failed to remove temp directory {temp_dir}: {exc}")

    if cfg.dry_run:
        planned_downloads = saved
        planned_transcriptions = len(transcription_jobs) if cfg.transcribe_missing else 0
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
