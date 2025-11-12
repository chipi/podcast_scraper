"""Core workflow orchestration: main pipeline execution."""

from __future__ import annotations

import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from . import config, filesystem, models, progress, whisper
from .episode_processor import process_episode_download, transcribe_media_to_text
from .rss_parser import create_episode_from_item, fetch_and_parse_rss

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
    effective_output_dir, run_suffix = filesystem.setup_output_directory(cfg)
    logger.debug("Effective output dir=%s (run_suffix=%s)", effective_output_dir, run_suffix)

    if cfg.clean_output and cfg.dry_run:
        if os.path.exists(effective_output_dir):
            logger.info(
                f"Dry-run: would remove existing output directory (--clean-output): {effective_output_dir}"
            )
    elif cfg.clean_output:
        try:
            if os.path.exists(effective_output_dir):
                shutil.rmtree(effective_output_dir)
                logger.info(f"Removed existing output directory (--clean-output): {effective_output_dir}")
        except OSError as exc:
            raise RuntimeError(f"Failed to clean output directory {effective_output_dir}: {exc}") from exc

    if cfg.dry_run:
        logger.info(f"Dry-run: not creating output directory {effective_output_dir}")
    else:
        os.makedirs(effective_output_dir, exist_ok=True)

    feed = fetch_and_parse_rss(cfg)
    logger.debug(
        "Fetched RSS feed title=%s (%s items)", feed.title, len(feed.items)
    )

    items = feed.items
    total_items = len(items)
    if cfg.max_episodes is not None:
        items = items[: cfg.max_episodes]

    logger.info(f"Episodes to process: {len(items)} of {total_items}")

    # Create Episode objects from RSS items
    episodes = [create_episode_from_item(item, idx, feed.base_url) for idx, item in enumerate(items, start=1)]
    logger.debug("Materialized %s episode objects", len(episodes))

    whisper_model = None
    if cfg.transcribe_missing and not cfg.dry_run:
        whisper_model = whisper.load_whisper_model(cfg)
        logger.debug("Whisper model loaded: %s", bool(whisper_model))

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
    download_args = [
        (
            episode,
            cfg,
            temp_dir,
            effective_output_dir,
            run_suffix,
            transcription_jobs,
            transcription_jobs_lock,
        )
        for episode in episodes
    ]

    if not download_args:
        saved = 0
    elif cfg.workers <= 1 or len(download_args) == 1:
        for args in download_args:
            if process_episode_download(*args):
                saved += 1
                logger.debug("Episode %s yielded transcript (saved=%s)", args[0].idx, saved)
    else:
        with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            future_map = {
                executor.submit(process_episode_download, *args): args[0].idx for args in download_args
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
                if transcribe_media_to_text(job, cfg, whisper_model, run_suffix, effective_output_dir):
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
        summary = (
            f"Dry run complete. transcripts_planned={planned_total} "
            f"(direct={planned_downloads}, whisper={planned_transcriptions}) in {effective_output_dir}"
        )
        logger.info(summary)
        return planned_total, summary
    else:
        summary = f"Done. transcripts_saved={saved} in {effective_output_dir}"
        logger.info(summary)
        return saved, summary
