"""Transcription stage for Whisper/OpenAI transcription processing.

This module handles transcription resource setup and job processing.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import List, Optional, Tuple

from ... import config, filesystem, metrics, models, progress
from ...episode_processor import transcribe_media_to_text as factory_transcribe_media_to_text
from ..helpers import update_metric_safely


# Use wrapper function if available (for testability)
def transcribe_media_to_text(*args, **kwargs):
    import sys

    workflow_pkg = sys.modules.get("podcast_scraper.workflow")
    if workflow_pkg and hasattr(workflow_pkg, "transcribe_media_to_text"):
        func = getattr(workflow_pkg, "transcribe_media_to_text")
        from unittest.mock import Mock

        if isinstance(func, Mock):
            return func(*args, **kwargs)
    return factory_transcribe_media_to_text(*args, **kwargs)


from ...transcription.factory import create_transcription_provider
from ..types import (
    FeedMetadata,
    HostDetectionResult,
    ProcessingJob,
    ProcessingResources,
    TranscriptionResources,
)

# Import metadata functions for generating metadata after transcription
from . import metadata as metadata_stage

logger = logging.getLogger(__name__)


def setup_transcription_resources(
    cfg: config.Config, effective_output_dir: str
) -> TranscriptionResources:
    """Setup transcription provider and temp directory for transcription.

    Args:
        cfg: Configuration object
        effective_output_dir: Output directory path

    Returns:
        TranscriptionResources object
    """
    transcription_provider = None

    if cfg.transcribe_missing and not cfg.dry_run:
        # Stage 2: Use provider pattern
        try:
            # Use wrapper function if available (for testability)
            import sys

            workflow_pkg = sys.modules.get("podcast_scraper.workflow")
            if workflow_pkg and hasattr(workflow_pkg, "create_transcription_provider"):
                func = getattr(workflow_pkg, "create_transcription_provider")
                from unittest.mock import Mock

                if isinstance(func, Mock):
                    transcription_provider = func(cfg)
                else:
                    transcription_provider = create_transcription_provider(cfg)
            else:
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

    return TranscriptionResources(
        transcription_provider,
        temp_dir,
        transcription_jobs,
        transcription_jobs_lock,
        saved_counter_lock,
    )


def process_transcription_jobs(
    transcription_resources: TranscriptionResources,
    download_args: List[Tuple],
    episodes: List[models.Episode],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: FeedMetadata,
    host_detection_result: HostDetectionResult,
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
        pipeline_metrics: Metrics collector
        summary_provider: SummarizationProvider instance

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
                    update_metric_safely(
                        pipeline_metrics, "bytes_downloaded_total", bytes_downloaded
                    )
                if success:
                    saved += 1
                    update_metric_safely(pipeline_metrics, "transcripts_transcribed", 1)

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
                            metadata_stage.call_generate_metadata(
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
                update_metric_safely(pipeline_metrics, "errors_total", 1)
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


# TODO: Reduce complexity - extract more helper functions for parallel processing logic
def process_transcription_jobs_concurrent(  # noqa: C901
    transcription_resources: TranscriptionResources,
    download_args: List[Tuple],
    episodes: List[models.Episode],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: FeedMetadata,
    host_detection_result: HostDetectionResult,
    processing_resources: ProcessingResources,
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
        processing_resources: Processing resources
        pipeline_metrics: Metrics collector
        summary_provider: SummarizationProvider instance
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

    def _find_next_unprocessed_transcription_job() -> Optional[models.TranscriptionJob]:
        """Find the next unprocessed transcription job from the queue.

        Returns:
            TranscriptionJob if found, None otherwise
        """
        if transcription_resources.transcription_jobs_lock:
            with transcription_resources.transcription_jobs_lock:
                with processed_job_indices_lock:
                    for job in transcription_resources.transcription_jobs:
                        if job.idx not in processed_job_indices:
                            processed_job_indices.add(job.idx)
                            return job
        else:
            with processed_job_indices_lock:
                for job in transcription_resources.transcription_jobs:
                    if job.idx not in processed_job_indices:
                        processed_job_indices.add(job.idx)
                        return job
        return None

    def _check_transcription_queue_empty() -> bool:
        """Check if transcription queue is empty.

        Returns:
            True if queue is empty, False otherwise
        """
        with processed_job_indices_lock:
            if transcription_resources.transcription_jobs_lock:
                with transcription_resources.transcription_jobs_lock:
                    total_jobs = len(transcription_resources.transcription_jobs)
            else:
                total_jobs = len(transcription_resources.transcription_jobs)
            return total_jobs == len(processed_job_indices)

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
                update_metric_safely(pipeline_metrics, "bytes_downloaded_total", bytes_downloaded)
            if success:
                update_metric_safely(pipeline_metrics, "transcripts_transcribed", 1)

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
                        processing_job = ProcessingJob(
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
            update_metric_safely(pipeline_metrics, "errors_total", 1)
            logger.error(f"[{job.idx}] transcription raised an unexpected error: {exc}")
            return False, None, 0

    # Process jobs as they become available
    # Continue until downloads are complete AND queue is empty
    if max_workers <= 1:
        # Sequential processing (Whisper default)
        while True:
            current_job = _find_next_unprocessed_transcription_job()
            if current_job:
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
                if _check_transcription_queue_empty():
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
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            def _submit_new_transcription_jobs() -> None:
                """Submit new transcription jobs as they become available."""
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

            def _process_completed_transcription_futures() -> None:
                """Process completed transcription futures."""
                for future in as_completed(list(futures.keys()), timeout=1.0):
                    job_idx = futures.pop(future)
                    try:
                        success, transcript_path, bytes_downloaded = future.result()
                        nonlocal saved, jobs_processed
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

            while True:
                _submit_new_transcription_jobs()
                try:
                    _process_completed_transcription_futures()
                except TimeoutError:
                    # Some futures are still pending - continue loop to check again
                    pass

                # Check if we should continue
                if downloads_complete_event and downloads_complete_event.is_set():
                    all_submitted = _check_transcription_queue_empty()
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
