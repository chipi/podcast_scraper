"""Transcription stage for Whisper/OpenAI transcription processing.

This module handles transcription resource setup and job processing.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ... import config, models

if TYPE_CHECKING:
    from ...models import Episode, RssFeed, TranscriptionJob
else:
    Episode = models.Episode  # type: ignore[assignment]
    RssFeed = models.RssFeed  # type: ignore[assignment]
    TranscriptionJob = models.TranscriptionJob  # type: ignore[assignment]
from ...providers.capabilities import get_provider_capabilities, is_local_provider
from ...utils import filesystem, progress
from .. import metrics
from ..episode_processor import transcribe_media_to_text as factory_transcribe_media_to_text
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
    cfg: config.Config,
    effective_output_dir: str,
    transcription_provider: Optional[Any] = None,
) -> TranscriptionResources:
    """Setup transcription provider and temp directory for transcription.

    Args:
        cfg: Configuration object
        effective_output_dir: Output directory path
        transcription_provider: Optional pre-initialized transcription provider instance.
            If None and transcribe_missing=True, will create one (for backward compatibility).

    Returns:
        TranscriptionResources object
    """
    # Use provided transcription provider, or create one if not provided (backward compatibility)
    if transcription_provider is None and cfg.transcribe_missing and not cfg.dry_run:
        # Fallback: create transcription provider if not provided (for backward compatibility)
        # This should not happen in normal flow - providers should be created in orchestration
        logger.warning(
            "transcription_provider not provided to setup_transcription_resources, "
            "creating new instance (this should be created in orchestration)"
        )
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

    # Create bounded queue for transcription jobs (prevents unbounded memory growth)
    transcription_jobs: queue.Queue[TranscriptionJob] = queue.Queue(  # type: ignore[valid-type]
        maxsize=cfg.transcription_queue_size
    )
    # Lock may become redundant with Queue (Queue is thread-safe), but keeping for now
    # to maintain compatibility and allow gradual migration
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
    episodes: List[Episode],  # type: ignore[valid-type]
    feed: RssFeed,  # type: ignore[valid-type]
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
    if not cfg.transcribe_missing:
        return 0

    # For sequential processing, we need to collect all jobs from the queue first
    # (since queue.get() removes items, we need to track them)
    jobs_list: List[TranscriptionJob] = []  # type: ignore[valid-type]
    while True:
        try:
            job = transcription_resources.transcription_jobs.get_nowait()
            jobs_list.append(job)
        except queue.Empty:
            break

    if not jobs_list:
        return 0

    saved = 0
    total_jobs = len(jobs_list)
    if cfg.dry_run:
        logger.info(f"Dry-run: would transcribe {total_jobs} episodes with Whisper")
    else:
        logger.info(f"Starting Whisper transcription for {total_jobs} episodes")

    with progress.progress_context(total_jobs, "Whisper transcription") as reporter:
        jobs_processed = 0
        for job in jobs_list:
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
                    # Increment transcripts_transcribed for both cache hits and actual
                    # transcriptions. This metric counts transcripts saved, not transcription
                    # work performed. When cache is used, transcripts_transcribed > 0 but
                    # transcribe_count = 0.
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
                                nlp=nlp,  # Pass spaCy model for reuse (Issue #387)
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
    episodes: List[Episode],  # type: ignore[valid-type]
    feed: RssFeed,  # type: ignore[valid-type]
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
    - Whisper provider: Respects config (default: 1). Values > 1 are experimental.
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

    # Get parallelism from config
    # All providers now respect transcription_parallelism for experimentation
    # Note: Local (ML) provider parallelism > 1 is experimental and not production-ready
    max_workers = cfg.transcription_parallelism
    transcription_provider = transcription_resources.transcription_provider
    is_local = is_local_provider(transcription_provider) if transcription_provider else False
    if is_local and max_workers > 1:
        logger.warning(
            "Local transcription provider: Using parallel processing (parallelism=%d) - "
            "EXPERIMENTAL: Not production-ready, may cause memory/GPU contention",
            max_workers,
        )
    else:
        provider_caps = (
            get_provider_capabilities(transcription_provider) if transcription_provider else None
        )
        provider_name = provider_caps.provider_name if provider_caps else "unknown"
        logger.info(
            "Transcription provider '%s': configured=%d, effective=%d",
            provider_name,
            cfg.transcription_parallelism,
            max_workers,
        )

    saved = 0
    jobs_processed = 0

    logger.debug("Concurrent transcription processor started (max_workers=%d)", max_workers)

    def _process_single_job(
        job: TranscriptionJob,  # type: ignore[valid-type]
    ) -> tuple[bool, Optional[str], int]:  # type: ignore[valid-type]
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
                # Increment transcripts_transcribed for both cache hits and actual transcriptions
                # This metric counts transcripts saved, not transcription work performed.
                # When cache is used, transcripts_transcribed > 0 but transcribe_count = 0.
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

    # Process jobs as they become available from the queue
    # Continue until downloads are complete AND queue is empty
    if max_workers <= 1:
        # Sequential processing (Whisper default)
        while True:
            try:
                # Block with timeout to allow checking if downloads are complete
                timeout = (
                    0.1
                    if not (downloads_complete_event and downloads_complete_event.is_set())
                    else 0.05
                )
                current_job = transcription_resources.transcription_jobs.get(
                    block=True, timeout=timeout
                )
                # Track queue wait time (Issue #387)
                queue_wait_start = time.time()
                success, transcript_path, bytes_downloaded = _process_single_job(current_job)
                queue_wait_duration = time.time() - queue_wait_start
                if pipeline_metrics is not None:
                    pipeline_metrics.record_queue_wait_time(queue_wait_duration)
                if success:
                    saved += 1
                jobs_processed += 1
                logger.debug(
                    "Processed transcription job idx=%s (saved=%s, processed=%s)",
                    current_job.idx,
                    saved,
                    jobs_processed,
                )
            except queue.Empty:
                # Queue is empty - check if we should continue waiting
                if downloads_complete_event and downloads_complete_event.is_set():
                    # Downloads complete and queue is empty, exit
                    break
                # Wait a bit before checking again (avoid busy-waiting)
                time.sleep(0.1)
    else:
        # Parallel processing (OpenAI provider, or Whisper with parallelism > 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: Dict[Any, int] = {}

            def _submit_new_transcription_jobs() -> None:
                """Submit new transcription jobs as they become available from the queue."""
                # Submit jobs up to max_workers limit
                while len(futures) < max_workers:
                    try:
                        # Non-blocking get to avoid blocking when queue is empty
                        job = transcription_resources.transcription_jobs.get_nowait()
                        future = executor.submit(_process_single_job, job)
                        futures[future] = job.idx
                    except queue.Empty:
                        # No more jobs available right now
                        break

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
                    # Downloads complete - check if queue is empty and all futures are done
                    if transcription_resources.transcription_jobs.empty() and len(futures) == 0:
                        # All jobs processed, exit
                        break

                # Wait a bit before checking again
                if not (downloads_complete_event and downloads_complete_event.is_set()):
                    time.sleep(0.1)
                else:
                    time.sleep(0.05)

    # Update saved counter
    saved_counter[0] = saved
    # Note: Queue size is not directly accessible, but we track jobs_processed
    logger.info(
        f"Concurrent transcription processing completed: {saved}/{jobs_processed} "
        f"transcripts saved (parallelism={max_workers})"
    )
