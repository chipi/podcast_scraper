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
from ...utils.log_redaction import format_exception_for_log, redact_for_log
from .. import metrics
from ..episode_processor import transcribe_media_to_text as factory_transcribe_media_to_text
from ..helpers import update_metric_safely


# Use wrapper function if available (for testability)
def transcribe_media_to_text(*args, **kwargs):
    """Delegate to workflow.transcribe_media_to_text or factory; allows tests to inject a mock."""
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


def _stamp_tx_active(pm: Optional[Any], start: float) -> None:
    """Emit the (start, now) transcription-thread interval to metrics (#1180).

    Extracted so `_process_single_job` can call it inline at each return site
    without pushing its cognitive complexity up. A no-op when metrics is None
    (test paths, dry-run).
    """
    if pm is not None:
        pm.record_transcription_thread_active(start, time.monotonic())


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
            logger.error(
                "Failed to initialize transcription provider: %s",
                format_exception_for_log(exc),
            )
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
            # Pre-authorise this episode's spend. See _authorise_transcription_spend: refusal
            # skips the job and latches the ledger rather than raising, and the main thread turns
            # that latch into the run's outcome after the transcription thread joins.
            if not _authorise_transcription_spend(job, cfg):
                jobs_processed += 1
                reporter.update(1)
                continue
            try:
                # Anchor the diarization roster with the feed-stated hosts (canonicalizes
                # ASR-garbled host surnames). cached_hosts already merges feed + config hosts.
                job.feed_hosts = sorted(host_detection_result.cached_hosts or [])
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
                logger.error(
                    "[%s] transcription raised an unexpected error: %s",
                    job.idx,
                    format_exception_for_log(exc),
                )
                # Record per-episode failure for run index (Issue #429)
                if pipeline_metrics is not None:
                    episode_obj = next((ep for ep in episodes if ep.idx == job.idx), None)
                    if episode_obj is not None:
                        from ..helpers import get_episode_id_from_episode

                        episode_id, _ = get_episode_id_from_episode(episode_obj, cfg.rss_url or "")
                        pipeline_metrics.update_episode_status(
                            episode_id=episode_id,
                            status="failed",
                            stage="transcription",
                            error_type=type(exc).__name__,
                            error_message=redact_for_log(str(exc), max_len=500),
                        )
                # Issue #429 Phase 2: stop on first failure or after N failures
                fail_fast = getattr(cfg, "fail_fast", False)
                max_failures = getattr(cfg, "max_failures", None)
                if fail_fast or (
                    max_failures is not None
                    and pipeline_metrics is not None
                    and pipeline_metrics.errors_total >= max_failures
                ):
                    logger.info(
                        "Stopping transcription: fail_fast=%s, max_failures=%s, errors_total=%s",
                        fail_fast,
                        max_failures,
                        pipeline_metrics.errors_total,
                    )
                    break

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
#: Wall-clock ceiling for a transcription loop, mirroring the processing loop's #1180 bound.
#: Transcription is the slowest stage (ASR is minutes per episode), so this is generous — its job
#: is to make "forever" impossible, not to be a scheduling policy.
DEFAULT_TRANSCRIPTION_LOOP_BUDGET_SECONDS = 6 * 60 * 60


def _transcription_loop_budget_seconds(cfg: Any) -> float:
    """Wall-clock budget for a transcription loop. Applies to every config, opted-in or not."""
    raw = getattr(cfg, "transcription_loop_budget_seconds", None)
    try:
        if raw is not None and float(raw) > 0:
            return float(raw)
    except (TypeError, ValueError):
        pass
    return float(DEFAULT_TRANSCRIPTION_LOOP_BUDGET_SECONDS)


def _transcription_supervision_exit_reason(
    started_at: float, budget_seconds: float
) -> Optional[str]:
    """Why this loop must stop regardless of queue state — or None to keep going.

    WHY THIS EXISTS (2026-08-19). The processing loop got these bounds after the 2026-08-12
    incident (#1180); the transcription loop did not, and it has the same shape and the same
    hazard. Its ONLY exit is `downloads_complete_event` plus an empty queue. If the main thread
    dies before setting that event — process_episodes re-raises CostCapExceeded
    (processing.py:1605) and ResilienceFuseOpenError (processing.py:1607) — this non-daemon
    thread waits forever and the process can never exit. Prod was found on 2026-08-19 with a
    container Up 7 days from exactly that failure in the sibling thread.

    orchestration now sets both events from a finally, which fixes the known escape. These bounds
    are the backstop that does not depend on the caller getting that right — no matter which line
    the main thread dies on, a worker must never outlive its parent.
    """
    if not threading.main_thread().is_alive():
        return "main thread exited"
    elapsed = time.time() - started_at
    if elapsed > budget_seconds:
        return f"wall-clock budget exceeded ({elapsed:.0f}s > {budget_seconds:.0f}s)"
    return None


def _authorise_transcription_spend(job: Any, cfg: Any) -> bool:
    """May this episode's transcription be paid for? False means the budget refused it.

    Provider-agnostic: the price comes from the configured transcription provider's row in the
    same pricing table that prices the real call, so nothing here names a vendor.

    Fails OPEN in exactly two cases, both deliberate:
      * no cap configured — there is nothing to enforce;
      * the episode's duration or price cannot be resolved — refusing work because a pricing row
        is missing would ground the pipeline on a config gap rather than a cost problem. The
        ledger still records what the call actually cost, so an unpriceable stream is bounded
        after the fact rather than not at all.
    """
    try:
        from ..run_budget import get_run_budget

        budget = get_run_budget()
        if budget.cap_usd is None:
            return True

        # Already refused earlier in this feed: stop scheduling without re-pricing.
        if budget.tripped:
            logger.warning(
                "run budget exhausted — NOT transcribing episode idx=%s (%s)",
                getattr(job, "idx", "?"),
                budget.trip_reason or "cap reached",
            )
            return False

        from ...utils.provider_metrics import transcription_model_for_cfg
        from ..episode_processor import _audio_sec_for_transcription_job
        from ..helpers import calculate_provider_cost

        audio_sec = _audio_sec_for_transcription_job(job)
        if not audio_sec or audio_sec <= 0:
            return True

        estimate = calculate_provider_cost(
            cfg=cfg,
            provider_type=str(getattr(cfg, "transcription_provider", None) or "whisper"),
            capability="transcription",
            model=transcription_model_for_cfg(cfg),
            audio_minutes=float(audio_sec) / 60.0,
        )
        if estimate is None:
            return True

        if budget.check_and_reserve(float(estimate)):
            return True

        logger.error(
            "REFUSING to transcribe episode idx=%s: it would cost about $%.4f and the run has "
            "$%.4f of its $%.2f budget left. Remaining episodes in this run are skipped; the "
            "batch stops after this feed.",
            getattr(job, "idx", "?"),
            float(estimate),
            budget.remaining_usd,
            budget.cap_usd,
        )
        return False
    except Exception:  # noqa: BLE001 - a broken guard must not silently block transcription
        logger.debug("transcription spend authorisation skipped", exc_info=True)
        return True


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
        logger.debug(
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

        #1180: brackets the actual work with a monotonic-clock interval so the
        end-of-run overlap ratio can be computed. Works for both the sequential
        (max_workers <= 1) and ThreadPoolExecutor paths — every job goes through
        here. The interval record fires from a helper (``_stamp_tx_active``) so
        this function's cognitive complexity is unchanged by the addition.
        """
        # PRE-AUTHORISE THE SPEND BEFORE MAKING IT (2026-08-18).
        #
        # This is the only point in the pipeline that is both provider-agnostic and BEFORE the
        # money moves. Everything downstream sees spend only after transcription has finished:
        # the first check that observes it is orchestration's post-join
        # check_cost_soft_cap_at_stage, by which time every episode in the feed has been
        # transcribed and billed. A feed reached $9.63 under a $5 cap that way.
        #
        # This covers both loops in THIS function (sequential and ThreadPoolExecutor); the
        # separate process_transcription_jobs above has the same guard at its own transcribe call
        # site, because it is a second entry point, not a caller of this one.
        # Refusal returns a normal "did not transcribe" result and latches the
        # ledger; it does NOT raise. An exception here kills only this worker thread while the
        # main thread waits in transcription_thread.join() and never learns, which is the
        # wedge that #1180's supervision work was written to stop (processing.py:2022-2029).
        # The main thread converts the latch into CostCapExceeded after the join.
        if not _authorise_transcription_spend(job, cfg):
            return False, None, 0

        _tx_active_start = time.monotonic()
        try:
            # Anchor the roster with the feed-stated hosts (see the sequential path above).
            job.feed_hosts = sorted(host_detection_result.cached_hosts or [])
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
                            # #1180: stamp for handoff-latency measurement.
                            queued_at=time.monotonic(),
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
            _stamp_tx_active(pipeline_metrics, _tx_active_start)
            return success, transcript_path, bytes_downloaded
        except Exception as exc:  # pragma: no cover
            update_metric_safely(pipeline_metrics, "errors_total", 1)
            logger.error(
                "[%s] transcription raised an unexpected error: %s",
                job.idx,
                format_exception_for_log(exc),
            )
            # Record per-episode failure for run index (Issue #429)
            if pipeline_metrics is not None:
                episode_obj = next((ep for ep in episodes if ep.idx == job.idx), None)
                if episode_obj is not None:
                    from ..helpers import get_episode_id_from_episode

                    episode_id, _ = get_episode_id_from_episode(episode_obj, cfg.rss_url or "")
                    pipeline_metrics.update_episode_status(
                        episode_id=episode_id,
                        status="failed",
                        stage="transcription",
                        error_type=type(exc).__name__,
                        error_message=redact_for_log(str(exc), max_len=500),
                    )
            _stamp_tx_active(pipeline_metrics, _tx_active_start)
            return False, None, 0

    # Process jobs as they become available from the queue
    # Continue until downloads are complete AND queue is empty
    # Supervision bounds for both loops below (2026-08-19). See
    # _transcription_supervision_exit_reason: these make "wait forever" impossible regardless of
    # whether the caller remembered to set downloads_complete_event.
    _tx_loop_started_at = time.time()
    _tx_loop_budget = _transcription_loop_budget_seconds(cfg)

    if max_workers <= 1:
        # Sequential processing (Whisper default)
        while True:
            _reason = _transcription_supervision_exit_reason(_tx_loop_started_at, _tx_loop_budget)
            if _reason is not None:
                logger.error(
                    "Transcription loop stopping: %s. %d job(s) processed; anything still queued "
                    "is NOT transcribed and a resumed run will pick it up.",
                    _reason,
                    jobs_processed,
                )
                break
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

            def _tx_must_stop() -> Optional[str]:
                return _transcription_supervision_exit_reason(_tx_loop_started_at, _tx_loop_budget)

            def _submit_new_transcription_jobs() -> None:
                """Submit new transcription jobs as they become available from the queue."""
                if _tx_must_stop() is not None:
                    return  # stop feeding a loop that is about to exit
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
                        logger.error(
                            "[%s] transcription future raised error: %s",
                            job_idx,
                            format_exception_for_log(exc),
                        )

            while True:
                # Supervision FIRST: the branches below can only fire once
                # downloads_complete_event is set, so when the main thread dies before setting it
                # this loop has no other way out. Same shape as the processing loop's #1180 bound.
                _reason = _tx_must_stop()
                if _reason is not None:
                    logger.error(
                        "Transcription loop stopping: %s. Abandoning %d in-flight job(s); they "
                        "are NOT transcribed and a resumed run will pick them up.",
                        _reason,
                        len(futures),
                    )
                    break

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
    logger.debug(
        "Concurrent transcription processing completed: %s/%s transcripts saved "
        "(parallelism=%s)",
        saved,
        jobs_processed,
        max_workers,
    )
