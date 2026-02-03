"""Summarization stage for episode summarization.

This module handles parallel and sequential episode summarization.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Any, List, Literal, Optional, Tuple

import yaml

from ... import config, models
from ...rss import extract_episode_metadata, extract_episode_published_date
from ...utils import filesystem
from .. import metrics
from ..metadata_generation import (
    _determine_metadata_path,
    generate_episode_metadata as metadata_generate_episode_metadata,
)
from ..types import FeedMetadata, HostDetectionResult

logger = logging.getLogger(__name__)


def _collect_episodes_for_summarization(
    episodes: List[models.Episode],
    download_args: Optional[List[Tuple]],
    effective_output_dir: str,
    run_suffix: Optional[str],
    cfg: config.Config,
) -> List[Tuple]:
    """Collect episodes that need summarization.

    Returns:
        List of tuples: (episode, transcript_path, metadata_path, detected_names)
    """
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
            metadata_path = _determine_metadata_path(episode, effective_output_dir, run_suffix, cfg)
            needs_summary = True
            if os.path.exists(metadata_path):
                # Check if summary already exists in metadata
                try:
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

    return episodes_to_summarize


def _process_episodes_sequentially(
    episodes_to_summarize: List[Tuple],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: FeedMetadata,
    host_detection_result: HostDetectionResult,
    summary_provider: Any,
    pipeline_metrics: Optional[metrics.Metrics],
) -> None:
    """Process episodes sequentially using API provider."""
    logger.debug("Using API provider for sequential summarization")
    for episode, transcript_path, metadata_path, detected_names in episodes_to_summarize:
        summarize_single_episode(
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


def _can_use_parallel_processing(
    summary_provider: Any,
) -> bool:
    """Check if provider supports parallel processing with worker instances.

    Args:
        summary_provider: Provider instance to check

    Returns:
        True if provider supports create_worker_instances(), False otherwise
    """
    # Check if provider has create_worker_instances method
    return hasattr(summary_provider, "create_worker_instances")


def _determine_max_workers(model_device: str, cfg: config.Config, num_episodes: int) -> int:
    """Determine maximum number of workers based on device and configuration.

    Returns:
        Maximum number of workers to use
    """
    max_workers = 1
    if model_device == "cpu":
        if cfg.summary_max_workers_cpu is not None:
            max_workers_limit = cfg.summary_max_workers_cpu
        else:
            is_test_env = os.environ.get("PYTEST_CURRENT_TEST") is not None
            max_workers_limit = 1 if is_test_env else 4
        max_workers = min(cfg.summary_batch_size or 1, max_workers_limit, num_episodes)
    elif model_device in ("mps", "cuda"):
        if cfg.summary_max_workers_gpu is not None:
            max_workers_limit = cfg.summary_max_workers_gpu
        else:
            is_test_env = os.environ.get("PYTEST_CURRENT_TEST") is not None
            max_workers_limit = 1 if is_test_env else 2
        max_workers = min(max_workers_limit, num_episodes)
    return max_workers


def summarize_single_episode(
    episode: models.Episode,
    transcript_path: str,
    metadata_path: Optional[str],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: FeedMetadata,
    host_detection_result: HostDetectionResult,
    summary_provider=None,  # SummarizationProvider instance (required)
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
        summary_provider: SummarizationProvider instance (required)
        detected_names: Detected guest names for this episode (optional)
        pipeline_metrics: Metrics collector
    """
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
    # Use wrapper function if available (for testability)
    import sys

    workflow_pkg = sys.modules.get("podcast_scraper.workflow")
    # Check for wrapper function first
    if workflow_pkg and hasattr(workflow_pkg, "_generate_episode_metadata"):
        func = getattr(workflow_pkg, "_generate_episode_metadata")
        from unittest.mock import Mock

        if isinstance(func, Mock):
            func(
                feed=feed,
                episode=episode,
                feed_url=cfg.rss_url or "",
                cfg=cfg,
                output_dir=effective_output_dir,
                run_suffix=run_suffix,
                transcript_file_path=os.path.relpath(transcript_path, effective_output_dir),
                transcript_source=transcript_source,
                whisper_model=None,
                summary_provider=summary_provider,
                detected_hosts=(
                    list(host_detection_result.cached_hosts)
                    if host_detection_result.cached_hosts
                    else None
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
                pipeline_metrics=pipeline_metrics,
            )
            return
    # Check for metadata module patch (tests patch workflow.metadata.generate_episode_metadata)
    if workflow_pkg and hasattr(workflow_pkg, "metadata"):
        metadata_mod = getattr(workflow_pkg, "metadata")
        if hasattr(metadata_mod, "generate_episode_metadata"):
            func = getattr(metadata_mod, "generate_episode_metadata")
            from unittest.mock import Mock

            if isinstance(func, Mock):
                func(
                    feed=feed,
                    episode=episode,
                    feed_url=cfg.rss_url or "",
                    cfg=cfg,
                    output_dir=effective_output_dir,
                    run_suffix=run_suffix,
                    transcript_file_path=os.path.relpath(transcript_path, effective_output_dir),
                    transcript_source=transcript_source,
                    whisper_model=None,
                    summary_provider=summary_provider,
                    detected_hosts=(
                        list(host_detection_result.cached_hosts)
                        if host_detection_result.cached_hosts
                        else None
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
                    pipeline_metrics=pipeline_metrics,
                )
                return
    metadata_generate_episode_metadata(
        feed=feed,
        episode=episode,
        feed_url=cfg.rss_url or "",
        cfg=cfg,
        output_dir=effective_output_dir,
        run_suffix=run_suffix,
        transcript_file_path=os.path.relpath(transcript_path, effective_output_dir),
        transcript_source=transcript_source,
        whisper_model=None,
        summary_provider=summary_provider,
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
        pipeline_metrics=pipeline_metrics,
    )


def _process_episodes_in_parallel(
    episodes_to_summarize: List[Tuple],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: FeedMetadata,
    host_detection_result: HostDetectionResult,
    summary_provider: Any,
    pipeline_metrics: Optional[metrics.Metrics],
    max_workers: int,
) -> None:
    """Process episodes in parallel with worker provider instances.

    This function creates worker provider instances for parallel processing.
    Each worker thread gets its own provider instance to ensure thread safety.

    Args:
        episodes_to_summarize: List of episode tuples to process
        feed: Parsed RssFeed object
        cfg: Configuration object
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix
        feed_metadata: Feed metadata tuple
        host_detection_result: Host detection result
        summary_provider: SummarizationProvider instance (must support create_worker_instances)
        pipeline_metrics: Metrics collector
        max_workers: Maximum number of parallel workers
    """
    logger.debug(
        f"Using {max_workers} workers for parallel episode summarization "
        f"(creating {max_workers} worker provider instances for thread safety)"
    )

    # Create worker provider instances for parallel processing
    # This ensures each worker thread has its own provider instance with its own models
    if not hasattr(summary_provider, "create_worker_instances"):
        provider_name = type(summary_provider).__name__
        logger.warning(
            f"Provider {provider_name} does not support "
            "create_worker_instances(). "
            "Falling back to sequential processing."
        )
        _process_episodes_sequentially(
            episodes_to_summarize,
            feed,
            cfg,
            effective_output_dir,
            run_suffix,
            feed_metadata,
            host_detection_result,
            summary_provider,
            pipeline_metrics,
        )
        return

    try:
        worker_providers = summary_provider.create_worker_instances(max_workers)
        logger.debug(f"Successfully created {len(worker_providers)} worker provider instances")
    except Exception as e:
        logger.error(f"Failed to create worker provider instances: {e}")
        logger.warning("Falling back to sequential processing due to worker creation failure")
        _process_episodes_sequentially(
            episodes_to_summarize,
            feed,
            cfg,
            effective_output_dir,
            run_suffix,
            feed_metadata,
            host_detection_result,
            summary_provider,
            pipeline_metrics,
        )
        return

    # Use thread-local storage to assign worker providers to worker threads
    thread_local = threading.local()
    _execute_parallel_summarization(
        episodes_to_summarize,
        feed,
        cfg,
        effective_output_dir,
        run_suffix,
        feed_metadata,
        host_detection_result,
        pipeline_metrics,
        worker_providers,
        thread_local,
        max_workers,
    )

    # Cleanup: cleanup all worker providers after parallel processing
    logger.debug(f"Cleaning up {len(worker_providers)} worker provider instances")
    for worker_provider in worker_providers:
        try:
            worker_provider.cleanup()
        except Exception as cleanup_error:
            logger.debug(f"Error cleaning up worker provider: {cleanup_error}")
    # Clear worker provider list to help GC
    worker_providers.clear()


def _execute_parallel_summarization(
    episodes_to_summarize: List[Tuple],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: FeedMetadata,
    host_detection_result: HostDetectionResult,
    pipeline_metrics: Optional[metrics.Metrics],
    worker_providers: List[Any],
    thread_local: threading.local,
    max_workers: int,
) -> None:
    """Execute parallel summarization with worker provider instances."""
    provider_index = [0]  # Use list to allow modification in nested function
    provider_index_lock = threading.Lock()

    def _get_worker_provider():
        """Get worker provider instance for current worker thread."""
        if not hasattr(thread_local, "provider"):
            with provider_index_lock:
                if provider_index[0] < len(worker_providers):
                    idx = provider_index[0]
                    thread_local.provider = worker_providers[idx]
                    provider_index[0] += 1
                else:
                    # Fallback: reuse last provider (shouldn't happen with proper worker count)
                    thread_local.provider = worker_providers[-1]
        return thread_local.provider

    def _summarize_with_worker_provider(args):
        """Wrapper to get worker-specific provider and summarize episode."""
        episode, transcript_path, metadata_path, detected_names = args
        worker_provider = _get_worker_provider()
        summarize_single_episode(
            episode=episode,
            transcript_path=transcript_path,
            metadata_path=metadata_path,
            feed=feed,
            cfg=cfg,
            effective_output_dir=effective_output_dir,
            run_suffix=run_suffix,
            feed_metadata=feed_metadata,
            host_detection_result=host_detection_result,
            summary_provider=worker_provider,  # Use worker-specific provider
            detected_names=detected_names,
            pipeline_metrics=pipeline_metrics,
        )

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for episode_data in episodes_to_summarize:
                future = executor.submit(_summarize_with_worker_provider, episode_data)
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
        # Force garbage collection after unloading models
        gc.collect()


def parallel_episode_summarization(
    episodes: List[models.Episode],
    feed: models.RssFeed,
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_metadata: FeedMetadata,
    host_detection_result: HostDetectionResult,
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
        download_args: Optional download args for detected names mapping
        pipeline_metrics: Metrics collector
    """
    # Collect episodes that need summarization
    episodes_to_summarize = _collect_episodes_for_summarization(
        episodes, download_args, effective_output_dir, run_suffix, cfg
    )

    if not episodes_to_summarize:
        logger.debug("No episodes need summarization (all already have summaries)")
        return

    if summary_provider is None:
        # Fail fast - summary_provider is required when generate_summaries=True
        if cfg.generate_summaries:
            error_msg = (
                "Summary provider not available but generate_summaries=True. "
                "Cannot proceed with parallel summarization without provider."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        # If generate_summaries=False, it's okay to skip
        logger.warning("Summary provider not available, skipping parallel summarization")
        return

    logger.info(f"Processing summarization for {len(episodes_to_summarize)} episodes in parallel")

    # Check if provider supports parallel processing with worker instances
    can_use_parallel = _can_use_parallel_processing(summary_provider)

    if not can_use_parallel:
        # Provider doesn't support parallel processing - use sequential
        logger.debug(
            f"Provider {type(summary_provider).__name__} does not support parallel processing, "
            "using sequential processing"
        )
        _process_episodes_sequentially(
            episodes_to_summarize,
            feed,
            cfg,
            effective_output_dir,
            run_suffix,
            feed_metadata,
            host_detection_result,
            summary_provider,
            pipeline_metrics,
        )
        return

    # Determine max workers based on provider type and configuration
    # For MLProvider, check device from internal model
    max_workers = 1
    if hasattr(summary_provider, "_map_model") and summary_provider._map_model:
        model_device = getattr(summary_provider._map_model, "device", "cpu")
        max_workers = _determine_max_workers(model_device, cfg, len(episodes_to_summarize))
    else:
        # For other providers, use default sequential processing
        max_workers = 1

    if max_workers <= 1:
        # Sequential processing - use provider directly
        _process_episodes_sequentially(
            episodes_to_summarize,
            feed,
            cfg,
            effective_output_dir,
            run_suffix,
            feed_metadata,
            host_detection_result,
            summary_provider,
            pipeline_metrics,
        )
    else:
        # Parallel processing - each worker gets its own provider instance
        _process_episodes_in_parallel(
            episodes_to_summarize,
            feed,
            cfg,
            effective_output_dir,
            run_suffix,
            feed_metadata,
            host_detection_result,
            summary_provider,
            pipeline_metrics,
            max_workers,
        )
