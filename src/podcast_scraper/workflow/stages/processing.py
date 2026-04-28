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
from typing import Any, cast, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from ... import config, models

if TYPE_CHECKING:
    from ...models import Episode, RssFeed
else:
    Episode = models.Episode  # type: ignore[assignment]
    RssFeed = models.RssFeed  # type: ignore[assignment]
from ...rss import BYTES_PER_MB, http_head, OPENAI_MAX_FILE_SIZE_BYTES
from ...utils.log_redaction import format_exception_for_log, redact_for_log
from .. import metrics
from ..episode_processor import process_episode_download as factory_process_episode_download


# Use wrapper function if available (for testability)
def process_episode_download(*args, **kwargs):
    """Delegate to workflow.process_episode_download or factory; allows tests to inject a mock."""
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
    """Delegate to workflow.extract_episode_description or RSS; allows tests to inject a mock."""
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

_PROCESSING_JOBS_WARN_THRESHOLD = 1000
_processing_jobs_warned = False


def _warn_if_jobs_large(jobs: List) -> None:
    """Emit a one-time warning when processing_jobs grows large."""
    global _processing_jobs_warned
    if _processing_jobs_warned:
        return
    n = len(jobs)
    if n > _PROCESSING_JOBS_WARN_THRESHOLD:
        _processing_jobs_warned = True
        logger.warning(
            "processing_jobs list has %d entries; " "consider reducing episode count",
            n,
        )


_EPISODE_RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def _is_episode_retryable(exc: Exception) -> bool:
    """Return True if the exception warrants an episode-level retry."""
    if isinstance(exc, _EPISODE_RETRYABLE_EXCEPTIONS):
        return True
    try:
        import requests

        if isinstance(exc, requests.RequestException):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return any(tok in msg for tok in ("timeout", "connection", "reset", "429", "503"))


_EpisodeResult = Tuple[bool, Optional[str], Optional[str], int]


def _process_episode_with_retry(
    process_fn: Any,
    args: Tuple,
    cfg: "config.Config",
    pipeline_metrics: "metrics.Metrics",
) -> _EpisodeResult:
    """Wrap a single episode download call with app-level retries.

    When ``cfg.episode_retry_max > 0`` and the download raises a
    transient network error, the entire episode operation is retried
    up to ``episode_retry_max`` times with exponential backoff starting
    at ``episode_retry_delay_sec``.

    Returns the same 4-tuple as ``process_episode_download``.
    """
    max_retries = getattr(cfg, "episode_retry_max", 0)
    if max_retries <= 0:
        result: _EpisodeResult = process_fn(*args, pipeline_metrics=pipeline_metrics)
        return result

    delay = getattr(cfg, "episode_retry_delay_sec", 5.0)
    episode = args[0]
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            result = process_fn(*args, pipeline_metrics=pipeline_metrics)
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries and _is_episode_retryable(exc):
                logger.warning(
                    "[%s] episode download attempt %d/%d failed: " "%s — retrying in %.1fs",
                    episode.idx,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    delay,
                )
                pipeline_metrics.record_episode_download_retry(delay)
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
            else:
                raise

    # Should not reach here, but satisfy type checker
    if last_exc:
        raise last_exc
    return False, None, None, 0


def _flatten_speaker_name_entries(value: Any) -> List[str]:
    """Normalize speaker-detector output to flat, non-empty strings.

    LLM JSON occasionally nests names (e.g. ``[\"A\", \"B\"]`` or mixed lists);
    those values are not hashable and must not be used in ``set`` membership
    checks without flattening.
    """
    if value is None:
        return []
    if isinstance(value, str):
        t = value.strip()
        return [t] if t else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for v in value:
            out.extend(_flatten_speaker_name_entries(v))
        return out
    t = str(value).strip()
    return [t] if t else []


def _speaker_names_to_str_set(members: Any) -> Set[str]:
    """Build a string set from iterable of possibly nested speaker/host labels."""
    out: Set[str] = set()
    if members is None:
        return out
    for item in members:
        for s in _flatten_speaker_name_entries(item):
            out.add(s)
    return out


def _handle_dry_run_host_detection(
    feed: RssFeed,  # type: ignore[valid-type]
) -> HostDetectionResult:
    """Handle host detection in dry-run mode.

    Args:
        feed: Parsed RssFeed object

    Returns:
        HostDetectionResult with hosts from RSS author tags if available
    """
    logger.info("(dry-run) would initialize speaker detector")
    cached_hosts: set[str] = set()
    # Still detect hosts from RSS author tags if available
    if feed.authors:
        cached_hosts = set(feed.authors)
        if cached_hosts:
            logger.info(
                "DETECTED HOSTS (from %s): %s",
                "RSS author tags",
                ", ".join(sorted(cached_hosts)),
            )
    return HostDetectionResult(cached_hosts, None, None)


def _create_speaker_detector_if_needed(
    cfg: config.Config, speaker_detector: Optional[Any]
) -> Optional[Any]:
    """Create speaker detector if not provided (backward compatibility).

    Args:
        cfg: Configuration object
        speaker_detector: Optional existing speaker detector

    Returns:
        Speaker detector instance or None if creation failed
    """
    if speaker_detector is not None:
        return speaker_detector

    # Fallback: create speaker detector if not provided (for backward compatibility)
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
        return speaker_detector
    except Exception as exc:
        logger.error(
            "Failed to initialize speaker detector: %s",
            format_exception_for_log(exc),
        )
        return None


def _detect_hosts_from_feed(
    feed: RssFeed,  # type: ignore[valid-type]
    speaker_detector: Any,
) -> set[str]:
    """Detect hosts from feed metadata using speaker detector.

    Args:
        feed: Parsed RssFeed object
        speaker_detector: Speaker detector instance

    Returns:
        Set of detected host names
    """
    feed_hosts = speaker_detector.detect_hosts(
        feed_title=feed.title,
        feed_description=None,  # TODO: Extract from feed XML if needed
        feed_authors=feed.authors if feed.authors else None,
    )
    return cast(set[str], feed_hosts)


def _validate_hosts_with_first_episode(
    feed_hosts: set[str],
    feed: RssFeed,  # type: ignore[valid-type]
    episodes: List[Episode],  # type: ignore[valid-type]
    speaker_detector: Any,
    pipeline_metrics: Optional[metrics.Metrics],
) -> set[str]:
    """Validate hosts by checking if they appear in first episode.

    Args:
        feed_hosts: Hosts detected from feed
        feed: Parsed RssFeed object
        episodes: List of Episode objects
        speaker_detector: Speaker detector instance
        pipeline_metrics: Optional metrics collector

    Returns:
        Validated set of host names
    """
    # Skip validation if hosts came from author tags (they're already reliable)
    if not feed_hosts or not episodes or feed.authors:
        return feed_hosts

    # Only validate if we used NER (not author tags)
    first_episode = episodes[0]
    first_episode_description = extract_episode_description(first_episode.item)
    # Validate hosts by checking if they appear in first episode
    # Use provider's detect_speakers to extract persons from first episode
    # Pass pipeline_metrics for LLM call tracking (if OpenAI provider)
    import inspect

    sig = inspect.signature(speaker_detector.detect_speakers)
    if "pipeline_metrics" in sig.parameters:
        first_episode_speakers, _, _, _ = (
            speaker_detector.detect_speakers(  # type: ignore[call-arg]
                episode_title=first_episode.title,
                episode_description=first_episode_description,
                known_hosts=set(),
                pipeline_metrics=pipeline_metrics,
            )
        )
    else:
        first_episode_speakers, _, _, _ = speaker_detector.detect_speakers(
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
    return validated_hosts if validated_hosts else feed_hosts


def _fallback_to_episode_authors(
    cfg: config.Config, episodes: List[Episode]  # type: ignore[valid-type]
) -> set[str]:
    """Fallback to episode-level authors if no feed-level hosts found.

    Args:
        cfg: Configuration object
        episodes: List of Episode objects

    Returns:
        Set of episode author names (filtered to exclude organizations)
    """
    episode_authors: set[str] = set()
    if not cfg.auto_speakers or not episodes:
        return episode_authors

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

    return episode_authors


def _log_detected_hosts(
    cached_hosts: set[str],
    feed: RssFeed,  # type: ignore[valid-type]
    episode_authors: set[str],
    cfg: config.Config,
) -> None:
    """Log detected hosts with their source.

    Args:
        cached_hosts: Set of detected host names
        feed: Parsed RssFeed object
        episode_authors: Set of episode-level authors
        cfg: Configuration object
    """
    if not cached_hosts:
        if cfg.auto_speakers:
            logger.debug(
                "No hosts detected from feed metadata, episode-level authors, or config known_hosts"
            )
        return

    # Determine source for logging
    if feed.authors:
        source = "RSS author tags"
    elif episode_authors and cached_hosts == episode_authors:
        source = "episode-level authors"
    elif cfg.known_hosts and cached_hosts == set(cfg.known_hosts):
        source = "config known_hosts (fallback)"
    else:
        source = "feed metadata (NER)"
    logger.info("DETECTED HOSTS (from %s): %s", source, ", ".join(sorted(cached_hosts)))


def detect_feed_hosts_and_patterns(
    cfg: config.Config,
    feed: RssFeed,  # type: ignore[valid-type]
    episodes: List[Episode],  # type: ignore[valid-type]
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
    if cfg.dry_run:
        return _handle_dry_run_host_detection(feed)

    # Use provided speaker detector, or create one if not provided (backward compatibility)
    speaker_detector = _create_speaker_detector_if_needed(cfg, speaker_detector)
    if speaker_detector is None:
        return HostDetectionResult(cached_hosts, heuristics, None)

    # Detect hosts: prefer RSS author tags, fall back to NER
    feed_hosts = _detect_hosts_from_feed(feed, speaker_detector)

    # Priority: Use known_hosts from config if provided (show-level override)
    if cfg.known_hosts:
        known_hosts_set = set(cfg.known_hosts)
        logger.info(
            "Using known_hosts from config: %s",
            ", ".join(sorted(known_hosts_set)),
        )
        # Merge with feed_hosts (known_hosts takes precedence)
        cached_hosts = known_hosts_set | feed_hosts
        if cached_hosts:
            logger.info(
                "DETECTED HOSTS (from config known_hosts + feed): %s",
                ", ".join(sorted(cached_hosts)),
            )
            # Skip validation since known_hosts are trusted
            return HostDetectionResult(cached_hosts, heuristics, speaker_detector)

    # Validate hosts with first episode: hosts should appear in first episode too
    cached_hosts = _validate_hosts_with_first_episode(
        feed_hosts, feed, episodes, speaker_detector, pipeline_metrics
    )

    # Fallback to episode-level authors if no feed-level hosts found (Issue #380)
    episode_authors: set[str] = set()
    if not cached_hosts:
        episode_authors = _fallback_to_episode_authors(cfg, episodes)
        if episode_authors:
            cached_hosts = episode_authors
            logger.info(
                "DETECTED HOSTS (from episode-level authors): %s",
                ", ".join(sorted(cached_hosts)),
            )

    # Fallback to known_hosts from config if no hosts detected (show-level override)
    if not cached_hosts and cfg.known_hosts:
        cached_hosts = set(cfg.known_hosts)
        logger.info(
            "DETECTED HOSTS (from config known_hosts fallback): %s",
            ", ".join(sorted(cached_hosts)),
        )

    # Log detected hosts with their source
    _log_detected_hosts(cached_hosts, feed, episode_authors, cfg)

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


def _check_episode_size_skip(
    cfg: config.Config,
    episode: Episode,  # type: ignore[valid-type]
) -> tuple[bool, bool]:
    """Check file size for API limits. Returns (skip_speaker_detection, skip_episode)."""
    if (
        cfg.dry_run
        or not cfg.transcribe_missing
        or cfg.transcription_provider not in ("openai", "gemini")
        or not episode.media_url
    ):
        return False, False
    resp = http_head(episode.media_url, cfg.user_agent, cfg.timeout)
    if not resp:
        return False, False
    content_length = resp.headers.get("Content-Length")
    if not content_length:
        return False, False
    try:
        file_size_bytes = int(content_length)
    except (ValueError, TypeError):
        return False, False
    if file_size_bytes <= OPENAI_MAX_FILE_SIZE_BYTES:
        return False, False
    file_size_mb = file_size_bytes / BYTES_PER_MB
    provider_name = "OpenAI" if cfg.transcription_provider == "openai" else "Gemini"
    if not episode.transcript_urls:
        logger.info(
            "[%d] Skipping episode: Audio file size (%.1f MB) exceeds %s API limit (25 MB) "
            "and no transcript URLs available.",
            episode.idx,
            file_size_mb,
            provider_name,
        )
        return True, True
    logger.info(
        "[%d] Skipping speaker detection: Audio file size (%.1f MB) exceeds %s API limit "
        "(25 MB), but transcript URLs available.",
        episode.idx,
        file_size_mb,
        provider_name,
    )
    return True, False


def _get_speaker_detector(
    host_detection_result: HostDetectionResult, cfg: config.Config
) -> Optional[Any]:
    """Get speaker detector from result or create fallback."""
    detector = host_detection_result.speaker_detector
    if detector:
        return detector
    logger.warning("speaker_detector not found in host_detection_result, creating new instance")
    import sys

    workflow_pkg = sys.modules.get("podcast_scraper.workflow")
    if workflow_pkg and hasattr(workflow_pkg, "create_speaker_detector"):
        func = getattr(workflow_pkg, "create_speaker_detector")
        from unittest.mock import Mock

        detector = func(cfg) if isinstance(func, Mock) else create_speaker_detector(cfg)
    else:
        from ...speaker_detectors.factory import (
            create_speaker_detector as factory_create_speaker_detector,
        )

        detector = factory_create_speaker_detector(cfg)
    if detector:
        detector.initialize()
    return detector


def _detect_speakers_for_episode(
    episode: Episode,  # type: ignore[valid-type]
    cfg: config.Config,
    host_detection_result: HostDetectionResult,
    pipeline_metrics: metrics.Metrics,
    skip_speaker_detection: bool = False,
) -> Optional[List[str]]:
    """Run speaker detection for one episode; return list of guest names or None."""
    if not cfg.auto_speakers:
        if cfg.screenplay_speaker_names and len(cfg.screenplay_speaker_names) > 1:
            return cfg.screenplay_speaker_names[1:]
        return None
    logger.debug("Episode %d: %s", episode.idx, episode.title)
    if skip_speaker_detection:
        return None
    if cfg.dry_run:
        episode_description = extract_episode_description(episode.item) or ""
        desc_preview = (
            episode_description[:50] + "..."
            if len(episode_description) > 50
            else episode_description
        )
        logger.info(
            "(dry-run) would detect speakers from: %s | %s",
            episode.title,
            desc_preview,
        )
        return None
    if skip_speaker_detection:
        return None
    episode_description = extract_episode_description(episode.item)
    extract_names_start = time.time()
    speaker_detector = _get_speaker_detector(host_detection_result, cfg)
    if not speaker_detector:
        return None
    cached_hosts = host_detection_result.cached_hosts if cfg.cache_detected_hosts else set()
    combined_hosts = set(cfg.known_hosts) | cached_hosts if cfg.known_hosts else cached_hosts
    import inspect

    sig = inspect.signature(speaker_detector.detect_speakers)
    if "pipeline_metrics" in sig.parameters:
        detected_speakers, detected_hosts_set, detection_succeeded, _ = (
            speaker_detector.detect_speakers(
                episode_title=episode.title,
                episode_description=episode_description,
                known_hosts=combined_hosts,
                pipeline_metrics=pipeline_metrics,
            )
        )
    else:
        detected_speakers, detected_hosts_set, detection_succeeded, _ = (
            speaker_detector.detect_speakers(
                episode_title=episode.title,
                episode_description=episode_description,
                known_hosts=combined_hosts,
            )
        )
    if pipeline_metrics is not None:
        pipeline_metrics.record_extract_names_time(time.time() - extract_names_start, episode.idx)
    if (
        not detection_succeeded
        and cfg.screenplay_speaker_names
        and len(cfg.screenplay_speaker_names) >= 2
    ):
        return cfg.screenplay_speaker_names[1:]
    if detection_succeeded:
        flat_speakers: List[str] = []
        for entry in detected_speakers or []:
            flat_speakers.extend(_flatten_speaker_name_entries(entry))
        host_strings = _speaker_names_to_str_set(detected_hosts_set)
        return [name for name in flat_speakers if name not in host_strings]
    return None


def prepare_episode_download_args(
    episodes: List[Episode],  # type: ignore[valid-type]
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
        skip_speaker_detection, skip_episode = _check_episode_size_skip(cfg, episode)
        if skip_episode:
            if pipeline_metrics is not None:
                from ..helpers import update_metric_safely

                update_metric_safely(pipeline_metrics, "episodes_skipped_total", 1)
            continue
        if getattr(cfg, "append", False):
            from ..append_resume import episode_complete_for_append_resume
            from ..helpers import get_episode_id_from_episode

            feed_url = cfg.rss_url or ""
            if episode_complete_for_append_resume(
                cfg, episode, feed_url, effective_output_dir, run_suffix
            ):
                logger.info(
                    "[%s] Append: skipping episode already complete on disk (episode_id resume)",
                    episode.idx,
                )
                if pipeline_metrics is not None:
                    episode_id, episode_number = get_episode_id_from_episode(episode, feed_url)
                    pipeline_metrics.record_episode_status(
                        episode_id=episode_id,
                        episode_number=episode_number or episode.idx,
                        status="ok",
                        stage="append_skipped_complete",
                    )
                continue
        detected_speaker_names = _detect_speakers_for_episode(
            episode,
            cfg,
            host_detection_result,
            pipeline_metrics,
            skip_speaker_detection=skip_speaker_detection,
        )
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


def _handle_episode_download_result(
    episode: Episode,  # type: ignore[valid-type]
    success: bool,
    transcript_path: Optional[str],
    transcript_source: Optional[str],
    bytes_downloaded: int,
    cfg: config.Config,
    processing_resources: ProcessingResources,
    pipeline_metrics: metrics.Metrics,
    detected_names: Optional[List[str]],
) -> int:
    """Handle result from episode download processing.

    Args:
        episode: Episode object
        success: Whether download/transcription succeeded
        transcript_path: Path to transcript file or None
        transcript_source: Source of transcript or None
        bytes_downloaded: Bytes downloaded
        cfg: Configuration object
        processing_resources: Processing resources
        pipeline_metrics: Metrics collector
        detected_names: Detected speaker names

    Returns:
        1 if transcript was saved, 0 otherwise
    """
    from ..helpers import update_metric_safely

    saved = 0
    if bytes_downloaded:
        update_metric_safely(pipeline_metrics, "bytes_downloaded_total", bytes_downloaded)

    if success:
        saved = 1
        # Track transcript source
        if transcript_source == "direct_download":
            update_metric_safely(pipeline_metrics, "transcripts_downloaded", 1)
        logger.debug("Episode %s yielded transcript (saved=%s)", episode.idx, saved)

        # Update episode status: downloaded (Issue #391)
        if pipeline_metrics is not None:
            from ..helpers import get_episode_id_from_episode

            episode_id, episode_number = get_episode_id_from_episode(episode, cfg.rss_url or "")
            pipeline_metrics.update_episode_status(episode_id=episode_id, stage="downloaded")

        # Queue processing job if metadata generation enabled and transcript available
        # Skip if transcript_source is None (Whisper pending) - queued after
        if cfg.generate_metadata and transcript_source is not None:
            from typing import cast, Literal

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
                    _warn_if_jobs_large(processing_resources.processing_jobs)
            else:
                processing_resources.processing_jobs.append(processing_job)
                _warn_if_jobs_large(processing_resources.processing_jobs)
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
                "[%s] Episode queued for transcription " "(not skipped, transcribe_missing=True)",
                episode.idx,
            )

    return saved


def _process_episodes_sequential(
    download_args: List[Tuple],
    cfg: config.Config,
    transcription_resources: TranscriptionResources,
    processing_resources: ProcessingResources,
    pipeline_metrics: metrics.Metrics,
) -> int:
    """Process episodes sequentially.

    Args:
        download_args: List of download argument tuples
        cfg: Configuration object
        transcription_resources: Transcription resources
        processing_resources: Processing resources
        pipeline_metrics: Metrics collector

    Returns:
        Number of transcripts saved
    """
    saved = 0
    for args in download_args:
        episode = args[0]
        detected_names = args[7]
        try:
            success, transcript_path, transcript_source, bytes_downloaded = (
                _process_episode_with_retry(
                    process_episode_download,
                    args,
                    cfg,
                    pipeline_metrics,
                )
            )
            saved += _handle_episode_download_result(
                episode,
                success,
                transcript_path,
                transcript_source,
                bytes_downloaded,
                cfg,
                processing_resources,
                pipeline_metrics,
                detected_names,
            )
        except Exception as exc:  # pragma: no cover
            from ..helpers import update_metric_safely

            update_metric_safely(pipeline_metrics, "errors_total", 1)
            logger.error(
                "[%s] episode processing raised an unexpected " "error: %s",
                episode.idx,
                exc,
                exc_info=True,
            )
    return saved


def _process_episodes_concurrent(
    download_args: List[Tuple],
    episodes: List[Episode],  # type: ignore[valid-type]
    cfg: config.Config,
    transcription_resources: TranscriptionResources,
    processing_resources: ProcessingResources,
    pipeline_metrics: metrics.Metrics,
) -> int:
    """Process episodes concurrently.

    Args:
        download_args: List of download argument tuples
        episodes: List of Episode objects
        cfg: Configuration object
        transcription_resources: Transcription resources
        processing_resources: Processing resources
        pipeline_metrics: Metrics collector

    Returns:
        Number of transcripts saved
    """
    from concurrent.futures import as_completed, ThreadPoolExecutor
    from typing import cast, Literal

    from ..helpers import update_metric_safely

    saved = 0
    saved_counter_lock = transcription_resources.saved_counter_lock
    # Note: processing_resources is accessed via closure
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        future_map = {
            executor.submit(
                _process_episode_with_retry,
                process_episode_download,
                args,
                cfg,
                pipeline_metrics,
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
                            pipeline_metrics,
                            "transcripts_downloaded",
                            1,
                            saved_counter_lock,
                        )
                    logger.debug("Episode %s yielded transcript (saved=%s)", idx, saved)

                    # Update episode status: downloaded (Issue #391)
                    if pipeline_metrics is not None:
                        from ..helpers import get_episode_id_from_episode

                        episode_obj = next((ep for ep in episodes if ep.idx == idx), None)
                        if episode_obj:
                            episode_id, episode_number = get_episode_id_from_episode(
                                episode_obj, cfg.rss_url or ""
                            )
                            pipeline_metrics.update_episode_status(
                                episode_id=episode_id, stage="downloaded"
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
                                    _warn_if_jobs_large(processing_resources.processing_jobs)
                            else:
                                processing_resources.processing_jobs.append(processing_job)
                                _warn_if_jobs_large(processing_resources.processing_jobs)
                            logger.debug(
                                "Queued processing job for episode %s (transcript_source=%s)",
                                episode_obj.idx,
                                transcript_source_typed,
                            )
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
            except Exception as exc:  # pragma: no cover
                update_metric_safely(pipeline_metrics, "errors_total", 1, saved_counter_lock)
                logger.error(
                    "[%s] episode processing raised an unexpected error: %s",
                    idx,
                    format_exception_for_log(exc),
                )

    return saved


def process_episodes(  # noqa: C901
    download_args: List[Tuple],
    episodes: List[Episode],  # type: ignore[valid-type]
    feed: RssFeed,  # type: ignore[valid-type]
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
    if not download_args:
        return 0

    if cfg.workers <= 1 or len(download_args) == 1:
        # Sequential processing
        saved = _process_episodes_sequential(
            download_args, cfg, transcription_resources, processing_resources, pipeline_metrics
        )
    else:
        # Concurrent processing
        saved = _process_episodes_concurrent(
            download_args,
            episodes,
            cfg,
            transcription_resources,
            processing_resources,
            pipeline_metrics,
        )

    return saved


def _drain_completed_processing_futures(
    futures: Dict[Any, int],
    cfg: config.Config,
    pipeline_metrics: Optional[metrics.Metrics],
) -> Tuple[int, int, bool]:
    """Drain completed futures from the executor, update counts, and detect stop request.

    Returns:
        Tuple of (ok_delta, failed_delta, stop_requested).
    """
    ok_delta, failed_delta = 0, 0
    stop_requested = False
    try:
        for future in as_completed(list(futures.keys()), timeout=1.0):
            episode_idx = futures.pop(future)
            try:
                success = future.result()
                if success:
                    ok_delta += 1
                else:
                    failed_delta += 1
                    fail_fast = getattr(cfg, "fail_fast", False)
                    max_failures = getattr(cfg, "max_failures", None)
                    if fail_fast or (
                        max_failures is not None
                        and pipeline_metrics is not None
                        and pipeline_metrics.errors_total >= max_failures
                    ):
                        stop_requested = True
                        logger.info(
                            "Stopping processing: fail_fast=%s, max_failures=%s, "
                            "errors_total=%s",
                            fail_fast,
                            max_failures,
                            pipeline_metrics.errors_total if pipeline_metrics else None,
                        )
                logger.debug(
                    "Processed processing job idx=%s (ok_delta=%s, failed_delta=%s)",
                    episode_idx,
                    ok_delta,
                    failed_delta,
                )
            except Exception as exc:  # pragma: no cover
                failed_delta += 1
                logger.error(
                    "[%s] processing future raised error: %s",
                    episode_idx,
                    format_exception_for_log(exc),
                )
                fail_fast = getattr(cfg, "fail_fast", False)
                max_failures = getattr(cfg, "max_failures", None)
                if fail_fast or (
                    max_failures is not None
                    and pipeline_metrics is not None
                    and pipeline_metrics.errors_total >= max_failures
                ):
                    stop_requested = True
    except TimeoutError:
        pass
    return (ok_delta, failed_delta, stop_requested)


def process_processing_jobs_concurrent(  # noqa: C901
    processing_resources: ProcessingResources,
    feed: RssFeed,  # type: ignore[valid-type]
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
    # Same as orchestration parallelism line when configured == effective; keep DEBUG only
    logger.debug(
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
        stop_requested = [False]  # Issue #429: set when fail_fast or max_failures reached

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            def _submit_new_jobs() -> None:
                """Submit new jobs as they become available."""
                if stop_requested[0]:
                    return
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
                """Process completed futures (delegate to module-level helper)."""
                ok_d, failed_d, stop = _drain_completed_processing_futures(
                    futures, cfg, pipeline_metrics
                )
                jobs_processed_ok[0] += ok_d
                jobs_processed_failed[0] += failed_d
                if stop:
                    stop_requested[0] = True

            def _should_continue_processing() -> bool:
                """Check if processing should continue."""
                if stop_requested[0] and len(futures) == 0:
                    return False
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

            # Enforce summarization timeout per episode (Issue #429)
            from ...utils.timeout import timeout_context, TimeoutError as SummarizationTimeoutError

            summarization_timeout = getattr(cfg, "summarization_timeout", 1200)
            with timeout_context(
                summarization_timeout,
                f"summarization for episode {job.episode.idx}",
            ):
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
        except SummarizationTimeoutError as exc:
            update_metric_safely(pipeline_metrics, "errors_total", 1)
            logger.error(
                "[%s] Summarization timeout after %ss: %s",
                job.episode.idx,
                getattr(cfg, "summarization_timeout", 1200),
                format_exception_for_log(exc),
            )
            if pipeline_metrics is not None:
                from ..helpers import get_episode_id_from_episode

                episode_id, _ = get_episode_id_from_episode(job.episode, cfg.rss_url or "")
                pipeline_metrics.update_episode_status(
                    episode_id=episode_id,
                    status="failed",
                    stage="summarization",
                    error_type="TimeoutError",
                    error_message=redact_for_log(str(exc), max_len=500),
                )
            return False
        except Exception as exc:  # pragma: no cover
            update_metric_safely(pipeline_metrics, "errors_total", 1)
            logger.error(
                "[%s] processing raised an unexpected error: %s",
                job.episode.idx,
                format_exception_for_log(exc),
            )
            # Record per-episode failure for run index (Issue #429)
            if pipeline_metrics is not None:
                from ..helpers import get_episode_id_from_episode

                episode_id, _ = get_episode_id_from_episode(job.episode, cfg.rss_url or "")
                pipeline_metrics.update_episode_status(
                    episode_id=episode_id,
                    status="failed",
                    stage="metadata",
                    error_type=type(exc).__name__,
                    error_message=redact_for_log(str(exc), max_len=500),
                )
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
                    # Issue #429: stop on first failure or after N failures (Phase 2)
                    fail_fast = getattr(cfg, "fail_fast", False)
                    max_failures = getattr(cfg, "max_failures", None)
                    if fail_fast or (
                        max_failures is not None
                        and pipeline_metrics is not None
                        and pipeline_metrics.errors_total >= max_failures
                    ):
                        logger.info(
                            "Stopping processing: fail_fast=%s, max_failures=%s, errors_total=%s",
                            fail_fast,
                            max_failures,
                            pipeline_metrics.errors_total if pipeline_metrics else 0,
                        )
                        break
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
        logger.debug(
            "Concurrent processing completed: %s succeeded, %s failed "
            "(%s/%s total, parallelism=%s)",
            jobs_processed_ok,
            jobs_processed_failed,
            jobs_processed,
            total_jobs,
            max_workers,
        )
    else:
        logger.debug(
            "Concurrent processing completed: %s/%s jobs processed (parallelism=%s)",
            jobs_processed_ok,
            total_jobs,
            max_workers,
        )
