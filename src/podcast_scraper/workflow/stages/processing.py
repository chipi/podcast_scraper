"""Processing stage for episode download and preparation.

This module handles episode processing, download argument preparation,
and host detection.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import as_completed, Future, ThreadPoolExecutor
from typing import Any, Callable, cast, Dict, List, NamedTuple, Optional, Set, Tuple, TYPE_CHECKING

from ... import config, models

if TYPE_CHECKING:
    from ...models import Episode, RssFeed
else:
    Episode = models.Episode  # type: ignore[assignment]
    RssFeed = models.RssFeed  # type: ignore[assignment]
from ...rss import BYTES_PER_MB, http_head, OPENAI_MAX_FILE_SIZE_BYTES
from ...utils.log_redaction import format_exception_for_log, redact_for_log
from ...utils.optional_deps import caused_by_missing_import
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


#: Fallback wall-clock ceiling for the parallel processing loop, in seconds.
#: Chosen as roughly 2x the longest legitimate run observed in prod (a 36-episode job takes
#: ~2h) so it never truncates real work, while still bounding the pathological case. The
#: sharper bound is main-thread liveness — this is the backstop for when the main thread is
#: alive but the loop can no longer make progress.
DEFAULT_PROCESSING_LOOP_BUDGET_SECONDS = 4 * 60 * 60


def _transcript_word_count(transcript_path: Optional[str]) -> int:
    """Word count of the transcript, or 0 when it cannot be read (#1920).

    Sizes the metadata-generation deadline. Never raises and never blocks the episode: an
    unreadable transcript yields 0, which makes the caller fall back to the flat configured
    budget — the pre-#1920 behaviour. Reading a ~100 KB text file is free next to a stage that
    routinely runs several hundred seconds.
    """
    if not transcript_path:
        return 0
    try:
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as fh:
            return len(fh.read().split())
    except OSError:
        return 0


def _processing_loop_budget_seconds(cfg: Any, max_workers: int) -> Optional[float]:
    """Wall-clock ceiling for ``_run_parallel_processing_loop``.

    Returns ``None`` to disable the bound entirely (explicit opt-out only). Set
    ``processing_loop_budget_seconds: 0`` in config to disable; any positive value
    overrides the default.
    """
    override = getattr(cfg, "processing_loop_budget_seconds", None)
    if override is not None:
        try:
            value = float(override)
        except (TypeError, ValueError):
            return float(DEFAULT_PROCESSING_LOOP_BUDGET_SECONDS)
        return None if value <= 0 else value
    return float(DEFAULT_PROCESSING_LOOP_BUDGET_SECONDS)


def _processing_job_key(job: Any) -> str:
    """Unique bookkeeping identity for one ProcessingJob.

    2026-08-25 incident (prod batch repair, 29 episodes): ``episode.idx`` is NOT
    unique across a multi-run work-list — a reprocess assigns each episode the
    idx from its on-disk ``NNNN - Title`` filename, which is only unique within
    the ORIGINAL ingest run. 29 episodes drawn from two 16-episode source runs
    shared idx values 1..16; keying the processed-set by idx then (a) silently
    skipped 13 episodes as "already processed" and (b) wedged the processing
    loop forever, because ``total_jobs == len(processed_idx_set)`` could never
    hold (29 != 16). The transcript path is unique by construction — it is the
    artifact identity the stage actually operates on. idx remains for display
    and for the on-disk ``{idx} - *`` glob contract, never for dedup.
    """
    return str(job.transcript_path)


def _mark_processed(processed_job_keys: Set[str], job: Any) -> None:
    """Record an episode as finished — ONCE — and emit a ``pipeline_progress`` event (ADR-119).

    The pipeline is an EVENTS source, not a scraped metrics target (an ephemeral ``run --rm``
    container has nothing to scrape). So per-run progress is an ``emit_event`` fact, not a
    gauge: it reaches VictoriaLogs the same way in both envs (dev pushes it; prod's Alloy tails the
    runner stdout) and Grafana charts it by ``run_id``. Idempotent (no double-emit); never raises.
    Keyed by :func:`_processing_job_key`, NOT ``episode.idx`` (see there for the incident).
    """
    key = _processing_job_key(job)
    if key in processed_job_keys:
        return
    processed_job_keys.add(key)
    try:
        from ...obs.events import emit_event

        emit_event("pipeline_progress", episodes_done=len(processed_job_keys))
    except Exception:  # noqa: BLE001 — telemetry must never break the run
        pass


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


from ...providers.resilience import ResilienceFuseOpenError
from ...speaker_detectors.corroboration import corroborate_guests
from ...speaker_detectors.factory import create_speaker_detector
from ...speaker_detectors.hosts import (
    detect_hosts_from_feed,
    hosts_from_feed_statement,
    is_network_or_org_author,
    normalize_host_names,
)
from ..cost_monitoring import CostCapExceeded
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


def _sleep_and_tally_idle(pm: Optional[Any], seconds: float) -> None:
    """Sleep then record the elapsed time as processing-thread queue-idle (#1180).

    Extracted so ``process_processing_jobs_concurrent`` stays within its
    cognitive-complexity budget. A no-op on the metrics side when ``pm`` is
    ``None``.
    """
    start = time.monotonic()
    time.sleep(seconds)
    if pm is not None:
        pm.record_processing_queue_idle_time(time.monotonic() - start)


def _time_processing_job(
    pm: Optional[Any],
    job: ProcessingJob,
    run: "Callable[[ProcessingJob], bool]",
) -> bool:
    """Wrap a per-episode processing invocation with #1180 instrumentation.

    Records: (a) the ProcessingProcessor thread's active interval around the
    call, (b) an inline-processed episode counter, and (c) the handoff latency
    from ``job.queued_at`` to now. Kept as a module-level helper so both the
    sequential and parallel call sites stay inside their cognitive-complexity
    budgets — see #1180 audit.
    """
    start = time.monotonic()
    if pm is not None and job.queued_at is not None:
        pm.record_handoff_latency(start - job.queued_at)
    result = run(job)
    if pm is not None:
        pm.record_processing_thread_active(start, time.monotonic())
        pm.record_inline_processed_episode()
    return result


def _enforce_cost_soft_cap_after_episode(
    cfg: config.Config, pipeline_metrics: Optional[metrics.Metrics]
) -> None:
    """Raise :class:`cost_monitoring.CostCapExceeded` when abort action is configured (#804)."""
    from ..cost_monitoring import enforce_cost_soft_cap

    enforce_cost_soft_cap(cfg, pipeline_metrics)


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
    if any(tok in msg for tok in ("timeout", "connection", "reset", "429", "503")):
        return True
    # NOTE: a structured-summary parse failure is NOT episode-retryable. Content-retry lives at
    # exactly one layer — the call (ADR-148): a transient invalid structured response gets one
    # bounded in-place re-roll on the same provider inside `_generate_episode_summary`, then
    # fallover, then fail. Re-running the whole episode (transcribe/diarize/GI/KG) to fix one bad
    # LLM call is the wrong layer; the earlier `"summary schema parsing failed"` string-match here
    # (commit 57ee206a) is replaced by that call-level re-roll.
    return False


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
    # #1053 / o11y: bind the episode id for THIS worker so the download stage's logs + incidents
    # carry it (the inline path previously never set it — only the safety-net summarizer did).
    from ...utils import correlation, otel_init
    from ..helpers import get_episode_id_from_episode

    episode = args[0]
    try:
        _corr_ep_id, _ = get_episode_id_from_episode(episode, cfg.rss_url or "")
    except Exception:  # never block processing on correlation
        _corr_ep_id = None

    # episode_scope binds the id ContextVar; episode_span opens the root OTEL span that parents the
    # provider HTTP spans and carries run/episode ids so traces are pivotable (both no-op when off).
    with (
        correlation.episode_scope(_corr_ep_id),
        otel_init.episode_span(
            run_id=correlation.get_run_id(),
            episode_id=_corr_ep_id,
            feed_id=getattr(cfg, "rss_url", None),
        ),
    ):
        return _process_episode_with_retry_inner(process_fn, args, cfg, pipeline_metrics, episode)


def _process_episode_with_retry_inner(
    process_fn: Any,
    args: Tuple,
    cfg: "config.Config",
    pipeline_metrics: "metrics.Metrics",
    episode: Any,
) -> _EpisodeResult:
    """Retry body for :func:`_process_episode_with_retry` (episode id already bound by caller)."""
    max_retries = getattr(cfg, "episode_retry_max", 0)
    if max_retries <= 0:
        result: _EpisodeResult = process_fn(*args, pipeline_metrics=pipeline_metrics)
        return result

    delay = getattr(cfg, "episode_retry_delay_sec", 5.0)
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
    # Statement-first, matching the real run (audit F4): the feed's "Hosted by ..." blurb is pure
    # regex (no ML — safe in dry-run), so an org-authored feed whose description names its hosts
    # previews the same hosts the real run would find, instead of reporting none. NER is skipped
    # here (it needs the model dry-run deliberately avoids); author tags are the second source.
    cached_hosts: set[str] = set(hosts_from_feed_statement(feed.title, feed.description))
    if not cached_hosts and feed.authors:
        cached_hosts = {a for a in feed.authors if not is_network_or_org_author(a)}
    if cached_hosts:
        logger.info(
            "DETECTED HOSTS (dry-run, feed statement / author tags): %s",
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
    # Statement-first order of authority (ADR-130 / Fable-5 audit F2). The deterministic parser
    # reads the feed's own "Hosted by ..." blurb out of the description, then org-filtered author
    # tags, then NER. EVERY LLM provider's detect_hosts short-circuits on RSS author tags — it
    # returns set(feed_authors) verbatim and never reads the description — so calling it first
    # mis-anchors org-authored feeds (the org is returned, then stripped to nothing) AND overrides a
    # feed whose description names hosts different from a personal author tag. Consult the provider
    # only when the deterministic parse finds nothing (it may still add an LLM-NER hit).
    stated = detect_hosts_from_feed(feed.title, feed.description, feed.authors or [])
    if stated:
        return stated
    try:
        feed_hosts = speaker_detector.detect_hosts(
            feed_title=feed.title,
            feed_description=feed.description,  # show blurb usually names the host (#1169)
            feed_authors=feed.authors if feed.authors else None,
        )
    except Exception as exc:
        if not caused_by_missing_import(exc):
            raise
        # A missing optional PACKAGE degrades; it does not end the run — the same rule the two
        # sites below already apply (utils.optional_deps). This one was missed, and it is reached
        # EARLIER than either: `_setup_pipeline_resources` calls it before any episode is touched,
        # so a box without the [ml] extra died at pipeline startup with a bare
        # ``ModuleNotFoundError: No module named 'spacy'`` out of ``run_pipeline`` — 12 e2e tests,
        # all of them describing error-recovery behaviour, none of them about spaCy.
        #
        # The deterministic parse above already ran and found nothing stated, so the honest result
        # is "no host inferred": whatever the episode metadata states is what it keeps. Host
        # detection is an enhancement, and the pipeline has a no-detector path
        # (``_create_speaker_detector_if_needed`` returning None takes it) — this is that path,
        # reached one step later.
        #
        # LOUDLY, per #1647: a feed whose hosts were never looked for must stay distinguishable
        # from one where the detector ran and honestly found nobody.
        logger.error(
            "Feed host detection UNAVAILABLE — an optional dependency is not installed (%s: %s). "
            "No host was inferred from the feed; the show keeps whatever its metadata states. "
            "Install the [ml] extra to enable it.",
            type(exc).__name__,
            exc,
        )
        return set()
    return _sanitize_detected_hosts(cast("set[str]", feed_hosts))


def _sanitize_detected_hosts(names: set[str]) -> set[str]:
    """Put a provider's host names through the same filter the deterministic path uses.

    The deterministic branch above splits multi-person strings (``split_author_names``) and
    rejects organisations (``is_network_or_org_author``). The provider branch did neither, and
    an LLM asked "who hosts this show?" answers in prose: on *The a16z Show* it returned the
    single string ``"Erik Torenberg, Ben Horowitz, Travis Kalanick"``, which became one
    ``Person`` node — ``person:erik-torenberg-ben-horowitz-travis-kalanick``, a human being who
    does not exist, anchoring the roster for the whole episode.

    A composite is worse than no host at all. It can never match a diarized voice (the roster
    compares per name), so it silently disables the known-hosts anchor *and* pollutes the graph
    with a fake entity that cross-episode queries will happily join on.

    Same conservative contract as ``split_author_names``: an over-eager split degrades to "no
    host", which is the safe direction (#876), never to an invented person.

    The rule itself now lives in :func:`~podcast_scraper.speaker_detectors.hosts.
    normalize_host_names`, shared with the episode-authors and config paths — fixing it here
    only, as the first attempt did, left the path that actually fired on a16z untouched.
    """
    out = normalize_host_names(names or set())
    if out != set(names or set()):
        logger.info(
            "host detection: provider names %s normalised to %s (split + org-filtered)",
            sorted(names or set()),
            sorted(out),
        )
    return out


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
    # Skip validation when the feed carries author tags: on such feeds the hosts came either from a
    # trusted author tag OR (post audit-F2, statement-first) from the feed's own "Hosted by ..."
    # statement — both authoritative, neither needing first-episode corroboration. Only NER-derived
    # hosts on a tag-less feed fall through to validation below.
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
        # normalize_host_names both SPLITS multi-person tags and applies the shared network/org
        # predicate. Filtering for orgs alone (what this did before) let *The a16z Show*'s
        # ``<itunes:author>Erik Torenberg, Ben Horowitz, Travis Kalanick</itunes:author>``
        # through whole — is_network_or_org_author returns False for it, since a three-person
        # string is neither a mononym nor org-marked — and this is the path that fired on the
        # acceptance run (#1652).
        episode_authors |= normalize_host_names(episode_author_list)

    return episode_authors


def _log_detected_hosts(
    cached_hosts: set[str],
    feed: RssFeed,  # type: ignore[valid-type]
    episode_authors: set[str],
    cfg: config.Config,
    source: Optional[str] = None,
) -> None:
    """Log detected hosts with their source.

    Args:
        cached_hosts: Set of detected host names
        feed: Parsed RssFeed object
        episode_authors: Set of episode-level authors
        cfg: Configuration object
        source: The branch that actually produced ``cached_hosts``. The caller knows this for
            certain; when omitted it is inferred, which is strictly a guess (see
            :func:`_infer_host_source`).
    """
    if not cached_hosts:
        if getattr(cfg, "auto_speakers", False):
            logger.debug(
                "No hosts detected from feed metadata, episode-level authors, or config known_hosts"
            )
        return

    if source is None:
        source = _infer_host_source(cached_hosts, feed, episode_authors, cfg)
    logger.info("DETECTED HOSTS (from %s): %s", source, ", ".join(sorted(cached_hosts)))


def _infer_host_source(
    cached_hosts: set[str],
    feed: RssFeed,  # type: ignore[valid-type]
    episode_authors: set[str],
    cfg: config.Config,
) -> str:
    """Best-effort guess at which branch produced ``cached_hosts``.

    Only for callers that cannot say. ``detect_feed_hosts_and_patterns`` passes ``source``
    explicitly, because inference here got it wrong in the way that costs debugging time:
    a non-empty ``feed.authors`` does NOT mean the hosts came from it — on an org-authored feed
    those tags are stripped as publisher metadata and the names arrive from the episode-authors
    fallback. Testing ``feed.authors`` first labelled exactly that case "RSS author tags", which
    aimed the a16z composite investigation at the wrong path for a full cycle (#1652).

    So: check the specific sources against what was actually produced, and only then fall back.
    """
    known = list(getattr(cfg, "known_hosts", None) or [])
    if episode_authors and cached_hosts == episode_authors:
        return "episode-level authors"
    # Both spellings: ``cached_hosts`` holds NORMALISED names, so a config entry that needed
    # splitting no longer equals its raw form — comparing only raw would mislabel it.
    if known and cached_hosts in (set(known), normalize_host_names(known)):
        return "config known_hosts (fallback)"
    if feed.authors:
        return "RSS author tags"
    return "feed metadata (NER)"


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
    # Strip network/publisher author tags the detector surfaces as hosts (e.g. "Colossus",
    # "Colossus | Investing & Business Podcasts"). For such shows the real host comes from the
    # transcript self-introduction at diarization time, not the feed metadata (#876). The
    # statement-first ordering inside _detect_hosts_from_feed means an org-authored feed whose
    # description names its hosts is already recovered before this strip runs (ADR-130 / audit F2).
    feed_hosts = {h for h in feed_hosts if not is_network_or_org_author(h)}

    # Priority: Use known_hosts from config if provided (show-level override)
    if cfg.known_hosts:
        # Operator-supplied, but not exempt: a composite entry ("A, B and C") in a show config
        # is just as unmatchable against a diarized voice as one from a feed, and would mint the
        # same fake Person. Normalising here keeps every seeding path on one rule (#1652).
        known_hosts_set = normalize_host_names(cfg.known_hosts)
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

    # Track which branch actually produced the hosts, rather than re-deriving it later from
    # circumstantial evidence — that inference is what mislabelled a16z's episode-author hosts
    # as "RSS author tags" (#1652).
    host_source: Optional[str] = None

    # Fallback to episode-level authors if no feed-level hosts found (Issue #380)
    episode_authors: set[str] = set()
    if not cached_hosts:
        episode_authors = {
            a
            for a in _fallback_to_episode_authors(cfg, episodes)
            if not is_network_or_org_author(a)
        }
        if episode_authors:
            cached_hosts = episode_authors
            host_source = "episode-level authors"
            logger.info(
                "DETECTED HOSTS (from episode-level authors): %s",
                ", ".join(sorted(cached_hosts)),
            )

    # Fallback to known_hosts from config if no hosts detected (show-level override)
    if not cached_hosts and cfg.known_hosts:
        cached_hosts = normalize_host_names(cfg.known_hosts)
        host_source = "config known_hosts (fallback)"
        logger.info(
            "DETECTED HOSTS (from config known_hosts fallback): %s",
            ", ".join(sorted(cached_hosts)),
        )

    # Log detected hosts with their source
    _log_detected_hosts(cached_hosts, feed, episode_authors, cfg, source=host_source)

    # Analyze patterns from first few episodes to extract heuristics
    if cfg.auto_speakers and episodes:
        try:
            heuristics_dict = speaker_detector.analyze_patterns(
                episodes=episodes, known_hosts=cached_hosts
            )
        except Exception as exc:
            if not caused_by_missing_import(exc):
                raise
            # A missing optional PACKAGE degrades; it does not end the run. THIS is the site that
            # actually killed things: `analyze_patterns` -> `_initialize_spacy` -> `import spacy`
            # runs once per FEED, before any episode is processed, so a missing spaCy took down
            # the whole run here — several stages before per-episode speaker detection could even
            # be reached.
            #
            # Building the detector already tolerated this and logged "speaker detection will be
            # unavailable"; this call then contradicted it. Unlike ffmpeg (#26, deliberately FATAL
            # because preprocessing decides whether a transcript is correct at all), pattern
            # analysis only produces title/description HEURISTICS — the pipeline runs without them
            # and `auto_speakers=False` skips them entirely by design.
            #
            # Loud, not silent: heuristics stay None and the run continues with metadata-stated
            # names only.
            logger.error(
                "Speaker pattern analysis UNAVAILABLE — an optional dependency is not installed "
                "(%s: %s). Continuing without title/description heuristics; episodes keep the "
                "names their metadata states. Install the [ml] extra to enable it.",
                type(exc).__name__,
                exc,
            )
            heuristics_dict = None
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


class EpisodeSizeSkip(NamedTuple):
    """Advisory about the media file's size relative to the transcription upload limit.

    ``skip_speaker_detection`` is retained at ``False`` (#1646). It used to be set ``True`` for
    any episode whose audio exceeded 25 MB, which disabled speaker detection — a stage that
    reads the episode TITLE and DESCRIPTION and never opens the media file. The field stays so
    callers keep a stable shape and so the ledger can still record a caller-requested skip, but
    the size gate no longer sets it.
    """

    skip_speaker_detection: bool
    skip_episode: bool
    reason: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None
    # Advisory only: the PUBLISHED media exceeds the upload limit. This says nothing about
    # whether the *uploaded* file will, because preprocessing runs in between and typically
    # reduces the file by ~90 %. Kept separate from the skip flags precisely so it can never
    # gate an unrelated stage again.
    media_oversize: bool = False


_NO_SIZE_SKIP = EpisodeSizeSkip(False, False)


def _check_episode_size_skip(
    cfg: config.Config,
    episode: Episode,  # type: ignore[valid-type]
) -> EpisodeSizeSkip:
    """Report whether the media exceeds the transcription upload limit. Advisory only.

    **#1646 — what changed and why.** This gate used to return
    ``skip_speaker_detection=True`` for any episode over ``OPENAI_MAX_FILE_SIZE_BYTES``
    (25 MB). Three things were wrong with that, and they compounded:

    1. **Wrong stage.** The limit is an upload cap for *transcription*. What it disabled was
       *speaker detection* — ``detect_speaker_names(episode_title=…, episode_description=…)``,
       which reads text metadata and never touches the audio. The size of the MP3 is
       irrelevant to it.
    2. **Wrong provider.** ``deepgram`` was added to the provider tuple in f846c502
       (2026-06-05) and inherited a cap that belongs to OpenAI Whisper. Deepgram has no 25 MB
       limit, and ``cloud_balanced`` transcribes with Deepgram.
    3. **Dead premise.** The guard came from #327 — *"skip speaker detection when
       transcription will be skipped due to file size limits"*. Transcription is no longer
       skipped: this function returns ``skip_episode=False`` on every path. Only the
       speaker-detection half of that decision was still firing, long after the condition it
       depended on stopped being true.
    4. **Wrong file.** This is the one that makes the other three worse, and it was missed in
       the first pass at #1646. The measurement is an HTTP ``HEAD`` on ``episode.media_url``
       — the size of the file the *publisher* serves. The 25 MB cap applies to what gets
       *uploaded*, and between those two points the pipeline preprocesses the audio (mono,
       16 kHz, silence-stripped, and for the API providers a bitrate ladder that steps down
       until the file is under ``_PREPROCESSING_API_REENCODE_TARGET_BYTES``). Measured on the
       acceptance corpus that is a consistent **~90 % reduction**: 91.5 MB → 9.1 MB,
       105.6 MB → 10.6 MB. So episodes were being judged against a cap using a number taken
       before the step that makes the number irrelevant — the uploaded files were never near
       the limit.

    Measured cost before the fix: 488 of 678 episodes (72 %) had speaker detection skipped;
    2,112 of 8,952 insights (23.6 %) became unsurfaceable; 82 episodes lost every insight.
    Given (4), essentially none of those episodes were ever too large to transcribe — the
    transcription they were "protected" from would have succeeded.

    The probe is kept because an operator still wants to see that a published file is large
    (#557), but it is advisory, it gates nothing, and it must not claim anything about the
    uploaded size — that is only knowable after preprocessing has run.
    """
    if (
        cfg.dry_run
        or not cfg.transcribe_missing
        or cfg.transcription_provider not in ("openai", "gemini", "mistral", "deepgram")
        or not episode.media_url
    ):
        return _NO_SIZE_SKIP
    resp = http_head(episode.media_url, cfg.user_agent, cfg.timeout)
    if not resp:
        return _NO_SIZE_SKIP
    content_length = resp.headers.get("Content-Length")
    if not content_length:
        return _NO_SIZE_SKIP
    try:
        file_size_bytes = int(content_length)
    except (ValueError, TypeError):
        return _NO_SIZE_SKIP
    if file_size_bytes <= OPENAI_MAX_FILE_SIZE_BYTES:
        return _NO_SIZE_SKIP
    file_size_mb = file_size_bytes / BYTES_PER_MB
    provider_labels = {
        "openai": "OpenAI",
        "gemini": "Gemini",
        "mistral": "Mistral",
        "deepgram": "Deepgram",
    }
    provider_name = provider_labels.get(cfg.transcription_provider, cfg.transcription_provider)
    preprocessing_on = bool(getattr(cfg, "preprocessing_enabled", False))
    detail: Dict[str, Any] = {
        # Named ``published_media_bytes``, not ``media_bytes``: this is the publisher's file,
        # NOT the file that gets uploaded. Conflating the two is what made the old ledger
        # entry misleading about every large episode.
        "published_media_bytes": file_size_bytes,
        "limit_bytes": OPENAI_MAX_FILE_SIZE_BYTES,
        "limit_applies_to": "uploaded_audio_after_preprocessing",
        "preprocessing_enabled": preprocessing_on,
        "transcription_provider": cfg.transcription_provider,
        "has_transcript_urls": bool(episode.transcript_urls),
    }
    # Advisory only (#1646). Reports what it actually measured — the PUBLISHED size — and
    # does not predict the uploaded size, which is only known after preprocessing. The
    # previous wording claimed "transcription will chunk after preprocess"; that was false in
    # the normal case, because preprocessing puts the file well under the cap and nothing
    # chunks. Speaker detection is unaffected either way: it never reads the audio.
    if preprocessing_on:
        logger.info(
            "[%d] Published media is %.1f MB, over the %s upload limit (25 MB). Preprocessing "
            "runs before upload and normally brings it well under; the cap applies to the "
            "preprocessed file, not this one. Speaker detection is unaffected (title + "
            "description).",
            episode.idx,
            file_size_mb,
            provider_name,
        )
    else:
        logger.warning(
            "[%d] Published media is %.1f MB, over the %s upload limit (25 MB), and "
            "preprocessing is DISABLED — the upload may genuinely exceed the cap. Speaker "
            "detection is unaffected (title + description).",
            episode.idx,
            file_size_mb,
            provider_name,
        )
    return EpisodeSizeSkip(
        skip_speaker_detection=False,
        skip_episode=False,
        reason=None,
        detail=detail,
        media_oversize=True,
    )


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


class DetectedSpeakers(NamedTuple):
    """What the detector found, and what SURVIVED corroboration — they are not the same list.

    ``guests`` is what may be painted onto a voice. ``stated`` is every name the episode metadata
    put forward, *including* the ones corroboration rejected, and it names nobody on its own.

    Keeping the rejects is what stops us laundering our own failures. When corroboration threw a
    real guest away ("the episode text names them but never introduces them as speaking"), the name
    disappeared entirely — so the roster saw no name going spare and concluded that *nobody could
    have named* the voice, filing a 35%-of-the-episode guest as "Unidentified speaker". The name
    was right there in the show notes. We could not place it, and that is a defect, not innocence.
    """

    guests: List[str]
    stated: List[str]


def _record_processing_incident(
    cfg: config.Config,
    job: Any,
    effective_output_dir: str,
    *,
    category: str,
    message: str,
    exception_type: str,
    stage: str,
) -> None:
    """Append one episode-scoped row to ``corpus_incidents.jsonl``.

    Neither notable outcome of the metadata stage used to leave a durable trace. The
    acceptance run's two slowest episodes produced not a single row, so the batch rollup
    reported zero incidents for feeds that had each spent over 20 minutes past budget; and the
    genuine-failure path recorded a status inside the run's own metrics and nothing else. In
    both cases the ERROR log is in-flight only and survives into no artifact, so the corpus
    summary read clean.

    Best-effort by construction: an incident write must never change the fate of the episode
    it is describing.
    """
    try:
        from ...utils.corpus_incidents import append_corpus_incident
        from ..helpers import get_episode_id_from_episode

        path = (getattr(cfg, "incident_log_path", None) or "").strip()
        if not path:
            path = os.path.join(effective_output_dir, "corpus_incidents.jsonl")
        episode = getattr(job, "episode", None)
        episode_id = None
        if episode is not None:
            # Its own guard: id resolution reads the RSS item and can raise on a malformed
            # feed. An incident WITHOUT an id is still worth having — losing the whole row
            # because the label could not be computed is how these went unrecorded in the
            # first place.
            try:
                episode_id, _ = get_episode_id_from_episode(episode, cfg.rss_url or "")
            except Exception:
                logger.debug("could not resolve episode id for incident", exc_info=True)
        append_corpus_incident(
            path,
            scope="episode",
            category=category,  # type: ignore[arg-type]
            message=message,
            exception_type=exception_type,
            stage=stage,
            feed_url=getattr(cfg, "rss_url", None),
            episode_id=episode_id,
            episode_idx=int(getattr(episode, "idx", 0) or 0),
        )
    except Exception:  # pragma: no cover - observability must never fail the episode
        logger.debug("could not record processing incident", exc_info=True)


def _record_summarization_overrun_incident(
    cfg: config.Config,
    job: Any,
    effective_output_dir: str,
    exc: BaseException,
) -> None:
    """An episode whose metadata generation blew its deadline but still finished.

    ``soft`` rather than ``policy``: a policy row means a documented, by-design skip (an API
    audio limit), whereas this is an anomaly worth chasing — the episode succeeded, but at a
    cost that says the budget or the workload needs attention.
    """
    _record_processing_incident(
        cfg,
        job,
        effective_output_dir,
        category="soft",
        message=(
            "Metadata generation exceeded its summarization deadline but COMPLETED; "
            "results kept (episode is not a failure)"
        ),
        exception_type="DeadlineExceededButCompleted",
        stage="summarization",
    )


def _record_metadata_failure_incident(
    cfg: config.Config,
    job: Any,
    effective_output_dir: str,
    exc: BaseException,
) -> None:
    """An episode whose metadata generation genuinely raised.

    This path recorded ``status=failed`` and nothing else. The status is real, but it lives
    only inside the run's own metrics — it never reached ``corpus_incidents.jsonl``, so the
    batch rollup counted zero incidents and the feed still reported ``ok: true``. An operator
    reading the corpus summary saw a clean run with one fewer episode and no reason given,
    which is the same silence that made the deadline bug take a full investigation to find.

    ``hard``: nothing about an unexpected exception is by design.
    """
    _record_processing_incident(
        cfg,
        job,
        effective_output_dir,
        category="hard",
        message=f"Metadata generation failed: {format_exception_for_log(exc)}",
        exception_type=type(exc).__name__,
        stage="metadata",
    )


def _record_naming_cost(pipeline_metrics: Any, cost_probe: Any, episode_idx: int) -> None:
    """Attribute the probe's captured naming cost to this episode.

    Only called once detection has actually run, so a recorded ``0.0`` means measured-and-free
    (the deterministic detector made no LLM call) — distinct from no entry at all, which means
    detection never ran and the cost is genuinely unknown. Best-effort: a metrics object without
    the recorder (older callers pass a plain object) must not fail the episode.
    """
    if pipeline_metrics is None or cost_probe is None:
        return
    recorder = getattr(pipeline_metrics, "record_speaker_detection_cost", None)
    if not callable(recorder):
        return
    try:
        recorder(float(cost_probe.speaker_detection_cost_usd), episode_idx)
    except Exception:  # pragma: no cover - metrics must never break processing
        logger.debug("[%s] could not record speaker detection cost", episode_idx)


def _detect_speakers_for_episode(
    episode: Episode,  # type: ignore[valid-type]
    cfg: config.Config,
    host_detection_result: HostDetectionResult,
    pipeline_metrics: metrics.Metrics,
    skip_speaker_detection: bool = False,
    skip_reason: Optional[str] = None,
    skip_detail: Optional[Dict[str, Any]] = None,
) -> Optional[DetectedSpeakers]:
    """Run speaker detection for one episode; return corroborated guests + every stated name.

    Every exit path records a stage outcome (#1647). Before that, three of these returns were
    silent and left no trace anywhere — no timing, no log, no error — so an episode whose
    speakers were never detected looked exactly like one that had none to detect. That is the
    ambiguity that let #1646 run unnoticed across 72 % of the corpus.
    """

    def _record(outcome: str, reason: Optional[str] = None, **kwargs: Any) -> None:
        if pipeline_metrics is not None:
            pipeline_metrics.record_stage_outcome(
                "speaker_detection", episode.idx, outcome, reason=reason, **kwargs
            )

    if not cfg.auto_speakers:
        if cfg.screenplay_speaker_names and len(cfg.screenplay_speaker_names) > 1:
            _record("skipped", "auto_speakers_disabled_using_configured_names")
            return DetectedSpeakers(guests=cfg.screenplay_speaker_names[1:], stated=[])
        _record("skipped", "auto_speakers_disabled")
        return None
    logger.debug("Episode %d: %s", episode.idx, episode.title)
    # One check, not two: this condition was tested again below with an identical body, so the
    # second test was unreachable. Collapsed while adding the ledger (#1647).
    if skip_speaker_detection:
        # WARNING, not silence: the caller decided to skip, and the consequence is that no
        # voice on this episode can be named. See #1646 for what that costs downstream.
        logger.warning(
            "[%s] Speaker detection SKIPPED (%s) — no voice on this episode can be named, "
            "so its insights will be marked unsurfaceable.",
            episode.idx,
            skip_reason or "reason not recorded",
        )
        _record("skipped", skip_reason or "skip_requested_by_caller", detail=skip_detail)
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
        _record("skipped", "dry_run")
        return None
    episode_description = extract_episode_description(episode.item)
    extract_names_start = time.time()
    speaker_detector = _get_speaker_detector(host_detection_result, cfg)
    if not speaker_detector:
        logger.warning(
            "[%s] Speaker detection SKIPPED — no speaker detector could be constructed "
            "(provider=%s).",
            episode.idx,
            getattr(cfg, "speaker_detector_provider", None),
        )
        _record(
            "skipped",
            "no_speaker_detector_available",
            detail={"speaker_detector_provider": getattr(cfg, "speaker_detector_provider", None)},
        )
        return None
    cached_hosts = host_detection_result.cached_hosts if cfg.cache_detected_hosts else set()
    # Per-episode seeding — a FIFTH path into known_hosts, found by the structural test rather
    # than by reading, and the reason that test exists. An un-normalised composite here reaches
    # the detector as the roster for every episode, which is where the fake Person is minted.
    combined_hosts = (
        normalize_host_names(cfg.known_hosts) | cached_hosts if cfg.known_hosts else cached_hosts
    )
    import inspect

    # Isolate THIS episode's naming cost. Providers record speaker-detection cost onto a
    # run-level accumulator shared by parallel episodes, so a delta on it is racy and cannot be
    # attributed to one episode — which is why naming.cost_usd was absent from every manifest of
    # the acceptance run while the run total kept climbing. The probe forwards everything to the
    # real metrics object (run totals stay correct) and captures this episode's share on the side.
    from ..processing_manifest import EpisodeCostProbe

    cost_probe = EpisodeCostProbe(pipeline_metrics) if pipeline_metrics is not None else None
    detect_metrics = cost_probe if cost_probe is not None else pipeline_metrics

    sig = inspect.signature(speaker_detector.detect_speakers)
    # A raising detector previously recorded NOTHING — no ledger entry at all — so an episode
    # whose speaker detection blew up was indistinguishable from one where it never ran. That
    # silence is the #1646 shape, and it is also why ``failed`` ended up being misused for the
    # empty-result path below: there was no real failure path competing for the word.
    #
    # Control flow is deliberately unchanged: record, then re-raise.
    try:
        if "pipeline_metrics" in sig.parameters:
            detected_speakers, detected_hosts_set, detection_succeeded, _ = (
                speaker_detector.detect_speakers(
                    episode_title=episode.title,
                    episode_description=episode_description,
                    known_hosts=combined_hosts,
                    pipeline_metrics=detect_metrics,
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
    except Exception as exc:
        # Cost is recorded even here: a detector that raised after its LLM call still spent the
        # money, and a manifest that omits it under-reports the episode's true cost.
        _record_naming_cost(pipeline_metrics, cost_probe, episode.idx)
        if caused_by_missing_import(exc):
            # A missing optional PACKAGE degrades; it does not end the run.
            #
            # The pipeline already builds the detector defensively — a missing spaCy is caught
            # there and logged as "speaker detection will be unavailable" — and then this path
            # killed the run anyway, several stages later. The code announced a degrade it did
            # not perform. That was never a decision: unlike ffmpeg (#26, deliberately FATAL
            # because preprocessing decides whether a transcript is correct at all), speaker
            # detection is an enhancement over episode metadata and the pipeline already has a
            # no-detector path — ``auto_speakers=False`` returns immediately.
            #
            # LOUDLY, though. `degraded`, not `ran`, and its own reason slug: an episode whose
            # speakers were never detected must stay distinguishable from one where the detector
            # ran and honestly found nobody. That distinction is the whole point of #1647, and a
            # silent skip would destroy it as surely as a crash.
            #
            # ``caused_by_missing_import`` is the SAME walker preload uses (95be1ec1) — see
            # utils.optional_deps. Only a missing package qualifies; a missing model file, a
            # gated token, a timeout or a bug still raises below.
            logger.error(
                "[%s] Speaker detection UNAVAILABLE — an optional dependency is not installed "
                "(%s: %s). The episode keeps whatever names its metadata states; no speaker was "
                "inferred. Install the [ml] extra to enable it.",
                episode.idx,
                type(exc).__name__,
                exc,
            )
            _record(
                "degraded",
                "speaker_detector_package_missing",
                detail={
                    "exception": type(exc).__name__,
                    "speaker_detector_provider": getattr(cfg, "speaker_detector_provider", None),
                },
                duration_seconds=time.time() - extract_names_start,
            )
            return None
        _record(
            "failed",
            "detector_raised",
            detail={
                "exception": type(exc).__name__,
                "speaker_detector_provider": getattr(cfg, "speaker_detector_provider", None),
            },
            duration_seconds=time.time() - extract_names_start,
        )
        raise
    elapsed = time.time() - extract_names_start
    _record_naming_cost(pipeline_metrics, cost_probe, episode.idx)
    if pipeline_metrics is not None:
        pipeline_metrics.record_extract_names_time(elapsed, episode.idx)
    if (
        not detection_succeeded
        and cfg.screenplay_speaker_names
        and len(cfg.screenplay_speaker_names) >= 2
    ):
        # The detector ran and came back empty; configured names are standing in for it.
        _record("degraded", "detection_failed_using_configured_names", duration_seconds=elapsed)
        return DetectedSpeakers(guests=cfg.screenplay_speaker_names[1:], stated=[])
    if not detection_succeeded:
        # RAN, not failed. ``detection_succeeded`` is ``bool(hosts or guests)``
        # (speaker_detectors/detection.py) — an EMPTINESS flag, not an error flag. Nothing
        # raised: the detector read the metadata and correctly found no names, which on a feed
        # that states no hosts is the designed outcome (#876 — NER on a description returns the
        # people an episode is ABOUT, so guessing is what put an advertiser's name on a show).
        #
        # Recording that as ``failed`` was a lie with two costs. A corpus report grouping by
        # outcome showed every host-less show as a permanent failure with nothing to fix; and
        # ``stage_did_run`` returns ``outcome in ("ran","degraded")``, so ``failed`` told the
        # roster the stage never ran — losing exactly the distinction between an UNMEASURED
        # voice and one measured as unnameable that #1647 exists to preserve.
        #
        # ``failed`` is reserved for a genuine exception, recorded by the caller's except path.
        _record("ran", "no_names_found_in_metadata", duration_seconds=elapsed)
    if detection_succeeded:
        flat_speakers: List[str] = []
        for entry in detected_speakers or []:
            flat_speakers.extend(_flatten_speaker_name_entries(entry))
        host_strings = _speaker_names_to_str_set(detected_hosts_set)
        proposed = [name for name in flat_speakers if name not in host_strings]

        # An LLM detector's name list is a PROPOSAL, not a result — it returns success=True whatever
        # it emits. Corroborate every proposed guest against the description before their name is
        # painted onto a diarized voice cluster.
        #
        # But keep the PROPOSAL as well. A rejected name is still a name the metadata stated, and
        # the roster needs to know it existed — otherwise a guest we could not place is filed as a
        # person nobody could have named.
        corroborated = corroborate_guests(
            proposed,
            episode_title=episode.title,
            episode_description=episode_description,
            known_hosts=host_strings | combined_hosts,
        )
        # Counts, not names: the ledger is a health signal, and a name list would make every
        # episode's record unbounded. `proposed` vs `corroborated` is the useful delta — a
        # detector proposing names that never survive corroboration is a distinct failure
        # from one that proposes nothing.
        _record(
            "ran",
            duration_seconds=elapsed,
            detail={
                "proposed_count": len(proposed),
                "corroborated_count": len(corroborated),
                "known_host_count": len(host_strings | combined_hosts),
            },
        )
        return DetectedSpeakers(guests=corroborated, stated=proposed)
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
        size_skip = _check_episode_size_skip(cfg, episode)
        if size_skip.skip_episode:
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
        detected = _detect_speakers_for_episode(
            episode,
            cfg,
            host_detection_result,
            pipeline_metrics,
            skip_speaker_detection=size_skip.skip_speaker_detection,
            skip_reason=size_skip.reason,
            skip_detail=size_skip.detail,
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
                list(detected.guests) if detected else None,
                list(detected.stated) if detected else None,
            )
        )

    # SELECTED N, PRODUCED ZERO — say so. A stage picks episodes, builds no work, and the run
    # still reports ok=1 and exits 0.
    #
    # SCOPE, honestly: this catches only the case where NO download arg was built at all. It
    # does NOT catch the `--pipeline-stage rederive_only` no-op, and an earlier version of this
    # comment wrongly claimed it did. That failure happens further downstream — `download_args`
    # is non-empty by the time `episode_processor`'s `if cfg.transcribe_missing and temp_dir:`
    # declines to enter the transcript-reuse path — so this guard is silent for it. Catching
    # that needs a counter after the download stage and is tracked on #1896.
    #
    # Only fires when the run EXPLICITLY asked for reprocessing. A normal incremental run where
    # everything is already ingested reaches here legitimately, and warning on that would make
    # the counter noise — non-zero on healthy nightly runs, which is how a signal gets ignored.
    reprocess_stages = {"rederive_only", "relabel_only", "rediarize_only"}
    asked_for_reprocess = (
        bool(getattr(cfg, "reprocess_existing_only", False))
        or str(getattr(cfg, "pipeline_stage", "full") or "full") in reprocess_stages
    )
    if episodes and not download_args and asked_for_reprocess:
        logger.warning(
            "SELECTED %d episode(s) for a REPROCESS but produced ZERO processing jobs — this "
            "run will do no work and still exit 0. pipeline_stage=%s reprocess_existing_only=%s "
            "transcribe_missing=%s skip_existing=%s",
            len(episodes),
            getattr(cfg, "pipeline_stage", "full"),
            getattr(cfg, "reprocess_existing_only", None),
            getattr(cfg, "transcribe_missing", None),
            getattr(cfg, "skip_existing", None),
        )
        update_metric_safely(pipeline_metrics, "selected_episodes_produced_no_jobs", 1)

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
                queued_at=time.monotonic(),  # #1180 handoff-latency stamp
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
        except CostCapExceeded:
            raise
        except ResilienceFuseOpenError:
            # ADR-122 item 3: a sustained fuse-open means the self-hosted endpoint is genuinely
            # down and — in reprocess mode — we do NOT fall over to another model. Continuing would
            # grind every remaining episode through the same hold-then-fail and yield a partial
            # corpus. Halt the batch (like the cost fuse above) so the operator can act; the run
            # resumes cleanly once the endpoint recovers.
            logger.error(
                "[%s] resilience fuse open (self-hosted endpoint down, reprocess mode) — halting "
                "the batch rather than grinding the rest of the corpus through a dead endpoint",
                episode.idx,
            )
            raise
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
                                queued_at=time.monotonic(),  # #1180 handoff-latency stamp
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
                    try:
                        _enforce_cost_soft_cap_after_episode(cfg, pipeline_metrics)
                    except CostCapExceeded:
                        raise
                    except Exception as cap_exc:
                        logger.error(
                            "cost soft cap check failed: %s",
                            format_exception_for_log(cap_exc),
                        )
                        raise
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
            except ResilienceFuseOpenError:
                # ADR-122 item 3: the self-hosted endpoint is down and reprocess mode never falls
                # over — halt the whole batch rather than failing every remaining future the same
                # way. Propagates up like the sequential loop's halt.
                logger.error(
                    "[%s] resilience fuse open (self-hosted endpoint down, reprocess mode) — "
                    "halting the batch",
                    episode_idx,
                )
                raise
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
    # Keyed by _processing_job_key (transcript path), NOT episode.idx — idx collides
    # across multi-run work-lists (2026-08-25 incident; see the helper's docstring).
    processed_job_indices: Set[str] = set()  # Track which jobs we've processed
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
                        if _processing_job_key(job) not in processed_job_indices:
                            _mark_processed(processed_job_indices, job)
                            return job
        else:
            with processed_job_indices_lock:
                for job in processing_resources.processing_jobs:
                    if _processing_job_key(job) not in processed_job_indices:
                        _mark_processed(processed_job_indices, job)
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
        abandoned_futures = [0]  # bounded-loop exit left these in flight

        # Supervision (2026-08-12 incident): this loop previously had no termination
        # guarantee. `_should_continue_processing` defaults to True, so if the main thread
        # died without setting `transcription_complete_event` — e.g. CostCapExceeded raised
        # out of `orchestration.check_cost_soft_cap_at_stage`, which sits in a region with
        # no try/finally — this thread span forever at 0.05s/iteration: live pid, 2.5% CPU,
        # zero progress, zero logs, indefinitely. Two independent bounds now apply:
        #   1. main-thread liveness — a worker must never outlive its parent
        #   2. a wall-clock budget   — nothing here may run unbounded
        loop_started_at = time.time()
        loop_budget_seconds = _processing_loop_budget_seconds(cfg, max_workers)

        def _supervision_exit_reason() -> Optional[str]:
            """Return a reason string when this loop must stop regardless of queue state."""
            if not threading.main_thread().is_alive():
                return "main thread exited"
            elapsed = time.time() - loop_started_at
            if loop_budget_seconds is not None and elapsed > loop_budget_seconds:
                return f"wall-clock budget exceeded ({elapsed:.0f}s > {loop_budget_seconds:.0f}s)"
            return None

        # NOT a `with` block: ThreadPoolExecutor.__exit__ calls shutdown(wait=True), which
        # blocks until every in-flight future finishes. On the supervision-abort path the
        # whole point is that a future may never finish, so `with` would reintroduce the
        # exact hang the bounds above exist to break. Shutdown mode is chosen in `finally`.
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            # Future -> episode idx, so a submit failure can report how many are still
            # in flight (see ``_try_submit``) without holding the jobs themselves.
            futures: Dict[Future, int] = {}

            def _try_submit(job: Any) -> bool:
                """Submit one job, tolerating a pool that can no longer accept work.

                Returns True when the job was scheduled. A submit during interpreter
                teardown raises ``RuntimeError: cannot schedule new futures after
                interpreter shutdown``; before 2026-08-12 that propagated out of the
                comprehension here and killed the entire run, discarding every episode
                still queued. The work itself was already defended (see
                ``_process_single_processing_job``) — only the scheduling was not.
                """
                _mark_processed(processed_job_indices, job)
                try:
                    future = executor.submit(process_job_func, job)
                except RuntimeError as exc:
                    # Un-mark so a resumed run re-submits this episode; skip_existing
                    # keeps that idempotent.
                    processed_job_indices.discard(_processing_job_key(job))
                    stop_requested[0] = True
                    logger.warning(
                        "Cannot schedule episode %s — executor no longer accepts work (%s). "
                        "Stopping submission; %d future(s) still in flight.",
                        job.episode.idx,
                        exc,
                        len(futures),
                    )
                    return False
                futures[future] = job.episode.idx
                return True

            def _submit_new_jobs() -> None:
                """Submit new jobs as they become available."""
                if stop_requested[0]:
                    return
                if processing_resources.processing_jobs_lock:
                    with processing_resources.processing_jobs_lock:
                        with processed_job_indices_lock:
                            for job in processing_resources.processing_jobs:
                                # Keyed by _processing_job_key, NOT episode.idx (idx collides
                                # across multi-run work-lists — see helper docstring). The old
                                # `idx not in futures` guard compared an int against Future
                                # KEYS (always true) — _mark_processed at submit time is the
                                # real double-submit guard, so the vestigial check is dropped.
                                if _processing_job_key(job) not in processed_job_indices:
                                    if not _try_submit(job):
                                        return
                else:
                    with processed_job_indices_lock:
                        for job in processing_resources.processing_jobs:
                            # Same key-not-idx rule as the locked branch above.
                            if _processing_job_key(job) not in processed_job_indices:
                                if not _try_submit(job):
                                    return

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
                """Check if processing should continue.

                Supervision bounds are evaluated FIRST and unconditionally. They must not
                be reachable only through the queue-state branches below, because the
                2026-08-12 wedge was precisely the case where those branches could never
                fire: `transcription_complete_event` was never set (the main thread died
                before reaching `.set()`), so the function fell through to `return True`
                on every iteration, forever.
                """
                reason = _supervision_exit_reason()
                if reason is not None:
                    if len(futures):
                        abandoned_futures[0] = len(futures)
                        logger.error(
                            "Processing loop stopping: %s. Abandoning %d in-flight "
                            "episode(s); they are not marked complete and a resumed run "
                            "will reprocess them (skip_existing keeps this idempotent).",
                            reason,
                            len(futures),
                        )
                    else:
                        logger.warning("Processing loop stopping: %s.", reason)
                    stop_requested[0] = True
                    return False
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
                # Track queue wait time (Issue #387) and, when the whole worker
                # pool is idle (no futures pending), the #1180 processing-thread
                # queue-idle counter. `len(futures) == 0` here means _submit_new_jobs
                # found nothing to submit AND no future is in-flight — the
                # ProcessingProcessor genuinely has no work.
                queue_wait_start = time.time()
                workers_all_idle = len(futures) == 0
                if not (transcription_complete_event and transcription_complete_event.is_set()):
                    time.sleep(0.1)
                    queue_wait_duration = time.time() - queue_wait_start
                else:
                    time.sleep(0.05)
                    queue_wait_duration = time.time() - queue_wait_start
                if pipeline_metrics is not None:
                    pipeline_metrics.record_queue_wait_time(queue_wait_duration)
                    if workers_all_idle:
                        pipeline_metrics.record_processing_queue_idle_time(queue_wait_duration)
        finally:
            if abandoned_futures[0]:
                # Do not wait: at least one future is presumed stuck, and waiting is the
                # failure mode we are escaping. cancel_futures drops anything still queued.
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

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
            from ...utils import timeout_config
            from ...utils.timeout import timeout_context, TimeoutError as SummarizationTimeoutError

            # The label says metadata generation, not "summarization", because that is what
            # this deadline actually wraps: call_generate_metadata is summary + GI + KG.
            # Measured 2026-08-31 on prod_dgx_full: summarisation peaked at 634.7s — half the
            # 1200s budget — while GI alone ran 1327s. Every one of the 22 overruns in that
            # batch was GI, reported under the summariser's name, which sends whoever reads it
            # to debug the innocent stage. The config key keeps its name for compatibility.
            #
            # #1920: the budget now scales with transcript length instead of being flat. The
            # work is linear in the transcript (r=0.868 over the 2026-09-01 batch) while the
            # deadline was a constant, so the longest episodes — the two-hour ones §5h of the
            # onboarding plan explicitly allows — were guaranteed to overrun and raise an
            # ERROR-level DEADLINE EXCEEDED after completing successfully. Floored at the
            # configured value, so short episodes keep exactly today's budget.
            metadata_timeout = int(
                timeout_config.get_metadata_generation_timeout(
                    cfg, _transcript_word_count(job.transcript_path)
                )
            )
            with timeout_context(
                metadata_timeout,
                f"metadata generation (summary+GI+KG) for episode {job.episode.idx}",
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
            # OVERRAN, not failed — and the difference is provable, not a judgement call.
            #
            # ``timeout_context`` observes; it cannot interrupt (see its docstring, and
            # tests/unit/podcast_scraper/utils/test_timeout_contract.py which pins that). It
            # raises ONLY after the wrapped block has already returned normally, so reaching
            # this handler means ``call_generate_metadata`` COMPLETED. If the work itself had
            # raised, control would be in the generic ``except Exception`` below instead. The
            # only other producer of this exception class, ``with_timeout``, has no callers.
            #
            # Marking the episode failed here was therefore untrue, and it was expensive. On the
            # acceptance run the two episodes whose GI exceeded 1200s — Dwarkesh (1529s) and
            # Latent Space (1209s) — were recorded ``failed @ summarization`` while their
            # artifacts were complete and valid on disk (summary schema_status=valid, 114 and 54
            # insights, 26 KG nodes each). ``_pipeline_return_episode_count`` then counted zero
            # ok episodes, so both feeds reported ``episodes_processed: 0`` with ``ok: true``,
            # and no incident was written anywhere. Two fully-processed episodes read as a
            # silent no-op. Every other feed in the run — the longest at 970s — was fine.
            #
            # Same shape as #1647: a signal that means "finished, but notable" was routed into
            # the word reserved for "did not finish". The overrun is real and stays loud (ERROR
            # log + an incident row so it appears in the batch rollup); what changes is that it
            # no longer erases a successful episode.
            update_metric_safely(pipeline_metrics, "summarization_deadline_overruns", 1)
            logger.error(
                "[%s] METADATA GENERATION (summary+GI+KG) OVERRAN its %ss deadline but "
                "COMPLETED; keeping the episode's results. This is a performance signal, not "
                "a failure. The dominant cost here is normally GI, not summarisation — "
                "compare gi_sec against summary_sec before suspecting the summariser: %s",
                job.episode.idx,
                getattr(cfg, "summarization_timeout", 1200),
                format_exception_for_log(exc),
            )
            _record_summarization_overrun_incident(cfg, job, effective_output_dir, exc)
            if pipeline_metrics is not None:
                from ..helpers import get_episode_id_from_episode

                episode_id, _ = get_episode_id_from_episode(job.episode, cfg.rss_url or "")
                pipeline_metrics.update_episode_status(
                    episode_id=episode_id,
                    status="ok",
                    stage="metadata_written",
                )
                try:
                    pipeline_metrics.record_stage_outcome(
                        "summarization",
                        job.episode.idx,
                        "degraded",
                        reason="deadline_exceeded_but_completed",
                        detail={
                            "deadline_seconds": getattr(cfg, "summarization_timeout", 1200),
                        },
                    )
                except Exception:
                    logger.debug("[%s] could not record summarization overrun", job.episode.idx)
            return True
        except Exception as exc:  # pragma: no cover
            update_metric_safely(pipeline_metrics, "errors_total", 1)
            logger.error(
                "[%s] processing raised an unexpected error: %s",
                job.episode.idx,
                format_exception_for_log(exc),
            )
            # A real failure, and until now it was recorded ONLY as an episode status inside
            # the run's own metrics. It never reached corpus_incidents.jsonl, so the batch
            # rollup counted zero incidents while the feed reported ok:true with one fewer
            # episode and no reason given — the same silence that made the deadline bug above
            # take a full investigation to find (#1657 acceptance item 4).
            _record_metadata_failure_incident(cfg, job, effective_output_dir, exc)
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

    # #1180: bind the wrapper once so both call sites (sequential + parallel
    # via _run_parallel_processing_loop) go through the same instrumentation.
    def _timed(job: ProcessingJob) -> bool:
        return _time_processing_job(pipeline_metrics, job, _process_single_processing_job)

    # Process jobs as they become available
    if max_workers <= 1:
        # Sequential processing
        while True:
            current_job = _find_next_unprocessed_job()
            if current_job:
                success = _timed(current_job)
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

            # Wait a bit before checking again. #1180: helper records the idle
            # interval so this loop's cognitive complexity stays under budget.
            _sleep_and_tally_idle(
                pipeline_metrics,
                (
                    0.05
                    if (transcription_complete_event and transcription_complete_event.is_set())
                    else 0.1
                ),
            )
    else:
        # Parallel processing
        parallel_jobs_ok, parallel_jobs_failed = _run_parallel_processing_loop(
            processing_resources,
            processed_job_indices,
            processed_job_indices_lock,
            _timed,  # #1180: instrument every job through the sequential + parallel loops
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
