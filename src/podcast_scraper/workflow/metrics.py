"""Simple in-memory metrics collector for pipeline performance tracking."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class EpisodeStatus:
    """Status tracking for a single episode."""

    episode_id: str
    episode_number: int
    status: Literal["ok", "failed", "skipped"]
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stage: Optional[str] = (
        None  # Stage where error occurred (download, transcribe, summarize, etc.)
    )
    retry_count: int = 0


@dataclass
class EpisodeMetrics:
    """Per-episode metrics for provider comparison."""

    episode_id: str
    episode_number: int
    audio_sec: Optional[float] = None  # Audio duration in seconds
    transcribe_sec: Optional[float] = None  # Transcription time in seconds
    summary_sec: Optional[float] = None  # Summarization time in seconds
    gi_sec: Optional[float] = None  # GIL artifact generation time in seconds
    kg_sec: Optional[float] = None  # KG artifact generation time in seconds
    retries: int = 0  # Number of retries (transcription + summarization)
    rate_limit_sleep_sec: float = 0.0  # Time spent sleeping due to rate limits
    prompt_tokens: Optional[int] = None  # Input tokens (transcription + summarization)
    completion_tokens: Optional[int] = None  # Output tokens (transcription + summarization)
    estimated_cost: Optional[float] = None  # Estimated cost in USD


@dataclass
class Metrics:
    """In-memory metrics collector for pipeline execution.

    Tracks performance metrics per run and per stage to enable
    performance analysis and bottleneck identification.
    """

    # Per-run metrics
    run_duration_seconds: float = 0.0
    episodes_scraped_total: int = 0
    episodes_skipped_total: int = 0
    errors_total: int = 0
    bytes_downloaded_total: int = 0

    # Processing statistics
    transcripts_downloaded: int = 0  # Direct transcript downloads
    transcripts_transcribed: int = 0  # Transcripts saved (from cache OR actual transcription)
    # Note: This counts transcripts saved regardless of source (cache hit or actual transcription).
    # When transcript cache is used, transcripts_transcribed will be > 0 but transcribe_count
    # and avg_transcribe_seconds will be 0 (no actual transcription work performed).
    episodes_summarized: int = 0  # Episodes with summaries generated
    metadata_files_generated: int = 0  # Metadata files created
    gi_artifacts_generated: int = 0  # GIL artifacts (gi.json) written
    gi_failures: int = 0  # GIL artifact generation failures (non-fatal)
    kg_artifacts_generated: int = 0  # KG artifacts (kg.json) written
    kg_failures: int = 0  # KG artifact generation failures (non-fatal)
    kg_provider_extractions: int = 0  # KG artifacts that used LLM extraction successfully
    # KG aggregate stats (per-artifact rollups; Issue KG parity with GIL metrics)
    kg_topic_nodes_total: int = 0  # Sum of Topic nodes across kg.json written this run
    kg_entity_nodes_total: int = 0  # Sum of Entity nodes across kg.json
    kg_episode_nodes_total: int = 0  # Sum of Episode nodes (typically == kg_artifacts_generated)
    kg_extractions_stub: int = 0  # Artifacts whose extraction.model_version is stub-like
    kg_extractions_summary_bullets: int = 0  # model_version == summary_bullets
    kg_extractions_provider: int = 0  # model_version startswith provider:
    # Subset: LLM JSON from summary bullets (not transcript extract_kg_graph)
    kg_extractions_provider_summary_bullets: int = 0
    gi_evidence_stack_completed: int = 0  # GIL artifacts that completed evidence QA+NLI path
    gi_evidence_extract_quotes_calls: int = 0  # extract_quotes calls on provider path
    # Candidates that passed QA threshold and were sent to NLI (may exceed completed NLI calls).
    gi_evidence_nli_candidates_queued: int = 0
    # score_entailment invocations that returned without raising (includes low scores).
    gi_evidence_score_entailment_calls: int = 0
    gi_episodes_zero_grounded_when_required: int = 0  # gi_require_grounding but 0 quotes
    gi_grounding_degraded: bool = False  # True if any episode had zero grounded quotes (above)
    # GIL success metrics (PRD-017): accumulated across artifacts this run
    gi_insights_total: int = 0  # Total Insight nodes across all artifacts
    gi_quotes_total: int = 0  # Total Quote nodes across all artifacts
    gi_insights_grounded: int = 0  # Insights with ≥1 SUPPORTED_BY edge
    gi_artifacts_with_insights_and_quotes: int = 0  # Artifacts with ≥1 insight and ≥1 quote
    gi_quotes_verbatim: int = 0  # Quotes whose text matches transcript[char_start:char_end]
    gi_quotes_checked: int = 0  # Quotes checked for verbatim (had transcript)

    # Per-stage metrics
    time_scraping: float = 0.0
    time_parsing: float = 0.0
    time_normalizing: float = 0.0
    io_and_waiting_thread_sum_seconds: float = (
        0.0  # Sum of IO/waiting time across all threads (aggregate waiting time).
        # This is the sum of all sub-buckets and can exceed run_duration_seconds
        # because it represents total thread-time, not wall-clock time (Issue #391).
        # For wall-clock time, see io_and_waiting_wall_seconds.
    )
    io_and_waiting_wall_seconds: float = (
        0.0  # Wall-clock time spent in IO/waiting stages (measured by wall-clock spans).
        # This represents actual elapsed time, not sum across threads (Issue #391).
    )
    # Sub-buckets for io_and_waiting (Issue #387)
    time_download_wait_seconds: float = 0.0  # Time waiting for downloads to complete
    time_transcription_wait_seconds: float = 0.0  # Time waiting for transcription jobs
    time_summarization_wait_seconds: float = 0.0  # Time waiting for summarization
    time_thread_sync_seconds: float = 0.0  # Time spent in thread synchronization (join() calls)
    time_queue_wait_seconds: float = 0.0  # Time waiting in queues
    time_writing_storage: float = (
        0.0  # Actual file write time only (open/write/flush) - should be tiny
    )

    # Per-episode operation timing (key = episode.idx, 1-based; safe under parallel workers)
    download_media_time_by_episode: Dict[int, float] = field(default_factory=dict)
    download_media_attempts: int = 0  # Total media download attempts (including reused/cached)
    # App-level episode download retries (after urllib3 retries; see Config episode_retry_*)
    episode_download_retries: int = 0
    episode_download_retry_sleep_seconds: float = 0.0
    _episode_download_retry_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    transcribe_time_by_episode: Dict[int, float] = field(
        default_factory=dict
    )  # Actual transcription only; omitted when transcript cache hit (no API transcribe)
    extract_names_time_by_episode: Dict[int, float] = field(default_factory=dict)
    summarize_time_by_episode: Dict[int, float] = field(default_factory=dict)
    cleaning_time_by_episode: Dict[int, float] = field(default_factory=dict)
    gi_times: List[float] = field(default_factory=list)  # GIL artifact generation times per episode
    kg_times: List[float] = field(default_factory=list)  # KG artifact generation times per episode
    # Wall time for maybe_index_corpus (RFC-064 frozen profiles); 0 if skipped or disabled
    vector_index_seconds: float = 0.0

    # LLM API call tracking (for cost estimation)
    llm_transcription_calls: int = 0  # Number of transcription API calls
    llm_transcription_audio_minutes: float = 0.0  # Total audio minutes transcribed
    llm_speaker_detection_calls: int = 0  # Number of speaker detection API calls
    llm_speaker_detection_input_tokens: int = 0  # Total input tokens for speaker detection
    llm_speaker_detection_output_tokens: int = 0  # Total output tokens for speaker detection
    llm_summarization_calls: int = 0  # Number of summarization API calls
    llm_summarization_input_tokens: int = 0  # Total input tokens for summarization
    llm_summarization_output_tokens: int = 0  # Total output tokens for summarization
    # Transcript semantic cleaning (LLM path; pattern-only runs do not increment these)
    llm_cleaning_calls: int = 0
    llm_cleaning_input_tokens: int = 0
    llm_cleaning_output_tokens: int = 0
    # Grounded insights layer: generate_insights + extract_quotes + score_entailment (LLM path)
    llm_gi_calls: int = 0
    llm_gi_input_tokens: int = 0
    llm_gi_output_tokens: int = 0
    llm_gi_evidence_retries: int = 0  # Retries across GIL evidence LLM calls (extract_quotes / NLI)
    llm_gi_evidence_rate_limit_sleep_sec: float = 0.0  # Rate-limit sleep on those calls
    # Knowledge graph: extract_kg_graph (LLM provider path)
    llm_kg_calls: int = 0
    llm_kg_input_tokens: int = 0
    llm_kg_output_tokens: int = 0
    # Single-call clean + summary + bullets (Issue #477 bundled pipeline experiment)
    llm_bundled_clean_summary_calls: int = 0
    llm_bundled_clean_summary_input_tokens: int = 0
    llm_bundled_clean_summary_output_tokens: int = 0
    llm_bundled_fallback_to_staged_count: int = 0

    # Audio preprocessing metrics (RFC-040, Issue #387)
    preprocessing_times: List[float] = field(
        default_factory=list
    )  # Preprocessing times per episode (actual processing time, not wall time)
    preprocessing_wall_times: List[float] = field(
        default_factory=list
    )  # Wall time for preprocessing per episode (includes cache checks, even for hits)
    preprocessing_cache_hit_times: List[float] = field(
        default_factory=list
    )  # Time spent on cache hits per episode
    preprocessing_cache_miss_times: List[float] = field(
        default_factory=list
    )  # Time spent on cache misses (actual preprocessing) per episode
    preprocessing_cache_hit_flags: List[bool] = field(
        default_factory=list
    )  # Boolean flag per episode indicating cache hit (True) or miss (False)
    preprocessing_original_sizes: List[int] = field(
        default_factory=list
    )  # Original audio file sizes in bytes
    preprocessing_preprocessed_sizes: List[int] = field(
        default_factory=list
    )  # Preprocessed audio file sizes in bytes
    preprocessing_saved_bytes: List[int] = field(
        default_factory=list
    )  # Bytes saved per episode (original - preprocessed)
    preprocessing_audio_metadata: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Audio metadata per episode (bitrate, sample_rate, codec, channels)
    preprocessing_attempts: int = 0  # Total preprocessing attempts (cache hits + misses)
    preprocessing_cache_hits: int = 0  # Number of cache hits
    preprocessing_cache_misses: int = 0  # Number of cache misses

    # Per-episode status tracking (Issue #379)
    episode_statuses: List[EpisodeStatus] = field(default_factory=list)
    _episode_statuses_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )  # Thread-safe access to episode_statuses

    # Per-episode metrics for provider comparison
    episode_metrics: List[EpisodeMetrics] = field(default_factory=list)
    _episode_metrics_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )  # Thread-safe access to episode_metrics

    # Device usage tracking per stage (Issue #387)
    transcription_device: Optional[str] = None  # Device used for transcription stage
    summarization_device: Optional[str] = None  # Device used for summarization stage

    _start_time: float = field(default_factory=time.time, init=False)

    @property
    def gi_evidence_path_provider(self) -> int:
        """Deprecated alias for :attr:`gi_evidence_stack_completed` (export/UI compat)."""
        return self.gi_evidence_stack_completed

    @gi_evidence_path_provider.setter
    def gi_evidence_path_provider(self, value: int) -> None:
        self.gi_evidence_stack_completed = value

    def record_stage(self, stage: str, duration: float) -> None:
        """Record time spent in a stage.

        Args:
            stage: Stage name ("scraping", "parsing", "normalizing", "writing_storage")
            duration: Duration in seconds
        """
        if stage == "scraping":
            self.time_scraping += duration
        elif stage == "parsing":
            self.time_parsing += duration
        elif stage == "normalizing":
            self.time_normalizing += duration
        elif stage == "io_and_waiting":
            self.io_and_waiting_thread_sum_seconds += duration
        elif stage == "writing_storage":
            # Actual file write time only (should be tiny)
            self.time_writing_storage += duration

    def record_download_wait_time(self, duration: float) -> None:
        """Record time waiting for downloads to complete (Issue #387).

        Args:
            duration: Duration in seconds
        """
        self.time_download_wait_seconds += duration
        # Also add to io_and_waiting_thread_sum for backward compatibility
        self.io_and_waiting_thread_sum_seconds += duration

    def record_transcription_wait_time(self, duration: float) -> None:
        """Record time waiting for transcription jobs (Issue #387).

        Args:
            duration: Duration in seconds
        """
        self.time_transcription_wait_seconds += duration
        # Also add to io_and_waiting_thread_sum for backward compatibility
        self.io_and_waiting_thread_sum_seconds += duration

    def record_summarization_wait_time(self, duration: float) -> None:
        """Record time waiting for summarization (Issue #387).

        Args:
            duration: Duration in seconds
        """
        self.time_summarization_wait_seconds += duration
        # Also add to io_and_waiting_thread_sum for backward compatibility
        self.io_and_waiting_thread_sum_seconds += duration

    def record_thread_sync_time(self, duration: float) -> None:
        """Record time spent in thread synchronization (join() calls) (Issue #387).

        Args:
            duration: Duration in seconds
        """
        self.time_thread_sync_seconds += duration
        # Also add to io_and_waiting_thread_sum for backward compatibility
        self.io_and_waiting_thread_sum_seconds += duration

    def record_queue_wait_time(self, duration: float) -> None:
        """Record time waiting in queues (Issue #387).

        Args:
            duration: Duration in seconds
        """
        self.time_queue_wait_seconds += duration
        # Also add to io_and_waiting_thread_sum for backward compatibility
        self.io_and_waiting_thread_sum_seconds += duration

    def record_io_waiting_wall_time(self, duration: float) -> None:
        """Record wall-clock time spent in IO/waiting (Issue #391).

        This measures actual elapsed time (not sum across threads) for waiting operations
        like thread joins, queue waits, etc. at the orchestration level.

        Args:
            duration: Wall-clock duration in seconds
        """
        self.io_and_waiting_wall_seconds += duration

    def record_transcription_device(self, device: str) -> None:
        """Record device used for transcription stage (Issue #387).

        Args:
            device: Device string (e.g., 'cpu', 'cuda', 'mps')
        """
        self.transcription_device = device

    def record_summarization_device(self, device: str) -> None:
        """Record device used for summarization stage (Issue #387).

        Args:
            device: Device string (e.g., 'cpu', 'cuda', 'mps')
        """
        self.summarization_device = device

    def record_download_media_attempt(self) -> None:
        """Record a media download attempt (called regardless of cache/reuse)."""
        self.download_media_attempts += 1

    def record_episode_download_retry(self, sleep_sec: float) -> None:
        """Record one application-level episode download retry and planned backoff sleep."""
        with self._episode_download_retry_lock:
            self.episode_download_retries += 1
            self.episode_download_retry_sleep_seconds += max(0.0, sleep_sec)

    def record_download_media_time(self, duration: float, episode_idx: int) -> None:
        """Record time spent downloading media for an episode (by episode.idx).

        Set duration=0 when reuse_media skips HTTP or transcript cache makes download
        non-billable for reporting (audio was only needed for cache lookup).
        """
        self.download_media_time_by_episode[episode_idx] = duration

    def record_transcribe_time(self, duration: float, episode_idx: int) -> None:
        """Record time spent transcribing an episode (by episode.idx)."""
        self.transcribe_time_by_episode[episode_idx] = duration

    def record_extract_names_time(self, duration: float, episode_idx: int) -> None:
        """Record time spent extracting speaker names for an episode (by episode.idx)."""
        self.extract_names_time_by_episode[episode_idx] = duration

    def record_summarize_time(self, duration: float, episode_idx: int) -> None:
        """Record time spent generating summary for an episode (by episode.idx)."""
        self.summarize_time_by_episode[episode_idx] = duration

    def record_cleaning_time(self, duration: float, episode_idx: int) -> None:
        """Record wall time for transcript cleaning before summarize (by episode.idx)."""
        self.cleaning_time_by_episode[episode_idx] = duration

    def record_gi_time(self, duration: float) -> None:
        """Record time spent generating GIL artifact for an episode.

        Args:
            duration: Duration in seconds
        """
        self.gi_times.append(duration)

    def record_kg_time(self, duration: float) -> None:
        """Record time spent generating KG artifact for an episode.

        Args:
            duration: Duration in seconds
        """
        self.kg_times.append(duration)

    def record_kg_artifact_stats(self, payload: Dict[str, Any]) -> None:
        """Accumulate KG node counts and extraction-mode tallies from one kg.json payload.

        Args:
            payload: Parsed KG artifact dict (schema_version, extraction, nodes, edges).
        """
        mv = str((payload.get("extraction") or {}).get("model_version", "") or "")
        if mv.startswith("provider:"):
            self.kg_extractions_provider += 1
            if mv.startswith("provider:summary_bullets:"):
                self.kg_extractions_provider_summary_bullets += 1
        elif mv == "summary_bullets":
            self.kg_extractions_summary_bullets += 1
        else:
            self.kg_extractions_stub += 1
        for n in payload.get("nodes") or []:
            t = n.get("type")
            if t == "Topic":
                self.kg_topic_nodes_total += 1
            elif t == "Entity":
                self.kg_entity_nodes_total += 1
            elif t == "Episode":
                self.kg_episode_nodes_total += 1

    def record_gi_success_counts(
        self,
        insights: int,
        quotes: int,
        grounded_insights: int,
        has_insights_and_quotes: bool,
        quotes_verbatim: int = 0,
        quotes_checked: int = 0,
    ) -> None:
        """Record GIL success metrics from one artifact (PRD-017).

        Call once per generated gi.json. Accumulates totals for grounding rate,
        quote validity rate, and extraction coverage.

        Args:
            insights: Number of Insight nodes in this artifact.
            quotes: Number of Quote nodes in this artifact.
            grounded_insights: Number of insights with ≥1 SUPPORTED_BY edge.
            has_insights_and_quotes: True if this artifact has ≥1 insight and ≥1 quote.
            quotes_verbatim: Number of quotes that matched transcript verbatim.
            quotes_checked: Number of quotes checked for verbatim (had transcript).
        """
        self.gi_insights_total += insights
        self.gi_quotes_total += quotes
        self.gi_insights_grounded += grounded_insights
        if has_insights_and_quotes:
            self.gi_artifacts_with_insights_and_quotes += 1
        self.gi_quotes_verbatim += quotes_verbatim
        self.gi_quotes_checked += quotes_checked

    def record_llm_transcription_call(self, audio_minutes: float) -> None:
        """Record an LLM transcription API call.

        Args:
            audio_minutes: Audio duration in minutes
        """
        self.llm_transcription_calls += 1
        self.llm_transcription_audio_minutes += audio_minutes

    def record_llm_speaker_detection_call(self, input_tokens: int, output_tokens: int) -> None:
        """Record an LLM speaker detection API call.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
        """
        self.llm_speaker_detection_calls += 1
        self.llm_speaker_detection_input_tokens += input_tokens
        self.llm_speaker_detection_output_tokens += output_tokens

    def record_llm_summarization_call(self, input_tokens: int, output_tokens: int) -> None:
        """Record an LLM summarization API call.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
        """
        self.llm_summarization_calls += 1
        self.llm_summarization_input_tokens += input_tokens
        self.llm_summarization_output_tokens += output_tokens

    def record_llm_cleaning_call(self, input_tokens: int, output_tokens: int) -> None:
        """Record an LLM transcript-cleaning API call (semantic cleaning)."""
        self.llm_cleaning_calls += 1
        self.llm_cleaning_input_tokens += input_tokens
        self.llm_cleaning_output_tokens += output_tokens

    def record_llm_gi_call(self, input_tokens: int, output_tokens: int) -> None:
        """Record an LLM call for the grounded-insights layer (insights / evidence stack)."""
        self.llm_gi_calls += 1
        self.llm_gi_input_tokens += input_tokens
        self.llm_gi_output_tokens += output_tokens

    def record_llm_gi_evidence_call_metrics(self, cm: Any) -> None:
        """Accumulate retries and rate-limit sleep from one GIL evidence LLM API call.

        Expects :class:`~podcast_scraper.utils.provider_metrics.ProviderCallMetrics`
        after :meth:`~podcast_scraper.utils.provider_metrics.ProviderCallMetrics.finalize`.
        """
        self.llm_gi_evidence_retries += int(getattr(cm, "retries", 0) or 0)
        self.llm_gi_evidence_rate_limit_sleep_sec += float(
            getattr(cm, "rate_limit_sleep_sec", 0.0) or 0.0
        )

    def record_llm_kg_call(self, input_tokens: int, output_tokens: int) -> None:
        """Record an LLM call for KG extraction (extract_kg_graph)."""
        self.llm_kg_calls += 1
        self.llm_kg_input_tokens += input_tokens
        self.llm_kg_output_tokens += output_tokens

    def record_llm_bundled_clean_summary_call(self, input_tokens: int, output_tokens: int) -> None:
        """Record one bundled LLM call (semantic clean + title + bullets, Issue #477)."""
        self.llm_bundled_clean_summary_calls += 1
        self.llm_bundled_clean_summary_input_tokens += input_tokens
        self.llm_bundled_clean_summary_output_tokens += output_tokens

    def record_llm_bundled_fallback_to_staged(self) -> None:
        """Increment count when bundled clean+summary fails and staged path is used."""
        self.llm_bundled_fallback_to_staged_count += 1

    def record_preprocessing_time(self, duration: float) -> None:
        """Record time spent preprocessing audio for an episode.

        Args:
            duration: Duration in seconds
        """
        self.preprocessing_times.append(duration)

    def record_preprocessing_wall_time(self, duration: float) -> None:
        """Record wall time for preprocessing (includes cache checks, even for hits) (Issue #387).

        Args:
            duration: Wall time in seconds
        """
        self.preprocessing_wall_times.append(duration)

    def record_preprocessing_cache_hit_time(self, duration: float) -> None:
        """Record time spent on cache hit (Issue #387).

        Args:
            duration: Time in seconds
        """
        self.preprocessing_cache_hit_times.append(duration)

    def record_preprocessing_cache_miss_time(self, duration: float) -> None:
        """Record time spent on cache miss (actual preprocessing) (Issue #387).

        Args:
            duration: Time in seconds
        """
        self.preprocessing_cache_miss_times.append(duration)

    def record_preprocessing_cache_hit_flag(self, is_hit: bool) -> None:
        """Record cache hit/miss flag per episode (Issue #387).

        Args:
            is_hit: True if cache hit, False if cache miss
        """
        self.preprocessing_cache_hit_flags.append(is_hit)

    def record_preprocessing_size_reduction(
        self, original_size: int, preprocessed_size: int
    ) -> None:
        """Record audio file size reduction from preprocessing.

        Args:
            original_size: Original audio file size in bytes
            preprocessed_size: Preprocessed audio file size in bytes
        """
        self.preprocessing_original_sizes.append(original_size)
        self.preprocessing_preprocessed_sizes.append(preprocessed_size)
        saved_bytes = original_size - preprocessed_size
        self.preprocessing_saved_bytes.append(saved_bytes)

    def record_preprocessing_audio_metadata(
        self,
        bitrate: Optional[int] = None,
        sample_rate: Optional[int] = None,
        codec: Optional[str] = None,
        channels: Optional[int] = None,
    ) -> None:
        """Record audio metadata for an episode (Issue #387).

        Args:
            bitrate: Audio bitrate in bps
            sample_rate: Sample rate in Hz
            codec: Audio codec name
            channels: Number of audio channels
        """
        metadata: Dict[str, Any] = {}
        if bitrate is not None:
            metadata["bitrate"] = bitrate
        if sample_rate is not None:
            metadata["sample_rate"] = sample_rate
        if codec is not None:
            metadata["codec"] = codec
        if channels is not None:
            metadata["channels"] = channels
        self.preprocessing_audio_metadata.append(metadata)

    def record_preprocessing_attempt(self) -> None:
        """Record a preprocessing attempt (called for both cache hits and misses)."""
        self.preprocessing_attempts += 1

    def record_preprocessing_cache_hit(self) -> None:
        """Record a preprocessing cache hit."""
        self.preprocessing_cache_hits += 1

    def record_preprocessing_cache_miss(self) -> None:
        """Record a preprocessing cache miss."""
        self.preprocessing_cache_misses += 1

    def record_episode_status(
        self,
        episode_id: str,
        episode_number: int,
        status: Literal["ok", "failed", "skipped"],
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        stage: Optional[str] = None,
        retry_count: int = 0,
    ) -> None:
        """Record status for an episode (thread-safe).

        Args:
            episode_id: Unique episode identifier
            episode_number: Episode number/index
            status: Episode status (ok, failed, skipped)
            error_type: Type of error if status is "failed"
            error_message: Error message if status is "failed"
            stage: Stage where error occurred
            retry_count: Number of retry attempts
        """
        episode_status = EpisodeStatus(
            episode_id=episode_id,
            episode_number=episode_number,
            status=status,
            error_type=error_type,
            error_message=error_message,
            stage=stage,
            retry_count=retry_count,
        )
        with self._episode_statuses_lock:
            self.episode_statuses.append(episode_status)

    def get_or_create_episode_status(self, episode_id: str, episode_number: int) -> EpisodeStatus:
        """Get existing episode status or create new one with 'queued' status (thread-safe).

        Args:
            episode_id: Unique episode identifier
            episode_number: Episode number/index

        Returns:
            EpisodeStatus object (existing or newly created)
        """
        with self._episode_statuses_lock:
            # Find existing status
            for status in self.episode_statuses:
                if status.episode_id == episode_id:
                    return status
            # Create new status with 'queued'
            episode_status = EpisodeStatus(
                episode_id=episode_id,
                episode_number=episode_number,
                status="ok",  # Start as 'ok' (will be updated as stages complete)
                stage="queued",
            )
            self.episode_statuses.append(episode_status)
            return episode_status

    def update_episode_status(
        self,
        episode_id: str,
        status: Optional[Literal["ok", "failed", "skipped"]] = None,
        stage: Optional[str] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update existing episode status (thread-safe).

        Args:
            episode_id: Unique episode identifier
            status: New status (ok, failed, skipped) - optional, only updates if provided
            stage: Stage name (e.g., 'downloaded', 'transcribed', 'summarized', 'metadata_written')
            error_type: Type of error if status is "failed"
            error_message: Error message if status is "failed"
        """
        with self._episode_statuses_lock:
            for episode_status in self.episode_statuses:
                if episode_status.episode_id == episode_id:
                    if status is not None:
                        episode_status.status = status
                    if stage is not None:
                        episode_status.stage = stage
                    if error_type is not None:
                        episode_status.error_type = error_type
                    if error_message is not None:
                        episode_status.error_message = error_message
                    return
            # If not found, create new status (defensive fallback - should be rare
            # after initialization)
            logger.debug(f"Episode status not found for {episode_id}, creating new entry")
            episode_status = EpisodeStatus(
                episode_id=episode_id,
                episode_number=0,  # Unknown number
                status=status or "ok",
                stage=stage,
                error_type=error_type,
                error_message=error_message,
            )
            self.episode_statuses.append(episode_status)

    def get_or_create_episode_metrics(self, episode_id: str, episode_number: int) -> EpisodeMetrics:
        """Get existing episode metrics or create new one (thread-safe).

        Args:
            episode_id: Unique episode identifier
            episode_number: Episode number/index

        Returns:
            EpisodeMetrics object (existing or newly created)
        """
        with self._episode_metrics_lock:
            # Find existing metrics
            for metrics in self.episode_metrics:
                if metrics.episode_id == episode_id:
                    return metrics
            # Create new metrics
            episode_metrics = EpisodeMetrics(
                episode_id=episode_id,
                episode_number=episode_number,
            )
            self.episode_metrics.append(episode_metrics)
            return episode_metrics

    def lookup_episode_metrics(self, episode_id: str) -> Optional[EpisodeMetrics]:
        """Return existing per-episode metrics row, if any (thread-safe)."""
        with self._episode_metrics_lock:
            for em in self.episode_metrics:
                if em.episode_id == episode_id:
                    return em
        return None

    def update_episode_metrics(
        self,
        episode_id: str,
        audio_sec: Optional[float] = None,
        transcribe_sec: Optional[float] = None,
        summary_sec: Optional[float] = None,
        gi_sec: Optional[float] = None,
        kg_sec: Optional[float] = None,
        retries: Optional[int] = None,
        rate_limit_sleep_sec: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        estimated_cost: Optional[float] = None,
    ) -> None:
        """Update existing episode metrics (thread-safe).

        Args:
            episode_id: Unique episode identifier
            audio_sec: Audio duration in seconds
            transcribe_sec: Transcription time in seconds
            summary_sec: Summarization time in seconds
            gi_sec: GIL artifact generation time in seconds
            kg_sec: KG artifact generation time in seconds
            retries: Number of retries (will be added to existing count)
            rate_limit_sleep_sec: Time spent sleeping due to rate limits (will be added)
            prompt_tokens: Input tokens (will be added to existing count)
            completion_tokens: Output tokens (will be added to existing count)
            estimated_cost: Estimated cost in USD (will be added to existing cost)
        """
        with self._episode_metrics_lock:
            for metrics in self.episode_metrics:
                if metrics.episode_id == episode_id:
                    if audio_sec is not None:
                        metrics.audio_sec = audio_sec
                    if transcribe_sec is not None:
                        metrics.transcribe_sec = transcribe_sec
                    if summary_sec is not None:
                        metrics.summary_sec = summary_sec
                    if gi_sec is not None:
                        metrics.gi_sec = gi_sec
                    if kg_sec is not None:
                        metrics.kg_sec = kg_sec
                    if retries is not None:
                        metrics.retries += retries
                    if rate_limit_sleep_sec is not None:
                        metrics.rate_limit_sleep_sec += rate_limit_sleep_sec
                    if prompt_tokens is not None:
                        if metrics.prompt_tokens is None:
                            metrics.prompt_tokens = 0
                        metrics.prompt_tokens += prompt_tokens
                    if completion_tokens is not None:
                        if metrics.completion_tokens is None:
                            metrics.completion_tokens = 0
                        metrics.completion_tokens += completion_tokens
                    if estimated_cost is not None:
                        if metrics.estimated_cost is None:
                            metrics.estimated_cost = 0.0
                        metrics.estimated_cost += estimated_cost
                    return
            # If not found, create new metrics (defensive fallback - should be rare
            # after initialization)
            logger.debug(f"Episode metrics not found for {episode_id}, creating new entry")
            episode_metrics = EpisodeMetrics(
                episode_id=episode_id,
                episode_number=0,  # Unknown number
                audio_sec=audio_sec,
                transcribe_sec=transcribe_sec,
                summary_sec=summary_sec,
                gi_sec=gi_sec,
                kg_sec=kg_sec,
                retries=retries or 0,
                rate_limit_sleep_sec=rate_limit_sleep_sec or 0.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost=estimated_cost,
            )
            self.episode_metrics.append(episode_metrics)

    def finish(self) -> Dict[str, Any]:
        """Calculate final metrics and return as dict.

        ``http_urllib3_retry_events`` comes from the process-wide downloader counter
        (reset when ``configure_downloader`` runs at pipeline start). It is not
        safe to interpret for a single run if multiple pipelines execute
        concurrently in one process.

        Returns:
            Dictionary of all metrics with final values
        """
        self.run_duration_seconds = time.time() - self._start_time

        from ..rss.downloader import get_http_retry_event_count
        from ..rss.http_policy import get_http_policy_metrics_snapshot

        http_urllib3_retry_events = get_http_retry_event_count()
        http_policy_metrics = get_http_policy_metrics_snapshot()

        def _avg_episode_dict(d: Dict[int, float]) -> float:
            return round(sum(d.values()) / len(d), 2) if d else 0.0

        # Calculate averages for per-episode operations (dict keyed by episode.idx)
        avg_download_media = _avg_episode_dict(self.download_media_time_by_episode)
        avg_transcribe = _avg_episode_dict(self.transcribe_time_by_episode)
        avg_extract_names = _avg_episode_dict(self.extract_names_time_by_episode)
        avg_summarize = _avg_episode_dict(self.summarize_time_by_episode)
        avg_cleaning = _avg_episode_dict(self.cleaning_time_by_episode)
        avg_gi = round(sum(self.gi_times) / len(self.gi_times), 2) if self.gi_times else 0.0
        avg_kg = round(sum(self.kg_times) / len(self.kg_times), 2) if self.kg_times else 0.0

        def _avg_tokens_per_call(total: int, calls: int) -> float:
            if calls <= 0:
                return 0.0
            return round(total / calls, 2)

        sp_in_avg = _avg_tokens_per_call(
            self.llm_speaker_detection_input_tokens, self.llm_speaker_detection_calls
        )
        sp_out_avg = _avg_tokens_per_call(
            self.llm_speaker_detection_output_tokens, self.llm_speaker_detection_calls
        )
        sum_in_avg = _avg_tokens_per_call(
            self.llm_summarization_input_tokens, self.llm_summarization_calls
        )
        sum_out_avg = _avg_tokens_per_call(
            self.llm_summarization_output_tokens, self.llm_summarization_calls
        )
        cl_in_avg = _avg_tokens_per_call(self.llm_cleaning_input_tokens, self.llm_cleaning_calls)
        cl_out_avg = _avg_tokens_per_call(self.llm_cleaning_output_tokens, self.llm_cleaning_calls)
        gi_in_avg = _avg_tokens_per_call(self.llm_gi_input_tokens, self.llm_gi_calls)
        gi_out_avg = _avg_tokens_per_call(self.llm_gi_output_tokens, self.llm_gi_calls)
        kg_in_avg = _avg_tokens_per_call(self.llm_kg_input_tokens, self.llm_kg_calls)
        kg_out_avg = _avg_tokens_per_call(self.llm_kg_output_tokens, self.llm_kg_calls)
        bd_in_avg = _avg_tokens_per_call(
            self.llm_bundled_clean_summary_input_tokens,
            self.llm_bundled_clean_summary_calls,
        )
        bd_out_avg = _avg_tokens_per_call(
            self.llm_bundled_clean_summary_output_tokens,
            self.llm_bundled_clean_summary_calls,
        )

        gi_llm_calls_per_artifact = (
            round(self.llm_gi_calls / self.gi_artifacts_generated, 2)
            if self.gi_artifacts_generated
            else 0.0
        )
        kg_llm_calls_per_artifact = (
            round(self.llm_kg_calls / self.kg_artifacts_generated, 2)
            if self.kg_artifacts_generated
            else 0.0
        )
        cleaning_llm_calls_per_recorded = (
            round(self.llm_cleaning_calls / len(self.cleaning_time_by_episode), 2)
            if self.cleaning_time_by_episode
            else 0.0
        )
        avg_preprocessing = (
            round(sum(self.preprocessing_times) / len(self.preprocessing_times), 2)
            if self.preprocessing_times
            else 0.0
        )

        # Calculate average size reduction
        total_original = (
            sum(self.preprocessing_original_sizes) if self.preprocessing_original_sizes else 0
        )
        total_preprocessed = (
            sum(self.preprocessing_preprocessed_sizes)
            if self.preprocessing_preprocessed_sizes
            else 0
        )
        avg_size_reduction_percent = (
            round((1 - total_preprocessed / total_original) * 100, 1) if total_original > 0 else 0.0
        )

        with self._episode_metrics_lock:
            total_episode_estimated_cost_usd = round(
                sum((em.estimated_cost or 0.0) for em in self.episode_metrics),
                6,
            )
            total_episode_prompt_tokens = sum(
                (em.prompt_tokens or 0) for em in self.episode_metrics
            )
            total_episode_completion_tokens = sum(
                (em.completion_tokens or 0) for em in self.episode_metrics
            )

        out: Dict[str, Any] = {
            "schema_version": "1.0.0",  # Versioned schema for metrics (Issue #379)
            "run_duration_seconds": round(self.run_duration_seconds, 2),
            "episodes_scraped_total": self.episodes_scraped_total,
            "episodes_skipped_total": self.episodes_skipped_total,
            "errors_total": self.errors_total,
            "bytes_downloaded_total": self.bytes_downloaded_total,
            "http_urllib3_retry_events": http_urllib3_retry_events,
            "episode_download_retries": self.episode_download_retries,
            "episode_download_retry_sleep_seconds": round(
                self.episode_download_retry_sleep_seconds, 3
            ),
            "transcripts_downloaded": self.transcripts_downloaded,
            "transcripts_transcribed": self.transcripts_transcribed,
            "episodes_summarized": self.episodes_summarized,
            "metadata_files_generated": self.metadata_files_generated,
            "gi_artifacts_generated": self.gi_artifacts_generated,
            "gi_failures": self.gi_failures,
            "kg_artifacts_generated": self.kg_artifacts_generated,
            "kg_failures": self.kg_failures,
            "kg_provider_extractions": self.kg_provider_extractions,
            "kg_topic_nodes_total": self.kg_topic_nodes_total,
            "kg_entity_nodes_total": self.kg_entity_nodes_total,
            "kg_episode_nodes_total": self.kg_episode_nodes_total,
            "kg_extractions_stub": self.kg_extractions_stub,
            "kg_extractions_summary_bullets": self.kg_extractions_summary_bullets,
            "kg_extractions_provider": self.kg_extractions_provider,
            "kg_extractions_provider_summary_bullets": (
                self.kg_extractions_provider_summary_bullets
            ),
            "kg_avg_topics_per_artifact": (
                round(self.kg_topic_nodes_total / self.kg_artifacts_generated, 2)
                if self.kg_artifacts_generated
                else 0.0
            ),
            "kg_avg_entities_per_artifact": (
                round(self.kg_entity_nodes_total / self.kg_artifacts_generated, 2)
                if self.kg_artifacts_generated
                else 0.0
            ),
            "gi_evidence_stack_completed": self.gi_evidence_stack_completed,
            "gi_evidence_path_provider": self.gi_evidence_path_provider,
            "gi_evidence_extract_quotes_calls": self.gi_evidence_extract_quotes_calls,
            "gi_evidence_nli_candidates_queued": self.gi_evidence_nli_candidates_queued,
            "gi_evidence_score_entailment_calls": self.gi_evidence_score_entailment_calls,
            "gi_episodes_zero_grounded_when_required": self.gi_episodes_zero_grounded_when_required,
            "gi_grounding_degraded": self.gi_grounding_degraded,
            "gi_insights_total": self.gi_insights_total,
            "gi_quotes_total": self.gi_quotes_total,
            "gi_insights_grounded": self.gi_insights_grounded,
            "gi_artifacts_with_insights_and_quotes": self.gi_artifacts_with_insights_and_quotes,
            "gi_quotes_verbatim": self.gi_quotes_verbatim,
            "gi_quotes_checked": self.gi_quotes_checked,
            "gi_grounding_rate_pct": round(
                (
                    (self.gi_insights_grounded / self.gi_insights_total * 100.0)
                    if self.gi_insights_total
                    else 0.0
                ),
                1,
            ),
            "gi_quote_validity_rate_pct": round(
                (
                    (self.gi_quotes_verbatim / self.gi_quotes_checked * 100.0)
                    if self.gi_quotes_checked
                    else 0.0
                ),
                1,
            ),
            "gi_extraction_coverage_pct": round(
                (
                    (
                        self.gi_artifacts_with_insights_and_quotes
                        / self.gi_artifacts_generated
                        * 100.0
                    )
                    if self.gi_artifacts_generated
                    else 0.0
                ),
                1,
            ),
            "gi_avg_insights_per_episode": round(
                (
                    self.gi_insights_total / self.gi_artifacts_generated
                    if self.gi_artifacts_generated
                    else 0.0
                ),
                2,
            ),
            "gi_avg_quotes_per_episode": round(
                (
                    self.gi_quotes_total / self.gi_artifacts_generated
                    if self.gi_artifacts_generated
                    else 0.0
                ),
                2,
            ),
            "time_scraping": round(self.time_scraping, 2),
            "time_parsing": round(self.time_parsing, 2),
            "time_normalizing": round(self.time_normalizing, 2),
            "io_and_waiting_thread_sum_seconds": round(self.io_and_waiting_thread_sum_seconds, 2),
            "io_and_waiting_wall_seconds": round(self.io_and_waiting_wall_seconds, 2),
            # Deprecated alias for io_and_waiting_thread_sum_seconds (kept for export compat)
            "time_io_and_waiting": round(self.io_and_waiting_thread_sum_seconds, 2),
            # Sub-buckets for io_and_waiting (Issue #387)
            "time_download_wait_seconds": round(self.time_download_wait_seconds, 2),
            "time_transcription_wait_seconds": round(self.time_transcription_wait_seconds, 2),
            "time_summarization_wait_seconds": round(self.time_summarization_wait_seconds, 2),
            "time_thread_sync_seconds": round(self.time_thread_sync_seconds, 2),
            "time_queue_wait_seconds": round(self.time_queue_wait_seconds, 2),
            "time_writing_storage": round(
                self.time_writing_storage, 2
            ),  # Actual file write time only
            # Per-episode averages
            "avg_download_media_seconds": avg_download_media,
            "avg_transcribe_seconds": avg_transcribe,
            "avg_extract_names_seconds": avg_extract_names,
            "avg_summarize_seconds": avg_summarize,
            "avg_cleaning_seconds": avg_cleaning,
            "avg_gi_seconds": avg_gi,
            "avg_kg_seconds": avg_kg,
            # Operation counts for context
            "download_media_count": len(self.download_media_time_by_episode),
            # Total download attempts (including reused/cached)
            "download_media_attempts": self.download_media_attempts,
            "transcribe_count": len(self.transcribe_time_by_episode),
            "extract_names_count": len(self.extract_names_time_by_episode),
            "summarize_count": len(self.summarize_time_by_episode),
            "cleaning_count": len(self.cleaning_time_by_episode),
            "gi_count": len(self.gi_times),
            "kg_count": len(self.kg_times),
            "vector_index_seconds": round(self.vector_index_seconds, 4),
            # LLM API call tracking
            "llm_transcription_calls": self.llm_transcription_calls,
            "llm_transcription_audio_minutes": round(self.llm_transcription_audio_minutes, 2),
            "llm_speaker_detection_calls": self.llm_speaker_detection_calls,
            "llm_speaker_detection_input_tokens": self.llm_speaker_detection_input_tokens,
            "llm_speaker_detection_output_tokens": self.llm_speaker_detection_output_tokens,
            "llm_speaker_detection_avg_input_tokens_per_call": sp_in_avg,
            "llm_speaker_detection_avg_output_tokens_per_call": sp_out_avg,
            "llm_summarization_calls": self.llm_summarization_calls,
            "llm_summarization_input_tokens": self.llm_summarization_input_tokens,
            "llm_summarization_output_tokens": self.llm_summarization_output_tokens,
            "llm_summarization_avg_input_tokens_per_call": sum_in_avg,
            "llm_summarization_avg_output_tokens_per_call": sum_out_avg,
            "llm_cleaning_calls": self.llm_cleaning_calls,
            "llm_cleaning_input_tokens": self.llm_cleaning_input_tokens,
            "llm_cleaning_output_tokens": self.llm_cleaning_output_tokens,
            "llm_cleaning_avg_input_tokens_per_call": cl_in_avg,
            "llm_cleaning_avg_output_tokens_per_call": cl_out_avg,
            "llm_cleaning_calls_per_recorded_cleaning_episode": cleaning_llm_calls_per_recorded,
            "llm_gi_calls": self.llm_gi_calls,
            "llm_gi_input_tokens": self.llm_gi_input_tokens,
            "llm_gi_output_tokens": self.llm_gi_output_tokens,
            "llm_gi_avg_input_tokens_per_call": gi_in_avg,
            "llm_gi_avg_output_tokens_per_call": gi_out_avg,
            "llm_gi_calls_per_gi_artifact": gi_llm_calls_per_artifact,
            "llm_gi_evidence_retries": self.llm_gi_evidence_retries,
            "llm_gi_evidence_rate_limit_sleep_sec": round(
                self.llm_gi_evidence_rate_limit_sleep_sec, 4
            ),
            "llm_kg_calls": self.llm_kg_calls,
            "llm_kg_input_tokens": self.llm_kg_input_tokens,
            "llm_kg_output_tokens": self.llm_kg_output_tokens,
            "llm_kg_avg_input_tokens_per_call": kg_in_avg,
            "llm_kg_avg_output_tokens_per_call": kg_out_avg,
            "llm_kg_calls_per_kg_artifact": kg_llm_calls_per_artifact,
            "llm_bundled_clean_summary_calls": self.llm_bundled_clean_summary_calls,
            "llm_bundled_clean_summary_input_tokens": (self.llm_bundled_clean_summary_input_tokens),
            "llm_bundled_clean_summary_output_tokens": (
                self.llm_bundled_clean_summary_output_tokens
            ),
            "llm_bundled_clean_summary_avg_input_tokens_per_call": bd_in_avg,
            "llm_bundled_clean_summary_avg_output_tokens_per_call": bd_out_avg,
            "llm_bundled_fallback_to_staged_count": self.llm_bundled_fallback_to_staged_count,
            "total_episode_estimated_cost_usd": total_episode_estimated_cost_usd,
            "total_episode_prompt_tokens": total_episode_prompt_tokens,
            "total_episode_completion_tokens": total_episode_completion_tokens,
            "llm_token_totals_by_stage": {
                "speaker_detection": {
                    "input": self.llm_speaker_detection_input_tokens,
                    "output": self.llm_speaker_detection_output_tokens,
                    "calls": self.llm_speaker_detection_calls,
                },
                "cleaning": {
                    "input": self.llm_cleaning_input_tokens,
                    "output": self.llm_cleaning_output_tokens,
                    "calls": self.llm_cleaning_calls,
                },
                "summarization": {
                    "input": self.llm_summarization_input_tokens,
                    "output": self.llm_summarization_output_tokens,
                    "calls": self.llm_summarization_calls,
                },
                "bundled_clean_summary": {
                    "input": self.llm_bundled_clean_summary_input_tokens,
                    "output": self.llm_bundled_clean_summary_output_tokens,
                    "calls": self.llm_bundled_clean_summary_calls,
                },
                "gi": {
                    "input": self.llm_gi_input_tokens,
                    "output": self.llm_gi_output_tokens,
                    "calls": self.llm_gi_calls,
                },
                "kg": {
                    "input": self.llm_kg_input_tokens,
                    "output": self.llm_kg_output_tokens,
                    "calls": self.llm_kg_calls,
                },
            },
            # Audio preprocessing metrics (Issue #387)
            "avg_preprocessing_seconds": avg_preprocessing,
            "preprocessing_count": len(self.preprocessing_times),
            "preprocessing_attempts": self.preprocessing_attempts,  # Total attempts (hits + misses)
            "avg_preprocessing_size_reduction_percent": avg_size_reduction_percent,
            "preprocessing_cache_hits": self.preprocessing_cache_hits,
            "preprocessing_cache_misses": self.preprocessing_cache_misses,
            # New preprocessing metrics (Issue #387)
            "avg_preprocessing_wall_ms": (
                round(
                    sum(self.preprocessing_wall_times) * 1000 / len(self.preprocessing_wall_times),
                    2,
                )
                if self.preprocessing_wall_times
                else 0.0
            ),
            "avg_preprocessing_cache_hit_ms": (
                round(
                    sum(self.preprocessing_cache_hit_times)
                    * 1000
                    / len(self.preprocessing_cache_hit_times),
                    2,
                )
                if self.preprocessing_cache_hit_times
                else 0.0
            ),
            "avg_preprocessing_cache_miss_ms": (
                round(
                    sum(self.preprocessing_cache_miss_times)
                    * 1000
                    / len(self.preprocessing_cache_miss_times),
                    2,
                )
                if self.preprocessing_cache_miss_times
                else 0.0
            ),
            "total_preprocessing_saved_bytes": (
                sum(self.preprocessing_saved_bytes) if self.preprocessing_saved_bytes else 0
            ),
            "avg_preprocessing_saved_bytes": (
                round(sum(self.preprocessing_saved_bytes) / len(self.preprocessing_saved_bytes), 2)
                if self.preprocessing_saved_bytes
                else 0.0
            ),
            "preprocessing_audio_metadata": self.preprocessing_audio_metadata,
            # Per-episode stage seconds (string keys for JSON); RFC-064 stage_truth snapshots
            "download_media_time_by_episode": {
                str(k): round(v, 4) for k, v in sorted(self.download_media_time_by_episode.items())
            },
            "transcribe_time_by_episode": {
                str(k): round(v, 4) for k, v in sorted(self.transcribe_time_by_episode.items())
            },
            "extract_names_time_by_episode": {
                str(k): round(v, 4) for k, v in sorted(self.extract_names_time_by_episode.items())
            },
            "summarize_time_by_episode": {
                str(k): round(v, 4) for k, v in sorted(self.summarize_time_by_episode.items())
            },
            "cleaning_time_by_episode": {
                str(k): round(v, 4) for k, v in sorted(self.cleaning_time_by_episode.items())
            },
            # Episode statuses
            "episode_statuses": [asdict(status) for status in self.episode_statuses],
            # Device usage per stage (Issue #387)
            "transcription_device": self.transcription_device,
            "summarization_device": self.summarization_device,
        }
        out.update(http_policy_metrics)
        return out

    def to_json(self) -> str:
        """Convert metrics to JSON string.

        Returns:
            JSON string representation of all metrics
        """
        metrics_dict = self.finish()
        return json.dumps(metrics_dict, indent=2)

    def save_to_file(self, filepath: str | Path) -> None:
        """Save metrics to JSON file with validation and atomic write (Issue #387).

        This method ensures:
        - Metrics are complete (all required fields present)
        - Atomic write (write to temp file, then rename)
        - Schema validation (all expected keys are present)

        Args:
            filepath: Path to output JSON file

        Raises:
            ValueError: If metrics validation fails
            OSError: If file writing fails
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get complete metrics dict
        metrics_dict = self.finish()

        # Validate metrics before writing (Issue #387)
        self._validate_metrics(metrics_dict)

        # Serialize to JSON
        metrics_json = json.dumps(metrics_dict, indent=2)

        # Atomic write: write to temp file, then rename (Issue #387)
        temp_filepath = filepath.with_suffix(filepath.suffix + ".tmp")
        try:
            temp_filepath.write_text(metrics_json, encoding="utf-8")
            temp_filepath.replace(filepath)  # Atomic rename
            logger.debug("Pipeline metrics saved to: %s", filepath)
        except Exception as e:
            # Clean up temp file on error
            if temp_filepath.exists():
                try:
                    temp_filepath.unlink()
                except Exception:
                    pass
            raise OSError(f"Failed to save metrics to {filepath}: {e}") from e

    def _validate_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """Validate metrics dict to ensure completeness (Issue #387).

        Checks that all expected keys are present and have valid values.

        Args:
            metrics_dict: Metrics dictionary to validate

        Raises:
            ValueError: If validation fails
        """
        # Required top-level keys (core metrics)
        required_keys = {
            "run_duration_seconds",
            "episodes_scraped_total",
            "episodes_skipped_total",
            "errors_total",
            "time_scraping",
            "time_parsing",
            "time_normalizing",
            "io_and_waiting_thread_sum_seconds",
            "io_and_waiting_wall_seconds",
            "time_io_and_waiting",  # Deprecated alias (export compat)
            "time_writing_storage",
            # Sub-buckets for io_and_waiting (Issue #387)
            "time_download_wait_seconds",
            "time_transcription_wait_seconds",
            "time_summarization_wait_seconds",
            "time_thread_sync_seconds",
            "time_queue_wait_seconds",
            "schema_version",
            "vector_index_seconds",
        }

        missing_keys = required_keys - set(metrics_dict.keys())
        if missing_keys:
            raise ValueError(
                f"Metrics validation failed: missing required keys: {sorted(missing_keys)}"
            )

        # Validate schema version is present and valid
        schema_version = metrics_dict.get("schema_version")
        if schema_version is None:
            raise ValueError("Metrics validation failed: schema_version is missing")
        if not isinstance(schema_version, str):
            raise ValueError(
                f"Metrics validation failed: schema_version must be string, "
                f"got {type(schema_version)}"
            )

        # Validate numeric fields are not None (they can be 0, but not None)
        numeric_fields = [
            "run_duration_seconds",
            "episodes_scraped_total",
            "episodes_skipped_total",
            "errors_total",
            "time_scraping",
            "time_parsing",
            "time_normalizing",
            "io_and_waiting_thread_sum_seconds",
            "io_and_waiting_wall_seconds",
            "time_io_and_waiting",  # Deprecated alias (export compat)
            "time_writing_storage",
            "time_download_wait_seconds",
            "time_transcription_wait_seconds",
            "time_summarization_wait_seconds",
            "time_thread_sync_seconds",
            "time_queue_wait_seconds",
            "vector_index_seconds",
        ]
        for field_name in numeric_fields:
            if metrics_dict.get(field_name) is None:
                raise ValueError(
                    f"Metrics validation failed: {field_name} is None (must be numeric)"
                )

    def log_metrics(self) -> None:
        """Log metrics with each metric on its own line for better readability.

        Emits multiple log lines, one per metric, in format:
        Pipeline finished:
          - key: value
          - key: value
        ...

        Uses DEBUG level per RFC-027 to avoid cluttering normal logs.
        Detailed metrics are available at DEBUG level, summary metrics at INFO level.
        """
        metrics_dict = self.finish()
        # Print each metric on its own line for better readability
        summary_lines = ["Pipeline finished (detailed metrics):"]
        for key, value in metrics_dict.items():
            # Format key names to be more readable (replace underscores with spaces, title case)
            readable_key = key.replace("_", " ").title()
            summary_lines.append(f"  - {readable_key}: {value}")
        summary = "\n".join(summary_lines)
        logger.debug(summary)  # Changed from logger.info() per RFC-027
