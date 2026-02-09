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

    # Per-episode operation timing (for A/B testing and performance analysis)
    download_media_times: List[float] = field(
        default_factory=list
    )  # Media download times per episode
    download_media_attempts: int = 0  # Total media download attempts (including reused/cached)
    transcribe_times: List[float] = field(
        default_factory=list
    )  # Transcription times per episode (only for actual transcription, NOT cache hits)
    # Note: When transcript cache is used, this list remains empty because record_transcribe_time()
    # is never called (cache hit returns early). transcribe_count = len(transcribe_times) will be 0.
    extract_names_times: List[float] = field(
        default_factory=list
    )  # Speaker detection times per episode
    summarize_times: List[float] = field(
        default_factory=list
    )  # Summary generation times per episode

    # LLM API call tracking (for cost estimation)
    llm_transcription_calls: int = 0  # Number of transcription API calls
    llm_transcription_audio_minutes: float = 0.0  # Total audio minutes transcribed
    llm_speaker_detection_calls: int = 0  # Number of speaker detection API calls
    llm_speaker_detection_input_tokens: int = 0  # Total input tokens for speaker detection
    llm_speaker_detection_output_tokens: int = 0  # Total output tokens for speaker detection
    llm_summarization_calls: int = 0  # Number of summarization API calls
    llm_summarization_input_tokens: int = 0  # Total input tokens for summarization
    llm_summarization_output_tokens: int = 0  # Total output tokens for summarization

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
    def time_io_and_waiting(self) -> float:
        """Backward compatibility property (deprecated, use io_and_waiting_thread_sum_seconds).

        Returns:
            Sum of IO/waiting time across all threads (same as io_and_waiting_thread_sum_seconds)
        """
        return self.io_and_waiting_thread_sum_seconds

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

    def record_download_media_time(self, duration: float) -> None:
        """Record time spent downloading media for an episode.

        Args:
            duration: Duration in seconds
        """
        self.download_media_times.append(duration)

    def record_transcribe_time(self, duration: float) -> None:
        """Record time spent transcribing an episode.

        Args:
            duration: Duration in seconds
        """
        self.transcribe_times.append(duration)

    def record_extract_names_time(self, duration: float) -> None:
        """Record time spent extracting speaker names for an episode.

        Args:
            duration: Duration in seconds
        """
        self.extract_names_times.append(duration)

    def record_summarize_time(self, duration: float) -> None:
        """Record time spent generating summary for an episode.

        Args:
            duration: Duration in seconds
        """
        self.summarize_times.append(duration)

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

    def update_episode_metrics(
        self,
        episode_id: str,
        audio_sec: Optional[float] = None,
        transcribe_sec: Optional[float] = None,
        summary_sec: Optional[float] = None,
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
                retries=retries or 0,
                rate_limit_sleep_sec=rate_limit_sleep_sec or 0.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost=estimated_cost,
            )
            self.episode_metrics.append(episode_metrics)

    def finish(self) -> Dict[str, Any]:
        """Calculate final metrics and return as dict.

        Returns:
            Dictionary of all metrics with final values
        """
        self.run_duration_seconds = time.time() - self._start_time

        # Calculate averages for per-episode operations
        avg_download_media = (
            round(sum(self.download_media_times) / len(self.download_media_times), 2)
            if self.download_media_times
            else 0.0
        )
        avg_transcribe = (
            round(sum(self.transcribe_times) / len(self.transcribe_times), 2)
            if self.transcribe_times
            else 0.0
        )
        avg_extract_names = (
            round(sum(self.extract_names_times) / len(self.extract_names_times), 2)
            if self.extract_names_times
            else 0.0
        )
        avg_summarize = (
            round(sum(self.summarize_times) / len(self.summarize_times), 2)
            if self.summarize_times
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

        return {
            "schema_version": "1.0.0",  # Versioned schema for metrics (Issue #379)
            "run_duration_seconds": round(self.run_duration_seconds, 2),
            "episodes_scraped_total": self.episodes_scraped_total,
            "episodes_skipped_total": self.episodes_skipped_total,
            "errors_total": self.errors_total,
            "bytes_downloaded_total": self.bytes_downloaded_total,
            "transcripts_downloaded": self.transcripts_downloaded,
            "transcripts_transcribed": self.transcripts_transcribed,
            "episodes_summarized": self.episodes_summarized,
            "metadata_files_generated": self.metadata_files_generated,
            "time_scraping": round(self.time_scraping, 2),
            "time_parsing": round(self.time_parsing, 2),
            "time_normalizing": round(self.time_normalizing, 2),
            "io_and_waiting_thread_sum_seconds": round(self.io_and_waiting_thread_sum_seconds, 2),
            "io_and_waiting_wall_seconds": round(self.io_and_waiting_wall_seconds, 2),
            # Backward compatibility (deprecated)
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
            # Operation counts for context
            "download_media_count": len(
                self.download_media_times
            ),  # Episodes with actual download time recorded
            # Total download attempts (including reused/cached)
            "download_media_attempts": self.download_media_attempts,
            "transcribe_count": len(
                self.transcribe_times
            ),  # Number of actual transcriptions (0 when cache is used)
            "extract_names_count": len(self.extract_names_times),
            "summarize_count": len(self.summarize_times),
            # LLM API call tracking
            "llm_transcription_calls": self.llm_transcription_calls,
            "llm_transcription_audio_minutes": round(self.llm_transcription_audio_minutes, 2),
            "llm_speaker_detection_calls": self.llm_speaker_detection_calls,
            "llm_speaker_detection_input_tokens": self.llm_speaker_detection_input_tokens,
            "llm_speaker_detection_output_tokens": self.llm_speaker_detection_output_tokens,
            "llm_summarization_calls": self.llm_summarization_calls,
            "llm_summarization_input_tokens": self.llm_summarization_input_tokens,
            "llm_summarization_output_tokens": self.llm_summarization_output_tokens,
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
            # Episode statuses
            "episode_statuses": [asdict(status) for status in self.episode_statuses],
            # Device usage per stage (Issue #387)
            "transcription_device": self.transcription_device,
            "summarization_device": self.summarization_device,
        }

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
            logger.info(f"Pipeline metrics saved to: {filepath}")
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
            "time_io_and_waiting",  # Backward compatibility (deprecated)
            "time_writing_storage",
            # Sub-buckets for io_and_waiting (Issue #387)
            "time_download_wait_seconds",
            "time_transcription_wait_seconds",
            "time_summarization_wait_seconds",
            "time_thread_sync_seconds",
            "time_queue_wait_seconds",
            "schema_version",
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
            "time_io_and_waiting",  # Backward compatibility (deprecated)
            "time_writing_storage",
            "time_download_wait_seconds",
            "time_transcription_wait_seconds",
            "time_summarization_wait_seconds",
            "time_thread_sync_seconds",
            "time_queue_wait_seconds",
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
