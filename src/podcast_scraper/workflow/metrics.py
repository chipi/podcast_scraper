"""Simple in-memory metrics collector for pipeline performance tracking."""

from __future__ import annotations

import json
import logging
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
    transcripts_transcribed: int = 0  # Whisper transcriptions
    episodes_summarized: int = 0  # Episodes with summaries generated
    metadata_files_generated: int = 0  # Metadata files created

    # Per-stage metrics
    time_scraping: float = 0.0
    time_parsing: float = 0.0
    time_normalizing: float = 0.0
    time_io_and_waiting: float = (
        0.0  # Renamed from time_writing_storage - includes downloads,
        # transcription waiting, thread sync, and actual I/O
    )
    time_writing_storage: float = (
        0.0  # Actual file write time only (open/write/flush) - should be tiny
    )

    # Per-episode operation timing (for A/B testing and performance analysis)
    download_media_times: List[float] = field(
        default_factory=list
    )  # Media download times per episode
    download_media_attempts: int = 0  # Total media download attempts (including reused/cached)
    transcribe_times: List[float] = field(default_factory=list)  # Transcription times per episode
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

    # Audio preprocessing metrics (RFC-040)
    preprocessing_times: List[float] = field(
        default_factory=list
    )  # Preprocessing times per episode
    preprocessing_original_sizes: List[int] = field(
        default_factory=list
    )  # Original audio file sizes in bytes
    preprocessing_preprocessed_sizes: List[int] = field(
        default_factory=list
    )  # Preprocessed audio file sizes in bytes
    preprocessing_attempts: int = 0  # Total preprocessing attempts (cache hits + misses)
    preprocessing_cache_hits: int = 0  # Number of cache hits
    preprocessing_cache_misses: int = 0  # Number of cache misses

    # Per-episode status tracking (Issue #379)
    episode_statuses: List[EpisodeStatus] = field(default_factory=list)

    _start_time: float = field(default_factory=time.time, init=False)

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
            self.time_io_and_waiting += duration
        elif stage == "writing_storage":
            # Actual file write time only (should be tiny)
            self.time_writing_storage += duration

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
        """Record status for an episode.

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
        self.episode_statuses.append(episode_status)

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
            "time_io_and_waiting": round(self.time_io_and_waiting, 2),
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
            "transcribe_count": len(self.transcribe_times),
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
            # Audio preprocessing metrics
            "avg_preprocessing_seconds": avg_preprocessing,
            "preprocessing_count": len(self.preprocessing_times),
            "preprocessing_attempts": self.preprocessing_attempts,  # Total attempts (hits + misses)
            "avg_preprocessing_size_reduction_percent": avg_size_reduction_percent,
            "preprocessing_cache_hits": self.preprocessing_cache_hits,
            "preprocessing_cache_misses": self.preprocessing_cache_misses,
            # Episode statuses
            "episode_statuses": [asdict(status) for status in self.episode_statuses],
        }

    def to_json(self) -> str:
        """Convert metrics to JSON string.

        Returns:
            JSON string representation of all metrics
        """
        metrics_dict = self.finish()
        return json.dumps(metrics_dict, indent=2)

    def save_to_file(self, filepath: str | Path) -> None:
        """Save metrics to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        metrics_json = self.to_json()
        filepath.write_text(metrics_json, encoding="utf-8")
        logger.info(f"Pipeline metrics saved to: {filepath}")

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
