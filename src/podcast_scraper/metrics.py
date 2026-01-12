"""Simple in-memory metrics collector for pipeline performance tracking."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


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
    time_writing_storage: float = 0.0

    # Per-episode operation timing (for A/B testing and performance analysis)
    download_media_times: List[float] = field(
        default_factory=list
    )  # Media download times per episode
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
        elif stage == "writing_storage":
            self.time_writing_storage += duration

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

        return {
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
            "time_writing_storage": round(self.time_writing_storage, 2),
            # Per-episode averages
            "avg_download_media_seconds": avg_download_media,
            "avg_transcribe_seconds": avg_transcribe,
            "avg_extract_names_seconds": avg_extract_names,
            "avg_summarize_seconds": avg_summarize,
            # Operation counts for context
            "download_media_count": len(self.download_media_times),
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
