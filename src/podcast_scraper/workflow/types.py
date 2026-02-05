"""Type definitions for workflow pipeline.

This module contains NamedTuple types used throughout the workflow pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Set

from .. import models


class FeedMetadata(NamedTuple):
    """Feed metadata for metadata generation."""

    description: Optional[str]
    image_url: Optional[str]
    last_updated: Optional[datetime]


class HostDetectionResult(NamedTuple):
    """Result of host detection and pattern analysis."""

    cached_hosts: Set[str]
    heuristics: Optional[Dict[str, Any]]
    speaker_detector: Any = None  # Stage 3: Optional SpeakerDetector instance


class TranscriptionResources(NamedTuple):
    """Resources needed for transcription."""

    transcription_provider: Any  # Stage 2: TranscriptionProvider instance
    temp_dir: Optional[str]
    transcription_jobs: List[models.TranscriptionJob]
    transcription_jobs_lock: Optional[Any]  # threading.Lock
    saved_counter_lock: Optional[Any]  # threading.Lock


class ProcessingJob(NamedTuple):
    """Job for processing (metadata/summarization) stage."""

    episode: models.Episode
    transcript_path: str
    transcript_source: Literal["direct_download", "whisper_transcription"]
    detected_names: Optional[List[str]]
    whisper_model: Optional[str]


class ProcessingResources(NamedTuple):
    """Resources needed for processing stage."""

    processing_jobs: List[ProcessingJob]
    processing_jobs_lock: Optional[Any]  # threading.Lock
    processing_complete_event: Optional[Any]  # threading.Event


class ProviderCallMetrics(NamedTuple):
    """Metrics from a single provider call (transcription or summarization)."""

    prompt_tokens: Optional[int] = None  # Input tokens used
    completion_tokens: Optional[int] = None  # Output tokens used
    retries: int = 0  # Number of retries attempted
    rate_limit_sleep_sec: float = 0.0  # Time spent sleeping due to rate limits
    estimated_cost: Optional[float] = None  # Estimated cost in USD
