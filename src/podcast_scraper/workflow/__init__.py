"""Workflow orchestration and pipeline execution.

This package provides:
- Pipeline orchestration (orchestration.py)
- Workflow stages (setup, scraping, transcription, processing, etc.)
- Episode processing utilities
- Metrics collection
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Optional, TYPE_CHECKING

# Re-export metrics for convenience
from . import metrics

# Export types for convenience
from .types import (
    FeedMetadata,
    HostDetectionResult,
    ProcessingJob,
    ProcessingResources,
    TranscriptionResources,
)

if TYPE_CHECKING:
    # Type stubs for attributes from orchestration
    _preloaded_ml_provider: Optional[Any]
    run_pipeline: Callable[..., Any]
    apply_log_level: Callable[..., Any]
    _preload_ml_models_if_needed: Callable[..., Any]
    transcribe_media_to_text: Callable[..., Any]
    create_summarization_provider: Callable[..., Any]
    create_speaker_detector: Callable[..., Any]
    create_transcription_provider: Callable[..., Any]
    extract_episode_description: Callable[..., Any]
    process_episode_download: Callable[..., Any]

# Re-export metadata_generation and orchestration; expose run_pipeline and apply_log_level
from . import metadata_generation, orchestration
from .orchestration import apply_log_level, run_pipeline


# Expose _preloaded_ml_provider as a reference to orchestration's attribute
# This ensures factories always get the current value, even after it's set in setup.py
# We use __getattr__ to make it work like a module-level attribute
def __getattr__(name: str) -> Any:
    """Dynamically access attributes from orchestration module."""
    if name == "_preloaded_ml_provider":
        return getattr(orchestration, "_preloaded_ml_provider", None)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FeedMetadata",
    "HostDetectionResult",
    "ProcessingJob",
    "ProcessingResources",
    "TranscriptionResources",
    "metrics",
    "metadata_generation",
    "run_pipeline",
    "apply_log_level",
]
