"""Provider parameter models and utilities for AI experiments.

This package provides typed parameter models for provider configuration,
enabling experiment-style parameter passing while maintaining backward
compatibility with Config-based usage.
"""

from __future__ import annotations

# Re-export for convenience
from .params import (
    SpeakerDetectionParams,
    SummarizationParams,
    TranscriptionParams,
)

__all__ = [
    "SummarizationParams",
    "TranscriptionParams",
    "SpeakerDetectionParams",
]
