"""Workflow orchestration and pipeline execution.

This package provides:
- Pipeline orchestration (orchestration.py)
- Workflow stages (setup, scraping, transcription, processing, etc.)
- Episode processing utilities
- Metrics collection
"""

from __future__ import annotations

import sys
from pathlib import Path
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
    # Type stubs for dynamically re-exported attributes from orchestration.py
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

# Import from orchestration.py in this package
_orchestration_py_path = Path(__file__).parent / "orchestration.py"


def _re_export_orchestration_functions(_orchestration_module: Any) -> None:
    """Re-export public functions from orchestration.py."""
    current_module = sys.modules[__name__]

    # Re-export run_pipeline and other public functions
    current_module.run_pipeline = _orchestration_module.run_pipeline  # type: ignore
    current_module.apply_log_level = _orchestration_module.apply_log_level  # type: ignore


if _orchestration_py_path.exists():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "podcast_scraper.workflow.orchestration", _orchestration_py_path
    )
    if spec and spec.loader:
        _orchestration_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_orchestration_module)
        _re_export_orchestration_functions(_orchestration_module)

# Re-export _preloaded_ml_provider from orchestration for factory reuse
# Import the module so we can access the attribute dynamically
# Re-export metadata_generation for convenience
from . import metadata_generation, orchestration


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
