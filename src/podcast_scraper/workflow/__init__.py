"""Workflow package for pipeline orchestration.

This package contains helper modules and stage-specific modules extracted
from workflow.py to improve maintainability.

Note: This __init__.py exists to make workflow/ a proper package so that
workflow.py can import from .workflow.helpers, .workflow.stages, etc.
The main workflow functions (run_pipeline) are in workflow.py, not here.
"""

# Re-export main functions from workflow.py to maintain backward compatibility
# This allows "from podcast_scraper.workflow import run_pipeline" to work
# even though workflow.py and workflow/ have the same name
import sys
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Type stubs for dynamically re-exported attributes from workflow.py
    # These are set dynamically at runtime via _re_export_workflow_functions
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

# Export types for convenience
from .types import (
    FeedMetadata,
    HostDetectionResult,
    ProcessingJob,
    ProcessingResources,
    TranscriptionResources,
)

# Import from the parent workflow.py file
_workflow_py_path = Path(__file__).parent.parent / "workflow.py"


def _re_export_workflow_functions(_workflow_module: Any) -> None:
    """Re-export functions from workflow.py for backward compatibility.

    This function reduces complexity by grouping all re-exports together.
    """
    current_module = sys.modules[__name__]

    # Re-export run_pipeline and other public functions
    current_module.run_pipeline = _workflow_module.run_pipeline  # type: ignore
    current_module.apply_log_level = _workflow_module.apply_log_level  # type: ignore
    # Re-export type aliases for backward compatibility
    current_module._FeedMetadata = _workflow_module._FeedMetadata  # type: ignore
    current_module._HostDetectionResult = _workflow_module._HostDetectionResult  # type: ignore
    current_module._TranscriptionResources = (  # type: ignore
        _workflow_module._TranscriptionResources
    )
    current_module._ProcessingJob = _workflow_module._ProcessingJob  # type: ignore
    current_module._ProcessingResources = _workflow_module._ProcessingResources  # type: ignore

    # Re-export backward compatibility wrappers for private functions
    # Only export functions that actually exist in workflow.py
    # Note: Most stage functions have been moved to workflow/stages modules
    # and are no longer defined in workflow.py
    _private_functions = [
        "_preload_ml_models_if_needed",
    ]
    for func_name in _private_functions:
        if hasattr(_workflow_module, func_name):
            setattr(current_module, func_name, getattr(_workflow_module, func_name))

    # Re-export functions that tests patch
    # Note: Helper functions (_cleanup_pipeline, _generate_pipeline_summary, _update_metric_safely)
    # and stage functions have been moved to workflow/helpers and workflow/stages modules
    _test_functions = [
        "transcribe_media_to_text",
        "create_summarization_provider",
        "_preloaded_ml_provider",
    ]
    for func_name in _test_functions:
        if hasattr(_workflow_module, func_name):
            setattr(current_module, func_name, getattr(_workflow_module, func_name))

    # Re-export modules that tests patch
    _test_modules = ["filesystem", "os", "time", "shutil", "progress", "metadata", "logger"]
    for module_name in _test_modules:
        if hasattr(_workflow_module, module_name):
            setattr(current_module, module_name, getattr(_workflow_module, module_name))

    # Re-export factory functions that tests patch
    _factory_functions = [
        "create_speaker_detector",
        "create_transcription_provider",
    ]
    for func_name in _factory_functions:
        if hasattr(_workflow_module, func_name):
            setattr(current_module, func_name, getattr(_workflow_module, func_name))

    # Re-export rss_parser functions that tests patch
    if hasattr(_workflow_module, "extract_episode_description"):
        current_module.extract_episode_description = (  # type: ignore
            _workflow_module.extract_episode_description
        )
    # Re-export episode_processor functions that tests patch
    if hasattr(_workflow_module, "process_episode_download"):
        current_module.process_episode_download = (  # type: ignore
            _workflow_module.process_episode_download
        )


if _workflow_py_path.exists():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "podcast_scraper.workflow_module", _workflow_py_path
    )
    if spec and spec.loader:
        _workflow_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_workflow_module)
        _re_export_workflow_functions(_workflow_module)

__all__ = [
    "FeedMetadata",
    "HostDetectionResult",
    "ProcessingJob",
    "ProcessingResources",
    "TranscriptionResources",
    "run_pipeline",
    "apply_log_level",
]
