"""Run summary generation combining run manifest and pipeline metrics.

This module creates run.json files that combine run manifest (system state)
with pipeline metrics (performance data) for complete run records.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def create_run_summary(
    run_manifest: Optional[Any],
    pipeline_metrics: Optional[Any],
    output_dir: str,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create run summary combining manifest and metrics.

    Args:
        run_manifest: RunManifest object (optional)
        pipeline_metrics: Metrics object (optional)
        output_dir: Output directory path
        run_id: Optional run identifier

    Returns:
        Dictionary containing run summary
    """
    summary: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_id": run_id or datetime.utcnow().isoformat() + "Z",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    # Add manifest if available
    if run_manifest:
        if hasattr(run_manifest, "to_dict"):
            summary["manifest"] = run_manifest.to_dict()
        else:
            summary["manifest"] = run_manifest

    # Add metrics if available
    if pipeline_metrics:
        metrics_summary: Dict[str, Any] = {}

        # Basic metrics
        if hasattr(pipeline_metrics, "finish"):
            metrics_dict = pipeline_metrics.finish()
            metrics_summary.update(metrics_dict)
        else:
            # Fallback: try to get metrics directly
            if hasattr(pipeline_metrics, "run_duration_seconds"):
                metrics_summary["run_duration_seconds"] = pipeline_metrics.run_duration_seconds
            if hasattr(pipeline_metrics, "episodes_scraped_total"):
                metrics_summary["episodes_scraped_total"] = pipeline_metrics.episodes_scraped_total
            if hasattr(pipeline_metrics, "episodes_skipped_total"):
                metrics_summary["episodes_skipped_total"] = pipeline_metrics.episodes_skipped_total
            if hasattr(pipeline_metrics, "errors_total"):
                metrics_summary["errors_total"] = pipeline_metrics.errors_total

        # Stage timings
        if hasattr(pipeline_metrics, "time_scraping"):
            metrics_summary["time_scraping_seconds"] = pipeline_metrics.time_scraping
        if hasattr(pipeline_metrics, "time_parsing"):
            metrics_summary["time_parsing_seconds"] = pipeline_metrics.time_parsing
        if hasattr(pipeline_metrics, "time_normalizing"):
            metrics_summary["time_normalizing_seconds"] = pipeline_metrics.time_normalizing
        if hasattr(pipeline_metrics, "io_and_waiting_thread_sum_seconds"):
            metrics_summary["io_and_waiting_thread_sum_seconds"] = (
                pipeline_metrics.io_and_waiting_thread_sum_seconds
            )
        if hasattr(pipeline_metrics, "io_and_waiting_wall_seconds"):
            metrics_summary["io_and_waiting_wall_seconds"] = (
                pipeline_metrics.io_and_waiting_wall_seconds
            )
        # Backward compatibility (deprecated)
        if hasattr(pipeline_metrics, "time_io_and_waiting"):
            metrics_summary["time_io_and_waiting_seconds"] = (
                pipeline_metrics.io_and_waiting_thread_sum_seconds
            )

        # Model loading times
        if hasattr(pipeline_metrics, "whisper_model_loading_time"):
            metrics_summary["whisper_model_loading_time_seconds"] = (
                pipeline_metrics.whisper_model_loading_time
            )
        if hasattr(pipeline_metrics, "summarization_model_loading_time"):
            metrics_summary["summarization_model_loading_time_seconds"] = (
                pipeline_metrics.summarization_model_loading_time
            )

        # Memory usage
        if hasattr(pipeline_metrics, "peak_memory_mb"):
            metrics_summary["peak_memory_mb"] = pipeline_metrics.peak_memory_mb
        if hasattr(pipeline_metrics, "initial_memory_mb"):
            metrics_summary["initial_memory_mb"] = pipeline_metrics.initial_memory_mb

        # Episode statuses
        if hasattr(pipeline_metrics, "episode_statuses"):
            metrics_summary["episode_statuses"] = pipeline_metrics.episode_statuses

        summary["metrics"] = metrics_summary

    return summary


def save_run_summary(
    run_summary: Dict[str, Any],
    output_dir: str,
    filename: str = "run.json",
) -> str:
    """Save run summary to JSON file.

    Args:
        run_summary: Run summary dictionary
        output_dir: Output directory path
        filename: Output filename (default: "run.json")

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_json = json.dumps(run_summary, indent=2, default=str)
    output_path.write_text(summary_json, encoding="utf-8")
    logger.info(f"Run summary saved to: {output_path}")

    return str(output_path)
