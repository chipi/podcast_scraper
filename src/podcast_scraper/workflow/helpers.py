"""Helper utilities for workflow pipeline.

This module contains utility functions used throughout the workflow pipeline.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Dict, List, Optional, Tuple

from .. import config, metrics
from ..workflow.types import TranscriptionResources

logger = logging.getLogger(__name__)


def update_metric_safely(
    pipeline_metrics: metrics.Metrics,
    metric_name: str,
    value: int,
    lock: Optional[threading.Lock] = None,
) -> None:
    """Update a metric value in a thread-safe manner.

    Args:
        pipeline_metrics: Metrics object to update
        metric_name: Name of the metric attribute to update
        value: Value to add to the metric
        lock: Optional lock for thread safety
    """
    if lock:
        with lock:
            current_value = getattr(pipeline_metrics, metric_name, 0)
            setattr(pipeline_metrics, metric_name, current_value + value)
    else:
        current_value = getattr(pipeline_metrics, metric_name, 0)
        setattr(pipeline_metrics, metric_name, current_value + value)


def cleanup_pipeline(temp_dir: Optional[str]) -> None:
    """Cleanup temporary files and directories.

    Args:
        temp_dir: Path to temporary directory (if any)
    """
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temp directory: {temp_dir}")
        except OSError as exc:
            logger.debug(f"Failed to remove temp directory {temp_dir}: {exc}")


def generate_pipeline_summary(
    cfg: config.Config,
    saved: int,
    transcription_resources: TranscriptionResources,
    effective_output_dir: str,
    pipeline_metrics: metrics.Metrics,
) -> Tuple[int, str]:
    """Generate pipeline summary message with detailed statistics.

    Creates a human-readable summary of pipeline execution, including counts
    of transcripts processed, performance metrics (download/transcription times),
    and output location. In dry-run mode, reports planned operations instead
    of actual results.

    Args:
        cfg: Configuration object (checks dry_run, transcribe_missing flags)
        saved: Number of transcripts successfully saved or planned
        transcription_resources: Transcription resources containing the job queue
        effective_output_dir: Full path to output directory for display
        pipeline_metrics: Metrics collector with timing data for operations

    Returns:
        Tuple[int, str]: A tuple containing:
            - count (int): Total episodes processed (saved + transcribed)
            - summary (str): Multi-line summary message with statistics and metrics
    """
    if cfg.dry_run:
        planned_downloads = saved
        planned_transcriptions = (
            len(transcription_resources.transcription_jobs) if cfg.transcribe_missing else 0
        )
        planned_total = planned_downloads + planned_transcriptions
        logger.debug(
            "Dry-run summary: planned_downloads=%s planned_transcriptions=%s",
            planned_downloads,
            planned_transcriptions,
        )
        # Print each metric on its own line for better readability
        summary_lines = [f"Dry run complete. transcripts_planned={planned_total}"]
        summary_lines.append(f"  - Direct downloads planned: {planned_downloads}")
        summary_lines.append(f"  - Whisper transcriptions planned: {planned_transcriptions}")
        summary_lines.append(f"  - Output directory: {effective_output_dir}")
        summary = "\n".join(summary_lines)
        # Don't log here - caller (cli.py) will log the summary
        return planned_total, summary
    else:
        # Build detailed summary with statistics
        # Print each metric on its own line for better readability
        summary_lines = [f"Done. transcripts_saved={saved}"]

        # Add breakdown by source
        if pipeline_metrics.transcripts_downloaded > 0:
            summary_lines.append(
                f"  - Transcripts downloaded: {pipeline_metrics.transcripts_downloaded}"
            )
        if pipeline_metrics.transcripts_transcribed > 0:
            summary_lines.append(
                f"  - Episodes transcribed: {pipeline_metrics.transcripts_transcribed}"
            )

        # Add metadata and summary statistics
        if cfg.generate_metadata and pipeline_metrics.metadata_files_generated > 0:
            summary_lines.append(
                f"  - Metadata files generated: {pipeline_metrics.metadata_files_generated}"
            )
        if cfg.generate_summaries and pipeline_metrics.episodes_summarized > 0:
            summary_lines.append(f"  - Episodes summarized: {pipeline_metrics.episodes_summarized}")

        # Add error count if any
        if pipeline_metrics.errors_total > 0:
            summary_lines.append(f"  - Errors: {pipeline_metrics.errors_total}")

        # Add skipped count if any
        if pipeline_metrics.episodes_skipped_total > 0:
            summary_lines.append(f"  - Episodes skipped: {pipeline_metrics.episodes_skipped_total}")

        # Add performance metrics (averages per episode) - one per line
        metrics_dict = pipeline_metrics.finish()
        if metrics_dict.get("avg_download_media_seconds", 0) > 0:
            avg_download = metrics_dict["avg_download_media_seconds"]
            summary_lines.append(f"  - Average download time: {avg_download:.1f}s/episode")
        if metrics_dict.get("avg_transcribe_seconds", 0) > 0:
            avg_transcribe = metrics_dict["avg_transcribe_seconds"]
            summary_lines.append(f"  - Average transcription time: {avg_transcribe:.1f}s/episode")
        if metrics_dict.get("avg_extract_names_seconds", 0) > 0:
            avg_extract = metrics_dict["avg_extract_names_seconds"]
            summary_lines.append(f"  - Average name extraction time: {avg_extract:.1f}s/episode")
        if metrics_dict.get("avg_summarize_seconds", 0) > 0:
            avg_summarize = metrics_dict["avg_summarize_seconds"]
            summary_lines.append(f"  - Average summary time: {avg_summarize:.1f}s/episode")
        if metrics_dict.get("avg_preprocessing_seconds", 0) > 0:
            avg_preprocessing = metrics_dict["avg_preprocessing_seconds"]
            preprocessing_count = metrics_dict.get("preprocessing_count", 0)
            size_reduction = metrics_dict.get("avg_preprocessing_size_reduction_percent", 0.0)
            cache_hits = metrics_dict.get("preprocessing_cache_hits", 0)
            cache_misses = metrics_dict.get("preprocessing_cache_misses", 0)
            summary_lines.append(
                f"  - Average preprocessing time: {avg_preprocessing:.1f}s/episode "
                f"({preprocessing_count} processed, {size_reduction:.1f}% size reduction)"
            )
            if cache_hits > 0 or cache_misses > 0:
                total_cache_ops = cache_hits + cache_misses
                cache_hit_rate = (
                    (cache_hits / total_cache_ops * 100) if total_cache_ops > 0 else 0.0
                )
                summary_lines.append(
                    f"  - Preprocessing cache: {cache_hits} hits, {cache_misses} misses "
                    f"({cache_hit_rate:.1f}% hit rate)"
                )

        summary_lines.append(f"  - Output directory: {effective_output_dir}")

        # Add LLM call summary and cost estimation if any LLM provider was used
        llm_summary = _generate_llm_call_summary(cfg, pipeline_metrics)
        if llm_summary:
            summary_lines.append("")
            summary_lines.append("LLM API Usage:")
            summary_lines.extend(llm_summary)

        # Join all lines
        summary = "\n".join(summary_lines)
        # Don't log here - caller (cli.py) will log the summary
        return saved, summary


def _get_provider_pricing(
    cfg: config.Config, provider_type: str, capability: str, model: str
) -> Dict[str, float]:
    """Get pricing information from the appropriate provider.

    Args:
        cfg: Configuration object
        provider_type: Provider type ("openai", "whisper", etc.)
        capability: Capability type ("transcription", "speaker_detection", "summarization")
        model: Model name

    Returns:
        Dictionary with pricing information, or empty dict if not available
    """
    if provider_type == "openai":
        from ..openai.openai_provider import OpenAIProvider

        return OpenAIProvider.get_pricing(model, capability)
    # Add other providers here as they're implemented
    # elif provider_type == "anthropic":
    #     from ..anthropic.anthropic_provider import AnthropicProvider
    #     return AnthropicProvider.get_pricing(model, capability)
    return {}


def _generate_llm_call_summary(cfg: config.Config, pipeline_metrics: metrics.Metrics) -> List[str]:
    """Generate summary of LLM API calls and estimated costs.

    Args:
        cfg: Configuration object to check which providers were used
        pipeline_metrics: Metrics object with LLM call tracking data

    Returns:
        List of summary lines, or empty list if no LLM calls were made
    """
    summary_lines: List[str] = []
    metrics_dict = pipeline_metrics.finish()

    # Check if any LLM provider was used
    uses_openai_transcription = cfg.transcription_provider == "openai"
    uses_openai_speaker = cfg.speaker_detector_provider == "openai"
    uses_openai_summarization = cfg.summary_provider == "openai"

    if not (uses_openai_transcription or uses_openai_speaker or uses_openai_summarization):
        return summary_lines

    total_cost = 0.0

    # Transcription calls
    if uses_openai_transcription:
        transcription_calls = metrics_dict.get("llm_transcription_calls", 0)
        audio_minutes = metrics_dict.get("llm_transcription_audio_minutes", 0.0)
        if transcription_calls > 0:
            model = getattr(cfg, "openai_transcription_model", "whisper-1")
            pricing = _get_provider_pricing(cfg, "openai", "transcription", model)
            if pricing and "cost_per_minute" in pricing:
                transcription_cost = audio_minutes * pricing["cost_per_minute"]
                total_cost += transcription_cost
                summary_lines.append(
                    f"  - Transcription: {transcription_calls} calls, "
                    f"{audio_minutes:.1f} minutes, ${transcription_cost:.4f} "
                    f"(model: {model})"
                )

    # Speaker detection calls
    if uses_openai_speaker:
        speaker_calls = metrics_dict.get("llm_speaker_detection_calls", 0)
        speaker_input_tokens = metrics_dict.get("llm_speaker_detection_input_tokens", 0)
        speaker_output_tokens = metrics_dict.get("llm_speaker_detection_output_tokens", 0)
        if speaker_calls > 0:
            model = getattr(cfg, "openai_speaker_model", "gpt-4o-mini")
            pricing = _get_provider_pricing(cfg, "openai", "speaker_detection", model)
            if pricing and "input_cost_per_1m_tokens" in pricing:
                input_cost = (speaker_input_tokens / 1_000_000) * pricing[
                    "input_cost_per_1m_tokens"
                ]
                output_cost = (speaker_output_tokens / 1_000_000) * pricing[
                    "output_cost_per_1m_tokens"
                ]
                speaker_cost = input_cost + output_cost
                total_cost += speaker_cost
                summary_lines.append(
                    f"  - Speaker Detection: {speaker_calls} calls, "
                    f"{speaker_input_tokens:,} input + {speaker_output_tokens:,} output tokens, "
                    f"${speaker_cost:.4f} (model: {model})"
                )

    # Summarization calls
    if uses_openai_summarization:
        summary_calls = metrics_dict.get("llm_summarization_calls", 0)
        summary_input_tokens = metrics_dict.get("llm_summarization_input_tokens", 0)
        summary_output_tokens = metrics_dict.get("llm_summarization_output_tokens", 0)
        if summary_calls > 0:
            model = getattr(cfg, "openai_summary_model", "gpt-4o-mini")
            pricing = _get_provider_pricing(cfg, "openai", "summarization", model)
            if pricing and "input_cost_per_1m_tokens" in pricing:
                input_cost = (summary_input_tokens / 1_000_000) * pricing[
                    "input_cost_per_1m_tokens"
                ]
                output_cost = (summary_output_tokens / 1_000_000) * pricing[
                    "output_cost_per_1m_tokens"
                ]
                summary_cost = input_cost + output_cost
                total_cost += summary_cost
                summary_lines.append(
                    f"  - Summarization: {summary_calls} calls, "
                    f"{summary_input_tokens:,} input + {summary_output_tokens:,} output tokens, "
                    f"${summary_cost:.4f} (model: {model})"
                )

    # Add total cost if any calls were made
    if total_cost > 0:
        summary_lines.append(f"  - Total estimated cost: ${total_cost:.4f}")

    return summary_lines
