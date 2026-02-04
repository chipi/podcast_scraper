"""Helper utilities for workflow pipeline.

This module contains utility functions used throughout the workflow pipeline.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .. import config
from . import metrics
from .types import TranscriptionResources

if TYPE_CHECKING:
    from .. import models

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


def get_episode_id_from_episode(
    episode: "models.Episode", feed_url: str
) -> Tuple[str, Optional[int]]:
    """Generate episode ID from episode object (helper for status tracking).

    Args:
        episode: Episode object
        feed_url: RSS feed URL

    Returns:
        Tuple of (episode_id, episode_number)
    """
    from ..rss.parser import extract_episode_published_date
    from .metadata_generation import generate_episode_id

    # Extract episode metadata for ID generation
    episode_guid = None
    episode_link = None
    episode_published_date = None
    episode_number = getattr(episode, "number", None)

    if hasattr(episode, "item") and episode.item is not None:
        # Extract GUID from RSS item
        guid_elem = episode.item.find("guid")
        if guid_elem is not None and guid_elem.text:
            episode_guid = guid_elem.text.strip()
        # Extract link
        link_elem = episode.item.find("link")
        if link_elem is not None and link_elem.text:
            episode_link = link_elem.text.strip()
        # Extract published date
        episode_published_date = extract_episode_published_date(episode.item)

    # Generate stable episode ID
    episode_id = generate_episode_id(
        feed_url=feed_url,
        episode_title=episode.title,
        episode_guid=episode_guid,
        published_date=episode_published_date,
        episode_link=episode_link,
        episode_number=episode_number,
    )

    return episode_id, episode.idx


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
    episodes: Optional[List["models.Episode"]] = None,
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
        episodes: Optional list of episodes for cost projection in dry-run mode

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

        # Add cost projection if OpenAI providers are configured (Issue #253)
        cost_projection = _generate_dry_run_cost_projection(cfg, episodes, planned_total)
        if cost_projection:
            summary_lines.append("")
            summary_lines.append("Cost Projection (Dry Run):")
            summary_lines.append("=" * 30)
            summary_lines.extend(cost_projection)

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
        from ..providers.openai.openai_provider import OpenAIProvider

        return OpenAIProvider.get_pricing(model, capability)
    elif provider_type == "gemini":
        from ..providers.gemini.gemini_provider import GeminiProvider

        return GeminiProvider.get_pricing(model, capability)
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
    uses_gemini_transcription = cfg.transcription_provider == "gemini"
    uses_gemini_speaker = cfg.speaker_detector_provider == "gemini"
    uses_gemini_summarization = cfg.summary_provider == "gemini"

    if not (
        uses_openai_transcription
        or uses_openai_speaker
        or uses_openai_summarization
        or uses_gemini_transcription
        or uses_gemini_speaker
        or uses_gemini_summarization
    ):
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
    elif uses_gemini_transcription:
        transcription_calls = metrics_dict.get("llm_transcription_calls", 0)
        audio_minutes = metrics_dict.get("llm_transcription_audio_minutes", 0.0)
        if transcription_calls > 0:
            model = getattr(cfg, "gemini_transcription_model", "gemini-1.5-pro")
            pricing = _get_provider_pricing(cfg, "gemini", "transcription", model)
            if pricing and "cost_per_minute" in pricing:
                transcription_cost = audio_minutes * pricing["cost_per_minute"]
                total_cost += transcription_cost
                summary_lines.append(
                    f"  - Transcription: {transcription_calls} calls, "
                    f"{audio_minutes:.1f} minutes, ${transcription_cost:.4f} "
                    f"(model: {model})"
                )
    else:
        # Show that transcription was done with ML (free) to make cost savings clear
        transcripts_transcribed = metrics_dict.get("transcripts_transcribed", 0)
        if transcripts_transcribed > 0:
            transcription_provider = getattr(cfg, "transcription_provider", "whisper")
            summary_lines.append(
                f"  - Transcription: {transcripts_transcribed} episodes, $0.0000 "
                f"(provider: {transcription_provider}, local processing)"
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
    elif uses_gemini_speaker:
        speaker_calls = metrics_dict.get("llm_speaker_detection_calls", 0)
        speaker_input_tokens = metrics_dict.get("llm_speaker_detection_input_tokens", 0)
        speaker_output_tokens = metrics_dict.get("llm_speaker_detection_output_tokens", 0)
        if speaker_calls > 0:
            model = getattr(cfg, "gemini_speaker_model", "gemini-1.5-pro")
            pricing = _get_provider_pricing(cfg, "gemini", "speaker_detection", model)
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
    else:
        # Show that speaker detection was done with ML (free) to make cost savings clear
        extract_names_count = metrics_dict.get("extract_names_count", 0)
        if extract_names_count > 0:
            speaker_provider = getattr(cfg, "speaker_detector_provider", "spacy")
            summary_lines.append(
                f"  - Speaker Detection: {extract_names_count} episodes, $0.0000 "
                f"(provider: {speaker_provider}, local processing)"
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
    elif uses_gemini_summarization:
        summary_calls = metrics_dict.get("llm_summarization_calls", 0)
        summary_input_tokens = metrics_dict.get("llm_summarization_input_tokens", 0)
        summary_output_tokens = metrics_dict.get("llm_summarization_output_tokens", 0)
        if summary_calls > 0:
            model = getattr(cfg, "gemini_summary_model", "gemini-1.5-pro")
            pricing = _get_provider_pricing(cfg, "gemini", "summarization", model)
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
    else:
        # Show that summarization was done with ML (free) to make cost savings clear
        episodes_summarized = metrics_dict.get("episodes_summarized", 0)
        if episodes_summarized > 0:
            summary_provider = getattr(cfg, "summary_provider", "transformers")
            summary_lines.append(
                f"  - Summarization: {episodes_summarized} episodes, $0.0000 "
                f"(provider: {summary_provider}, local processing)"
            )

    # Add total cost if any calls were made
    if total_cost > 0:
        summary_lines.append(f"  - Total estimated cost: ${total_cost:.4f}")

    return summary_lines


def _generate_dry_run_cost_projection(
    cfg: config.Config,
    episodes: Optional[List["models.Episode"]],
    episode_count: int,
) -> List[str]:
    """Generate cost projection for dry-run mode based on configured LLM providers.

    Estimates API costs for OpenAI/Gemini transcription, speaker detection, and summarization
    based on episode count and available metadata (duration). Only displays projection
    when LLM providers are configured.

    Args:
        cfg: Configuration object to check provider settings
        episodes: Optional list of episodes for duration-based estimation
        episode_count: Total number of episodes to process

    Returns:
        List of cost projection lines, or empty list if no LLM providers configured
    """
    summary_lines: List[str] = []

    # Check if any LLM provider is configured
    uses_openai_transcription = cfg.transcription_provider == "openai"
    uses_openai_speaker = cfg.speaker_detector_provider == "openai"
    uses_openai_summarization = cfg.summary_provider == "openai"
    uses_gemini_transcription = cfg.transcription_provider == "gemini"
    uses_gemini_speaker = cfg.speaker_detector_provider == "gemini"
    uses_gemini_summarization = cfg.summary_provider == "gemini"

    if not (
        uses_openai_transcription
        or uses_openai_speaker
        or uses_openai_summarization
        or uses_gemini_transcription
        or uses_gemini_speaker
        or uses_gemini_summarization
    ):
        return summary_lines

    # Extract episode durations if available
    episode_durations: List[int] = []
    if episodes:
        from ..rss.parser import extract_episode_metadata

        # base_url not needed for duration extraction (only used for URL resolution)
        for episode in episodes:
            # Extract duration from RSS item metadata
            # extract_episode_metadata returns: (description, guid, link, duration_seconds, ...)
            _, _, _, duration, _, _ = extract_episode_metadata(episode.item, "")
            if duration:
                episode_durations.append(duration)

    # Calculate average duration for estimation
    avg_duration_minutes = 0.0
    if episode_durations:
        avg_duration_seconds = sum(episode_durations) / len(episode_durations)
        avg_duration_minutes = avg_duration_seconds / 60.0
    else:
        # Conservative fallback: assume 30 minutes per episode if no duration available
        avg_duration_minutes = 30.0

    total_cost = 0.0

    # Transcription cost estimation
    if uses_openai_transcription:
        transcription_episodes = episode_count
        model = getattr(cfg, "openai_transcription_model", "whisper-1")
        pricing = _get_provider_pricing(cfg, "openai", "transcription", model)
        if pricing and "cost_per_minute" in pricing:
            total_audio_minutes = transcription_episodes * avg_duration_minutes
            transcription_cost = total_audio_minutes * pricing["cost_per_minute"]
            total_cost += transcription_cost
            summary_lines.append(
                f"Transcription ({model}):\n"
                f"  - Episodes: {transcription_episodes}\n"
                f"  - Estimated audio: {total_audio_minutes:.1f} minutes\n"
                f"  - Estimated cost: ${transcription_cost:.4f}"
            )
    elif uses_gemini_transcription:
        transcription_episodes = episode_count
        model = getattr(cfg, "gemini_transcription_model", "gemini-1.5-pro")
        pricing = _get_provider_pricing(cfg, "gemini", "transcription", model)
        if pricing and "cost_per_minute" in pricing:
            total_audio_minutes = transcription_episodes * avg_duration_minutes
            transcription_cost = total_audio_minutes * pricing["cost_per_minute"]
            total_cost += transcription_cost
            summary_lines.append(
                f"Transcription ({model}):\n"
                f"  - Episodes: {transcription_episodes}\n"
                f"  - Estimated audio: {total_audio_minutes:.1f} minutes\n"
                f"  - Estimated cost: ${transcription_cost:.4f}"
            )

    # Speaker detection cost estimation
    if uses_openai_speaker:
        speaker_episodes = episode_count
        model = getattr(cfg, "openai_speaker_model", "gpt-4o-mini")
        pricing = _get_provider_pricing(cfg, "openai", "speaker_detection", model)
        if pricing and "input_cost_per_1m_tokens" in pricing:
            # Estimate tokens: ~150 words/minute speaking rate, ~1.3 tokens/word
            # Plus prompt overhead (~200 tokens for system + user prompt)
            words_per_minute = 150.0
            tokens_per_word = 1.3
            prompt_overhead_tokens = 200

            # Estimate transcript tokens from duration
            transcript_tokens_per_episode = int(
                avg_duration_minutes * words_per_minute * tokens_per_word
            )
            input_tokens_per_episode = transcript_tokens_per_episode + prompt_overhead_tokens
            # Output: typically small JSON response (~50 tokens)
            output_tokens_per_episode = 50

            total_input_tokens = speaker_episodes * input_tokens_per_episode
            total_output_tokens = speaker_episodes * output_tokens_per_episode

            input_cost = (total_input_tokens / 1_000_000) * pricing["input_cost_per_1m_tokens"]
            output_cost = (total_output_tokens / 1_000_000) * pricing["output_cost_per_1m_tokens"]
            speaker_cost = input_cost + output_cost
            total_cost += speaker_cost

            summary_lines.append(
                f"Speaker Detection ({model}):\n"
                f"  - Episodes: {speaker_episodes}\n"
                f"  - Estimated tokens: ~{total_input_tokens:,} input + "
                f"~{total_output_tokens:,} output\n"
                f"  - Estimated cost: ${speaker_cost:.4f}"
            )
    elif uses_gemini_speaker:
        speaker_episodes = episode_count
        model = getattr(cfg, "gemini_speaker_model", "gemini-1.5-pro")
        pricing = _get_provider_pricing(cfg, "gemini", "speaker_detection", model)
        if pricing and "input_cost_per_1m_tokens" in pricing:
            # Estimate tokens: ~150 words/minute speaking rate, ~1.3 tokens/word
            # Plus prompt overhead (~200 tokens for system + user prompt)
            words_per_minute = 150.0
            tokens_per_word = 1.3
            prompt_overhead_tokens = 200

            # Estimate transcript tokens from duration
            transcript_tokens_per_episode = int(
                avg_duration_minutes * words_per_minute * tokens_per_word
            )
            input_tokens_per_episode = transcript_tokens_per_episode + prompt_overhead_tokens
            # Output: typically small JSON response (~50 tokens)
            output_tokens_per_episode = 50

            total_input_tokens = speaker_episodes * input_tokens_per_episode
            total_output_tokens = speaker_episodes * output_tokens_per_episode

            input_cost = (total_input_tokens / 1_000_000) * pricing["input_cost_per_1m_tokens"]
            output_cost = (total_output_tokens / 1_000_000) * pricing["output_cost_per_1m_tokens"]
            speaker_cost = input_cost + output_cost
            total_cost += speaker_cost

            summary_lines.append(
                f"Speaker Detection ({model}):\n"
                f"  - Episodes: {speaker_episodes}\n"
                f"  - Estimated tokens: ~{total_input_tokens:,} input + "
                f"~{total_output_tokens:,} output\n"
                f"  - Estimated cost: ${speaker_cost:.4f}"
            )

    # Summarization cost estimation
    if uses_openai_summarization:
        summary_episodes = episode_count
        model = getattr(cfg, "openai_summary_model", "gpt-4o-mini")
        pricing = _get_provider_pricing(cfg, "openai", "summarization", model)
        if pricing and "input_cost_per_1m_tokens" in pricing:
            # Estimate tokens: same as speaker detection for input (transcript)
            # Plus prompt overhead (~300 tokens for summarization prompt)
            words_per_minute = 150.0
            tokens_per_word = 1.3
            prompt_overhead_tokens = 300

            transcript_tokens_per_episode = int(
                avg_duration_minutes * words_per_minute * tokens_per_word
            )
            input_tokens_per_episode = transcript_tokens_per_episode + prompt_overhead_tokens
            # Output: summary is typically 100-200 words (~150 tokens)
            output_tokens_per_episode = 150

            total_input_tokens = summary_episodes * input_tokens_per_episode
            total_output_tokens = summary_episodes * output_tokens_per_episode

            input_cost = (total_input_tokens / 1_000_000) * pricing["input_cost_per_1m_tokens"]
            output_cost = (total_output_tokens / 1_000_000) * pricing["output_cost_per_1m_tokens"]
            summary_cost = input_cost + output_cost
            total_cost += summary_cost

            summary_lines.append(
                f"Summarization ({model}):\n"
                f"  - Episodes: {summary_episodes}\n"
                f"  - Estimated tokens: ~{total_input_tokens:,} input + "
                f"~{total_output_tokens:,} output\n"
                f"  - Estimated cost: ${summary_cost:.4f}"
            )
    elif uses_gemini_summarization:
        summary_episodes = episode_count
        model = getattr(cfg, "gemini_summary_model", "gemini-1.5-pro")
        pricing = _get_provider_pricing(cfg, "gemini", "summarization", model)
        if pricing and "input_cost_per_1m_tokens" in pricing:
            # Estimate tokens: same as speaker detection for input (transcript)
            # Plus prompt overhead (~300 tokens for summarization prompt)
            words_per_minute = 150.0
            tokens_per_word = 1.3
            prompt_overhead_tokens = 300

            transcript_tokens_per_episode = int(
                avg_duration_minutes * words_per_minute * tokens_per_word
            )
            input_tokens_per_episode = transcript_tokens_per_episode + prompt_overhead_tokens
            # Output: summary is typically 100-200 words (~150 tokens)
            output_tokens_per_episode = 150

            total_input_tokens = summary_episodes * input_tokens_per_episode
            total_output_tokens = summary_episodes * output_tokens_per_episode

            input_cost = (total_input_tokens / 1_000_000) * pricing["input_cost_per_1m_tokens"]
            output_cost = (total_output_tokens / 1_000_000) * pricing["output_cost_per_1m_tokens"]
            summary_cost = input_cost + output_cost
            total_cost += summary_cost

            summary_lines.append(
                f"Summarization ({model}):\n"
                f"  - Episodes: {summary_episodes}\n"
                f"  - Estimated tokens: ~{total_input_tokens:,} input + "
                f"~{total_output_tokens:,} output\n"
                f"  - Estimated cost: ${summary_cost:.4f}"
            )

    # Add total cost and disclaimer
    if total_cost > 0:
        summary_lines.append("")
        summary_lines.append(f"Total Estimated Cost: ${total_cost:.4f}")
        summary_lines.append("")
        summary_lines.append(
            "Note: Estimates are approximate and based on average episode duration. "
            "Actual costs may vary based on actual audio length and transcript complexity."
        )

    return summary_lines
