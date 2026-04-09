"""Helper utilities for workflow pipeline.

This module contains utility functions used throughout the workflow pipeline.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .. import config
from . import metrics
from .types import TranscriptionResources

if TYPE_CHECKING:
    from ..models import Episode
else:
    from .. import models

    Episode = models.Episode  # type: ignore[assignment]

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
    episode: Episode, feed_url: str  # type: ignore[valid-type]
) -> Tuple[str, Optional[int]]:  # type: ignore[valid-type]
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


def _pipeline_return_episode_count(saved: int, pipeline_metrics: metrics.Metrics) -> int:
    """Episodes to report to callers (CLI, service multi-feed corpus summary).

    ``saved`` counts net-new transcript files written; cache-only or deduped runs can
    have ``saved == 0`` while ``episode_statuses`` still records ok=N (multi-feed #506).
    """
    statuses = getattr(pipeline_metrics, "episode_statuses", None) or []
    if statuses:
        return sum(1 for s in statuses if getattr(s, "status", None) == "ok")
    return saved


def generate_pipeline_summary(  # noqa: C901
    cfg: config.Config,
    saved: int,
    transcription_resources: TranscriptionResources,
    effective_output_dir: str,
    pipeline_metrics: metrics.Metrics,
    episodes: Optional[List[Episode]] = None,  # type: ignore[valid-type]
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
            - count (int): Episodes completed successfully (``episode_statuses`` ok
              count when present; otherwise ``saved`` transcript files)
            - summary (str): Multi-line summary message with statistics and metrics
    """
    if cfg.dry_run:
        planned_downloads = saved
        # Use qsize() for Queue (approximate size, but fine for dry-run reporting)
        # Handle both Queue and list for backward compatibility
        if hasattr(transcription_resources.transcription_jobs, "qsize"):
            planned_transcriptions = (
                transcription_resources.transcription_jobs.qsize() if cfg.transcribe_missing else 0
            )
        else:
            # Fallback for list (should not happen in production, but handle for tests)
            planned_transcriptions = (
                len(transcription_resources.transcription_jobs)  # type: ignore[arg-type]
                if cfg.transcribe_missing
                else 0
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
        if getattr(cfg, "generate_gi", False):
            if pipeline_metrics.gi_artifacts_generated > 0:
                summary_lines.append(
                    f"  - GIL artifacts generated: {pipeline_metrics.gi_artifacts_generated}"
                )
            if pipeline_metrics.gi_failures > 0:
                summary_lines.append(f"  - GIL failures: {pipeline_metrics.gi_failures}")
            if getattr(pipeline_metrics, "gi_evidence_stack_completed", 0) > 0:
                n = pipeline_metrics.gi_evidence_stack_completed
                summary_lines.append(f"  - GIL evidence stack completed: {n}")
            if getattr(pipeline_metrics, "gi_evidence_extract_quotes_calls", 0) > 0:
                summary_lines.append(
                    f"  - GIL evidence extract_quotes calls: "
                    f"{pipeline_metrics.gi_evidence_extract_quotes_calls}"
                )
            if getattr(pipeline_metrics, "gi_evidence_nli_candidates_queued", 0) > 0:
                summary_lines.append(
                    f"  - GIL evidence NLI candidates (QA-pass): "
                    f"{pipeline_metrics.gi_evidence_nli_candidates_queued}"
                )
            if getattr(pipeline_metrics, "gi_evidence_score_entailment_calls", 0) > 0:
                summary_lines.append(
                    f"  - GIL evidence NLI completed calls: "
                    f"{pipeline_metrics.gi_evidence_score_entailment_calls}"
                )
            if getattr(pipeline_metrics, "gi_episodes_zero_grounded_when_required", 0) > 0:
                summary_lines.append(
                    f"  - GIL episodes with 0 grounded quotes (require_grounding): "
                    f"{pipeline_metrics.gi_episodes_zero_grounded_when_required}"
                )
            if getattr(pipeline_metrics, "gi_grounding_degraded", False):
                summary_lines.append(
                    "  - GIL grounding degraded: at least one episode had no quotes"
                )
        if getattr(cfg, "generate_kg", False):
            n_kg = pipeline_metrics.kg_artifacts_generated
            if n_kg > 0:
                tt = pipeline_metrics.kg_topic_nodes_total
                ee = pipeline_metrics.kg_entity_nodes_total
                avg_topics = round(tt / n_kg, 2) if n_kg else 0.0
                avg_entities = round(ee / n_kg, 2) if n_kg else 0.0
                summary_lines.append(
                    f"  - KG: {n_kg} episode graph(s), "
                    f"{tt} topic + {ee} entity nodes "
                    f"(avg {avg_topics} topics, {avg_entities} entities per graph)"
                )
            if pipeline_metrics.kg_failures > 0:
                summary_lines.append(f"  - KG failures: {pipeline_metrics.kg_failures}")
            if getattr(pipeline_metrics, "kg_provider_extractions", 0) > 0:
                summary_lines.append(
                    f"  - KG LLM JSON extractions (succeeded): "
                    f"{pipeline_metrics.kg_provider_extractions}"
                )

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
        return _pipeline_return_episode_count(saved, pipeline_metrics), summary


def _transcription_estimated_cost_usd(
    pricing: Dict[str, float], audio_minutes: float
) -> Optional[float]:
    """Return USD cost for audio given per-minute or per-second rates."""
    if "cost_per_minute" in pricing:
        return float(audio_minutes) * float(pricing["cost_per_minute"])
    if "cost_per_second" in pricing:
        return float(audio_minutes) * 60.0 * float(pricing["cost_per_second"])
    return None


def calculate_provider_cost(
    cfg: config.Config,
    provider_type: str,
    capability: str,
    model: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    audio_minutes: Optional[float] = None,
) -> Optional[float]:
    """Calculate estimated cost for a provider call.

    Args:
        cfg: Configuration object
        provider_type: Provider type ("openai", "gemini", etc.)
        capability: Capability type ("transcription", "speaker_detection", "summarization")
        model: Model name
        prompt_tokens: Input tokens (for text-based capabilities)
        completion_tokens: Output tokens (for text-based capabilities)
        audio_minutes: Audio duration in minutes (for transcription)

    Returns:
        Estimated cost in USD, or None if cost cannot be calculated
    """
    pricing = _get_provider_pricing(cfg, provider_type, capability, model)
    if not pricing:
        return None

    cost = 0.0

    # Text-based pricing (tokens)
    if prompt_tokens is not None or completion_tokens is not None:
        if "input_cost_per_1m_tokens" in pricing and prompt_tokens:
            cost += (prompt_tokens / 1_000_000) * pricing["input_cost_per_1m_tokens"]
        if "output_cost_per_1m_tokens" in pricing and completion_tokens:
            cost += (completion_tokens / 1_000_000) * pricing["output_cost_per_1m_tokens"]

    # Audio-based pricing (minutes or per-second rates)
    if audio_minutes is not None:
        audio_cost = _transcription_estimated_cost_usd(pricing, float(audio_minutes))
        if audio_cost is not None:
            cost += audio_cost

    return cost if cost > 0 else None


def _cleaning_model_for_summary_provider(cfg: config.Config) -> Tuple[str, str]:
    """Return (provider_type, cleaning_model) for the active summary provider."""
    p = str(getattr(cfg, "summary_provider", "transformers"))
    attr_by_provider = {
        "openai": "openai_cleaning_model",
        "gemini": "gemini_cleaning_model",
        "anthropic": "anthropic_cleaning_model",
        "mistral": "mistral_cleaning_model",
        "deepseek": "deepseek_cleaning_model",
        "grok": "grok_cleaning_model",
        "ollama": "ollama_cleaning_model",
    }
    defaults = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-pro",
        "anthropic": "claude-3-5-sonnet-20241022",
        "mistral": "mistral-small",
        "deepseek": "deepseek-chat",
        "grok": "grok-beta",
        "ollama": "llama3.1:8b",
    }
    attr = attr_by_provider.get(p)
    if not attr:
        return p, ""
    model = getattr(cfg, attr, None) or defaults.get(p, "")
    return p, str(model)


def _effective_kg_extraction_provider(cfg: config.Config) -> str:
    """Provider name used for KG LLM extraction when source is ``provider``."""
    explicit = getattr(cfg, "kg_extraction_provider", None)
    if explicit is not None:
        return str(explicit)
    return str(getattr(cfg, "summary_provider", "transformers"))


def _kg_model_for_pricing(cfg: config.Config, kg_provider: str) -> str:
    """Model id for KG cost estimate (override or provider default summary model)."""
    override = getattr(cfg, "kg_extraction_model", None)
    if override:
        return str(override)
    model_attr = {
        "openai": "openai_summary_model",
        "gemini": "gemini_summary_model",
        "anthropic": "anthropic_summary_model",
        "mistral": "mistral_summary_model",
        "deepseek": "deepseek_summary_model",
        "grok": "grok_summary_model",
        "ollama": "ollama_summary_model",
    }.get(kg_provider)
    defaults = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-pro",
        "anthropic": "claude-3-5-sonnet-20241022",
        "mistral": "mistral-small",
        "deepseek": "deepseek-chat",
        "grok": "grok-beta",
        "ollama": "llama3.1:8b",
    }
    if not model_attr:
        return ""
    return str(getattr(cfg, model_attr, defaults.get(kg_provider, "")))


def _builtin_provider_pricing(provider_type: str, capability: str, model: str) -> Dict[str, float]:
    """Built-in USD rates from provider modules (no YAML)."""
    if provider_type == "openai":
        from ..providers.openai.openai_provider import OpenAIProvider

        return OpenAIProvider.get_pricing(model, capability)
    if provider_type == "gemini":
        from ..providers.gemini.gemini_provider import GeminiProvider

        return GeminiProvider.get_pricing(model, capability)
    if provider_type == "anthropic":
        from ..providers.anthropic.anthropic_provider import AnthropicProvider

        return AnthropicProvider.get_pricing(model, capability)
    if provider_type == "mistral":
        from ..providers.mistral.mistral_provider import MistralProvider

        return MistralProvider.get_pricing(model, capability)
    if provider_type == "deepseek":
        from ..providers.deepseek.deepseek_provider import DeepSeekProvider

        return DeepSeekProvider.get_pricing(model, capability)
    if provider_type == "grok":
        from ..providers.grok.grok_provider import GrokProvider

        return GrokProvider.get_pricing(model, capability)
    if provider_type == "ollama":
        from ..providers.ollama.ollama_provider import OllamaProvider

        return OllamaProvider.get_pricing(model, capability)
    return {}


def _get_provider_pricing(
    cfg: config.Config, provider_type: str, capability: str, model: str
) -> Dict[str, float]:
    """Get pricing information from the appropriate provider.

    Merges optional ``pricing_assumptions_file`` YAML overrides on top of built-in rates.

    Args:
        cfg: Configuration object
        provider_type: Provider type ("openai", "whisper", etc.)
        capability: Capability type ("transcription", "speaker_detection", "summarization")
        model: Model name

    Returns:
        Dictionary with pricing information, or empty dict if not available
    """
    base = _builtin_provider_pricing(provider_type, capability, model)
    path_cfg = str(getattr(cfg, "pricing_assumptions_file", "") or "").strip()
    if not path_cfg:
        return base
    from .. import pricing_assumptions

    table, _resolved = pricing_assumptions.get_loaded_table(path_cfg)
    if not table:
        return base
    ext = pricing_assumptions.lookup_external_pricing(table, provider_type, capability, model)
    if not ext:
        return base
    merged = dict(base)
    merged.update(ext)
    return merged


def _metrics_wall_suffix(
    metrics_dict: Dict[str, Any],
    avg_key: str,
    count_key: str,
    label: str,
) -> str:
    """Suffix for cost lines: avg wall seconds over episodes with recorded stage timings."""
    cnt = int(metrics_dict.get(count_key) or 0)
    if cnt <= 0:
        return ""
    avg = float(metrics_dict.get(avg_key) or 0.0)
    return f", {label} avg wall {avg:.2f}s ({cnt} eps)"


def _llm_cost_cleaning_section(
    cfg: config.Config,
    metrics_dict: Dict[str, Any],
    cleaning_strategy: str,
    uses_llm_summarization: bool,
    llm_cleaning_calls: int,
) -> Tuple[List[str], float]:
    """LLM transcript-cleaning subsection for the pipeline cost summary."""
    lines: List[str] = []
    extra_cost = 0.0
    cleaning_provider_type, cleaning_model = _cleaning_model_for_summary_provider(cfg)
    uses_semantic_cleaning_cfg = (
        cleaning_strategy in ("llm", "hybrid") and uses_llm_summarization and bool(cleaning_model)
    )
    if uses_semantic_cleaning_cfg:
        c_in = int(metrics_dict.get("llm_cleaning_input_tokens", 0) or 0)
        c_out = int(metrics_dict.get("llm_cleaning_output_tokens", 0) or 0)
        clean_wall = _metrics_wall_suffix(
            metrics_dict, "avg_cleaning_seconds", "cleaning_count", "transcript clean"
        )
        cl_in_avg = float(metrics_dict.get("llm_cleaning_avg_input_tokens_per_call") or 0.0)
        cl_out_avg = float(metrics_dict.get("llm_cleaning_avg_output_tokens_per_call") or 0.0)
        tok_avg = ""
        if llm_cleaning_calls > 0:
            tok_avg = (
                f", avg {cl_in_avg:.1f} in + {cl_out_avg:.1f} out tok/call"
                if (cl_in_avg > 0 or cl_out_avg > 0)
                else ""
            )
        if llm_cleaning_calls > 0:
            pricing = _get_provider_pricing(
                cfg, cleaning_provider_type, "summarization", cleaning_model
            )
            if pricing and "input_cost_per_1m_tokens" in pricing:
                clean_cost = (c_in / 1_000_000) * pricing["input_cost_per_1m_tokens"] + (
                    c_out / 1_000_000
                ) * pricing["output_cost_per_1m_tokens"]
                extra_cost += clean_cost
                lines.append(
                    f"  - Cleaning: {llm_cleaning_calls} calls, "
                    f"{c_in:,} input + {c_out:,} output tokens{tok_avg}{clean_wall}, "
                    f"${clean_cost:.4f} (model: {cleaning_model})"
                )
            else:
                lines.append(
                    f"  - Cleaning: {llm_cleaning_calls} calls, "
                    f"{c_in:,} input + {c_out:,} output tokens{tok_avg}{clean_wall} "
                    f"(model: {cleaning_model}; pricing unavailable)"
                )
        elif cleaning_strategy == "llm":
            lines.append(
                f"  - Cleaning: strategy=llm, 0 LLM calls recorded, $0.0000 "
                f"(model: {cleaning_model}){clean_wall}"
            )
        else:
            lines.append(
                "  - Cleaning: hybrid (0 LLM calls; pattern-only this run), " f"$0.0000{clean_wall}"
            )
    elif cleaning_strategy == "pattern":
        lines.append("  - Cleaning: pattern-based (no LLM), $0.0000")
    return lines, extra_cost


def _llm_cost_gil_section(
    cfg: config.Config,
    metrics_dict: Dict[str, Any],
    llm_summarization_provider: str,
    uses_llm_summarization: bool,
    llm_gi_calls: int,
) -> Tuple[List[str], float]:
    """GIL LLM subsection (insights + provider evidence stack)."""
    lines: List[str] = []
    extra_cost = 0.0
    if llm_gi_calls <= 0:
        return lines, extra_cost
    g_in = int(metrics_dict.get("llm_gi_input_tokens", 0) or 0)
    g_out = int(metrics_dict.get("llm_gi_output_tokens", 0) or 0)
    gi_wall = _metrics_wall_suffix(metrics_dict, "avg_gi_seconds", "gi_count", "GIL total")
    gi_in_avg = float(metrics_dict.get("llm_gi_avg_input_tokens_per_call") or 0.0)
    gi_out_avg = float(metrics_dict.get("llm_gi_avg_output_tokens_per_call") or 0.0)
    gi_tok_avg = (
        f", avg {gi_in_avg:.1f} in + {gi_out_avg:.1f} out tok/call"
        if (gi_in_avg > 0 or gi_out_avg > 0)
        else ""
    )
    if uses_llm_summarization:
        model_attr = f"{llm_summarization_provider}_summary_model"
        default_models = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-1.5-pro",
            "anthropic": "claude-3-5-sonnet-20241022",
            "mistral": "mistral-small",
            "deepseek": "deepseek-chat",
            "grok": "grok-beta",
            "ollama": "llama3.1:8b",
        }
        gi_model = getattr(cfg, model_attr, default_models.get(llm_summarization_provider, ""))
        pricing = _get_provider_pricing(
            cfg, str(llm_summarization_provider), "summarization", gi_model
        )
        if pricing and "input_cost_per_1m_tokens" in pricing:
            gi_cost = (g_in / 1_000_000) * pricing["input_cost_per_1m_tokens"] + (
                g_out / 1_000_000
            ) * pricing["output_cost_per_1m_tokens"]
            extra_cost += gi_cost
            lines.append(
                f"  - GIL (insights/evidence LLM): {llm_gi_calls} calls, "
                f"{g_in:,} input + {g_out:,} output tokens{gi_tok_avg}{gi_wall}, "
                f"${gi_cost:.4f} (model: {gi_model})"
            )
        else:
            lines.append(
                f"  - GIL (insights/evidence LLM): {llm_gi_calls} calls, "
                f"{g_in:,} input + {g_out:,} output tokens{gi_tok_avg}{gi_wall} "
                f"(model: {gi_model}; pricing unavailable)"
            )
    else:
        lines.append(
            f"  - GIL (insights/evidence LLM): {llm_gi_calls} calls, "
            f"{g_in:,} input + {g_out:,} output tokens{gi_tok_avg}{gi_wall} "
            "(summary provider is not a billable LLM; cost n/a)"
        )
    return lines, extra_cost


def _kg_llm_cost_headline(metrics_dict: Dict[str, Any]) -> str:
    """Human-readable KG LLM mode for CLI summary (uses finished metrics rollups)."""
    k_sb = int(metrics_dict.get("kg_extractions_provider_summary_bullets", 0) or 0)
    k_pv = int(metrics_dict.get("kg_extractions_provider", 0) or 0)
    if k_pv <= 0:
        return "LLM extraction"
    k_sb = min(k_sb, k_pv)
    k_tx = k_pv - k_sb
    if k_sb == k_pv:
        return "summary bullets → topics"
    if k_tx == k_pv:
        return "transcript"
    return f"mixed ({k_tx} transcript, {k_sb} bullet-derived)"


def _llm_cost_kg_section(
    cfg: config.Config,
    metrics_dict: Dict[str, Any],
    llm_providers: Set[str],
    llm_kg_calls: int,
) -> Tuple[List[str], float]:
    """KG provider-extraction LLM subsection."""
    lines: List[str] = []
    extra_cost = 0.0
    if llm_kg_calls <= 0:
        return lines, extra_cost
    kg_headline = _kg_llm_cost_headline(metrics_dict)
    kg_provider = _effective_kg_extraction_provider(cfg)
    k_in = int(metrics_dict.get("llm_kg_input_tokens", 0) or 0)
    k_out = int(metrics_dict.get("llm_kg_output_tokens", 0) or 0)
    kg_wall = _metrics_wall_suffix(metrics_dict, "avg_kg_seconds", "kg_count", "KG total")
    kg_in_avg = float(metrics_dict.get("llm_kg_avg_input_tokens_per_call") or 0.0)
    kg_out_avg = float(metrics_dict.get("llm_kg_avg_output_tokens_per_call") or 0.0)
    kg_tok_avg = (
        f", avg {kg_in_avg:.1f} in + {kg_out_avg:.1f} out tok/call"
        if (kg_in_avg > 0 or kg_out_avg > 0)
        else ""
    )
    if kg_provider in llm_providers:
        kg_model = _kg_model_for_pricing(cfg, kg_provider)
        if kg_model:
            pricing = _get_provider_pricing(cfg, kg_provider, "summarization", kg_model)
            if pricing and "input_cost_per_1m_tokens" in pricing:
                kg_cost = (k_in / 1_000_000) * pricing["input_cost_per_1m_tokens"] + (
                    k_out / 1_000_000
                ) * pricing["output_cost_per_1m_tokens"]
                extra_cost += kg_cost
                lines.append(
                    f"  - KG ({kg_headline}): {llm_kg_calls} calls, "
                    f"{k_in:,} input + {k_out:,} output tokens{kg_tok_avg}{kg_wall}, "
                    f"${kg_cost:.4f} (provider: {kg_provider}, model: {kg_model})"
                )
            else:
                lines.append(
                    f"  - KG ({kg_headline}): {llm_kg_calls} calls, "
                    f"{k_in:,} input + {k_out:,} output tokens{kg_tok_avg}{kg_wall} "
                    f"(model: {kg_model}; pricing unavailable)"
                )
        else:
            lines.append(
                f"  - KG ({kg_headline}): {llm_kg_calls} calls, "
                f"{k_in:,} input + {k_out:,} output tokens{kg_tok_avg}{kg_wall}"
            )
    else:
        lines.append(
            f"  - KG ({kg_headline}): {llm_kg_calls} calls, "
            f"{k_in:,} input + {k_out:,} output tokens{kg_tok_avg}{kg_wall} "
            f"(provider: {kg_provider}, local processing)"
        )
    return lines, extra_cost


def _llm_cost_summary_lines_and_total(
    cfg: config.Config,
    metrics_dict: Dict[str, Any],
) -> Tuple[List[str], float]:
    """Build LLM cost summary lines and total USD (same rules as pipeline end summary).

    Args:
        cfg: Active configuration (determines which providers are billable LLMs).
        metrics_dict: Output of ``Metrics.finish()`` for the run.

    Returns:
        (summary_lines, total_estimated_cost_usd). Total excludes non-billable stages.
    """
    summary_lines: List[str] = []

    llm_transcription_provider = cfg.transcription_provider
    llm_speaker_provider = cfg.speaker_detector_provider
    llm_summarization_provider = cfg.summary_provider

    llm_providers = {"openai", "gemini", "mistral", "anthropic", "deepseek", "grok", "ollama"}

    uses_llm_transcription = llm_transcription_provider in llm_providers
    uses_llm_speaker = llm_speaker_provider in llm_providers
    uses_llm_summarization = llm_summarization_provider in llm_providers

    cleaning_strategy = str(getattr(cfg, "transcript_cleaning_strategy", "hybrid") or "hybrid")
    llm_cleaning_calls = int(metrics_dict.get("llm_cleaning_calls", 0) or 0)
    llm_gi_calls = int(metrics_dict.get("llm_gi_calls", 0) or 0)
    llm_kg_calls = int(metrics_dict.get("llm_kg_calls", 0) or 0)
    any_extra_llm_usage = llm_cleaning_calls > 0 or llm_gi_calls > 0 or llm_kg_calls > 0

    if not (
        uses_llm_transcription or uses_llm_speaker or uses_llm_summarization or any_extra_llm_usage
    ):
        return summary_lines, 0.0

    total_cost = 0.0

    if uses_llm_transcription:
        transcription_calls = metrics_dict.get("llm_transcription_calls", 0)
        audio_minutes = metrics_dict.get("llm_transcription_audio_minutes", 0.0)
        if transcription_calls > 0:
            model_attr = f"{llm_transcription_provider}_transcription_model"
            default_models = {
                "openai": "whisper-1",
                "gemini": "gemini-1.5-pro",
                "mistral": "voxtral",
            }
            model = getattr(cfg, model_attr, default_models.get(llm_transcription_provider, ""))
            pricing = _get_provider_pricing(cfg, llm_transcription_provider, "transcription", model)
            if pricing:
                transcription_cost = _transcription_estimated_cost_usd(
                    pricing, float(audio_minutes)
                )
                if transcription_cost is not None:
                    total_cost += transcription_cost
                    summary_lines.append(
                        f"  - Transcription: {transcription_calls} calls, "
                        f"{audio_minutes:.1f} minutes, ${transcription_cost:.4f} "
                        f"(model: {model})"
                    )
    else:
        transcripts_transcribed = metrics_dict.get("transcripts_transcribed", 0)
        if transcripts_transcribed > 0:
            transcription_provider = getattr(cfg, "transcription_provider", "whisper")
            summary_lines.append(
                f"  - Transcription: {transcripts_transcribed} episodes, $0.0000 "
                f"(provider: {transcription_provider}, local processing)"
            )

    if uses_llm_speaker:
        speaker_calls = metrics_dict.get("llm_speaker_detection_calls", 0)
        speaker_input_tokens = metrics_dict.get("llm_speaker_detection_input_tokens", 0)
        speaker_output_tokens = metrics_dict.get("llm_speaker_detection_output_tokens", 0)
        if speaker_calls > 0:
            model_attr = f"{llm_speaker_provider}_speaker_model"
            default_models = {
                "openai": "gpt-4o-mini",
                "gemini": "gemini-1.5-pro",
                "anthropic": "claude-3-5-sonnet-20241022",
                "mistral": "mistral-small",
                "deepseek": "deepseek-chat",
                "grok": "grok-beta",
                "ollama": "llama3.2",
            }
            model = getattr(cfg, model_attr, default_models.get(llm_speaker_provider, ""))
            pricing = _get_provider_pricing(cfg, llm_speaker_provider, "speaker_detection", model)
            sp_wall = _metrics_wall_suffix(
                metrics_dict, "avg_extract_names_seconds", "extract_names_count", "NER/speaker"
            )
            sp_in_avg = float(
                metrics_dict.get("llm_speaker_detection_avg_input_tokens_per_call") or 0.0
            )
            sp_out_avg = float(
                metrics_dict.get("llm_speaker_detection_avg_output_tokens_per_call") or 0.0
            )
            sp_tok_avg = (
                f", avg {sp_in_avg:.1f} in + {sp_out_avg:.1f} out tok/call"
                if (sp_in_avg > 0 or sp_out_avg > 0)
                else ""
            )
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
                    f"{speaker_input_tokens:,} input + {speaker_output_tokens:,} "
                    f"output tokens{sp_tok_avg}{sp_wall}, "
                    f"${speaker_cost:.4f} (model: {model})"
                )
    else:
        extract_names_count = metrics_dict.get("extract_names_count", 0)
        if extract_names_count > 0:
            speaker_provider = getattr(cfg, "speaker_detector_provider", "spacy")
            summary_lines.append(
                f"  - Speaker Detection: {extract_names_count} episodes, $0.0000 "
                f"(provider: {speaker_provider}, local processing)"
            )

    if uses_llm_summarization:
        summary_calls = metrics_dict.get("llm_summarization_calls", 0)
        summary_input_tokens = metrics_dict.get("llm_summarization_input_tokens", 0)
        summary_output_tokens = metrics_dict.get("llm_summarization_output_tokens", 0)
        if summary_calls > 0:
            model_attr = f"{llm_summarization_provider}_summary_model"
            default_models = {
                "openai": "gpt-4o-mini",
                "gemini": "gemini-1.5-pro",
                "anthropic": "claude-3-5-sonnet-20241022",
                "mistral": "mistral-small",
                "deepseek": "deepseek-chat",
                "grok": "grok-beta",
                "ollama": "llama3.1:8b",
            }
            model = getattr(cfg, model_attr, default_models.get(llm_summarization_provider, ""))
            pricing = _get_provider_pricing(cfg, llm_summarization_provider, "summarization", model)
            sum_wall = _metrics_wall_suffix(
                metrics_dict, "avg_summarize_seconds", "summarize_count", "summarize"
            )
            s_in_avg = float(metrics_dict.get("llm_summarization_avg_input_tokens_per_call") or 0.0)
            s_out_avg = float(
                metrics_dict.get("llm_summarization_avg_output_tokens_per_call") or 0.0
            )
            sum_tok_avg = (
                f", avg {s_in_avg:.1f} in + {s_out_avg:.1f} out tok/call"
                if (s_in_avg > 0 or s_out_avg > 0)
                else ""
            )
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
                    f"{summary_input_tokens:,} input + {summary_output_tokens:,} "
                    f"output tokens{sum_tok_avg}{sum_wall}, "
                    f"${summary_cost:.4f} (model: {model})"
                )
    else:
        episodes_summarized = metrics_dict.get("episodes_summarized", 0)
        if episodes_summarized > 0:
            summary_provider = getattr(cfg, "summary_provider", "transformers")
            summary_lines.append(
                f"  - Summarization: {episodes_summarized} episodes, $0.0000 "
                f"(provider: {summary_provider}, local processing)"
            )

    clines, cextra = _llm_cost_cleaning_section(
        cfg,
        metrics_dict,
        cleaning_strategy,
        uses_llm_summarization,
        llm_cleaning_calls,
    )
    summary_lines.extend(clines)
    total_cost += cextra

    glines, gextra = _llm_cost_gil_section(
        cfg,
        metrics_dict,
        str(llm_summarization_provider),
        uses_llm_summarization,
        llm_gi_calls,
    )
    summary_lines.extend(glines)
    total_cost += gextra

    klines, kextra = _llm_cost_kg_section(cfg, metrics_dict, llm_providers, llm_kg_calls)
    summary_lines.extend(klines)
    total_cost += kextra

    if total_cost > 0:
        summary_lines.append(f"  - Total estimated cost: ${total_cost:.4f}")

    return summary_lines, total_cost


def estimated_llm_cost_usd_from_metrics_dict(
    cfg: config.Config,
    metrics_dict: Dict[str, Any],
) -> Optional[float]:
    """Total estimated LLM API spend in USD from a ``metrics.finish()`` dict.

    Uses the same pricing rules as the pipeline's LLM cost summary. Returns ``None``
    when there is no billable LLM usage (all local/ML or zero tokens/audio).

    Args:
        cfg: Configuration for the run.
        metrics_dict: Serialized metrics (e.g. from ``metrics.json``).

    Returns:
        Estimated total in USD, or ``None`` if not applicable.
    """
    _, total = _llm_cost_summary_lines_and_total(cfg, metrics_dict)
    return total if total > 0 else None


def _generate_llm_call_summary(cfg: config.Config, pipeline_metrics: metrics.Metrics) -> List[str]:
    """Generate summary of LLM API calls and estimated costs.

    This is a shared utility function that handles cost summaries for ALL LLM providers
    (OpenAI, Gemini, Anthropic, Mistral, DeepSeek, Grok, Ollama). It checks which provider
    was configured and generates the appropriate cost summary for that provider.

    Args:
        cfg: Configuration object to check which providers were used
        pipeline_metrics: Metrics object with LLM call tracking data

    Returns:
        List of summary lines, or empty list if no LLM calls were made
    """
    lines, _ = _llm_cost_summary_lines_and_total(cfg, pipeline_metrics.finish())
    return lines


def _generate_dry_run_cost_projection(
    cfg: config.Config,
    episodes: Optional[List[Episode]],  # type: ignore[valid-type]
    episode_count: int,
) -> List[str]:
    """Generate cost projection for dry-run mode based on configured LLM providers.

    Estimates API costs for OpenAI/Gemini/Anthropic transcription, speaker detection,
    and summarization based on episode count and available metadata (duration).
    Only displays projection when LLM providers are configured.

    Args:
        cfg: Configuration object to check provider settings
        episodes: Optional list of episodes for duration-based estimation
        episode_count: Total number of episodes to process

    Returns:
        List of cost projection lines, or empty list if no LLM providers configured
    """
    summary_lines: List[str] = []

    # Determine which LLM providers are configured (generic approach)
    llm_transcription_provider = cfg.transcription_provider
    llm_speaker_provider = cfg.speaker_detector_provider
    llm_summarization_provider = cfg.summary_provider

    # Supported LLM providers for each capability
    llm_providers = {"openai", "gemini", "mistral", "anthropic", "deepseek", "grok", "ollama"}

    # Check if any LLM provider is configured
    uses_llm_transcription = llm_transcription_provider in llm_providers
    uses_llm_speaker = llm_speaker_provider in llm_providers
    uses_llm_summarization = llm_summarization_provider in llm_providers

    if not (uses_llm_transcription or uses_llm_speaker or uses_llm_summarization):
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
    if uses_llm_transcription:
        transcription_episodes = episode_count
        # Get model and default based on provider
        model_attr = f"{llm_transcription_provider}_transcription_model"
        default_models = {
            "openai": "whisper-1",
            "gemini": "gemini-1.5-pro",
            "mistral": "voxtral",
        }
        model = getattr(cfg, model_attr, default_models.get(llm_transcription_provider, ""))
        pricing = _get_provider_pricing(cfg, llm_transcription_provider, "transcription", model)
        if pricing:
            total_audio_minutes = transcription_episodes * avg_duration_minutes
            transcription_cost = _transcription_estimated_cost_usd(pricing, total_audio_minutes)
            if transcription_cost is not None:
                total_cost += transcription_cost
                summary_lines.append(
                    f"Transcription ({model}):\n"
                    f"  - Episodes: {transcription_episodes}\n"
                    f"  - Estimated audio: {total_audio_minutes:.1f} minutes\n"
                    f"  - Estimated cost: ${transcription_cost:.4f}"
                )

    # Speaker detection cost estimation
    if uses_llm_speaker:
        speaker_episodes = episode_count
        # Get model and default based on provider
        model_attr = f"{llm_speaker_provider}_speaker_model"
        default_models = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-1.5-pro",
            "anthropic": "claude-3-5-sonnet-20241022",
            "mistral": "mistral-small",
            "deepseek": "deepseek-chat",
            "grok": "grok-beta",
            "ollama": "llama3.2",
        }
        model = getattr(cfg, model_attr, default_models.get(llm_speaker_provider, ""))
        pricing = _get_provider_pricing(cfg, llm_speaker_provider, "speaker_detection", model)
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
    if uses_llm_summarization:
        summary_episodes = episode_count
        # Get model and default based on provider
        model_attr = f"{llm_summarization_provider}_summary_model"
        default_models = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-1.5-pro",
            "anthropic": "claude-3-5-sonnet-20241022",
            "mistral": "mistral-small",
            "deepseek": "deepseek-chat",
            "grok": "grok-beta",
            "ollama": "llama3.1:8b",
        }
        model = getattr(cfg, model_attr, default_models.get(llm_summarization_provider, ""))
        pricing = _get_provider_pricing(cfg, llm_summarization_provider, "summarization", model)
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
