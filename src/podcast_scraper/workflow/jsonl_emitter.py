"""JSONL metrics emitter for streaming metrics output.

This module provides a JSONL emitter that wraps the existing Metrics class
to provide streaming metrics output during pipeline execution, enabling
real-time monitoring and analysis.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from . import metrics

logger = logging.getLogger(__name__)


class JSONLEmitter:
    """JSONL metrics emitter that wraps existing Metrics instance.

    The emitter provides streaming JSONL output during pipeline execution,
    complementing the existing single JSON file output at the end.
    All data flows through the existing Metrics and EpisodeMetrics classes.
    """

    def __init__(
        self,
        metrics_instance: metrics.Metrics,
        jsonl_path: str,
        *,
        echo_stdout: bool = False,
    ):
        """Initialize JSONL emitter.

        Args:
            metrics_instance: Existing Metrics instance (data source)
            jsonl_path: Path to JSONL output file
            echo_stdout: When True, duplicate each JSON line to stdout (for Loki via
                docker logs; GitHub #746).
        """
        self.metrics = metrics_instance
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._echo_stdout = echo_stdout
        self._file_handle = None

    def __enter__(self):
        """Context manager entry - open file for writing (append if exists)."""
        # Append mode if file exists (for episode events), otherwise create new
        mode = "a" if self.jsonl_path.exists() else "w"
        self._file_handle = open(self.jsonl_path, mode, encoding="utf-8")
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit - close file (signature required by context manager protocol)."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def _write_line(self, event: Dict[str, Any]) -> None:
        """Write a single JSON line to the JSONL file.

        Args:
            event: Event dictionary to write
        """
        if not self._file_handle:
            raise RuntimeError("JSONL emitter not opened (use as context manager)")
        json_line = json.dumps(event, ensure_ascii=False)
        self._file_handle.write(json_line + "\n")
        self._file_handle.flush()  # Ensure immediate write for streaming
        if self._echo_stdout:
            sys.stdout.write(json_line + "\n")
            sys.stdout.flush()

    def emit_run_started(
        self,
        cfg: Any,  # config.Config
        run_id: Optional[str] = None,
    ) -> None:
        """Emit run_started event.

        Args:
            cfg: Configuration object
            run_id: Optional run ID (defaults to timestamp)
        """
        if run_id is None:
            run_id = datetime.utcnow().isoformat() + "Z"

        # Create config fingerprint (non-sensitive fields only)
        config_fingerprint = {
            "rss_url": cfg.rss_url,
            "transcription_provider": cfg.transcription_provider,
            "summary_provider": getattr(cfg, "summary_provider", None),
            "workers": cfg.workers,
            "transcription_parallelism": cfg.transcription_parallelism,
            "max_episodes": cfg.max_episodes,
            "episode_order": cfg.episode_order,
            "episode_offset": cfg.episode_offset,
            "episode_since": (
                cfg.episode_since.isoformat() if cfg.episode_since is not None else None
            ),
            "episode_until": (
                cfg.episode_until.isoformat() if cfg.episode_until is not None else None
            ),
            "preprocessing_enabled": cfg.preprocessing_enabled,
            "transcript_cache_enabled": cfg.transcript_cache_enabled,
        }

        event = {
            "event_type": "run_started",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": run_id,
            "config": config_fingerprint,
        }
        self._write_line(event)
        logger.debug("Emitted run_started event (run_id=%s)", run_id)

    def emit_episode_finished(self, episode_id: str) -> None:
        """Emit episode_finished event.

        Pulls data from existing EpisodeMetrics in the Metrics instance.

        Args:
            episode_id: Episode ID to emit metrics for
        """
        # Find episode metrics in the metrics instance
        episode_metrics = None
        for em in self.metrics.episode_metrics:
            if em.episode_id == episode_id:
                episode_metrics = em
                break

        if not episode_metrics:
            logger.warning("Episode metrics not found for episode_id=%s", episode_id)
            return

        # Convert EpisodeMetrics to dict
        event = {
            "event_type": "episode_finished",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "episode_id": episode_metrics.episode_id,
            "episode_number": episode_metrics.episode_number,
            "audio_sec": episode_metrics.audio_sec,
            "transcribe_sec": episode_metrics.transcribe_sec,
            "summary_sec": episode_metrics.summary_sec,
            "gi_sec": episode_metrics.gi_sec,
            "kg_sec": episode_metrics.kg_sec,
            "retries": episode_metrics.retries,
            "rate_limit_sleep_sec": episode_metrics.rate_limit_sleep_sec,
            "prompt_tokens": episode_metrics.prompt_tokens,
            "completion_tokens": episode_metrics.completion_tokens,
            "estimated_cost": episode_metrics.estimated_cost,
        }
        self._write_line(event)
        logger.debug("Emitted episode_finished event (episode_id=%s)", episode_id)

    def emit_run_finished(self) -> None:
        """Emit run_finished event.

        Pulls data from Metrics.finish() to get final run metrics.
        """
        # Get final metrics dict (same as save_to_file uses)
        metrics_dict = self.metrics.finish()

        # Extract key run-level metrics (not all fields, just important ones)
        event = {
            "event_type": "run_finished",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_duration_seconds": metrics_dict.get("run_duration_seconds"),
            "episodes_scraped_total": metrics_dict.get("episodes_scraped_total"),
            "episodes_skipped_total": metrics_dict.get("episodes_skipped_total"),
            "errors_total": metrics_dict.get("errors_total"),
            "bytes_downloaded_total": metrics_dict.get("bytes_downloaded_total"),
            "transcripts_downloaded": metrics_dict.get("transcripts_downloaded"),
            "transcripts_transcribed": metrics_dict.get("transcripts_transcribed"),
            "episodes_summarized": metrics_dict.get("episodes_summarized"),
            "metadata_files_generated": metrics_dict.get("metadata_files_generated"),
            "gi_artifacts_generated": metrics_dict.get("gi_artifacts_generated"),
            "gi_failures": metrics_dict.get("gi_failures"),
            "gi_evidence_stack_completed": metrics_dict.get("gi_evidence_stack_completed"),
            "gi_evidence_path_provider": metrics_dict.get("gi_evidence_path_provider"),
            "gi_evidence_extract_quotes_calls": metrics_dict.get(
                "gi_evidence_extract_quotes_calls"
            ),
            "gi_evidence_nli_candidates_queued": metrics_dict.get(
                "gi_evidence_nli_candidates_queued"
            ),
            "gi_evidence_score_entailment_calls": metrics_dict.get(
                "gi_evidence_score_entailment_calls"
            ),
            "gi_episodes_zero_grounded_when_required": metrics_dict.get(
                "gi_episodes_zero_grounded_when_required"
            ),
            "gi_grounding_degraded": metrics_dict.get("gi_grounding_degraded"),
            "kg_artifacts_generated": metrics_dict.get("kg_artifacts_generated"),
            "kg_failures": metrics_dict.get("kg_failures"),
            "kg_provider_extractions": metrics_dict.get("kg_provider_extractions"),
            "kg_topic_nodes_total": metrics_dict.get("kg_topic_nodes_total"),
            "kg_entity_nodes_total": metrics_dict.get("kg_entity_nodes_total"),
            "kg_extractions_stub": metrics_dict.get("kg_extractions_stub"),
            "kg_extractions_summary_bullets": metrics_dict.get("kg_extractions_summary_bullets"),
            "kg_extractions_provider": metrics_dict.get("kg_extractions_provider"),
            "kg_extractions_provider_summary_bullets": metrics_dict.get(
                "kg_extractions_provider_summary_bullets"
            ),
            "avg_cleaning_seconds": metrics_dict.get("avg_cleaning_seconds"),
            "cleaning_count": metrics_dict.get("cleaning_count"),
            "avg_gi_seconds": metrics_dict.get("avg_gi_seconds"),
            "gi_count": metrics_dict.get("gi_count"),
            "avg_kg_seconds": metrics_dict.get("avg_kg_seconds"),
            "kg_count": metrics_dict.get("kg_count"),
            "avg_summarize_seconds": metrics_dict.get("avg_summarize_seconds"),
            "summarize_count": metrics_dict.get("summarize_count"),
            "llm_cleaning_calls": metrics_dict.get("llm_cleaning_calls"),
            "llm_cleaning_input_tokens": metrics_dict.get("llm_cleaning_input_tokens"),
            "llm_cleaning_output_tokens": metrics_dict.get("llm_cleaning_output_tokens"),
            "llm_cleaning_avg_input_tokens_per_call": metrics_dict.get(
                "llm_cleaning_avg_input_tokens_per_call"
            ),
            "llm_cleaning_avg_output_tokens_per_call": metrics_dict.get(
                "llm_cleaning_avg_output_tokens_per_call"
            ),
            "llm_cleaning_calls_per_recorded_cleaning_episode": metrics_dict.get(
                "llm_cleaning_calls_per_recorded_cleaning_episode"
            ),
            "llm_gi_calls": metrics_dict.get("llm_gi_calls"),
            "llm_gi_input_tokens": metrics_dict.get("llm_gi_input_tokens"),
            "llm_gi_output_tokens": metrics_dict.get("llm_gi_output_tokens"),
            "llm_gi_avg_input_tokens_per_call": metrics_dict.get(
                "llm_gi_avg_input_tokens_per_call"
            ),
            "llm_gi_avg_output_tokens_per_call": metrics_dict.get(
                "llm_gi_avg_output_tokens_per_call"
            ),
            "llm_gi_calls_per_gi_artifact": metrics_dict.get("llm_gi_calls_per_gi_artifact"),
            "llm_gi_evidence_retries": metrics_dict.get("llm_gi_evidence_retries"),
            "llm_gi_evidence_rate_limit_sleep_sec": metrics_dict.get(
                "llm_gi_evidence_rate_limit_sleep_sec"
            ),
            "llm_kg_calls": metrics_dict.get("llm_kg_calls"),
            "llm_kg_input_tokens": metrics_dict.get("llm_kg_input_tokens"),
            "llm_kg_output_tokens": metrics_dict.get("llm_kg_output_tokens"),
            "llm_kg_avg_input_tokens_per_call": metrics_dict.get(
                "llm_kg_avg_input_tokens_per_call"
            ),
            "llm_kg_avg_output_tokens_per_call": metrics_dict.get(
                "llm_kg_avg_output_tokens_per_call"
            ),
            "llm_kg_calls_per_kg_artifact": metrics_dict.get("llm_kg_calls_per_kg_artifact"),
            "llm_summarization_calls": metrics_dict.get("llm_summarization_calls"),
            "llm_summarization_input_tokens": metrics_dict.get("llm_summarization_input_tokens"),
            "llm_summarization_output_tokens": metrics_dict.get("llm_summarization_output_tokens"),
            "llm_summarization_avg_input_tokens_per_call": metrics_dict.get(
                "llm_summarization_avg_input_tokens_per_call"
            ),
            "llm_summarization_avg_output_tokens_per_call": metrics_dict.get(
                "llm_summarization_avg_output_tokens_per_call"
            ),
            "llm_bundled_clean_summary_calls": metrics_dict.get("llm_bundled_clean_summary_calls"),
            "llm_bundled_clean_summary_input_tokens": metrics_dict.get(
                "llm_bundled_clean_summary_input_tokens"
            ),
            "llm_bundled_clean_summary_output_tokens": metrics_dict.get(
                "llm_bundled_clean_summary_output_tokens"
            ),
            "llm_bundled_fallback_to_staged_count": metrics_dict.get(
                "llm_bundled_fallback_to_staged_count"
            ),
            "total_episode_estimated_cost_usd": metrics_dict.get(
                "total_episode_estimated_cost_usd"
            ),
            "total_episode_prompt_tokens": metrics_dict.get("total_episode_prompt_tokens"),
            "total_episode_completion_tokens": metrics_dict.get("total_episode_completion_tokens"),
            "schema_version": metrics_dict.get("schema_version"),
        }
        self._write_line(event)
        logger.debug("Emitted run_finished event")
