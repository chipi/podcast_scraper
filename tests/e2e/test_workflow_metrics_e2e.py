"""Pytest E2E: JSONL metrics during ``run_pipeline`` and degradation policy code paths."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import podcast_scraper
from podcast_scraper import config
from podcast_scraper.workflow.degradation import DegradationPolicy, handle_stage_failure

pytestmark = [pytest.mark.e2e]


class TestDegradationHandleStageFailureE2E:
    """Exercise ``workflow/degradation.py`` branches (no ML)."""

    @pytest.mark.critical_path
    def test_handle_stage_failure_all_stages(self) -> None:
        policy = DegradationPolicy()
        assert (
            handle_stage_failure(
                "summarization",
                RuntimeError("fail"),
                policy,
                episode_idx=0,
            )
            is True
        )
        strict_sum = DegradationPolicy(save_transcript_on_summarization_failure=False)
        assert (
            handle_stage_failure(
                "summarization",
                RuntimeError("fail"),
                strict_sum,
                episode_idx=0,
            )
            is False
        )
        assert (
            handle_stage_failure(
                "entity_extraction",
                RuntimeError("fail"),
                policy,
                episode_idx=0,
            )
            is True
        )
        strict_ent = DegradationPolicy(save_summary_on_entity_extraction_failure=False)
        assert (
            handle_stage_failure(
                "entity_extraction",
                RuntimeError("fail"),
                strict_ent,
                episode_idx=0,
            )
            is False
        )
        assert (
            handle_stage_failure(
                "transcription",
                RuntimeError("fail"),
                policy,
                episode_idx=0,
            )
            is False
        )
        assert (
            handle_stage_failure(
                "metadata",
                RuntimeError("fail"),
                policy,
                episode_idx=0,
            )
            is True
        )
        assert (
            handle_stage_failure(
                cast(Any, "unknown_stage"),
                RuntimeError("fail"),
                policy,
                episode_idx=0,
            )
            is True
        )


@pytest.mark.ml_models
class TestJSONLMetricsPipelineE2E:
    """``run_pipeline`` with JSONL metrics (uses cached transformers + spaCy)."""

    @pytest.mark.critical_path
    def test_run_pipeline_writes_jsonl_events(self, e2e_server) -> None:
        from tests.integration.ml_model_cache_helpers import (
            require_transformers_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast1_with_transcript")
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "run_metrics.jsonl")
            cfg = config.Config(
                rss_urls=[rss_url],
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=False,
                auto_speakers=True,
                generate_summaries=True,
                summary_provider="transformers",
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
                summary_reduce_model=config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,
                summary_chunk_size=350,
                generate_metadata=True,
                metadata_format="json",
                jsonl_metrics_enabled=True,
                jsonl_metrics_path=jsonl_path,
            )
            podcast_scraper.run_pipeline(cfg)

            jpath = Path(jsonl_path)
            assert jpath.is_file(), "JSONL file should exist after run_pipeline"
            lines = [ln for ln in jpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert len(lines) >= 2, "Expect run_started and run_finished (and possibly episode)"
            types = {json.loads(ln).get("event_type") for ln in lines}
            assert "run_started" in types
            assert "run_finished" in types
