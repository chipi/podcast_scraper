"""Integration: degradation policy, JSONL metrics streaming, orchestration JSONL hookup.

The integration coverage job only executes ``tests/integration/``. These flows are
operator-relevant (partial-run telemetry and failure policy) and stay free of real ML.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from podcast_scraper import config as config_module
from podcast_scraper.workflow import jsonl_emitter, metrics as metrics_module, orchestration
from podcast_scraper.workflow.degradation import DegradationPolicy, handle_stage_failure

pytestmark = [pytest.mark.integration, pytest.mark.module_workflow]


class TestDegradationPolicyIntegration:
    """Exercise ``workflow.degradation`` branches used by the live pipeline."""

    def test_default_policy_constructible(self) -> None:
        policy = DegradationPolicy()
        assert policy.continue_on_stage_failure is True
        assert policy.fallback_provider_on_failure is None

    def test_summarization_respects_continue_when_saving_transcript(self) -> None:
        policy = DegradationPolicy(
            save_transcript_on_summarization_failure=True,
            continue_on_stage_failure=False,
        )
        assert handle_stage_failure("summarization", ValueError("x"), policy) is False

    def test_entity_extraction_respects_continue_when_saving_summary(self) -> None:
        policy = DegradationPolicy(
            save_summary_on_entity_extraction_failure=True,
            continue_on_stage_failure=False,
        )
        assert handle_stage_failure("entity_extraction", ValueError("x"), policy) is False

    def test_metadata_fail_fast_when_configured(self) -> None:
        policy = DegradationPolicy(continue_on_stage_failure=False)
        assert handle_stage_failure("metadata", ValueError("x"), policy) is False

    def test_unknown_stage_follows_continue_flag(self) -> None:
        loose = DegradationPolicy(continue_on_stage_failure=True)
        assert handle_stage_failure(cast(Any, "future_stage"), ValueError("x"), loose) is True
        strict = DegradationPolicy(continue_on_stage_failure=False)
        assert handle_stage_failure(cast(Any, "future_stage"), ValueError("x"), strict) is False

    def test_transcription_always_fatal_without_episode_prefix(self) -> None:
        policy = DegradationPolicy()
        assert handle_stage_failure("transcription", RuntimeError("mic"), policy) is False


class TestJSONLStreamingIntegration:
    """Exercise ``JSONLEmitter`` with a real ``Metrics`` instance (filesystem + JSON)."""

    def test_full_jsonl_cycle_with_metrics(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "run.jsonl"
        cfg = config_module.Config(
            rss_urls=["https://example.com/feed.xml"],
            jsonl_metrics_enabled=True,
            jsonl_metrics_path=str(jsonl_path),
        )
        collector = metrics_module.Metrics()
        emitter = jsonl_emitter.JSONLEmitter(collector, str(jsonl_path))
        with emitter:
            emitter.emit_run_started(cfg, run_id="integration-run")
            collector.get_or_create_episode_metrics("ep-a", 1)
            collector.update_episode_metrics(episode_id="ep-a", audio_sec=12.0)
            emitter.emit_episode_finished("ep-a")
            emitter.emit_run_finished()

        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["event_type"] == "run_started"
        assert json.loads(lines[1])["event_type"] == "episode_finished"
        assert json.loads(lines[2])["event_type"] == "run_finished"

    def test_episode_finished_missing_metrics_skips_write(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "partial.jsonl"
        cfg = config_module.Config(
            rss_urls=["https://example.com/feed.xml"],
            jsonl_metrics_enabled=True,
            jsonl_metrics_path=str(jsonl_path),
        )
        collector = metrics_module.Metrics()
        with jsonl_emitter.JSONLEmitter(collector, str(jsonl_path)) as emitter:
            emitter.emit_run_started(cfg, "rid")
            emitter.emit_episode_finished("no-such-episode")

        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["event_type"] == "run_started"

    def test_emit_without_context_raises(self, tmp_path: Path) -> None:
        cfg = config_module.Config(
            rss_urls=["https://example.com/feed.xml"],
            jsonl_metrics_enabled=True,
            jsonl_metrics_path=str(tmp_path / "x.jsonl"),
        )
        emitter = jsonl_emitter.JSONLEmitter(metrics_module.Metrics(), str(tmp_path / "x.jsonl"))
        with pytest.raises(RuntimeError, match="not opened"):
            emitter.emit_run_started(cfg, "rid")


class TestOrchestrationJsonlHookupIntegration:
    """``_setup_jsonl_emitter`` wiring (default path + disabled)."""

    def test_setup_disabled_returns_none(self, tmp_path: Path) -> None:
        collector = metrics_module.Metrics()
        cfg = config_module.Config(
            rss_urls=["https://example.com/feed.xml"],
            jsonl_metrics_enabled=False,
        )
        assert orchestration._setup_jsonl_emitter(cfg, str(tmp_path), collector) is None

    def test_setup_enabled_writes_default_run_jsonl(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "out"
        run_dir.mkdir()
        cfg = config_module.Config(
            rss_urls=["https://example.com/feed.xml"],
            jsonl_metrics_enabled=True,
            jsonl_metrics_path=None,
            run_id="orch-jsonl",
        )
        collector = metrics_module.Metrics()
        emitter = orchestration._setup_jsonl_emitter(cfg, str(run_dir), collector)
        assert emitter is not None
        try:
            run_jsonl = run_dir / "run.jsonl"
            assert run_jsonl.is_file()
            first = json.loads(run_jsonl.read_text(encoding="utf-8").splitlines()[0])
            assert first["event_type"] == "run_started"
            assert first["run_id"] == "orch-jsonl"
        finally:
            emitter.__exit__(None, None, None)
