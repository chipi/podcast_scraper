"""Unit tests for ``enrichment.events`` — JSONL event vocabulary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.events import (
    ALL_EVENT_TYPES,
    append_event,
    build_auto_disabled,
    build_cancelled,
    build_circuit_opened,
    build_enricher_completed,
    build_enricher_retry,
    build_enricher_started,
    build_health_re_enabled,
    build_run_completed,
    build_run_skipped,
    build_run_started,
    build_stall_warning,
    EVENT_ENRICHER_AUTO_DISABLED,
    EVENT_ENRICHER_CANCELLED,
    EVENT_ENRICHER_CIRCUIT_OPENED,
    EVENT_ENRICHER_COMPLETED,
    EVENT_ENRICHER_RETRY,
    EVENT_ENRICHER_STALL_WARNING,
    EVENT_ENRICHER_STARTED,
    EVENT_HEALTH_RE_ENABLED,
    EVENT_RUN_COMPLETED,
    EVENT_RUN_SKIPPED,
    EVENT_RUN_STARTED,
)
from podcast_scraper.enrichment.protocol import RunContext


def _ctx() -> RunContext:
    return RunContext(
        run_id="run-1",
        parent_run_id="parent-1",
        enricher_id="topic_similarity",
        enricher_version="1.0.0",
        tier="embedding",
        attempt=2,
        job_id="job-1",
        cancel_event=asyncio.Event(),
    )


# ---------------------------------------------------------------------------
# Vocabulary completeness
# ---------------------------------------------------------------------------


def test_all_event_types_set_is_complete() -> None:
    expected = {
        EVENT_RUN_STARTED,
        EVENT_RUN_COMPLETED,
        EVENT_RUN_SKIPPED,
        EVENT_ENRICHER_STARTED,
        EVENT_ENRICHER_RETRY,
        EVENT_ENRICHER_COMPLETED,
        EVENT_ENRICHER_CIRCUIT_OPENED,
        EVENT_ENRICHER_AUTO_DISABLED,
        EVENT_ENRICHER_CANCELLED,
        EVENT_ENRICHER_STALL_WARNING,
        EVENT_HEALTH_RE_ENABLED,
    }
    assert ALL_EVENT_TYPES == expected


def test_event_types_carry_enrichment_namespace_prefix() -> None:
    for et in ALL_EVENT_TYPES:
        assert et.startswith("enrichment.")


# ---------------------------------------------------------------------------
# Builders — payload shape contracts
# ---------------------------------------------------------------------------


def test_build_run_started_carries_envelope() -> None:
    p = build_run_started(
        run_id="r",
        parent_run_id="p",
        profile="cloud_thin",
        enricher_set=["topic_cooccurrence", "topic_similarity"],
    )
    assert p["event_type"] == EVENT_RUN_STARTED
    assert p["run_id"] == "r"
    assert p["parent_run_id"] == "p"
    assert p["profile"] == "cloud_thin"
    assert p["enricher_set"] == ["topic_cooccurrence", "topic_similarity"]
    assert "ts" in p


def test_build_run_started_drops_none_optional_fields() -> None:
    """Standalone enrichment runs have parent_run_id=None; the key
    is omitted from the payload to keep the JSONL stream clean."""
    p = build_run_started(run_id="r", parent_run_id=None, profile=None, enricher_set=[])
    assert "parent_run_id" not in p
    assert "profile" not in p


def test_build_run_completed_carries_per_enricher_totals() -> None:
    totals = {
        "topic_cooccurrence": {"ok": 1, "failed": 0},
        "topic_consensus": {"ok": 0, "failed": 1, "retries": 2},
    }
    p = build_run_completed(
        run_id="r",
        parent_run_id=None,
        duration_ms=42000,
        per_enricher_totals=totals,
    )
    assert p["event_type"] == EVENT_RUN_COMPLETED
    assert p["duration_ms"] == 42000
    assert p["per_enricher_totals"] == totals


def test_build_run_skipped_carries_reason() -> None:
    p = build_run_skipped(run_id="r", reason="core_pipeline_failed")
    assert p["event_type"] == EVENT_RUN_SKIPPED
    assert p["reason"] == "core_pipeline_failed"


def test_build_enricher_started_carries_ctx_envelope() -> None:
    p = build_enricher_started(_ctx(), scope="corpus")
    assert p["event_type"] == EVENT_ENRICHER_STARTED
    assert p["scope"] == "corpus"
    # Correlation fields from ctx.
    assert p["run_id"] == "run-1"
    assert p["enricher_id"] == "topic_similarity"
    assert p["tier"] == "embedding"
    assert p["attempt"] == 2


def test_build_enricher_retry_carries_backoff_and_reason() -> None:
    p = build_enricher_retry(
        _ctx(),
        backoff_s=1.0,
        reason="transient_dependency_error",
        error_class="DependencyAccessError",
    )
    assert p["event_type"] == EVENT_ENRICHER_RETRY
    assert p["backoff_s"] == pytest.approx(1.0)
    assert p["reason"] == "transient_dependency_error"
    assert p["error_class"] == "DependencyAccessError"
    assert p["enricher_id"] == "topic_similarity"


def test_build_enricher_completed_carries_status_and_metrics() -> None:
    p = build_enricher_completed(
        _ctx(),
        status="ok",
        duration_ms=412,
        records_written=42,
        retries=0,
    )
    assert p["event_type"] == EVENT_ENRICHER_COMPLETED
    assert p["status"] == "ok"
    assert p["duration_ms"] == 412
    assert p["records_written"] == 42
    assert p["retries"] == 0


def test_build_circuit_opened_carries_failure_count() -> None:
    p = build_circuit_opened(
        _ctx(),
        consecutive_failures=5,
        cooldown_until="2026-06-26T16:01:42Z",
    )
    assert p["event_type"] == EVENT_ENRICHER_CIRCUIT_OPENED
    assert p["consecutive_failures"] == 5
    assert p["cooldown_until"] == "2026-06-26T16:01:42Z"
    assert "opened_at" in p


def test_build_auto_disabled_carries_failed_runs_and_reason() -> None:
    p = build_auto_disabled(
        _ctx(),
        consecutive_failed_runs=2,
        reason="2 consecutive failed runs (circuit opened twice)",
    )
    assert p["event_type"] == EVENT_ENRICHER_AUTO_DISABLED
    assert p["consecutive_failed_runs"] == 2
    assert "2 consecutive" in p["reason"]
    assert "disabled_at" in p


def test_build_cancelled_carries_partial_records() -> None:
    p = build_cancelled(_ctx(), reason="cancel_requested", partial_records_written=7)
    assert p["event_type"] == EVENT_ENRICHER_CANCELLED
    assert p["partial_records_written"] == 7
    assert p["reason"] == "cancel_requested"


def test_build_stall_warning_carries_expected_interval() -> None:
    p = build_stall_warning(
        _ctx(),
        last_heartbeat_at="2026-06-26T15:03:14Z",
        expected_interval_s=5.0,
    )
    assert p["event_type"] == EVENT_ENRICHER_STALL_WARNING
    assert p["expected_interval_s"] == pytest.approx(5.0)


def test_build_health_re_enabled_carries_audit_fields() -> None:
    p = build_health_re_enabled(
        enricher_id="topic_consensus",
        operator_id="ops@example.com",
        reset_counter=True,
        cleared_cooldown=True,
        reason="confirmed transient HF outage",
    )
    assert p["event_type"] == EVENT_HEALTH_RE_ENABLED
    assert p["enricher_id"] == "topic_consensus"
    assert p["operator_id"] == "ops@example.com"
    assert p["reset_counter"] is True
    assert p["cleared_cooldown"] is True
    assert "confirmed transient" in p["reason"]


# ---------------------------------------------------------------------------
# append_event — JSONL appender
# ---------------------------------------------------------------------------


def test_append_event_writes_one_line_per_event(tmp_path: Path) -> None:
    path = tmp_path / "subdir" / "run.jsonl"
    append_event(path, {"event_type": "test", "v": 1})
    append_event(path, {"event_type": "test", "v": 2})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["v"] == 1
    assert json.loads(lines[1])["v"] == 2


def test_append_event_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "run.jsonl"
    append_event(path, {"event_type": "test", "v": 1})
    assert path.is_file()


def test_append_event_does_not_raise_on_unwritable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OSError on write must NOT propagate (o11y is best-effort)."""
    path = tmp_path / "run.jsonl"

    real_open = open

    def boom(*args, **kwargs):
        if str(args[0]).endswith("run.jsonl"):
            raise OSError("disk full")
        return real_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", boom)
    # Must not raise.
    append_event(path, {"event_type": "test"})


def test_append_event_serializes_unicode(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    append_event(path, {"event_type": "test", "person": "Müller-García"})
    line = path.read_text(encoding="utf-8").strip()
    decoded = json.loads(line)
    assert decoded["person"] == "Müller-García"
