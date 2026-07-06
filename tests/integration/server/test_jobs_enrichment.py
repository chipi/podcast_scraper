"""Unit tests for the enrichment-job helpers in ``server.jobs``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.server.jobs import (
    build_enrichment_argv,
    COMMAND_ENRICHMENT,
    COMMAND_FULL,
    enqueue_enrichment_job,
    enqueue_pipeline_job,
    list_jobs_snapshot,
    STATUS_QUEUED,
    STATUS_RUNNING,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# build_enrichment_argv
# ---------------------------------------------------------------------------


def test_build_enrichment_argv_minimal(tmp_path: Path) -> None:
    argv = build_enrichment_argv(tmp_path)
    assert "-m" in argv
    # #1069 consistency: enrichment runs via the main-CLI ``enrich`` subcommand
    # (``-m podcast_scraper.cli enrich``), so it invokes + dockers like the pipeline.
    assert "podcast_scraper.cli" in argv
    assert "enrich" in argv
    assert "--output-dir" in argv
    assert str(tmp_path) in argv
    assert "--log-level" in argv


def test_build_enrichment_argv_passes_only_skip_corpus_only(tmp_path: Path) -> None:
    argv = build_enrichment_argv(
        tmp_path,
        only=["topic_cooccurrence", "temporal_velocity"],
        skip=["nli_contradiction"],
        corpus_only=True,
    )
    s = " ".join(argv)
    assert "--only topic_cooccurrence,temporal_velocity" in s
    assert "--skip nli_contradiction" in s
    assert "--corpus-only" in argv


def test_build_enrichment_argv_includes_config_when_present(tmp_path: Path) -> None:
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("enrichment:\n  enabled: true\n", encoding="utf-8")
    argv = build_enrichment_argv(tmp_path, operator_yaml=op)
    assert "--config" in argv
    assert str(op) in argv


# ---------------------------------------------------------------------------
# enqueue_enrichment_job — command_type + registry round-trip
# ---------------------------------------------------------------------------


def test_enqueue_enrichment_job_creates_corpus_enrichment_command_type(
    tmp_path: Path,
) -> None:
    rec = enqueue_enrichment_job(tmp_path)
    assert rec["command_type"] == COMMAND_ENRICHMENT
    assert rec["job_id"]
    assert rec["status"] in (STATUS_RUNNING, STATUS_QUEUED)


def test_enqueue_enrichment_job_serializes_into_registry(tmp_path: Path) -> None:
    rec = enqueue_enrichment_job(tmp_path, only=["a"])
    snap = list_jobs_snapshot(tmp_path)
    assert any(r["job_id"] == rec["job_id"] for r in snap)
    same = next(r for r in snap if r["job_id"] == rec["job_id"])
    assert same["command_type"] == COMMAND_ENRICHMENT


def test_enqueue_enrichment_job_argv_summary_has_enrichment_cli(tmp_path: Path) -> None:
    rec = enqueue_enrichment_job(tmp_path)
    # #1069 consistency: the stored command is the main-CLI ``enrich`` subcommand.
    assert "podcast_scraper.cli" in rec["argv_summary"]
    assert "enrich" in rec["argv_summary"]


# ---------------------------------------------------------------------------
# Pipeline + enrichment job kinds coexist in the same registry
# ---------------------------------------------------------------------------


def test_pipeline_and_enrichment_jobs_share_registry(tmp_path: Path) -> None:
    """Both kinds enqueue + appear in the same JSONL registry — the
    promote-queued / cancel / reconcile / pid-alive logic stays
    command_type-agnostic."""
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("profile: airgapped_thin\n", encoding="utf-8")
    pipe_rec = enqueue_pipeline_job(tmp_path, op)
    enrich_rec = enqueue_enrichment_job(tmp_path)

    snap = list_jobs_snapshot(tmp_path)
    ids = {r["job_id"]: r["command_type"] for r in snap}
    assert ids[pipe_rec["job_id"]] == COMMAND_FULL
    assert ids[enrich_rec["job_id"]] == COMMAND_ENRICHMENT


def test_enqueue_enrichment_job_queues_when_running_at_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the running count >= cap, new enrichment jobs land queued."""
    # Force cap to 1.
    monkeypatch.setenv("PODCAST_VIEWER_MAX_PIPELINE_JOBS", "1")
    first = enqueue_enrichment_job(tmp_path)
    second = enqueue_enrichment_job(tmp_path)
    assert first["status"] == STATUS_RUNNING
    assert second["status"] == STATUS_QUEUED


# ---------------------------------------------------------------------------
# reconcile + cancel work for the new kind without code changes there
# ---------------------------------------------------------------------------


def test_cancel_works_for_enrichment_job(tmp_path: Path) -> None:
    """Cancel is command_type-agnostic — it walks the registry by id only."""
    from podcast_scraper.server.jobs import cancel_job

    rec = enqueue_enrichment_job(tmp_path)
    outcome, updated = cancel_job(tmp_path, str(rec["job_id"]))
    # Either signal_running (then post-finalize cancels) or already-cancelled
    # transition — both are acceptable mid-test outcomes.
    assert outcome in ("cancelled", "signal_running")
    assert updated is not None
    assert updated["command_type"] == COMMAND_ENRICHMENT


def test_reconcile_does_not_distinguish_command_type(tmp_path: Path) -> None:
    """Reconcile walks all running jobs regardless of command_type."""
    from podcast_scraper.server.jobs import apply_reconcile

    enqueue_enrichment_job(tmp_path)
    # Reconcile must not raise; no stale jobs to mark on a fresh registry.
    updated, details = apply_reconcile(tmp_path)
    assert isinstance(updated, int)
    assert isinstance(details, list)


# ---------------------------------------------------------------------------
# Argv summary helpers — round-trip serialization
# ---------------------------------------------------------------------------


def test_enrichment_job_record_has_required_fields(tmp_path: Path) -> None:
    rec = enqueue_enrichment_job(tmp_path)
    REQUIRED: set[str] = {
        "job_id",
        "command_type",
        "status",
        "created_at",
        "argv_summary",
        "cancel_requested",
        "log_relpath",
    }
    missing = REQUIRED - set(rec.keys())
    assert not missing, f"missing fields in record: {missing}"


def test_argv_summary_is_json_serialised(tmp_path: Path) -> None:
    import json

    rec = enqueue_enrichment_job(tmp_path, only=["a", "b"], corpus_only=True)
    argv = json.loads(rec["argv_summary"])
    assert isinstance(argv, list)
    assert "--only" in argv
    assert "a,b" in argv
    assert "--corpus-only" in argv


_ = Any  # quell ruff/flake8 unused-import suspicion on Any
