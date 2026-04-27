"""Unit coverage for ``pipeline_jobs`` helpers (env parsing, argv, reconcile)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from podcast_scraper.server.pipeline_job_registry import with_jobs_locked_mutate
from podcast_scraper.server.pipeline_jobs import (

    argv_summary,
    build_pipeline_argv,
    cancel_job,
    max_concurrent_jobs,
    pid_alive,
    reconcile_jobs_inplace,
    stale_after_seconds,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_STALE,
    STATUS_SUCCEEDED,
)

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm]/[server]
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


def test_max_concurrent_jobs_invalid_env_defaults_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_VIEWER_MAX_PIPELINE_JOBS", "not-an-int")
    assert max_concurrent_jobs() == 1
    monkeypatch.setenv("PODCAST_VIEWER_MAX_PIPELINE_JOBS", "0")
    assert max_concurrent_jobs() == 1
    monkeypatch.setenv("PODCAST_VIEWER_MAX_PIPELINE_JOBS", "4")
    assert max_concurrent_jobs() == 4


def test_stale_after_seconds_invalid_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_JOB_STALE_SECONDS", "x")
    assert stale_after_seconds() == 86400
    monkeypatch.setenv("PODCAST_JOB_STALE_SECONDS", "-5")
    assert stale_after_seconds() == 0


@pytest.mark.parametrize(
    "pid,expected",
    [
        (None, False),
        (0, False),
        (-1, False),
    ],
)
def test_pid_alive_false_for_invalid_pid(pid: int | None, expected: bool) -> None:
    assert pid_alive(pid) is expected


def test_pid_alive_true_for_current_process() -> None:
    assert pid_alive(os.getpid()) is True


def test_build_pipeline_argv_includes_profile_and_feeds_spec(tmp_path: Path) -> None:
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("profile: local\nbatch_size: 1\n", encoding="utf-8")
    spec = tmp_path / "feeds.spec.yaml"
    spec.write_text("feeds: []\n", encoding="utf-8")
    argv = build_pipeline_argv(tmp_path, op)
    assert "--profile" in argv
    assert "local" in argv
    assert "--config" in argv
    assert str(op.resolve()) in argv
    assert "--feeds-spec" in argv


def test_build_pipeline_argv_without_feeds_spec_file(tmp_path: Path) -> None:
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("max_episodes: 2\n", encoding="utf-8")
    argv = build_pipeline_argv(tmp_path, op)
    assert "--feeds-spec" not in argv


def test_build_pipeline_argv_operator_read_oserror(tmp_path: Path) -> None:
    op = tmp_path / "missing.yaml"
    argv = build_pipeline_argv(tmp_path, op)
    assert argv.count("--profile") == 0
    assert "--config" in argv


def test_reconcile_jobs_inplace_dead_pid() -> None:
    jobs = [
        {
            "job_id": "j1",
            "status": STATUS_RUNNING,
            "started_at": "2020-01-01T00:00:00Z",
            "pid": 9_999_001,
        }
    ]
    details = reconcile_jobs_inplace(jobs, stale_seconds=0)
    assert jobs[0]["status"] == STATUS_FAILED
    assert "dead pid" in details[0]


def test_argv_summary_roundtrip_json() -> None:
    s = argv_summary(["a", "b", "c"])
    assert json.loads(s) == ["a", "b", "c"]


def test_cancel_job_noop_terminal_for_succeeded(tmp_path: Path) -> None:
    jid = "terminal-job"

    def seed(jobs: list) -> None:
        jobs.append(
            {
                "job_id": jid,
                "status": STATUS_SUCCEEDED,
                "created_at": "2020-01-01T00:00:00Z",
                "ended_at": "2020-01-01T00:01:00Z",
                "pid": None,
                "argv_summary": "[]",
                "exit_code": 0,
                "log_relpath": f".viewer/jobs/{jid}.log",
                "error_reason": None,
                "cancel_requested": False,
            }
        )

    with_jobs_locked_mutate(tmp_path, seed)
    outcome, rec = cancel_job(tmp_path, jid)
    assert outcome == "noop_terminal"
    assert rec is not None
    assert rec["status"] == STATUS_SUCCEEDED


def test_cancel_job_queued_marks_cancelled(tmp_path: Path) -> None:
    jid = "queued-cancel"

    def seed(jobs: list) -> None:
        jobs.append(
            {
                "job_id": jid,
                "status": STATUS_QUEUED,
                "created_at": "2026-01-01T00:00:00Z",
                "started_at": None,
                "ended_at": None,
                "pid": None,
                "argv_summary": "[]",
                "exit_code": None,
                "log_relpath": f".viewer/jobs/{jid}.log",
                "error_reason": None,
                "cancel_requested": False,
            }
        )

    with_jobs_locked_mutate(tmp_path, seed)
    outcome, rec = cancel_job(tmp_path, jid)
    assert outcome == "cancelled"
    assert rec is not None
    assert rec["status"] == STATUS_CANCELLED


def test_reconcile_jobs_inplace_wall_clock_stale() -> None:
    """Running job: alive PID, old ``started_at`` → stale when threshold is low."""
    jobs = [
        {
            "job_id": "j2",
            "status": STATUS_RUNNING,
            "started_at": "2000-01-01T00:00:00Z",
            "pid": os.getpid(),
        }
    ]
    details = reconcile_jobs_inplace(jobs, stale_seconds=120)
    assert jobs[0]["status"] == STATUS_STALE
    assert any("stale" in d.lower() for d in details)
