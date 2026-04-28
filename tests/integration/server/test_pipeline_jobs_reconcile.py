"""Unit tests for pipeline job reconcile ordering."""

from __future__ import annotations

import pytest

from podcast_scraper.server.pipeline_jobs import (
    reconcile_jobs_inplace,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STALE,
)

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm]/[server]
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


def test_reconcile_prefers_dead_pid_over_wall_clock_stale() -> None:
    jobs = [
        {
            "job_id": "a",
            "status": STATUS_RUNNING,
            "started_at": "1999-01-01T00:00:00Z",
            "pid": 9_999_001,
        }
    ]
    details = reconcile_jobs_inplace(jobs, stale_seconds=60)
    assert jobs[0]["status"] == STATUS_FAILED
    assert jobs[0]["error_reason"] == "orphan_reconciled_dead_pid"
    assert "dead pid" in details[0]


def test_reconcile_marks_stale_when_pid_missing() -> None:
    jobs = [
        {
            "job_id": "b",
            "status": STATUS_RUNNING,
            "started_at": "1999-01-01T00:00:00Z",
            "pid": None,
        }
    ]
    details = reconcile_jobs_inplace(jobs, stale_seconds=60)
    assert jobs[0]["status"] == STATUS_STALE
    assert "stale" in details[0].lower()
