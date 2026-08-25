"""The emergency brake: stop everything, in the only order that works (#1785).

2026-08-18/19: a pipeline container ran unattended for 14 hours spending on Deepgram, and no
prod surface could stop it without SSH. The incident fixed the design:

- **Pause BEFORE signalling.** The sweeper promotes queued work every 30s, so stopping first
  just hands the slot to the next queued job and looks like the stop failed.
- **SIGTERM with grace, not kill** — in-flight provider cost must flush.
- **Verify, don't assume** — re-check and report anything that survived.
- Queued jobs stay queued (held by the pause), NOT cancelled: the operator resumes later.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

from podcast_scraper.server import jobs as jobs_core, queue_sweeper
from podcast_scraper.server.pipeline_job_registry import with_jobs_locked_mutate
from podcast_scraper.server.routes import jobs as jobs_routes

pytestmark = [pytest.mark.integration]


def _app(corpus: Path) -> TestClient:
    app = FastAPI()
    app.state.output_dir = corpus
    app.state.jobs_api_enabled = True
    app.include_router(jobs_routes.router, prefix="/api")
    return TestClient(app)


def _seed(corpus: Path, *rows: dict) -> None:
    def fn(jobs: list[dict]) -> None:
        jobs.extend(rows)

    with_jobs_locked_mutate(corpus, fn)


def _running_row(job_id: str, pid: int) -> dict:
    return {
        "job_id": job_id,
        "status": jobs_core.STATUS_RUNNING,
        "created_at": "2026-08-25T00:00:00Z",
        "started_at": "2026-08-25T00:00:00Z",
        "pid": pid,
        "boot_id": jobs_core.current_boot_id(),
        "argv_summary": "pipeline --config x",
    }


def _queued_row(job_id: str) -> dict:
    return {
        "job_id": job_id,
        "status": jobs_core.STATUS_QUEUED,
        "created_at": "2026-08-25T00:00:00Z",
        "argv_summary": "pipeline --config y",
    }


def test_stop_pauses_the_queue_before_any_sigterm(tmp_path: Path, monkeypatch) -> None:
    """The incident's core lesson: the pause flag must exist when the signal fires."""
    _seed(tmp_path, _running_row("run-1", 4242))
    flag = tmp_path / queue_sweeper.PAUSE_FLAG_RELPATH

    flag_state_at_kill: list[bool] = []

    def fake_kill(pid: int, sig: int) -> None:
        flag_state_at_kill.append(flag.exists())

    monkeypatch.setattr(os, "kill", fake_kill)

    r = _app(tmp_path).post("/api/jobs/stop", params={"verify_seconds": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["queue_paused"] is True
    assert flag.exists(), "the stop must leave the queue held"
    assert flag_state_at_kill == [True], (
        "SIGTERM fired before the pause flag existed — the sweeper's 30s loop can promote "
        "the next queued job into the slot the stop just freed (#1785)"
    )
    assert [j["job_id"] for j in body["stopped"]] == ["run-1"]


def test_stop_leaves_queued_jobs_queued(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    _seed(tmp_path, _running_row("run-1", 4242), _queued_row("q-1"))

    r = _app(tmp_path).post("/api/jobs/stop", params={"verify_seconds": 0})
    assert r.status_code == 200

    snap = {j["job_id"]: j["status"] for j in jobs_core.list_jobs_snapshot(tmp_path)}
    assert snap["q-1"] == jobs_core.STATUS_QUEUED, (
        "a queued job was cancelled by the brake — pause holds it; the operator decides on "
        "resume, not the stop"
    )


def test_stop_verifies_and_reports_survivors(tmp_path: Path, monkeypatch) -> None:
    """A stop that assumes is the old failure with fewer keystrokes. Survivors must be named."""
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(jobs_core, "pid_alive", lambda pid: True)
    _seed(tmp_path, _running_row("run-1", 4242))

    r = _app(tmp_path).post("/api/jobs/stop", params={"verify_seconds": 0.05})
    body = r.json()
    assert body["all_stopped"] is False
    assert [j["job_id"] for j in body["survivors"]] == ["run-1"]


def test_stop_confirms_when_nothing_survives(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(jobs_core, "pid_alive", lambda pid: False)
    _seed(tmp_path, _running_row("run-1", 4242))

    r = _app(tmp_path).post("/api/jobs/stop", params={"verify_seconds": 0.05})
    body = r.json()
    assert body["all_stopped"] is True
    assert body["survivors"] == []


def test_resume_releases_the_queue(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    client = _app(tmp_path)
    client.post("/api/jobs/stop", params={"verify_seconds": 0})
    assert (tmp_path / queue_sweeper.PAUSE_FLAG_RELPATH).exists()

    r = client.post("/api/jobs/resume")
    assert r.status_code == 200
    assert r.json()["queue_paused"] is False
    assert not (tmp_path / queue_sweeper.PAUSE_FLAG_RELPATH).exists()


def test_running_lists_only_what_is_executing(tmp_path: Path) -> None:
    """GET /jobs/running: the 'what is spending right now' view — no SSH, no docker ps."""
    _seed(
        tmp_path,
        _running_row("run-1", 4242),
        _queued_row("q-1"),
        {
            "job_id": "done-1",
            "status": "completed",
            "created_at": "2026-08-25T00:00:00Z",
            "argv_summary": "pipeline",
        },
    )
    r = _app(tmp_path).get("/api/jobs/running")
    assert r.status_code == 200
    assert [j["job_id"] for j in r.json()["jobs"]] == ["run-1"]
