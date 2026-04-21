"""Integration tests for opt-in /api/jobs pipeline API (RFC-077 Phase 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

from podcast_scraper.server.app import create_app
from podcast_scraper.server.pipeline_job_registry import with_jobs_locked_mutate

pytestmark = [pytest.mark.integration]


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "ep1.gi.json").write_text(
        json.dumps({"grounded_insights": {"version": "1.0"}}), encoding="utf-8"
    )
    return tmp_path


class _FakeProcImmediate:
    """Subprocess stand-in that finishes immediately (TestClient runs response backgrounds)."""

    pid = 91002

    async def wait(self) -> int:
        return 0


@pytest.fixture()
def fake_factory_immediate() -> object:
    async def _factory(argv: list[str], corpus_root: Path, log_abs: Path):  # noqa: ARG001
        log_abs.parent.mkdir(parents=True, exist_ok=True)
        log_abs.write_bytes(b"fake-log\n")
        return _FakeProcImmediate()

    return _factory


def test_jobs_not_mounted_by_default(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False)
    client = TestClient(app)
    assert client.get("/api/health").json().get("jobs_api") is False
    assert client.post("/api/jobs", params={"path": str(corpus)}).status_code == 404


def test_jobs_only_uses_per_corpus_operator_path(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    assert getattr(app.state, "operator_config_fixed_path", None) is None


def test_jobs_health_and_submit_completes(corpus: Path, fake_factory_immediate: object) -> None:
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    app.state.jobs_subprocess_factory = fake_factory_immediate
    client = TestClient(app)
    h = client.get("/api/health")
    assert h.status_code == 200
    assert h.json().get("jobs_api") is True

    r = client.post("/api/jobs", params={"path": str(corpus)})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    lst = client.get("/api/jobs", params={"path": str(corpus)})
    assert lst.status_code == 200
    rows = lst.json()["jobs"]
    assert any(j["job_id"] == job_id for j in rows)

    one = client.get(f"/api/jobs/{job_id}", params={"path": str(corpus)})
    assert one.status_code == 200
    assert one.json().get("status") == "succeeded"

    log_r = client.get(f"/api/jobs/{job_id}/log", params={"path": str(corpus)})
    assert log_r.status_code == 200
    assert log_r.text.strip() == "fake-log"

    log_q = client.get("/api/jobs/subprocess-log", params={"path": str(corpus), "job_id": job_id})
    assert log_q.status_code == 200
    assert log_q.text.strip() == "fake-log"


def test_jobs_reconcile_marks_dead_pid(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    client = TestClient(app)

    def seed(jobs: list) -> None:
        jobs.append(
            {
                "job_id": "dead-pid-job",
                "command_type": "full_incremental_pipeline",
                "status": "running",
                "created_at": "2020-01-01T00:00:00Z",
                "started_at": "2020-01-01T00:00:01Z",
                "ended_at": None,
                "pid": 9_876_543,
                "argv_summary": "[]",
                "exit_code": None,
                "log_relpath": ".viewer/jobs/dead-pid-job.log",
                "error_reason": None,
                "cancel_requested": False,
            }
        )

    with_jobs_locked_mutate(corpus, seed)
    rec = client.post("/api/jobs/reconcile", params={"path": str(corpus)})
    assert rec.status_code == 200
    body = rec.json()
    assert body["updated"] >= 1
    row = client.get("/api/jobs/dead-pid-job", params={"path": str(corpus)}).json()
    assert row["status"] == "failed"
    assert row.get("error_reason") == "orphan_reconciled_dead_pid"


def test_jobs_cancel_queued(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    client = TestClient(app)
    jid = "00000000-0000-4000-8000-000000000099"

    def seed(jobs: list) -> None:
        jobs.append(
            {
                "job_id": jid,
                "command_type": "full_incremental_pipeline",
                "status": "queued",
                "created_at": "2026-04-19T12:00:00Z",
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

    with_jobs_locked_mutate(corpus, seed)
    c = client.post(f"/api/jobs/{jid}/cancel", params={"path": str(corpus)})
    assert c.status_code == 200
    assert c.json()["status"] == "cancelled"
