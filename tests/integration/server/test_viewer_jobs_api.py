"""Integration tests for opt-in /api/jobs pipeline API (Phase 2)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

from podcast_scraper.server.app import create_app
from podcast_scraper.server.pipeline_job_registry import with_jobs_locked_mutate
from podcast_scraper.server.routes import jobs as jobs_mod

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


def test_jobs_list_requires_corpus_when_server_has_no_default_output_dir() -> None:
    app = FastAPI()
    app.state.output_dir = None
    app.state.jobs_api_enabled = True
    app.include_router(jobs_mod.router, prefix="/api")
    client = TestClient(app)
    r = client.get("/api/jobs")
    assert r.status_code == 400
    assert "Corpus path" in r.json().get("detail", "")


def test_jobs_rejects_corpus_path_outside_anchor_via_shared_exception_handler(
    corpus: Path,
) -> None:
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    client = TestClient(app)
    r = client.get("/api/jobs", params={"path": "/etc"})
    assert r.status_code == 400
    assert "subdirectory" in r.json().get("detail", "").lower()


def test_jobs_api_disabled_returns_500_when_router_mounted_without_create_app_guard(
    tmp_path: Path,
) -> None:
    """``_corpus_and_operator`` rejects calls when ``jobs_api_enabled`` is false (misconfiguration)."""
    app = FastAPI()
    app.state.output_dir = tmp_path
    app.state.jobs_api_enabled = False
    app.include_router(jobs_mod.router, prefix="/api")
    client = TestClient(app)
    r = client.get("/api/jobs", params={"path": str(tmp_path)})
    assert r.status_code == 500
    assert "jobs_api" in r.json().get("detail", "").lower()


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

    tail = client.get(
        f"/api/jobs/{job_id}/log-tail", params={"path": str(corpus), "max_bytes": 4096}
    )
    assert tail.status_code == 200
    body = tail.json()
    assert body.get("truncated") is False
    assert "fake-log" in body.get("text", "")

    tail_q = client.get(
        "/api/jobs/subprocess-log-tail",
        params={"path": str(corpus), "job_id": job_id, "max_bytes": 4096},
    )
    assert tail_q.status_code == 200
    body_q = tail_q.json()
    assert body_q.get("truncated") is False
    assert "fake-log" in body_q.get("text", "")


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


def test_jobs_get_unknown_returns_404(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    client = TestClient(app)
    r = client.get("/api/jobs/does-not-exist-uuid", params={"path": str(corpus)})
    assert r.status_code == 404


def test_jobs_log_returns_404_when_file_missing(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    client = TestClient(app)
    jid = "00000000-0000-4000-8000-000000000088"

    def seed(jobs: list) -> None:
        jobs.append(
            {
                "job_id": jid,
                "command_type": "full_incremental_pipeline",
                "status": "succeeded",
                "created_at": "2026-04-19T12:00:00Z",
                "started_at": "2026-04-19T12:00:01Z",
                "ended_at": "2026-04-19T12:00:02Z",
                "pid": None,
                "argv_summary": "[]",
                "exit_code": 0,
                "log_relpath": f".viewer/jobs/{jid}.log",
                "error_reason": None,
                "cancel_requested": False,
            }
        )

    with_jobs_locked_mutate(corpus, seed)
    r = client.get(f"/api/jobs/{jid}/log", params={"path": str(corpus)})
    assert r.status_code == 404


def test_jobs_cancel_unknown_returns_404(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    client = TestClient(app)
    r = client.post(
        "/api/jobs/00000000-0000-4000-8000-000000000077/cancel", params={"path": str(corpus)}
    )
    assert r.status_code == 404


def test_jobs_reconcile_wall_clock_stale(monkeypatch: pytest.MonkeyPatch, corpus: Path) -> None:
    monkeypatch.setenv("PODCAST_JOB_STALE_SECONDS", "120")
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    client = TestClient(app)

    def seed(jobs: list) -> None:
        jobs.append(
            {
                "job_id": "stale-wall",
                "command_type": "full_incremental_pipeline",
                "status": "running",
                "created_at": "2000-01-01T00:00:00Z",
                "started_at": "2000-01-01T00:00:01Z",
                "ended_at": None,
                "pid": os.getpid(),
                "argv_summary": "[]",
                "exit_code": None,
                "log_relpath": ".viewer/jobs/stale-wall.log",
                "error_reason": None,
                "cancel_requested": False,
            }
        )

    with_jobs_locked_mutate(corpus, seed)
    rec = client.post("/api/jobs/reconcile", params={"path": str(corpus)})
    assert rec.status_code == 200
    assert rec.json()["updated"] >= 1
    row = client.get("/api/jobs/stale-wall", params={"path": str(corpus)}).json()
    assert row["status"] == "stale"


def test_jobs_docker_mode_rejects_missing_pipeline_install_extras(
    monkeypatch: pytest.MonkeyPatch, corpus: Path
) -> None:
    monkeypatch.setenv("PODCAST_PIPELINE_EXEC_MODE", "docker")
    (corpus / "viewer_operator.yaml").write_text("max_episodes: 1\n", encoding="utf-8")
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    client = TestClient(app)
    r = client.post("/api/jobs", params={"path": str(corpus)})
    assert r.status_code == 400
    assert "pipeline_install_extras" in str(r.json().get("detail", ""))


def test_jobs_docker_mode_accepts_pipeline_install_extras(
    monkeypatch: pytest.MonkeyPatch, corpus: Path, fake_factory_immediate: object
) -> None:
    monkeypatch.setenv("PODCAST_PIPELINE_EXEC_MODE", "docker")
    (corpus / "viewer_operator.yaml").write_text(
        "pipeline_install_extras: ml\nmax_episodes: 1\n", encoding="utf-8"
    )
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    app.state.jobs_subprocess_factory = fake_factory_immediate
    client = TestClient(app)
    r = client.post("/api/jobs", params={"path": str(corpus)})
    assert r.status_code == 202


def test_jobs_submit_omits_queue_position_when_snapshot_has_non_int(
    corpus: Path, fake_factory_immediate: object
) -> None:
    jid = "00000000-0000-4000-8000-00000000000b"
    queued_rec = {"job_id": jid, "status": "queued"}
    snap = [{"job_id": jid, "queue_position": "4", "status": "queued"}]
    (corpus / "viewer_operator.yaml").write_text("max_episodes: 1\n", encoding="utf-8")

    with patch.object(jobs_mod, "enqueue_pipeline_job", return_value=queued_rec):
        with patch.object(jobs_mod, "list_jobs_snapshot", return_value=snap):
            app = create_app(corpus, static_dir=False, enable_jobs_api=True)
            app.state.jobs_subprocess_factory = fake_factory_immediate
            client = TestClient(app)
            r = client.post("/api/jobs", params={"path": str(corpus)})
    assert r.status_code == 202
    assert r.json().get("queue_position") is None


def test_jobs_submit_includes_queue_position_from_snapshot_when_queued(
    corpus: Path, fake_factory_immediate: object
) -> None:
    """``POST /api/jobs`` copies ``queue_position`` from ``list_jobs_snapshot`` for queued rows."""
    jid = "00000000-0000-4000-8000-00000000000a"
    queued_rec = {"job_id": jid, "status": "queued"}
    snap = [{"job_id": jid, "queue_position": 4, "status": "queued"}]
    (corpus / "viewer_operator.yaml").write_text("max_episodes: 1\n", encoding="utf-8")

    with patch.object(jobs_mod, "enqueue_pipeline_job", return_value=queued_rec):
        with patch.object(jobs_mod, "list_jobs_snapshot", return_value=snap):
            app = create_app(corpus, static_dir=False, enable_jobs_api=True)
            app.state.jobs_subprocess_factory = fake_factory_immediate
            client = TestClient(app)
            r = client.post("/api/jobs", params={"path": str(corpus)})
    assert r.status_code == 202
    assert r.json()["queue_position"] == 4


def test_jobs_docker_mode_accepts_pipeline_install_extras_llm(
    monkeypatch: pytest.MonkeyPatch, corpus: Path, fake_factory_immediate: object
) -> None:
    monkeypatch.setenv("PODCAST_PIPELINE_EXEC_MODE", "docker")
    (corpus / "viewer_operator.yaml").write_text(
        "pipeline_install_extras: llm\nmax_episodes: 1\n", encoding="utf-8"
    )
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    app.state.jobs_subprocess_factory = fake_factory_immediate
    client = TestClient(app)
    r = client.post("/api/jobs", params={"path": str(corpus)})
    assert r.status_code == 202
