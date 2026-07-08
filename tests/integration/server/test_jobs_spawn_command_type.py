"""Each job kind spawns its OWN CLI, not always the pipeline (#1069 consistency).

Before the stored-argv fix, the spawn path rebuilt ``build_pipeline_argv`` regardless
of ``command_type`` (and ``promote_queued_if_slot`` overwrote the stored argv), so an
enrichment job silently ran the *pipeline*. These tests capture the argv the subprocess
factory receives and assert the pipeline job spawns the pipeline CLI and the enrichment
job spawns the enrichment CLI — the invariant that makes enrichment a true peer of
ingestion across the trigger surfaces.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

from podcast_scraper.server.app import create_app
from podcast_scraper.server.jobs import (
    argv_from_record,
    argv_summary,
    drain_queue_async,
    promote_queued_if_slot,
)
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


class _FakeProc:
    pid = 91500

    async def wait(self) -> int:
        return 0


def _capturing_factory(captured: list[list[str]]):
    async def _factory(argv, corpus_root: Path, log_abs: Path):  # noqa: ARG001
        captured.append(list(argv))
        log_abs.parent.mkdir(parents=True, exist_ok=True)
        log_abs.write_bytes(b"fake-log\n")
        return _FakeProc()

    return _factory


def test_pipeline_job_spawns_the_pipeline_cli(corpus: Path) -> None:
    captured: list[list[str]] = []
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    app.state.jobs_subprocess_factory = _capturing_factory(captured)
    client = TestClient(app)

    assert client.post("/api/jobs", params={"path": str(corpus)}).status_code == 202
    client.get("/api/jobs", params={"path": str(corpus)})  # drain any pending kickoff

    assert captured, "subprocess factory was never invoked"
    argv = captured[0]
    assert "podcast_scraper.cli" in argv
    assert "enrich" not in argv  # a pipeline job is not the enrich subcommand


def test_enrichment_job_spawns_the_enrichment_cli_not_pipeline(corpus: Path) -> None:
    captured: list[list[str]] = []
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    app.state.jobs_subprocess_factory = _capturing_factory(captured)
    client = TestClient(app)

    assert client.post("/api/jobs/enrichment", params={"path": str(corpus)}).status_code == 202
    client.get("/api/jobs", params={"path": str(corpus)})  # drain any pending kickoff

    assert captured, "subprocess factory was never invoked"
    argv = captured[0]
    # THE FIX: an enrichment job runs the enrich CLI verb, not the plain pipeline.
    assert "podcast_scraper.cli" in argv and "enrich" in argv, argv


def test_promote_preserves_queued_job_command(corpus: Path) -> None:
    """Promotion flips lifecycle state but must NOT rewrite the command to pipeline (#1069).

    Before the fix, promote_queued_if_slot rebuilt build_pipeline_argv and overwrote the
    stored argv, so a queued enrichment job would run the pipeline once promoted.
    """
    enrich_argv = ["py", "-m", "podcast_scraper.cli", "enrich", "--output-dir", str(corpus)]
    jid = "00000000-0000-4000-8000-0000000000ee"

    def seed(jobs: list) -> None:
        jobs.append(
            {
                "job_id": jid,
                "command_type": "corpus_enrichment",
                "status": "queued",
                "created_at": "2026-04-19T12:00:00Z",
                "started_at": None,
                "ended_at": None,
                "pid": None,
                "argv_summary": argv_summary(enrich_argv),
                "exit_code": None,
                "log_relpath": f".viewer/jobs/{jid}.log",
                "error_reason": None,
                "cancel_requested": False,
            }
        )

    with_jobs_locked_mutate(corpus, seed)
    promoted = promote_queued_if_slot(corpus, corpus / "viewer_operator.yaml")

    assert promoted is not None
    assert promoted["status"] == "running"
    assert json.loads(promoted["argv_summary"]) == enrich_argv  # command preserved, not pipeline


def test_argv_from_record_reads_the_stored_command() -> None:
    assert argv_from_record({"argv_summary": json.dumps(["py", "-m", "x", "--y"])}) == [
        "py",
        "-m",
        "x",
        "--y",
    ]
    # Legacy / blank / malformed rows -> None so the caller falls back to pipeline argv.
    assert argv_from_record({}) is None
    assert argv_from_record({"argv_summary": ""}) is None
    assert argv_from_record({"argv_summary": "not-json"}) is None
    assert argv_from_record({"argv_summary": "{}"}) is None  # object, not a list
    assert argv_from_record({"argv_summary": "[]"}) is None  # empty list
    assert argv_from_record({"argv_summary": json.dumps([1, 2])}) is None  # non-str items


def _job(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "job_id": "",
        "command_type": "full",
        "status": "queued",
        "created_at": "2026-04-19T12:00:00Z",
        "started_at": None,
        "ended_at": None,
        "pid": None,
        "argv_summary": "[]",
        "exit_code": None,
        "log_relpath": "",
        "error_reason": None,
        "cancel_requested": False,
    }
    base.update(over)
    base["log_relpath"] = f".viewer/jobs/{base['job_id']}.log"
    return base


def test_queued_enrichment_job_spawns_enrich_when_promoted(corpus: Path) -> None:
    """End-to-end promote path: an enrichment job queued BEHIND a running job spawns the
    ENRICH CLI (not the pipeline) once the slot frees (#1069, R1-L1).

    The units were covered (initial enrichment spawn, promote-preserves-argv) but not the
    full submit→queue→drain→spawn scenario that was the actual regression. Drives the real
    ``drain_queue_async`` (promote_queued_if_slot → start_job_if_running_record) with a
    captured subprocess factory, at the default concurrency cap of 1.
    """
    captured: list[list[str]] = []
    app = create_app(corpus, static_dir=False, enable_jobs_api=True)
    app.state.jobs_subprocess_factory = _capturing_factory(captured)

    running_id = "00000000-0000-4000-8000-00000000aaaa"
    queued_id = "00000000-0000-4000-8000-00000000bbbb"
    enrich_argv = ["py", "-m", "podcast_scraper.cli", "enrich", "--output-dir", str(corpus)]

    def seed(jobs: list) -> None:
        jobs.append(
            _job(job_id=running_id, status="running", pid=42, started_at="2026-04-19T12:00:01Z")
        )
        jobs.append(
            _job(
                job_id=queued_id,
                command_type="corpus_enrichment",
                argv_summary=argv_summary(enrich_argv),
            )
        )

    with_jobs_locked_mutate(corpus, seed)

    # Slot full (cap=1, one running) → drain promotes/spawns nothing.
    asyncio.run(drain_queue_async(app, corpus))
    assert captured == [], "queued enrichment job spawned while the slot was still full"

    # The running job finishes → the slot frees.
    def finish(jobs: list) -> None:
        for j in jobs:
            if j["job_id"] == running_id:
                j["status"] = "succeeded"
                j["ended_at"] = "2026-04-19T12:05:00Z"
                j["exit_code"] = 0

    with_jobs_locked_mutate(corpus, finish)

    # Drain again → the queued enrichment job is promoted and spawns its OWN stored argv.
    asyncio.run(drain_queue_async(app, corpus))
    assert captured, "promotion never spawned the queued enrichment job"
    argv = captured[-1]
    assert "podcast_scraper.cli" in argv and "enrich" in argv, argv
