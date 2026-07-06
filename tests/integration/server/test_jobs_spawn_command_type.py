"""Each job kind spawns its OWN CLI, not always the pipeline (#1069 consistency).

Before the stored-argv fix, the spawn path rebuilt ``build_pipeline_argv`` regardless
of ``command_type`` (and ``promote_queued_if_slot`` overwrote the stored argv), so an
enrichment job silently ran the *pipeline*. These tests capture the argv the subprocess
factory receives and assert the pipeline job spawns the pipeline CLI and the enrichment
job spawns the enrichment CLI — the invariant that makes enrichment a true peer of
ingestion across the trigger surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

from podcast_scraper.server.app import create_app
from podcast_scraper.server.jobs import argv_from_record

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
