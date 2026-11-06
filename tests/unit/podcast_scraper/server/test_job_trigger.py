"""``--trigger`` for the pipeline's ``pipeline_run.*`` events: argv, job row, both spawn paths."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from podcast_scraper.server import jobs, pipeline_docker_factory as pdf
from podcast_scraper.server.jobs import (
    argv_from_record,
    build_pipeline_argv,
    enqueue_pipeline_job,
    spawn_pipeline_subprocess,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "viewer_operator.yaml").write_text("max_episodes: 1\n", encoding="utf-8")
    return tmp_path


def _flag(argv: list[str]) -> str | None:
    return argv[argv.index("--trigger") + 1] if "--trigger" in argv else None


def test_build_argv_carries_trigger(corpus: Path) -> None:
    argv = build_pipeline_argv(corpus, corpus / "viewer_operator.yaml", trigger="nightly")
    assert _flag(argv) == "nightly"


def test_build_argv_omits_flag_without_trigger_and_drops_unknown_values(corpus: Path) -> None:
    op = corpus / "viewer_operator.yaml"
    assert _flag(build_pipeline_argv(corpus, op)) is None
    assert _flag(build_pipeline_argv(corpus, op, trigger="rm -rf")) is None


def test_manual_reprocess_stage_is_reported_as_reprocess(corpus: Path) -> None:
    op = corpus / "viewer_operator.yaml"
    manual = build_pipeline_argv(corpus, op, trigger="manual", pipeline_stage="rederive_only")
    nightly = build_pipeline_argv(corpus, op, trigger="nightly", pipeline_stage="rederive_only")
    assert _flag(manual) == "reprocess"
    assert _flag(nightly) == "nightly"


def test_enqueue_defaults_to_manual_and_stores_trigger_on_the_row(corpus: Path) -> None:
    rec = enqueue_pipeline_job(corpus, corpus / "viewer_operator.yaml")
    assert rec["trigger"] == "manual"
    assert _flag(argv_from_record(rec) or []) == "manual"


def test_trigger_survives_argv_from_record(corpus: Path) -> None:
    rec = enqueue_pipeline_job(corpus, corpus / "viewer_operator.yaml", trigger="nightly")
    assert rec["trigger"] == "nightly"
    assert _flag(argv_from_record(rec) or []) == "nightly"


def test_subprocess_path_receives_the_trigger(corpus: Path) -> None:
    rec = enqueue_pipeline_job(corpus, corpus / "viewer_operator.yaml", trigger="nightly")
    argv = argv_from_record(rec)
    assert argv is not None
    exec_mock = AsyncMock(return_value=SimpleNamespace(stdout=object(), pid=1))
    app = SimpleNamespace(state=SimpleNamespace())

    async def _go() -> None:
        with (
            patch.object(jobs.asyncio, "create_subprocess_exec", exec_mock),
            patch.object(jobs, "_pump_subprocess_to_log", AsyncMock()),
        ):
            await spawn_pipeline_subprocess(app, corpus, rec["job_id"], argv, corpus / "j.log")

    asyncio.run(_go())
    assert exec_mock.await_args is not None
    assert _flag(list(exec_mock.await_args.args)) == "nightly"


def test_docker_path_receives_the_trigger(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_DOCKER_PROJECT_DIR", str(corpus))
    (corpus / "compose").mkdir()
    (corpus / "compose" / "docker-compose.stack.yml").write_text("services: {}\n", encoding="utf-8")
    op = corpus / "viewer_operator.yaml"
    op.write_text("pipeline_install_extras: ml\n", encoding="utf-8")
    monkeypatch.setattr(pdf.shutil, "which", lambda _name: "/bin/docker")
    rec = enqueue_pipeline_job(corpus, op, trigger="nightly")
    argv = argv_from_record(rec)
    assert argv is not None
    exec_mock = AsyncMock(return_value=SimpleNamespace())

    async def _go() -> None:
        with patch.object(pdf.asyncio, "create_subprocess_exec", exec_mock):
            await pdf._docker_jobs_factory(
                argv, corpus, corpus / ".viewer" / "jobs" / "d.log", operator_yaml=op
            )

    asyncio.run(_go())
    assert exec_mock.await_args is not None
    assert _flag(list(exec_mock.await_args.args)) == "nightly"
