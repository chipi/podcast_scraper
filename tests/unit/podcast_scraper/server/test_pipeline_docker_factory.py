"""Unit tests for Docker pipeline job factory helpers (#660 Phase 2)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from podcast_scraper.server import pipeline_docker_factory as pdf


def test_service_for_extras_llm() -> None:
    assert pdf._service_for_extras("llm") == ("pipeline-llm", "pipeline-llm")


def test_service_for_extras_ml_and_default() -> None:
    assert pdf._service_for_extras("ml") == ("pipeline", "pipeline")
    assert pdf._service_for_extras("anything_else") == ("pipeline", "pipeline")


def test_compose_files_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODCAST_DOCKER_COMPOSE_FILES", raising=False)
    assert pdf._compose_files() == ["compose/docker-compose.stack.yml"]


def test_compose_files_override_single(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PODCAST_DOCKER_COMPOSE_FILES",
        "compose/docker-compose.stack.yml,compose/docker-compose.jobs-docker.yml",
    )
    assert pdf._compose_files() == [
        "compose/docker-compose.stack.yml",
        "compose/docker-compose.jobs-docker.yml",
    ]


def test_compose_files_strips_and_skips_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_DOCKER_COMPOSE_FILES", " a.yaml ,  , b.yaml ")
    assert pdf._compose_files() == ["a.yaml", "b.yaml"]


def test_cli_argv_tail_after_python_m_cli() -> None:
    argv = [
        "/usr/bin/python3",
        "-m",
        "podcast_scraper.cli",
        "pipeline",
        "--help",
    ]
    assert pdf._cli_argv_tail(argv) == ["pipeline", "--help"]


def test_cli_argv_tail_with_interpreter_prefix() -> None:
    argv = [
        "ignored",
        "/venv/bin/python",
        "-m",
        "podcast_scraper.cli",
        "run",
        "x",
    ]
    assert pdf._cli_argv_tail(argv) == ["run", "x"]


def test_cli_argv_tail_no_cli_marker_returns_slice_after_first() -> None:
    argv = ["/bin/python", "script.py", "a"]
    assert pdf._cli_argv_tail(argv) == ["script.py", "a"]


def test_cli_argv_tail_single_element() -> None:
    assert pdf._cli_argv_tail(["only"]) == []


def test_cli_argv_tail_empty() -> None:
    assert pdf._cli_argv_tail([]) == []


def test_project_dir_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODCAST_DOCKER_PROJECT_DIR", raising=False)
    with pytest.raises(RuntimeError, match="PODCAST_DOCKER_PROJECT_DIR"):
        pdf._project_dir()


def test_project_dir_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_DOCKER_PROJECT_DIR", str(tmp_path))
    assert pdf._project_dir() == tmp_path.resolve()


def test_assert_operator_pipeline_extras_accepts_ml(tmp_path: Path) -> None:
    p = tmp_path / "op.yaml"
    p.write_text("pipeline_install_extras: ml\n", encoding="utf-8")
    assert pdf.assert_operator_pipeline_extras(p) == "ml"


def test_assert_operator_pipeline_extras_accepts_llm(tmp_path: Path) -> None:
    p = tmp_path / "op.yaml"
    p.write_text('pipeline_install_extras: "llm"\n', encoding="utf-8")
    assert pdf.assert_operator_pipeline_extras(p) == "llm"


def test_assert_operator_pipeline_extras_rejects_missing_or_other(tmp_path: Path) -> None:
    p = tmp_path / "op.yaml"
    p.write_text("max_episodes: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pipeline_install_extras"):
        pdf.assert_operator_pipeline_extras(p)

    p.write_text("pipeline_install_extras: cuda\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pipeline_install_extras"):
        pdf.assert_operator_pipeline_extras(p)


def test_attach_docker_jobs_factory_registers_callable() -> None:
    app = type("A", (), {})()
    app.state = type("S", (), {})()
    pdf.attach_docker_jobs_factory(app)
    factory = getattr(app.state, "jobs_subprocess_factory", None)
    assert callable(factory)
    assert asyncio.iscoroutinefunction(factory)


def test_attach_docker_jobs_factory_closure_calls_project_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure imports ``viewer_operator_extras_source`` and awaits ``_docker_jobs_factory``."""
    app = type("A", (), {})()
    app.state = type("S", (), {})()
    pdf.attach_docker_jobs_factory(app)
    factory = app.state.jobs_subprocess_factory

    op = tmp_path / "viewer_operator.yaml"
    op.write_text("pipeline_install_extras: ml\n", encoding="utf-8")
    log_abs = tmp_path / "j.log"

    async_mock = AsyncMock(return_value=SimpleNamespace())

    async def _go() -> None:
        with patch.object(pdf, "_docker_jobs_factory", async_mock):
            await factory(["/py", "-m", "podcast_scraper.cli", "x"], tmp_path, log_abs)

    asyncio.run(_go())
    async_mock.assert_awaited_once()
    _args, kwargs = async_mock.call_args
    assert kwargs["operator_yaml"] == op


def test_docker_jobs_factory_requires_docker_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODCAST_DOCKER_PROJECT_DIR", str(tmp_path))
    (tmp_path / "compose").mkdir()
    (tmp_path / "compose" / "docker-compose.stack.yml").write_text(
        "services: {}\n", encoding="utf-8"
    )
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("pipeline_install_extras: ml\n", encoding="utf-8")
    log_abs = tmp_path / ".viewer" / "jobs" / "x.log"

    monkeypatch.setattr(pdf.shutil, "which", lambda _name: None)

    async def _go() -> None:
        with pytest.raises(RuntimeError, match="docker CLI not found"):
            await pdf._docker_jobs_factory(
                ["/py", "-m", "podcast_scraper.cli", "pipeline", "--help"],
                tmp_path,
                log_abs,
                operator_yaml=op,
            )

    asyncio.run(_go())


def test_docker_jobs_factory_builds_compose_argv_and_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODCAST_DOCKER_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv(
        "PODCAST_DOCKER_COMPOSE_FILES",
        "compose/docker-compose.stack.yml,compose/extra.yml",
    )
    (tmp_path / "compose").mkdir()
    (tmp_path / "compose" / "docker-compose.stack.yml").write_text(
        "services: {}\n", encoding="utf-8"
    )
    (tmp_path / "compose" / "extra.yml").write_text("services: {}\n", encoding="utf-8")
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("pipeline_install_extras: llm\n", encoding="utf-8")
    log_abs = tmp_path / ".viewer" / "jobs" / "spawn.log"
    monkeypatch.setattr(pdf.shutil, "which", lambda _name: "/bin/docker")

    mock_proc = SimpleNamespace()
    exec_mock = AsyncMock(return_value=mock_proc)

    async def _go() -> object:
        with patch.object(pdf.asyncio, "create_subprocess_exec", exec_mock):
            return await pdf._docker_jobs_factory(
                ["/venv/bin/python", "-m", "podcast_scraper.cli", "run", "--help"],
                tmp_path,
                log_abs,
                operator_yaml=op,
            )

    proc = asyncio.run(_go())
    assert proc is mock_proc
    args, kwargs = exec_mock.call_args
    flat = list(args)
    assert flat[:2] == ["docker", "compose"]
    assert str(tmp_path / "compose" / "docker-compose.stack.yml") in flat
    assert str(tmp_path / "compose" / "extra.yml") in flat
    assert "--profile" in flat and "pipeline-llm" in flat
    assert "run" in flat and "-T" in flat and "--rm" in flat and "--no-deps" in flat
    cli_i = flat.index("podcast_scraper.cli")
    assert flat[cli_i + 1 : cli_i + 3] == ["run", "--help"]
    assert kwargs["cwd"] == str(tmp_path)
    assert log_abs.exists()
    fp = getattr(proc, "_ps_log_fp", None)
    assert fp is not None
    fp.close()
