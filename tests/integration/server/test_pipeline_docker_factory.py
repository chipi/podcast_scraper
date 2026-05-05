"""Unit tests for Docker pipeline job factory helpers (#660 Phase 2)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from podcast_scraper.server import pipeline_docker_factory as pdf

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm]/[server]
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


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


def test_compose_files_rejects_absolute_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """#666 review #4: an env-controlled absolute path could point at any host
    file (``/etc/hosts``, ``/etc/passwd``, …). Guard rejects such values."""
    monkeypatch.setenv("PODCAST_DOCKER_COMPOSE_FILES", "/etc/hosts")
    with pytest.raises(RuntimeError, match="absolute"):
        pdf._compose_files()


def test_compose_files_rejects_parent_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    """#666 review #4: ``../…`` entries could escape the project directory."""
    monkeypatch.setenv(
        "PODCAST_DOCKER_COMPOSE_FILES",
        "compose/docker-compose.stack.yml,../outside/docker-compose.yml",
    )
    with pytest.raises(RuntimeError, match=r"\.\."):
        pdf._compose_files()


def test_resolve_compose_path_blocks_symlink_escape(tmp_path) -> None:
    """#666 review #4: defense-in-depth against a symlink that points outside
    the project root."""
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (outside / "evil.yml").write_text("services: {}\n")
    (project / "evil.yml").symlink_to(outside / "evil.yml")
    with pytest.raises(RuntimeError, match="escapes"):
        pdf._resolve_compose_path(project.resolve(), "evil.yml")


def test_resolve_compose_path_happy_path(tmp_path) -> None:
    project = tmp_path / "project"
    (project / "compose").mkdir(parents=True)
    (project / "compose" / "a.yml").write_text("x: 1\n")
    resolved = pdf._resolve_compose_path(project.resolve(), "compose/a.yml")
    assert resolved == (project / "compose" / "a.yml").resolve()


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


class TestValidateOperatorPipelineExtras:
    """#666 review #13: symmetric-value / mode-specific-presence validator."""

    def test_docker_mode_requires_field(self, tmp_path: Path) -> None:
        p = tmp_path / "op.yaml"
        p.write_text("max_episodes: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Docker pipeline jobs require"):
            pdf.validate_operator_pipeline_extras(p, "docker")

    def test_docker_mode_accepts_ml(self, tmp_path: Path) -> None:
        p = tmp_path / "op.yaml"
        p.write_text("pipeline_install_extras: ml\n", encoding="utf-8")
        assert pdf.validate_operator_pipeline_extras(p, "docker") == "ml"

    def test_docker_mode_accepts_llm(self, tmp_path: Path) -> None:
        p = tmp_path / "op.yaml"
        p.write_text('pipeline_install_extras: "llm"\n', encoding="utf-8")
        assert pdf.validate_operator_pipeline_extras(p, "docker") == "llm"

    def test_subprocess_mode_allows_missing(self, tmp_path: Path) -> None:
        p = tmp_path / "op.yaml"
        p.write_text("max_episodes: 1\n", encoding="utf-8")
        # Native `make serve-api` YAMLs don't declare extras; empty pipe_mode
        # should not reject.
        assert pdf.validate_operator_pipeline_extras(p, "") is None

    def test_subprocess_mode_accepts_valid_value(self, tmp_path: Path) -> None:
        p = tmp_path / "op.yaml"
        p.write_text("pipeline_install_extras: ml\n", encoding="utf-8")
        assert pdf.validate_operator_pipeline_extras(p, "") == "ml"

    def test_both_modes_reject_invalid_value(self, tmp_path: Path) -> None:
        """Symmetry: a bad value is rejected in every mode, not just Docker."""
        p = tmp_path / "op.yaml"
        p.write_text("pipeline_install_extras: cuda\n", encoding="utf-8")
        with pytest.raises(ValueError, match="'ml' or 'llm'"):
            pdf.validate_operator_pipeline_extras(p, "docker")
        with pytest.raises(ValueError, match="'ml' or 'llm'"):
            pdf.validate_operator_pipeline_extras(p, "")

    def test_subprocess_mode_tolerates_missing_file(self, tmp_path: Path) -> None:
        """A fresh corpus without ``viewer_operator.yaml`` must not 500
        ``POST /api/jobs`` in subprocess mode (native ``make serve-api``).
        Regression for main breakage after #675 / #666 follow-ups."""
        missing = tmp_path / "viewer_operator.yaml"
        assert not missing.exists()
        assert pdf.validate_operator_pipeline_extras(missing, "") is None

    def test_docker_mode_still_rejects_missing_file(self, tmp_path: Path) -> None:
        """Docker mode treats a missing file the same as a present file with
        no ``pipeline_install_extras``: explicit error so the operator picks
        ``ml`` or ``llm``."""
        missing = tmp_path / "viewer_operator.yaml"
        with pytest.raises(ValueError, match="Docker pipeline jobs require"):
            pdf.validate_operator_pipeline_extras(missing, "docker")


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
    # Compose v2 resolves bind-mount sources relative to project-directory.
    # Passing ``--project-directory <repo-root>`` would re-anchor the
    # ``../<...>`` paths in our compose files one level above the repo —
    # silent host-side mkdir, then the pipeline entrypoint fails its
    # ``[ ! -f /app/config.yaml ]`` check. Letting Compose default the
    # project directory to ``dirname(first -f)`` keeps the existing
    # relative paths working.
    assert "--project-directory" not in flat
    cli_i = flat.index("podcast_scraper.cli")
    assert flat[cli_i + 1 : cli_i + 3] == ["run", "--help"]
    assert kwargs["cwd"] == str(tmp_path)
    assert log_abs.exists()
    fp = getattr(proc, "_ps_log_fp", None)
    assert fp is not None
    fp.close()
    # By default (no .env at project root), --env-file should NOT be passed
    # — keeps codespace + stack-test behaviour unchanged.
    assert "--env-file" not in flat


def test_docker_jobs_factory_passes_env_file_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``<project>/.env`` exists, ``--env-file`` is added so the nested
    compose can resolve YAML refs like ``${PODCAST_CORPUS_HOST_PATH:?...}``.

    Real-world cause: VPS deploys stage ``/srv/podcast-scraper/.env`` per
    PROD_RUNBOOK and rely on it for the prod overlay's volume interpolation.
    Without the flag, compose's auto-load looks at ``<project>/compose/.env``
    (project dir = ``dirname(first -f)``), which doesn't exist, and the
    spawn fails with ``required variable PODCAST_CORPUS_HOST_PATH is missing``.
    """
    monkeypatch.setenv("PODCAST_DOCKER_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv(
        "PODCAST_DOCKER_COMPOSE_FILES",
        "compose/docker-compose.stack.yml",
    )
    (tmp_path / "compose").mkdir()
    (tmp_path / "compose" / "docker-compose.stack.yml").write_text(
        "services: {}\n", encoding="utf-8"
    )
    # The flag-trigger: a real .env file at the project root.
    (tmp_path / ".env").write_text("PODCAST_CORPUS_HOST_PATH=/some/path\n", encoding="utf-8")
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("pipeline_install_extras: llm\n", encoding="utf-8")
    log_abs = tmp_path / ".viewer" / "jobs" / "spawn.log"
    monkeypatch.setattr(pdf.shutil, "which", lambda _name: "/bin/docker")

    exec_mock = AsyncMock(return_value=SimpleNamespace())

    async def _go() -> None:
        with patch.object(pdf.asyncio, "create_subprocess_exec", exec_mock):
            await pdf._docker_jobs_factory(
                ["python", "-m", "podcast_scraper.cli", "run"],
                tmp_path,
                log_abs,
                operator_yaml=op,
            )

    asyncio.run(_go())
    flat = list(exec_mock.call_args.args)
    assert "--env-file" in flat
    env_file_idx = flat.index("--env-file")
    assert flat[env_file_idx + 1] == str(tmp_path / ".env")
    # Sanity: --env-file precedes the -f flags so compose parses YAML with
    # the right env-var resolution context.
    first_f_idx = flat.index("-f")
    assert env_file_idx < first_f_idx
    fp = getattr(exec_mock.return_value, "_ps_log_fp", None)
    if fp is not None:
        fp.close()


class TestEnvDefaultPipelineInstallExtras:
    """Tests for the ``PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS`` fallback."""

    def test_returns_llm_when_env_is_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS", "llm")
        assert pdf._env_default_pipeline_install_extras() == "llm"

    def test_returns_ml_when_env_is_ml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS", "ml")
        assert pdf._env_default_pipeline_install_extras() == "ml"

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS", raising=False)
        assert pdf._env_default_pipeline_install_extras() is None

    def test_returns_none_for_invalid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Anything other than ml/llm falls through to the strict YAML-required
        # error path so operators get a clear failure rather than silently
        # picking a wrong service.
        monkeypatch.setenv("PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS", "cuda")
        assert pdf._env_default_pipeline_install_extras() is None

    def test_strips_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS", "  llm  ")
        assert pdf._env_default_pipeline_install_extras() == "llm"


class TestExtrasEnvFallback:
    """Tests for the env-var fallback in the two extras validators."""

    def test_assert_falls_back_to_env_when_yaml_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS", "llm")
        p = tmp_path / "operator.yaml"
        p.write_text("max_episodes: 1\n", encoding="utf-8")  # no extras field
        assert pdf.assert_operator_pipeline_extras(p) == "llm"

    def test_assert_still_raises_when_no_env_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS", raising=False)
        p = tmp_path / "operator.yaml"
        p.write_text("max_episodes: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="pipeline_install_extras"):
            pdf.assert_operator_pipeline_extras(p)

    def test_validate_falls_back_to_env_in_docker_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS", "llm")
        p = tmp_path / "operator.yaml"
        p.write_text("max_episodes: 1\n", encoding="utf-8")
        assert pdf.validate_operator_pipeline_extras(p, "docker") == "llm"

    def test_validate_still_raises_when_no_env_default_in_docker_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS", raising=False)
        p = tmp_path / "operator.yaml"
        p.write_text("max_episodes: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="pipeline_install_extras"):
            pdf.validate_operator_pipeline_extras(p, "docker")

    def test_yaml_field_takes_precedence_over_env_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Operator's explicit choice in YAML wins over the host-wide
        # default — env is only a fallback for first-run UX.
        monkeypatch.setenv("PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS", "ml")
        p = tmp_path / "operator.yaml"
        p.write_text("pipeline_install_extras: llm\n", encoding="utf-8")
        assert pdf.assert_operator_pipeline_extras(p) == "llm"
        assert pdf.validate_operator_pipeline_extras(p, "docker") == "llm"
