"""Unit tests for Docker pipeline job factory helpers (#660 Phase 2)."""

from __future__ import annotations

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
