"""Integration tests for FastAPI app factory.

Requires ``fastapi`` (``pip install -e '.[server]'``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

from podcast_scraper.server.app import create_app, create_app_for_uvicorn

pytestmark = [pytest.mark.integration]


def test_create_app_output_dir_none() -> None:
    app = create_app(None, static_dir=False)
    assert app.state.output_dir is None


def test_create_app_resolves_output_dir(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    assert app.state.output_dir == tmp_path.resolve()


def test_create_app_static_false_skips_mount() -> None:
    app = create_app(None, static_dir=False)
    names = [getattr(r, "name", None) for r in app.routes]
    assert "viewer" not in names


def test_create_app_enable_platform_noop() -> None:
    app = create_app(None, static_dir=False, enable_platform=True)
    assert app is not None


def test_create_app_for_uvicorn_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODCAST_SERVE_OUTPUT_DIR", raising=False)
    with pytest.raises(RuntimeError, match="PODCAST_SERVE_OUTPUT_DIR"):
        create_app_for_uvicorn()


def test_create_app_for_uvicorn_uses_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_SERVE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("PODCAST_SERVE_ENABLE_FEEDS_API", raising=False)
    monkeypatch.delenv("PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API", raising=False)
    app = create_app_for_uvicorn()
    assert app.state.output_dir == tmp_path.resolve()


def test_create_app_enable_feeds_without_output_dir_raises() -> None:
    with pytest.raises(ValueError, match="enable_feeds_api"):
        create_app(None, static_dir=False, enable_feeds_api=True)


def test_create_app_enable_operator_config_without_output_dir_raises() -> None:
    with pytest.raises(ValueError, match="enable_operator_config_api"):
        create_app(None, static_dir=False, enable_operator_config_api=True)


def test_create_app_enable_jobs_without_output_dir_raises() -> None:
    with pytest.raises(ValueError, match="enable_jobs_api"):
        create_app(None, static_dir=False, enable_jobs_api=True)


def test_create_app_wires_docker_jobs_factory_when_pipeline_exec_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODCAST_PIPELINE_EXEC_MODE", "docker")
    app = create_app(tmp_path, static_dir=False, enable_jobs_api=True)
    factory = getattr(app.state, "jobs_subprocess_factory", None)
    assert callable(factory)
    assert asyncio.iscoroutinefunction(factory)


def test_create_app_jobs_pins_operator_config_file_argument(tmp_path: Path) -> None:
    op = tmp_path / "shared.yaml"
    op.write_text("k: v\n", encoding="utf-8")
    app = create_app(
        tmp_path,
        static_dir=False,
        enable_jobs_api=True,
        operator_config_file=str(op),
    )
    assert app.state.operator_config_fixed_path == op.resolve()


def test_create_app_jobs_reads_operator_config_from_serve_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    op = tmp_path / "from_env.yaml"
    op.write_text("k: v\n", encoding="utf-8")
    monkeypatch.setenv("PODCAST_SERVE_CONFIG_FILE", str(op))
    app = create_app(tmp_path, static_dir=False, enable_jobs_api=True)
    assert app.state.operator_config_fixed_path == op.resolve()


def test_create_app_for_uvicorn_reads_feed_flags_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODCAST_SERVE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("PODCAST_SERVE_ENABLE_FEEDS_API", "1")
    monkeypatch.setenv("PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API", "true")
    app = create_app_for_uvicorn()
    assert app.state.feeds_api_enabled is True
    assert app.state.operator_config_api_enabled is True


def test_create_app_custom_static_dir_not_dir(tmp_path: Path) -> None:
    ghost = tmp_path / "missing_dist"
    app = create_app(tmp_path, static_dir=ghost)
    names = [getattr(r, "name", None) for r in app.routes]
    assert "viewer" not in names


def test_create_app_true_static_uses_default_when_present(tmp_path: Path) -> None:
    with patch("podcast_scraper.server.app._default_static_dir", return_value=tmp_path):
        app = create_app(None, static_dir=True)
    names = [getattr(r, "name", None) for r in app.routes]
    assert "viewer" in names


def test_api_corpus_text_file_not_shadowed_by_static_mount(tmp_path: Path) -> None:
    """Regression: ``mount('/', StaticFiles)`` must not swallow ``/api/corpus/text-file``."""
    with patch("podcast_scraper.server.app._default_static_dir", return_value=tmp_path):
        app = create_app(tmp_path, static_dir=True)
    (tmp_path / "t.txt").write_text("hi", encoding="utf-8")
    client = TestClient(app)
    r = client.get("/api/corpus/text-file", params={"relpath": "t.txt"})
    assert r.status_code == 200
    assert r.text == "hi"


def test_api_health_and_search_no_corpus(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    h = client.get("/api/health")
    assert h.status_code == 200
    assert h.json().get("status") == "ok"
    s = client.get("/api/search?q=hello")
    assert s.status_code == 200
    body = s.json()
    assert body.get("query") == "hello"


def test_api_artifacts_lists_empty_corpus(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/artifacts", params={"path": str(tmp_path)})
    assert r.status_code == 200
    data = r.json()
    assert data.get("artifacts") == []
