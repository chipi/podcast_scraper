"""Integration coverage for ``make_app_spawn_callback`` (#708).

Exercises the production spawn path that ``test_scheduler_lifecycle.py``
substitutes with a fake. Mounts the real callback against a FastAPI
TestClient with a fake ``jobs_subprocess_factory`` (same pattern as
``test_viewer_jobs_api.py``) and verifies the scheduled fire lands as
a row in ``GET /api/jobs``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("apscheduler")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.server.scheduler import make_app_spawn_callback

pytestmark = [pytest.mark.integration]


class _FakeProcImmediate:
    pid = 91003

    async def wait(self) -> int:
        return 0


@pytest.fixture()
def fake_factory() -> object:
    async def _factory(argv: list[str], corpus_root: Path, log_abs: Path):  # noqa: ARG001
        log_abs.parent.mkdir(parents=True, exist_ok=True)
        log_abs.write_bytes(b"scheduled-fire-fake-log\n")
        return _FakeProcImmediate()

    return _factory


def test_app_spawn_callback_enqueues_via_real_path(tmp_path: Path, fake_factory: object) -> None:
    """Calling the real spawn callback enqueues a job and TestClient sees it."""
    (tmp_path / "viewer_operator.yaml").write_text("max_episodes: 1\n", encoding="utf-8")
    app = create_app(tmp_path, static_dir=False, enable_jobs_api=True)
    app.state.jobs_subprocess_factory = fake_factory

    with TestClient(app) as client:
        # ``with TestClient(app)`` runs the lifespan, which sets app.state.event_loop.
        spawn = make_app_spawn_callback(app)
        operator_yaml = tmp_path / "viewer_operator.yaml"

        # Fire from a worker thread so we exercise the
        # ``run_coroutine_threadsafe`` hand-off path. APScheduler runs
        # on a daemon thread; this mirrors that.
        def _fire() -> None:
            spawn("test-schedule", tmp_path, operator_yaml)

        t = threading.Thread(target=_fire)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "spawn callback hung"

        # Give the event loop a beat to drain the scheduled coroutine.
        time.sleep(0.2)

        rows = client.get("/api/jobs", params={"path": str(tmp_path)}).json()["jobs"]
        assert len(rows) >= 1
        # Job lands in the same registry shape as POST /api/jobs.
        assert any(j.get("status") in ("queued", "running", "succeeded") for j in rows)


def test_app_spawn_callback_raises_when_event_loop_missing(tmp_path: Path) -> None:
    """When ``app.state.event_loop`` isn't set yet (lifespan hasn't run), spawn raises."""
    (tmp_path / "viewer_operator.yaml").write_text("max_episodes: 1\n", encoding="utf-8")
    # Build a bare app without entering the lifespan context.
    app = create_app(tmp_path, static_dir=False, enable_jobs_api=True)
    # event_loop is set by lifespan; bypass it deliberately.
    app.state.jobs_subprocess_factory = MagicMock()

    spawn = make_app_spawn_callback(app)
    with pytest.raises(RuntimeError, match="event loop unavailable"):
        spawn("name", tmp_path, tmp_path / "viewer_operator.yaml")


def test_app_spawn_callback_raises_when_event_loop_not_running(tmp_path: Path) -> None:
    """Closed loops are rejected (running=False)."""
    (tmp_path / "viewer_operator.yaml").write_text("max_episodes: 1\n", encoding="utf-8")
    app = create_app(tmp_path, static_dir=False, enable_jobs_api=True)
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()
    app.state.event_loop = closed_loop

    spawn = make_app_spawn_callback(app)
    with pytest.raises(RuntimeError, match="event loop unavailable"):
        spawn("name", tmp_path, tmp_path / "viewer_operator.yaml")


def test_lifespan_swallows_scheduler_startup_exception(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When ``scheduler.start()`` raises, the lifespan logs and continues (app.py:114-115)."""
    (tmp_path / "viewer_operator.yaml").write_text("max_episodes: 1\n", encoding="utf-8")
    app = create_app(tmp_path, static_dir=False, enable_jobs_api=True)
    broken = MagicMock()
    broken.start.side_effect = RuntimeError("apscheduler init failed")
    broken.shutdown = MagicMock()  # used by lifespan teardown
    app.state.scheduler = broken

    with caplog.at_level("WARNING"):
        with TestClient(app) as client:
            # App still serves requests despite scheduler startup failure.
            r = client.get("/api/health")
            assert r.status_code == 200

    assert broken.start.called
    assert any("scheduler startup failed" in rec.message for rec in caplog.records)
    # Lifespan teardown still calls shutdown().
    assert broken.shutdown.called
