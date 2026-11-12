"""``serve --workers N``: the multi-process path, and what it refuses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.server import cli_handlers, leader

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _restore_serve_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_run_workers`` writes these into ``os.environ`` for its workers; put them back after.

    ``monkeypatch.delenv(name, raising=False)`` alone does not: on a name that is not set it
    records nothing to undo, so the ``PODCAST_SERVE_WORKERS=3`` the code then writes outlived the
    test and sent every later app in that worker down the leader path (the scheduler lifespan
    test failed in the full suite). ``setenv`` first records the absent value, so it is restored.
    """
    for name in (leader.ENV_WORKERS, leader.ENV_LOCK_FILE, "PODCAST_SERVE_NO_STATIC"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


class _Uvicorn:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def run(self, app: Any, **kwargs: Any) -> None:
        self.calls.append((app, kwargs))


def _args(tmp_path: Path, workers: int) -> Any:
    return cli_handlers.parse_serve_argv(
        ["--output-dir", str(tmp_path), "--workers", str(workers), "--port", "8123"]
    )


def test_more_than_one_worker_starts_the_factory_with_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PODCAST_METRICS_ENABLED", raising=False)
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.delenv(leader.ENV_LOCK_FILE, raising=False)
    fake = _Uvicorn()
    rc = cli_handlers._run_workers(fake, _args(tmp_path, 3), "127.0.0.1", logging.getLogger())
    assert rc == 0
    app, kwargs = fake.calls[0]
    assert app == "podcast_scraper.server.app:create_app_for_uvicorn"
    assert kwargs["factory"] is True and kwargs["workers"] == 3
    import os

    assert os.environ[leader.ENV_WORKERS] == "3"
    assert os.environ[leader.ENV_LOCK_FILE].endswith("podcast-serve-leader-8123.lock")


def test_metrics_without_a_multiprocess_dir_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODCAST_METRICS_ENABLED", "1")
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    fake = _Uvicorn()
    rc = cli_handlers._run_workers(fake, _args(tmp_path, 2), "127.0.0.1", logging.getLogger())
    assert rc == 2 and fake.calls == []


def test_the_multiprocess_dir_is_emptied_before_workers_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mdir = tmp_path / "prom"
    mdir.mkdir()
    (mdir / "counter_123.db").write_bytes(b"old")
    monkeypatch.setenv("PODCAST_METRICS_ENABLED", "1")
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(mdir))
    fake = _Uvicorn()
    rc = cli_handlers._run_workers(fake, _args(tmp_path, 2), "127.0.0.1", logging.getLogger())
    assert rc == 0 and list(mdir.glob("*.db")) == []


def test_one_worker_is_the_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PODCAST_SERVE_WORKERS", raising=False)
    args = cli_handlers.parse_serve_argv(["--output-dir", str(tmp_path)])
    assert args.workers == 1
