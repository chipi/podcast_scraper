"""One leader per host among ``serve --workers N`` workers (``server/leader.py``)."""

from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from podcast_scraper.server import leader

pytestmark = [pytest.mark.unit]


def _hold(path: str, ready: "multiprocessing.synchronize.Event") -> None:  # pragma: no cover
    lead = leader.Leadership(Path(path), lambda: None)
    lead.start()
    assert lead.is_leader
    ready.set()
    time.sleep(60)


def test_one_worker_leads_and_the_next_follows(tmp_path: Path) -> None:
    lock = tmp_path / "lead.lock"
    started: list[str] = []
    first = leader.Leadership(lock, lambda: started.append("first"), retry_seconds=0.05)
    second = leader.Leadership(lock, lambda: started.append("second"), retry_seconds=0.05)
    first.start()
    second.start()
    assert first.is_leader and not second.is_leader
    assert started == ["first"]
    first.stop()
    deadline = time.monotonic() + 2
    while not second.is_leader and time.monotonic() < deadline:
        time.sleep(0.02)
    assert second.is_leader
    assert started == ["first", "second"]
    second.stop()


def test_the_lock_is_released_when_the_leaders_process_dies(tmp_path: Path) -> None:
    lock = tmp_path / "lead.lock"
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(target=_hold, args=(str(lock), ready))
    proc.start()
    try:
        assert ready.wait(30)
        follower = leader.Leadership(lock, lambda: None, retry_seconds=0.05)
        follower.start()
        assert not follower.is_leader
        os.kill(proc.pid, 9)
        proc.join(10)
        deadline = time.monotonic() + 3
        while not follower.is_leader and time.monotonic() < deadline:
            time.sleep(0.02)
        assert follower.is_leader
        follower.stop()
    finally:
        if proc.is_alive():
            proc.kill()


def test_a_leader_whose_work_fails_gives_the_lock_back(tmp_path: Path) -> None:
    lock = tmp_path / "lead.lock"
    calls: list[int] = []

    def flaky() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("scheduler did not start")

    lead = leader.Leadership(lock, flaky, retry_seconds=0.05)
    lead.start()  # the first start raises inside: logged, not propagated
    deadline = time.monotonic() + 2
    while not lead.is_leader and time.monotonic() < deadline:
        time.sleep(0.02)
    assert lead.is_leader and len(calls) == 2
    lead.stop()


def test_worker_count_defaults_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(leader.ENV_WORKERS, raising=False)
    assert leader.worker_count() == 1
    monkeypatch.setenv(leader.ENV_WORKERS, "4")
    assert leader.worker_count() == 4
    monkeypatch.setenv(leader.ENV_WORKERS, "x")
    assert leader.worker_count() == 1
