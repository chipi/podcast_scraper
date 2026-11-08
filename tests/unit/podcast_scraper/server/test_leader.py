"""One leader per host among ``serve --workers N`` workers (``server/leader.py``)."""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from podcast_scraper.server import leader

pytestmark = [pytest.mark.unit]


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


def _held_by_another(path: Path) -> bool:
    with open(path, "ab") as fd:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        return False


#: The leader that dies: a Python child holding the lock with leader.Leadership, then waiting.
_HOLD = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "from podcast_scraper.server import leader\n"
    "lead = leader.Leadership(Path(sys.argv[1]), lambda: None)\n"
    "lead.start()\n"
    "assert lead.is_leader\n"
    "time.sleep(30)\n"
)


def _without_coverage(env: dict[str, str]) -> dict[str, str]:
    """The child's environment without pytest-cov's subprocess hooks. Under them the child leaves a
    ``.coverage.*`` data file, and CI's ``coverage combine`` then takes that file for the unit job's
    whole coverage (47% instead of 83% on PR #2444's first runs)."""
    return {k: v for k, v in env.items() if not k.startswith(("COV_CORE_", "COVERAGE_"))}


def test_the_lock_is_released_when_the_leaders_process_dies(tmp_path: Path) -> None:
    lock = tmp_path / "lead.lock"
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLD, str(lock)], env=_without_coverage(dict(os.environ))
    )
    try:
        deadline = time.monotonic() + 30
        while not _held_by_another(lock) and time.monotonic() < deadline:
            assert proc.poll() is None, "the leader child exited before taking the lock"
            time.sleep(0.05)
        assert _held_by_another(lock)
        follower = leader.Leadership(lock, lambda: None, retry_seconds=0.05)
        follower.start()
        assert not follower.is_leader
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(10)
        deadline = time.monotonic() + 3
        while not follower.is_leader and time.monotonic() < deadline:
            time.sleep(0.02)
        assert follower.is_leader
        follower.stop()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(10)


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
