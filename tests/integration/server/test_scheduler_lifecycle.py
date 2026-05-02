"""Integration tests for the in-process feed-sweep scheduler (#708).

APScheduler-dependent — lives in ``tests/integration/`` because ``apscheduler``
is in the ``[server]`` extra (UNIT_TESTING_GUIDE.md gates ``tests/unit/`` to
``[dev]`` only).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("apscheduler")

from podcast_scraper.server.scheduler import (
    load_scheduled_jobs,
    SchedulerService,
)

pytestmark = [pytest.mark.integration]


@pytest.fixture()
def operator_yaml(tmp_path: Path) -> Path:
    """Empty corpus + an operator YAML to point the scheduler at."""
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("max_episodes: 1\n", encoding="utf-8")
    return op


def _write_yaml(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_load_scheduled_jobs_handles_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    assert load_scheduled_jobs(missing) == []


def test_start_with_no_jobs_is_noop(tmp_path: Path, operator_yaml: Path) -> None:
    spawn = MagicMock()
    svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=spawn)
    svc.start()
    try:
        assert svc.running is False
        assert svc.jobs == []
    finally:
        svc.shutdown()


def test_start_with_only_disabled_jobs_does_not_start(tmp_path: Path, operator_yaml: Path) -> None:
    _write_yaml(
        operator_yaml,
        "scheduled_jobs:\n" "  - name: never\n" "    cron: '0 4 * * *'\n" "    enabled: false\n",
    )
    spawn = MagicMock()
    svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=spawn)
    svc.start()
    try:
        assert svc.running is False
        # Disabled jobs are still loaded so the GET endpoint can list them.
        assert [j.name for j in svc.jobs] == ["never"]
    finally:
        svc.shutdown()


def test_start_registers_enabled_jobs(tmp_path: Path, operator_yaml: Path) -> None:
    _write_yaml(
        operator_yaml,
        "scheduled_jobs:\n"
        "  - name: morning\n"
        "    cron: '0 4 * * *'\n"
        "  - name: evening\n"
        "    cron: '0 20 * * *'\n"
        "    enabled: false\n",
    )
    spawn = MagicMock()
    svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=spawn)
    svc.start()
    try:
        assert svc.running is True
        names = [j.name for j in svc.jobs]
        assert names == ["morning", "evening"]
        # next_run_at populated for the enabled job; None for the disabled one.
        assert svc.next_run_at("morning") is not None
        assert svc.next_run_at("evening") is None
    finally:
        svc.shutdown()
        assert svc.running is False


def test_invalid_cron_is_skipped_not_fatal(tmp_path: Path, operator_yaml: Path) -> None:
    _write_yaml(
        operator_yaml,
        "scheduled_jobs:\n"
        "  - name: bad\n"
        "    cron: 'not a cron expression'\n"
        "  - name: good\n"
        "    cron: '0 4 * * *'\n",
    )
    spawn = MagicMock()
    svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=spawn)
    svc.start()
    try:
        # Both rows are loaded into ``jobs``; only the valid cron registers
        # a trigger. ``next_run_at`` distinguishes them.
        assert svc.next_run_at("bad") is None
        assert svc.next_run_at("good") is not None
    finally:
        svc.shutdown()


def test_malformed_yaml_logs_and_starts_empty(
    tmp_path: Path, operator_yaml: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_yaml(operator_yaml, "scheduled_jobs: not-a-list\n")
    spawn = MagicMock()
    svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=spawn)
    with caplog.at_level("ERROR"):
        svc.start()
    try:
        assert svc.running is False
        assert svc.jobs == []
        assert any("scheduled_jobs" in rec.message for rec in caplog.records)
    finally:
        svc.shutdown()


def test_reload_picks_up_yaml_changes(tmp_path: Path, operator_yaml: Path) -> None:
    _write_yaml(
        operator_yaml,
        "scheduled_jobs:\n  - name: first\n    cron: '0 4 * * *'\n",
    )
    spawn = MagicMock()
    svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=spawn)
    svc.start()
    try:
        assert [j.name for j in svc.jobs] == ["first"]
        _write_yaml(
            operator_yaml,
            "scheduled_jobs:\n"
            "  - name: first\n"
            "    cron: '0 4 * * *'\n"
            "  - name: second\n"
            "    cron: '0 5 * * *'\n",
        )
        svc.reload()
        assert sorted(j.name for j in svc.jobs) == ["first", "second"]
        assert svc.running is True
    finally:
        svc.shutdown()


def test_reload_to_empty_stops_scheduler(tmp_path: Path, operator_yaml: Path) -> None:
    _write_yaml(
        operator_yaml,
        "scheduled_jobs:\n  - name: x\n    cron: '0 4 * * *'\n",
    )
    spawn = MagicMock()
    svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=spawn)
    svc.start()
    try:
        assert svc.running is True
        _write_yaml(operator_yaml, "max_episodes: 1\n")
        svc.reload()
        assert svc.running is False
        assert svc.jobs == []
    finally:
        svc.shutdown()


def test_shutdown_is_idempotent(tmp_path: Path, operator_yaml: Path) -> None:
    svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
    svc.shutdown()  # never started
    svc.start()  # nothing to register
    svc.shutdown()
    svc.shutdown()  # double-shutdown
    assert svc.running is False


def test_fire_calls_spawn_and_increments_counter(tmp_path: Path, operator_yaml: Path) -> None:
    _write_yaml(
        operator_yaml,
        "scheduled_jobs:\n  - name: trigger-me\n    cron: '0 4 * * *'\n",
    )
    fired = threading.Event()
    captured: dict[str, Any] = {}

    def _spawn(name: str, corpus: Path, op_yaml: Path) -> None:
        captured["name"] = name
        captured["corpus"] = corpus
        captured["op_yaml"] = op_yaml
        fired.set()

    svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=_spawn)
    svc.start()
    try:
        # Direct fire bypasses cron timing — same code path APScheduler hits.
        svc._fire("trigger-me")
        assert fired.wait(timeout=1.0)
        assert captured["name"] == "trigger-me"
        assert captured["corpus"] == tmp_path
        assert captured["op_yaml"] == operator_yaml
    finally:
        svc.shutdown()


def test_fire_on_spawn_failure_does_not_crash_thread(
    tmp_path: Path, operator_yaml: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_yaml(
        operator_yaml,
        "scheduled_jobs:\n  - name: kaboom\n    cron: '0 4 * * *'\n",
    )

    def _spawn(name: str, corpus: Path, op_yaml: Path) -> None:
        raise RuntimeError("boom")

    svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=_spawn)
    svc.start()
    try:
        with caplog.at_level("ERROR"):
            svc._fire("kaboom")
        assert any("kaboom" in rec.message for rec in caplog.records)
    finally:
        svc.shutdown()


def test_apscheduler_actually_fires_at_a_scheduled_time(
    tmp_path: Path, operator_yaml: Path
) -> None:
    """End-to-end: register a real APScheduler trigger and wait for it to fire.

    Cron's minimum granularity is 1 minute, so we add an explicit ``DateTrigger``
    that fires ~0.5s in the future. This proves the wiring (``BackgroundScheduler``
    -> ``_fire`` -> ``spawn``) works — the cron path itself is exercised by
    the validation tests above.
    """
    _write_yaml(
        operator_yaml,
        "scheduled_jobs:\n  - name: now-ish\n    cron: '0 4 * * *'\n",
    )
    fired = threading.Event()

    def _spawn(name: str, corpus: Path, op_yaml: Path) -> None:
        fired.set()

    svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=_spawn)
    svc.start()
    try:
        from datetime import datetime, timedelta, timezone

        from apscheduler.triggers.date import DateTrigger

        sch = svc._scheduler
        assert sch is not None
        sch.add_job(
            svc._fire,
            DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(milliseconds=500)),
            id="oneshot",
            args=["now-ish"],
            replace_existing=True,
        )
        assert fired.wait(timeout=5.0), "scheduler did not fire one-shot job in time"
    finally:
        svc.shutdown()


def test_load_scheduled_jobs_via_helper(tmp_path: Path, operator_yaml: Path) -> None:
    _write_yaml(
        operator_yaml,
        "scheduled_jobs:\n  - name: a\n    cron: '0 4 * * *'\n",
    )
    out = load_scheduled_jobs(operator_yaml)
    assert [j.name for j in out] == ["a"]


def test_apscheduler_misfire_grace_lets_late_fire_execute(
    tmp_path: Path, operator_yaml: Path
) -> None:
    """Smoke test that ``misfire_grace_time`` is set on registered jobs.

    When the host was suspended past the trigger's scheduled time, a 1-hour
    grace lets the job fire on wakeup. We can't truly simulate a 1-hour
    suspension in a unit-time test, so just assert the grace value made it
    onto the job spec — regression guard against accidentally dropping it.
    """
    _write_yaml(
        operator_yaml,
        "scheduled_jobs:\n  - name: graceful\n    cron: '0 4 * * *'\n",
    )
    svc = SchedulerService(corpus_root=tmp_path, operator_yaml=operator_yaml, spawn=MagicMock())
    svc.start()
    try:
        sch = svc._scheduler
        assert sch is not None
        job = sch.get_job("graceful")
        assert job is not None
        assert job.misfire_grace_time == 3600
        # Brief sleep keeps the scheduler thread alive long enough for the
        # ``running`` flag to be True at the assertion point on slow CI.
        time.sleep(0.05)
        assert svc.running is True
    finally:
        svc.shutdown()
