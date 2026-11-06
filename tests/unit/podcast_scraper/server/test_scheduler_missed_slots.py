"""``pipeline_run.skipped`` for cron slots that never fired, and the nightly trigger on the fire.

APScheduler's in-memory jobstore forgets every slot that passed while the API was down, so the
scheduler keeps its own fire times beside the job registry and compares them on ``start()``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.server import scheduler as sched
from podcast_scraper.server.scheduler import SCHEDULER_STATE_RELPATH, SchedulerService

pytestmark = [pytest.mark.unit]

# Yearly, so APScheduler itself never fires inside a test; the latest slot is always Jan 1 of
# this year, long before "now".
_YEARLY = "0 0 1 1 *"


def _service(tmp_path: Path, *, cron: str = _YEARLY, enabled: bool = True) -> SchedulerService:
    op = tmp_path / "viewer_operator.yaml"
    op.write_text(
        "scheduled_jobs:\n"
        f"  - name: nightly-sweep\n    cron: '{cron}'\n    enabled: {str(enabled).lower()}\n",
        encoding="utf-8",
    )
    return SchedulerService(tmp_path, op, spawn=MagicMock())


def _write_state(tmp_path: Path, **entry: str) -> None:
    path = tmp_path / SCHEDULER_STATE_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"jobs": {"nightly-sweep": entry}}), encoding="utf-8")


def _read_state(tmp_path: Path) -> dict[str, Any]:
    jobs: dict[str, Any] = json.loads(
        (tmp_path / SCHEDULER_STATE_RELPATH).read_text(encoding="utf-8")
    )["jobs"]
    return jobs


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _skipped(emit: MagicMock) -> list[dict[str, Any]]:
    return [dict(c.kwargs) for c in emit.call_args_list if c.args[0] == "pipeline_run.skipped"]


def _start_stop(svc: SchedulerService) -> None:
    svc.start()
    svc.shutdown()


def test_missed_slot_is_reported_once_across_restarts(tmp_path: Path) -> None:
    two_years_ago = datetime.now(timezone.utc) - timedelta(days=730)
    _write_state(tmp_path, since=_iso(two_years_ago), last_fire=_iso(two_years_ago))
    svc = _service(tmp_path)
    with patch("podcast_scraper.obs.events.emit_event") as emit:
        _start_stop(svc)
        _start_stop(_service(tmp_path))
    events = _skipped(emit)
    assert len(events) == 1
    ev = events[0]
    assert ev["job"] == "nightly-sweep"
    assert ev["kind"] == "pipeline"
    assert ev["reason"] == "scheduler_not_running_at_slot"
    assert ev["slot_at"] == f"{datetime.now(timezone.utc).year}-01-01T00:00:00Z"
    assert ev["detected_at"].endswith("Z")
    assert _read_state(tmp_path)["nightly-sweep"]["last_reported_slot"] == ev["slot_at"]


def test_slot_that_fired_is_not_reported(tmp_path: Path) -> None:
    two_years_ago = datetime.now(timezone.utc) - timedelta(days=730)
    _write_state(tmp_path, since=_iso(two_years_ago), last_fire=_iso(datetime.now(timezone.utc)))
    with patch("podcast_scraper.obs.events.emit_event") as emit:
        _start_stop(_service(tmp_path))
    assert _skipped(emit) == []


def test_first_ever_start_reports_nothing_and_records_a_baseline(tmp_path: Path) -> None:
    with patch("podcast_scraper.obs.events.emit_event") as emit:
        _start_stop(_service(tmp_path))
    assert _skipped(emit) == []
    assert "since" in _read_state(tmp_path)["nightly-sweep"]


def test_fire_records_its_time_and_spawns_as_before(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc._fire("nightly-sweep")
    svc._spawn.assert_called_once()
    fired = datetime.fromisoformat(_read_state(tmp_path)["nightly-sweep"]["last_fire"])
    assert abs(datetime.now(timezone.utc) - fired) < timedelta(minutes=1)


def test_disabled_job_state_is_dropped_so_reenabling_starts_fresh(tmp_path: Path) -> None:
    old = datetime.now(timezone.utc) - timedelta(days=730)
    _write_state(tmp_path, since=_iso(old), last_fire=_iso(old))
    op = tmp_path / "viewer_operator.yaml"
    op.write_text(
        "scheduled_jobs:\n"
        f"  - name: nightly-sweep\n    cron: '{_YEARLY}'\n    enabled: false\n"
        f"  - name: other\n    cron: '{_YEARLY}'\n    enabled: true\n",
        encoding="utf-8",
    )
    svc = SchedulerService(tmp_path, op, spawn=MagicMock())
    with patch("podcast_scraper.obs.events.emit_event") as emit:
        _start_stop(svc)
    assert _skipped(emit) == []
    assert "nightly-sweep" not in _read_state(tmp_path)


def test_misfire_listener_emits_once_per_slot(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc._jobs = sched.load_scheduled_jobs(tmp_path / "viewer_operator.yaml")
    slot = datetime(2026, 11, 5, 3, 0, tzinfo=timezone.utc)
    event = SimpleNamespace(job_id="nightly-sweep", scheduled_run_time=slot)
    with patch("podcast_scraper.obs.events.emit_event") as emit:
        svc._on_job_missed(event)
        svc._on_job_missed(event)
    events = _skipped(emit)
    assert len(events) == 1
    assert events[0]["reason"] == "misfire"
    assert events[0]["slot_at"] == "2026-11-05T03:00:00Z"


def test_listener_is_registered_with_apscheduler(tmp_path: Path) -> None:
    from apscheduler.events import EVENT_JOB_MISSED

    svc = _service(tmp_path)
    svc.start()
    try:
        listeners = svc._scheduler._listeners  # type: ignore[union-attr]
        assert any(mask & EVENT_JOB_MISSED for _cb, mask in listeners)
    finally:
        svc.shutdown()


def test_emit_failure_never_breaks_start(tmp_path: Path) -> None:
    old = datetime.now(timezone.utc) - timedelta(days=730)
    _write_state(tmp_path, since=_iso(old), last_fire=_iso(old))
    svc = _service(tmp_path)
    with patch("podcast_scraper.obs.events.emit_event", side_effect=RuntimeError("sink down")):
        svc.start()
    try:
        assert svc.running
    finally:
        svc.shutdown()


def test_scheduled_fire_enqueues_with_the_nightly_trigger(tmp_path: Path) -> None:
    loop = MagicMock()
    loop.is_running.return_value = True
    app = SimpleNamespace(state=SimpleNamespace(event_loop=loop))
    with (
        patch(
            "podcast_scraper.server.jobs.enqueue_pipeline_job", return_value={"job_id": "j"}
        ) as enqueue,
        patch("asyncio.run_coroutine_threadsafe") as run_threadsafe,
        patch("podcast_scraper.server.jobs.schedule_post_submit", MagicMock()),
    ):
        spawn = sched.make_app_spawn_callback(app)
        spawn("nightly-sweep", tmp_path, tmp_path / "viewer_operator.yaml")
        run_threadsafe.call_args.args[0].close()
    assert enqueue.call_args.kwargs["trigger"] == "nightly"
