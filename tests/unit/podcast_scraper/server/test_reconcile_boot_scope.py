"""Reconcile is only as trustworthy as the evidence it acts on (#1653 review).

``reconcile_jobs_inplace`` used to run only when an operator posted ``/api/jobs/reconcile``.
Since the queue sweeper landed it also runs unattended every 30 s, which raises the bar on
every rule in it: a wrong call is now silent and repeated.

The pid it reasons about is not what it looks like. Production runs
``PODCAST_PIPELINE_EXEC_MODE=docker`` (``compose/docker-compose.prod.yml``), where the
recorded pid belongs to the ``docker compose run`` *client* inside the API container while the
work happens in a container on the host daemon. So after an API restart:

* the client pid is gone but the job is very much alive → failing the row on a dead pid frees
  a slot and lets a second writer into the corpus mid-write;
* the new container's PID namespace recycles low pid numbers → an unrelated process can make a
  dead job look alive, and ``cancel_job`` would SIGTERM whatever now owns that number.

These tests pin the resulting rules: a pid is evidence only for the boot that recorded it,
prior-boot rows are judged by a probe, absence of evidence never frees a slot, and the
wall-clock rule may only evict a live process when a human asked for it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from podcast_scraper.server import jobs as jobs_mod
from podcast_scraper.server.jobs import (
    current_boot_id,
    reconcile_jobs_inplace,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STALE,
)

pytestmark = [pytest.mark.unit]


def _row(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": "job-1",
        "command_type": "corpus_enrichment",
        "status": STATUS_RUNNING,
        "created_at": "2026-08-01T00:00:00Z",
        "started_at": _ago(seconds=30),
        "ended_at": None,
        "pid": 4242,
        "boot_id": current_boot_id(),
        "argv_summary": "[]",
        "exit_code": None,
        "log_relpath": ".viewer/jobs/job-1.log",
        "error_reason": None,
        "cancel_requested": False,
    }
    base.update(over)
    return base


def _ago(*, seconds: int) -> str:
    ts = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


class TestThisBootRows:
    def test_a_dead_pid_from_this_boot_is_still_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The original #1653 behaviour must survive: this is how a wedged slot gets freed."""
        monkeypatch.setattr(jobs_mod, "pid_alive", lambda _pid: False)
        jobs = [_row()]
        details = reconcile_jobs_inplace(jobs, stale_seconds=0)
        assert jobs[0]["status"] == STATUS_FAILED
        assert jobs[0]["error_reason"] == "orphan_reconciled_dead_pid"
        assert details

    def test_a_live_pid_from_this_boot_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(jobs_mod, "pid_alive", lambda _pid: True)
        jobs = [_row()]
        assert reconcile_jobs_inplace(jobs, stale_seconds=0) == []
        assert jobs[0]["status"] == STATUS_RUNNING

    def test_the_promote_to_spawn_window_is_not_an_orphan(self) -> None:
        """Between promotion and ``set_job_pid`` the row is RUNNING with no pid. Failing it
        there would kill jobs that are seconds from starting."""
        jobs = [_row(pid=None)]
        assert reconcile_jobs_inplace(jobs, stale_seconds=0) == []
        assert jobs[0]["status"] == STATUS_RUNNING


class TestPriorBootRows:
    def test_a_dead_looking_pid_from_a_previous_boot_is_not_trusted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The false-failed case: in Docker mode the compose client died with the old API
        container while the job container kept working. Failing the row here would free the
        slot and admit a second concurrent writer to the corpus."""
        monkeypatch.setattr(jobs_mod, "pid_alive", lambda _pid: False)
        jobs = [_row(boot_id="a-previous-boot")]
        details = reconcile_jobs_inplace(jobs, stale_seconds=0, prior_boot_alive=lambda _row: True)
        assert jobs[0]["status"] == STATUS_RUNNING
        assert details == []

    def test_a_live_looking_pid_from_a_previous_boot_is_not_trusted_either(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The false-alive case: pid numbers are recycled in the new PID namespace, so an
        unrelated process must not keep a ghost row holding its slot."""
        monkeypatch.setattr(jobs_mod, "pid_alive", lambda _pid: True)
        jobs = [_row(boot_id="a-previous-boot")]
        reconcile_jobs_inplace(jobs, stale_seconds=0, prior_boot_alive=lambda _row: False)
        assert jobs[0]["status"] == STATUS_FAILED

    def test_a_row_with_no_boot_id_counts_as_prior_boot(self) -> None:
        """Rows written before this change carry no boot id; treating them as "this boot"
        would silently reinstate the untrustworthy pid rule for exactly those rows."""
        jobs = [_row()]
        jobs[0].pop("boot_id")
        reconcile_jobs_inplace(jobs, stale_seconds=0, prior_boot_alive=lambda _row: False)
        assert jobs[0]["status"] == STATUS_FAILED

    def test_the_recorded_reason_names_the_evidence_actually_used(self) -> None:
        jobs = [_row(boot_id="old")]
        reconcile_jobs_inplace(
            jobs,
            stale_seconds=0,
            prior_boot_alive=lambda _row: False,
            prior_boot_reason="orphan_reconciled_no_container",
        )
        assert jobs[0]["error_reason"] == "orphan_reconciled_no_container"

    def test_unknown_liveness_keeps_the_slot(self) -> None:
        """None means "cannot tell" and must not be collapsed into "dead". Guessing wrong in
        that direction puts a second writer into the corpus; guessing wrong the other way
        costs one sweep interval."""
        jobs = [_row(boot_id="old")]
        assert reconcile_jobs_inplace(jobs, stale_seconds=0, prior_boot_alive=lambda _r: None) == []
        assert jobs[0]["status"] == STATUS_RUNNING

    def test_no_probe_at_all_keeps_the_slot(self) -> None:
        jobs = [_row(boot_id="old")]
        assert reconcile_jobs_inplace(jobs, stale_seconds=0) == []
        assert jobs[0]["status"] == STATUS_RUNNING


class TestWallClockStale:
    def test_an_operator_may_still_evict_a_live_long_runner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``POST /api/jobs/reconcile`` is a deliberate act and keeps its old behaviour."""
        monkeypatch.setattr(jobs_mod, "pid_alive", lambda _pid: True)
        jobs = [_row(started_at=_ago(seconds=7200))]
        reconcile_jobs_inplace(jobs, stale_seconds=60, stale_marks_live_processes=True)
        assert jobs[0]["status"] == STATUS_STALE

    def test_the_unattended_sweep_may_not(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Freeing the slot of a process we just proved ALIVE is how an automatic sweep
        manufactures two concurrent corpus writers. The longest observed job was ~2.16 h, but
        that was not a 678-episode reprocess — this must not depend on the window being
        generous enough."""
        monkeypatch.setattr(jobs_mod, "pid_alive", lambda _pid: True)
        jobs = [_row(started_at=_ago(seconds=7200))]
        assert (
            reconcile_jobs_inplace(jobs, stale_seconds=60, stale_marks_live_processes=False) == []
        )
        assert jobs[0]["status"] == STATUS_RUNNING

    def test_a_dead_job_past_the_window_is_still_released(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard is about *live* processes only — it must not turn the stale rule off."""
        monkeypatch.setattr(jobs_mod, "pid_alive", lambda _pid: False)
        jobs = [_row(boot_id="old", started_at=_ago(seconds=7200))]
        reconcile_jobs_inplace(
            jobs,
            stale_seconds=60,
            prior_boot_alive=lambda _row: None,
            stale_marks_live_processes=False,
        )
        assert jobs[0]["status"] == STATUS_STALE
        assert jobs[0]["error_reason"] == "wall_clock_stale"


class TestNonRunningRowsAreUntouched:
    @pytest.mark.parametrize("status", ["queued", "succeeded", "failed", "cancelled"])
    def test_terminal_and_queued_rows_are_ignored(self, status: str) -> None:
        jobs = [_row(status=status, started_at=_ago(seconds=7200))]
        assert reconcile_jobs_inplace(jobs, stale_seconds=60) == []
        assert jobs[0]["status"] == status
