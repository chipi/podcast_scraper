"""Cancel must not signal a pid this process did not record (#1653 review, ADR-152).

``cancel_job`` sends SIGTERM to the pid on the registry row. That is correct for a job this
server started, and dangerous for one it inherited: after a restart the API container has a
fresh PID namespace that recycles low pid numbers, so the number on an old row is very likely
owned by something else entirely. Reconcile getting this wrong produces a wrong status;
**cancel** getting it wrong kills an innocent process.

Docker mode has a handle that does survive the restart — the ``ps.job_id`` container label —
so a prior-boot cancel stops the container instead of signalling the pid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.server import jobs as jobs_mod
from podcast_scraper.server.jobs import cancel_job, current_boot_id, STATUS_RUNNING
from podcast_scraper.server.pipeline_job_registry import with_jobs_locked_mutate

pytestmark = [pytest.mark.unit]


def _seed(corpus: Path, **over: Any) -> None:
    row: dict[str, Any] = {
        "job_id": "cancel-me",
        "command_type": "corpus_enrichment",
        "status": STATUS_RUNNING,
        "created_at": "2026-08-01T00:00:00Z",
        "started_at": "2026-08-01T00:00:01Z",
        "ended_at": None,
        "pid": 4242,
        "boot_id": current_boot_id(),
        "argv_summary": "[]",
        "exit_code": None,
        "log_relpath": ".viewer/jobs/cancel-me.log",
        "error_reason": None,
        "cancel_requested": False,
    }
    row.update(over)
    with_jobs_locked_mutate(corpus, lambda jobs: jobs.append(row))


class TestCancelSignalling:
    def test_a_job_from_this_boot_is_signalled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        killed: list[int] = []
        monkeypatch.setattr(jobs_mod.os, "kill", lambda pid, _sig: killed.append(pid))
        _seed(tmp_path)
        outcome, _rec = cancel_job(tmp_path, "cancel-me")
        assert outcome == "cancelled"
        assert killed == [4242]

    def test_a_job_from_a_previous_boot_is_never_signalled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: that pid probably belongs to something else now."""
        killed: list[int] = []
        monkeypatch.setattr(jobs_mod.os, "kill", lambda pid, _sig: killed.append(pid))
        monkeypatch.setattr(jobs_mod, "docker_exec_mode", lambda: False)
        _seed(tmp_path, boot_id="a-previous-boot")
        outcome, _rec = cancel_job(tmp_path, "cancel-me")
        assert outcome == "cancelled"
        assert killed == []

    def test_a_legacy_row_without_a_boot_id_is_never_signalled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        killed: list[int] = []
        monkeypatch.setattr(jobs_mod.os, "kill", lambda pid, _sig: killed.append(pid))
        monkeypatch.setattr(jobs_mod, "docker_exec_mode", lambda: False)
        _seed(tmp_path)

        def _drop_boot_id(jobs: list[dict[str, Any]]) -> None:
            for j in jobs:
                j.pop("boot_id", None)

        with_jobs_locked_mutate(tmp_path, _drop_boot_id)
        outcome, _rec = cancel_job(tmp_path, "cancel-me")
        assert outcome == "cancelled"
        assert killed == []

    def test_in_docker_mode_a_prior_boot_cancel_stops_the_container(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        killed: list[int] = []
        stopped: list[str] = []
        monkeypatch.setattr(jobs_mod.os, "kill", lambda pid, _sig: killed.append(pid))
        monkeypatch.setattr(jobs_mod, "docker_exec_mode", lambda: True)

        def _spy_docker_stop(job_id: str) -> bool:
            """Record the stop and report success, without touching Docker."""
            stopped.append(job_id)
            return True

        monkeypatch.setattr(
            "podcast_scraper.server.pipeline_docker_factory.docker_stop_job",
            _spy_docker_stop,
        )
        _seed(tmp_path, boot_id="a-previous-boot")
        cancel_job(tmp_path, "cancel-me")
        assert stopped == ["cancel-me"]
        assert killed == []


class TestUnaffectedPaths:
    def test_a_queued_job_still_cancels_without_any_signal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        killed: list[int] = []
        monkeypatch.setattr(jobs_mod.os, "kill", lambda pid, _sig: killed.append(pid))
        _seed(tmp_path, status="queued", pid=None, started_at=None)
        outcome, rec = cancel_job(tmp_path, "cancel-me")
        assert outcome == "cancelled"
        assert rec is not None and rec["error_reason"] == "cancelled_before_start"
        assert killed == []

    def test_a_terminal_job_is_a_noop(self, tmp_path: Path) -> None:
        _seed(tmp_path, status="succeeded")
        outcome, _rec = cancel_job(tmp_path, "cancel-me")
        assert outcome == "noop_terminal"

    def test_an_unknown_job_is_not_found(self, tmp_path: Path) -> None:
        _seed(tmp_path)
        assert cancel_job(tmp_path, "no-such-job") == ("not_found", None)
