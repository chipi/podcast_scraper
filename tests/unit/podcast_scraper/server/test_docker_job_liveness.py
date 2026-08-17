"""Ground truth for "is this job still running?" in Docker exec mode (#1653 review).

Boot-scoping the pid check stops reconcile acting on misleading evidence, but on its own it
leaves a hole: a prior-boot ``running`` row is then never failed, so a genuinely dead job
holds its concurrency slot until the 24 h wall-clock window — which is the original wedge,
unfixed, for the restart case that causes it most often. A prior-boot row also has no monitor
task any more, so nothing will ever finalize it even if its container exits cleanly; reconcile
is its only route to a terminal state.

So the answer has to be *correct* reconciliation rather than suppressed reconciliation, and
that needs a fact the pid cannot supply: does a container for this job still exist? The
container is labelled with its registry ``job_id`` at spawn, and the API container already has
the docker socket mounted, so the daemon can be asked directly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.server import pipeline_docker_factory as fac

pytestmark = [pytest.mark.unit]


class _Completed:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture
def _docker_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fac.shutil, "which", lambda _name: "/usr/bin/docker")


class TestDockerJobAlive:
    def test_a_listed_container_means_alive(
        self, monkeypatch: pytest.MonkeyPatch, _docker_present: None
    ) -> None:
        monkeypatch.setattr(fac.subprocess, "run", lambda *a, **k: _Completed(stdout="abc123\n"))
        assert fac.docker_job_alive("job-1") is True

    def test_no_container_means_dead(
        self, monkeypatch: pytest.MonkeyPatch, _docker_present: None
    ) -> None:
        monkeypatch.setattr(fac.subprocess, "run", lambda *a, **k: _Completed(stdout="\n"))
        assert fac.docker_job_alive("job-1") is False

    def test_it_filters_on_the_job_id_label(
        self, monkeypatch: pytest.MonkeyPatch, _docker_present: None
    ) -> None:
        seen: list[list[str]] = []

        def _run(cmd: list[str], **_kw: Any) -> _Completed:
            seen.append(cmd)
            return _Completed(stdout="")

        monkeypatch.setattr(fac.subprocess, "run", _run)
        fac.docker_job_alive("job-xyz")
        assert seen and f"label={fac.JOB_ID_LABEL}=job-xyz" in seen[0]

    def test_a_missing_docker_cli_is_unknown_not_dead(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reporting "dead" on no evidence would free the slot and admit a second writer."""
        monkeypatch.setattr(fac.shutil, "which", lambda _name: None)
        assert fac.docker_job_alive("job-1") is None

    def test_a_failing_daemon_is_unknown_not_dead(
        self, monkeypatch: pytest.MonkeyPatch, _docker_present: None
    ) -> None:
        monkeypatch.setattr(
            fac.subprocess,
            "run",
            lambda *a, **k: _Completed(returncode=1, stderr="cannot connect to daemon"),
        )
        assert fac.docker_job_alive("job-1") is None

    def test_a_timeout_is_unknown_not_dead(
        self, monkeypatch: pytest.MonkeyPatch, _docker_present: None
    ) -> None:
        def _boom(*_a: Any, **_k: Any) -> _Completed:
            raise subprocess.TimeoutExpired(cmd="docker ps", timeout=10)

        monkeypatch.setattr(fac.subprocess, "run", _boom)
        assert fac.docker_job_alive("job-1") is None

    def test_the_probe_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch, _docker_present: None
    ) -> None:
        """An unbounded probe would hang the sweep, and with it the queue it is unwedging."""
        seen: dict[str, Any] = {}

        def _run(cmd: list[str], **kw: Any) -> _Completed:
            seen.update(kw)
            return _Completed(stdout="")

        monkeypatch.setattr(fac.subprocess, "run", _run)
        fac.docker_job_alive("job-1")
        assert seen.get("timeout")


class TestDockerStopJob:
    """Cancel needs a handle that outlives the API container, same as reconcile does."""

    def test_it_stops_the_listed_containers(
        self, monkeypatch: pytest.MonkeyPatch, _docker_present: None
    ) -> None:
        calls: list[list[str]] = []

        def _run(cmd: list[str], **_kw: Any) -> _Completed:
            calls.append(cmd)
            return _Completed(stdout="c1\nc2\n" if cmd[1] == "ps" else "")

        monkeypatch.setattr(fac.subprocess, "run", _run)
        assert fac.docker_stop_job("job-1") is True
        assert calls[-1][:2] == ["docker", "stop"]
        assert "c1" in calls[-1] and "c2" in calls[-1]

    def test_nothing_to_stop_reports_false(
        self, monkeypatch: pytest.MonkeyPatch, _docker_present: None
    ) -> None:
        monkeypatch.setattr(fac.subprocess, "run", lambda *a, **k: _Completed(stdout=""))
        assert fac.docker_stop_job("job-1") is False

    def test_no_docker_cli_reports_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fac.shutil, "which", lambda _n: None)
        assert fac.docker_stop_job("job-1") is False


class TestTheSpawnedContainerIsLabelled:
    """Without the label at spawn there is nothing to query later."""

    def test_the_run_command_carries_the_job_id_label(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        async def _fake_exec(*cmd: str, **_kw: Any) -> Any:
            captured.append(list(cmd))

            class _P:
                pid = 4321

            return _P()

        monkeypatch.setattr(fac.asyncio, "create_subprocess_exec", _fake_exec)
        monkeypatch.setattr(fac.shutil, "which", lambda _n: "/usr/bin/docker")
        monkeypatch.setattr(fac, "assert_operator_pipeline_extras", lambda _p: "llm")
        monkeypatch.setattr(fac, "_project_dir", lambda: tmp_path)
        monkeypatch.setattr(fac, "_compose_files", lambda: [])

        import asyncio as _asyncio

        _asyncio.run(
            fac._docker_jobs_factory(
                ["python", "-m", "podcast_scraper.cli", "enrich"],
                tmp_path,
                tmp_path / "log.txt",
                operator_yaml=tmp_path / "op.yaml",
                job_id="job-abc",
            )
        )
        assert captured
        cmd = captured[0]
        assert "--label" in cmd
        assert f"{fac.JOB_ID_LABEL}=job-abc" in cmd
        # The label must precede the service name, or compose reads it as a container arg.
        assert cmd.index("--label") < cmd.index("python")

    def test_no_label_is_added_without_a_job_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        async def _fake_exec(*cmd: str, **_kw: Any) -> Any:
            captured.append(list(cmd))

            class _P:
                pid = 4321

            return _P()

        monkeypatch.setattr(fac.asyncio, "create_subprocess_exec", _fake_exec)
        monkeypatch.setattr(fac.shutil, "which", lambda _n: "/usr/bin/docker")
        monkeypatch.setattr(fac, "assert_operator_pipeline_extras", lambda _p: "llm")
        monkeypatch.setattr(fac, "_project_dir", lambda: tmp_path)
        monkeypatch.setattr(fac, "_compose_files", lambda: [])

        import asyncio as _asyncio

        _asyncio.run(
            fac._docker_jobs_factory(
                ["python", "-m", "podcast_scraper.cli"],
                tmp_path,
                tmp_path / "log.txt",
                operator_yaml=tmp_path / "op.yaml",
            )
        )
        assert "--label" not in captured[0]
