"""Tests for scripts/ops/prod_ops_health_scheduled.sh, the box-side daily run (#1877).

The wrapper replaced the GitHub cron, so it owns the alert: a red or broken health run must raise
a GlitchTip event from inside compose-api-1 and still exit with the health script's code (that is
what `systemctl status prod-ops-health` shows). A green run must not alert. Here the wrapper runs
from a copy next to a stub prod_ops_health.sh, with `docker` stubbed on PATH to record the call.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ops" / "prod_ops_health_scheduled.sh"
UNITS = REPO_ROOT / "infra" / "systemd"

pytestmark = pytest.mark.unit


def _run(
    tmp_path: Path, health_rc: int, docker_rc: int = 0
) -> tuple[subprocess.CompletedProcess[str], Path]:
    ops = tmp_path / "ops"
    ops.mkdir()
    shutil.copy(SCRIPT, ops / SCRIPT.name)
    (ops / "prod_ops_health.sh").write_text(
        f"#!/usr/bin/env bash\necho health ran\nexit {health_rc}\n"
    )
    (ops / "prod_ops_health.sh").chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "docker.call"
    (bin_dir / "docker").write_text(
        f'#!/usr/bin/env bash\n{{ echo "ARGS: $*"; cat; }} > "{record}"\nexit {docker_rc}\n'
    )
    (bin_dir / "docker").chmod(0o755)
    result = subprocess.run(
        ["/usr/bin/env", "bash", str(ops / SCRIPT.name)],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    return result, record


def test_a_green_run_exits_0_and_sends_no_alert(tmp_path: Path) -> None:
    r, record = _run(tmp_path, health_rc=0)
    assert r.returncode == 0, r.stderr
    assert "health ran" in r.stdout
    assert not record.exists()


@pytest.mark.parametrize("rc", [1, 5])
def test_a_red_or_broken_run_alerts_from_the_api_container_and_keeps_its_exit_code(
    tmp_path: Path, rc: int
) -> None:
    r, record = _run(tmp_path, health_rc=rc)
    assert r.returncode == rc
    call = record.read_text()
    assert call.startswith("ARGS: exec -i ")
    assert f"-e RC={rc}" in call and "compose-api-1 python -" in call
    assert 'os.environ["PODCAST_SENTRY_DSN_API"]' in call
    assert "capture_message" in call and 'level="error"' in call
    assert "GlitchTip alert sent" in r.stderr


def test_an_alert_that_cannot_be_sent_is_said_so_and_keeps_the_exit_code(tmp_path: Path) -> None:
    r, _ = _run(tmp_path, health_rc=1, docker_rc=1)
    assert r.returncode == 1
    assert "GlitchTip alert NOT sent" in r.stderr


def test_the_units_run_the_wrapper_daily_as_deploy() -> None:
    service = (UNITS / "prod-ops-health.service").read_text()
    timer = (UNITS / "prod-ops-health.timer").read_text()
    assert "ExecStart=/srv/podcast-scraper/scripts/ops/prod_ops_health_scheduled.sh" in service
    assert "User=deploy" in service and "Type=oneshot" in service
    assert "OnCalendar=*-*-* 06:00:00 UTC" in timer and "Persistent=true" in timer
    assert os.access(SCRIPT, os.X_OK)
