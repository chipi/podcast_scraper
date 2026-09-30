"""The read-only prod check: the remediation fleet's only signal (agentic-ai-homelab RFC-0005).

If this says "fine" on a broken box, prod stays down all night. If it says "broken" on a fine box,
the fleet dispatches recoveries that do nothing, until its daily cap latches it. And if it disagrees
with what ``restage_prod_recreate.sh`` would recreate, every recovery is a no-op — hence the parity
test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.unit.scripts.ops.prod_box_stub import Containers, make_box, make_shm, run_script

pytestmark = pytest.mark.unit

_ALL_SECRETS = {"podcast-secrets": 11, "operator-secrets": 3, "player-secrets": 4}

_HEALTHY: Containers = {
    "compose-api-1": ("compose", "api", "Up 2 hours (healthy)", "stack-api:sha-9949699"),
    "compose-viewer-1": ("compose", "viewer", "Up 2 hours (healthy)", "viewer:sha-9949699"),
    "operator-api-1": ("operator", "api", "Up 2 hours (healthy)", "stack-api:sha-9949699"),
    "player-api-1": ("player", "api", "Up 2 hours (healthy)", "stack-api:sha-9949699"),
}

#: prod at 05:50 UTC on 2026-09-30, rebuilt from `docker ps -a` that morning.
_AFTER_THE_REBOOT: Containers = {
    "compose-api-1": ("compose", "api", "Up 2 hours (healthy)", "stack-api:sha-6f5bfd5"),
    "compose-viewer-1": ("compose", "viewer", "Up 2 hours (healthy)", "viewer:sha-6f5bfd5"),
    "operator-api-1": ("operator", "api", "Exited (127) 2 hours ago", "stack-api:sha-61a1450"),
    "operator-viewer-1": ("operator", "viewer", "Restarting (1) 45 seconds ago", "v:sha-61a1450"),
    "player-api-1": ("player", "api", "Exited (127) 2 hours ago", "stack-api:sha-61a1450"),
    "player-obs-1": ("player", "obs", "Up 2 hours (healthy)", "podcast-obs:sha-6f5bfd5"),
}


def _check(
    tmp_path: Path,
    containers: Containers,
    secrets: dict[str, int],
    *,
    with_secrets: tuple[str, ...] = ("compose-api-1",),
    alloy: bool = True,
) -> dict[str, Any]:
    stub = make_box(
        tmp_path, containers, with_secrets=with_secrets, running=("alloy",) if alloy else ()
    )
    proc_stat = tmp_path / "stat"
    proc_stat.write_text("cpu  1 2 3\nbtime 1790740976\n", encoding="utf-8")
    rc, out, err, calls = run_script(
        "prod_recovery_check.sh",
        tmp_path,
        stub,
        make_shm(tmp_path, secrets),
        {"PROC_STAT": str(proc_stat)},
    )
    assert rc == 0, err
    assert calls == [], "the check must never create or recreate anything"
    lines = out.strip().splitlines()
    assert len(lines) == 1, f"one JSON object on stdout, got: {out!r}"
    parsed: dict[str, Any] = json.loads(lines[0])
    return parsed


def test_a_healthy_box_needs_nothing(tmp_path: Path) -> None:
    got = _check(tmp_path, _HEALTHY, _ALL_SECRETS)
    assert got == {
        "schema": "prod_recovery_check/v1",
        "boot_time": 1790740976,
        "secrets_dirs": {"podcast": 11, "operator": 3, "player": 4},
        "down_containers": [],
        "keyless_control_plane": False,
        "alloy_running": True,
        "needs_recovery": False,
    }


def test_the_box_after_the_reboot_needs_recovery(tmp_path: Path) -> None:
    """THE incident: secrets gone, both public apis dead, control plane up with no keys."""
    got = _check(tmp_path, _AFTER_THE_REBOOT, {}, with_secrets=(), alloy=False)
    assert got["needs_recovery"] is True
    assert got["secrets_dirs"] == {"podcast": 0, "operator": 0, "player": 0}
    assert set(got["down_containers"]) == {
        "compose-api-1",
        "operator-api-1",
        "operator-viewer-1",
        "player-api-1",
    }
    assert got["keyless_control_plane"] is True
    assert got["alloy_running"] is False


def test_missing_secrets_alone_need_recovery(tmp_path: Path) -> None:
    """Every container still up on its mounted copies: nothing looks wrong, and the next restart
    of any of them fails. Prod was exactly here from 06:14 on 2026-09-30."""
    got = _check(tmp_path, _HEALTHY, {})
    assert got["down_containers"] == []
    assert got["needs_recovery"] is True


def test_an_empty_secret_dir_counts_as_missing(tmp_path: Path) -> None:
    got = _check(tmp_path, _HEALTHY, {**_ALL_SECRETS, "player-secrets": 0})
    assert got["secrets_dirs"]["player"] == 0
    assert got["needs_recovery"] is True


def test_alloy_down_alone_is_reported_but_not_a_recovery(tmp_path: Path) -> None:
    """Restaging does not start alloy; flagging it would dispatch a recovery that fixes nothing."""
    got = _check(tmp_path, _HEALTHY, _ALL_SECRETS, alloy=False)
    assert got["alloy_running"] is False
    assert got["needs_recovery"] is False


def test_the_check_reports_exactly_what_the_recovery_recreates(tmp_path: Path) -> None:
    """PARITY: detection and action come from one rule set (prod_health_lib.sh)."""
    check = tmp_path / "check"
    check.mkdir()
    got = _check(check, _AFTER_THE_REBOOT, {}, with_secrets=())

    act = tmp_path / "act"
    act.mkdir()
    stub = make_box(act, _AFTER_THE_REBOOT)
    rc, out, err, calls = run_script(
        "restage_prod_recreate.sh",
        act,
        stub,
        make_shm(act, {"podcast-secrets": 1, "operator-secrets": 1, "player-secrets": 1}),
        {"SELECTED": "podcast operator player"},
    )
    assert rc == 0, out + err
    # "  player: recreating api (player-api-1) at PODCAST_IMAGE_TAG=…"
    recreated = {
        line.split(" (", 1)[1].split(")", 1)[0] for line in out.splitlines() if "recreating" in line
    }
    assert recreated == set(got["down_containers"]), (recreated, got["down_containers"])
    assert len(calls) == len(recreated)
