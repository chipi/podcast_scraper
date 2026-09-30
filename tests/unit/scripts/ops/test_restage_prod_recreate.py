"""The post-reboot recreate must bring back what is down, on the image it was already on.

Driven against a stub ``docker`` that serves canned ``ps``/``inspect`` output and records every
``compose up``. Each test is one of the three ways the 2026-09-30 recovery went wrong on prod.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "ops" / "restage_prod_recreate.sh"

_STUB = r"""#!/usr/bin/env bash
set -u
S="$STUB_DIR"
case "$1" in
  ps)
    for a in "$@"; do case "$a" in label=com.docker.compose.project=*) p="${a##*=}";; esac; done
    cat "$S/ps.$p" 2>/dev/null || true ;;
  inspect)
    fmt="$3"; name="$4"
    case "$fmt" in
      *Config.Image*) cat "$S/image.$name" ;;
      *Mounts*) cat "$S/mounts.$name" 2>/dev/null || true ;;
    esac ;;
  compose)
    echo "TAG=${PODCAST_IMAGE_TAG:-} $*" >> "$S/calls" ;;
esac
"""


def _box(tmp_path: Path, containers: dict[str, tuple[str, str, str, str]]) -> Path:
    """containers: name -> (project, service, status, image)."""
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "docker").write_text(_STUB, encoding="utf-8")
    (stub / "docker").chmod(0o755)
    by_proj: dict[str, list[str]] = {}
    for name, (proj, svc, status, image) in containers.items():
        by_proj.setdefault(proj, []).append(f"{name}|{svc}|{status}")
        (stub / f"image.{name}").write_text(image + "\n", encoding="utf-8")
    for proj, rows in by_proj.items():
        (stub / f"ps.{proj}").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return stub


def _run(tmp_path: Path, stub: Path, selected: str, shm: list[str]) -> tuple[int, str, list[str]]:
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir(exist_ok=True)
    for d in shm:
        (shm_dir / d).mkdir()
    env = {
        **os.environ,
        "PATH": f"{stub}:{os.environ['PATH']}",
        "STUB_DIR": str(stub),
        "REPO_DIR": str(tmp_path),
        "SHM_DIR": str(shm_dir),
        "SELECTED": selected,
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=False
    )
    calls_file = stub / "calls"
    calls = calls_file.read_text(encoding="utf-8").splitlines() if calls_file.exists() else []
    return proc.returncode, proc.stdout + proc.stderr, calls


def test_each_container_comes_back_on_its_own_tag_not_a_siblings(tmp_path: Path) -> None:
    """THE image move: player-obs was newer than player-api, and api was pulled up to it."""
    stub = _box(
        tmp_path,
        {
            "player-obs-1": ("player", "obs", "Up 2 hours (healthy)", "podcast-obs:sha-6f5bfd5"),
            "player-api-1": ("player", "api", "Exited (127) 2 hours ago", "stack-api:sha-61a1450"),
            "player-learning-app-1": (
                "player",
                "learning-app",
                "Restarting (1) 3 seconds ago",
                "learning-app:sha-61a1450",
            ),
        },
    )
    rc, out, calls = _run(tmp_path, stub, "player", ["player-secrets"])
    assert rc == 0, out
    assert len(calls) == 2, calls
    assert all(c.startswith("TAG=sha-61a1450 ") for c in calls), calls
    assert not any(c.endswith(" obs") for c in calls), "a healthy sibling was recreated"


def _two_down_surfaces(tmp_path: Path) -> Path:
    return _box(
        tmp_path,
        {
            "operator-api-1": ("operator", "api", "Exited (127) 2 hours ago", "api:sha-61a1450"),
            "player-api-1": ("player", "api", "Exited (127) 2 hours ago", "api:sha-61a1450"),
        },
    )


def test_an_unknown_surface_fails_the_run(tmp_path: Path) -> None:
    """THE silent skip: 'operator\\' was dropped as unknown and the run still said success."""
    stub = _two_down_surfaces(tmp_path)
    # Only the mangled name, so nothing else in the run can fail it for an unrelated reason.
    rc, out, calls = _run(tmp_path, stub, "operator\\", ["operator-secrets"])
    assert rc != 0, "an unknown surface must fail the run"
    assert "unknown surface" in out
    assert calls == []


def test_every_selected_surface_runs(tmp_path: Path) -> None:
    stub = _two_down_surfaces(tmp_path)
    rc, out, calls = _run(tmp_path, stub, "operator player", [])
    assert rc == 0, out
    assert sum(" -p operator " in c for c in calls) == 1, calls
    assert sum(" -p player " in c for c in calls) == 1, calls


def test_a_keyless_control_plane_counts_as_down(tmp_path: Path) -> None:
    """THE 'healthy' api with no keys: running, passing health checks, no /run/secrets mounts."""
    stub = _box(
        tmp_path,
        {
            "compose-api-1": ("compose", "api", "Up 2 hours (healthy)", "stack-api:sha-6f5bfd5"),
            "compose-viewer-1": ("compose", "viewer", "Up 2 hours (healthy)", "viewer:sha-6f5bfd5"),
        },
    )
    rc, out, calls = _run(tmp_path, stub, "podcast", ["podcast-secrets"])
    assert rc == 0, out
    assert len(calls) == 1, calls
    assert calls[0].startswith("TAG=sha-6f5bfd5 ") and calls[0].endswith(" api"), calls
    assert "docker-compose.secrets.yml" in calls[0], "recreated without the secrets overlay"


def test_a_control_plane_with_keys_is_left_alone(tmp_path: Path) -> None:
    stub = _box(
        tmp_path,
        {"compose-api-1": ("compose", "api", "Up 2 hours (healthy)", "stack-api:sha-6f5bfd5")},
    )
    (stub / "mounts.compose-api-1").write_text(
        "/app/output /run/secrets/litellm_api_key \n", encoding="utf-8"
    )
    rc, out, calls = _run(tmp_path, stub, "podcast", ["podcast-secrets"])
    assert rc == 0, out
    assert calls == []


def test_the_control_plane_is_not_recreated_without_its_secrets(tmp_path: Path) -> None:
    """Recreating with the overlay and no source dir fails mid-way; say so instead."""
    stub = _box(
        tmp_path,
        {"compose-api-1": ("compose", "api", "Up 2 hours (healthy)", "stack-api:sha-6f5bfd5")},
    )
    rc, out, calls = _run(tmp_path, stub, "podcast", [])
    assert rc != 0
    assert "podcast-secrets is missing" in out
    assert calls == []
