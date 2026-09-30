"""A fake prod box for the shell scripts that inspect it.

A stub ``docker`` on PATH serves canned ``ps`` / ``inspect`` output and records every
``compose up``; a temp dir stands in for ``/dev/shm``. Shared by the recovery script's tests and the
read-only check's tests so both are judged against the SAME box.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

OPS = Path(__file__).resolve().parents[4] / "scripts" / "ops"

_STUB = r"""#!/usr/bin/env bash
set -u
S="$STUB_DIR"
case "$1" in
  ps)
    p=""; n=""; running=""
    for a in "$@"; do
      case "$a" in
        label=com.docker.compose.project=*) p="${a##*=}" ;;
        name=*) n="${a#name=}"; n="${n#^}"; n="${n%$}" ;;
        status=running) running=1 ;;
      esac
    done
    if [ -n "$p" ]; then cat "$S/ps.$p" 2>/dev/null || true
    elif [ -n "$n" ] && [ -n "$running" ]; then [ -f "$S/running.$n" ] && echo "$n"
    elif [ -n "$n" ]; then cat "$S/status.$n" 2>/dev/null || true
    fi ;;
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

#: name -> (compose project, service, `docker ps` status, image)
Containers = dict[str, tuple[str, str, str, str]]


def make_box(
    tmp_path: Path,
    containers: Containers,
    *,
    with_secrets: tuple[str, ...] = (),
    running: tuple[str, ...] = (),
) -> Path:
    """Build the stub dir. ``with_secrets`` names containers that have /run/secrets mounts."""
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "docker").write_text(_STUB, encoding="utf-8")
    (stub / "docker").chmod(0o755)
    by_proj: dict[str, list[str]] = {}
    for name, (proj, svc, status, image) in containers.items():
        by_proj.setdefault(proj, []).append(f"{name}|{svc}|{status}")
        (stub / f"image.{name}").write_text(image + "\n", encoding="utf-8")
        (stub / f"status.{name}").write_text(status + "\n", encoding="utf-8")
    for proj, rows in by_proj.items():
        (stub / f"ps.{proj}").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for name in with_secrets:
        (stub / f"mounts.{name}").write_text(
            "/app/output /run/secrets/litellm_api_key \n", encoding="utf-8"
        )
    for name in running:
        (stub / f"running.{name}").write_text("", encoding="utf-8")
    return stub


def make_shm(tmp_path: Path, dirs: dict[str, int]) -> Path:
    """``dirs``: e.g. {"player-secrets": 4} — a dir with N non-empty files (0 = empty dir)."""
    shm = tmp_path / "shm"
    shm.mkdir(exist_ok=True)
    for d, n in dirs.items():
        (shm / d).mkdir()
        for i in range(n):
            (shm / d / f"secret_{i}").write_text("x", encoding="utf-8")
    return shm


def run_script(
    script: str, tmp_path: Path, stub: Path, shm: Path, extra_env: dict[str, str] | None = None
) -> tuple[int, str, str, list[str]]:
    """Run ``scripts/ops/<script>`` against the fake box: (rc, stdout, stderr, compose calls)."""
    env = {
        **os.environ,
        "PATH": f"{stub}:{os.environ['PATH']}",
        "STUB_DIR": str(stub),
        "REPO_DIR": str(tmp_path),
        "SHM_DIR": str(shm),
        **(extra_env or {}),
    }
    proc = subprocess.run(
        ["bash", str(OPS / script)], env=env, capture_output=True, text=True, check=False
    )
    calls_file = stub / "calls"
    calls = calls_file.read_text(encoding="utf-8").splitlines() if calls_file.exists() else []
    return proc.returncode, proc.stdout, proc.stderr, calls
