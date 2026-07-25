"""Guard for ``scripts/ops/emit_ops_event.sh`` — the Tier-1 CI/ops event emitter (#1297).

Runs the real script against a **fake ``curl``** on PATH that captures the POST body, and
asserts the canonical ``ops_event/v1`` envelope (ADR-119): stream fields ``{app, env,
event_type}``, ``_time``/``_msg``, and typed field coercion (ints/bools, not strings).
No network — the sink is stubbed.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ops" / "emit_ops_event.sh"

_FAKE_CURL = """#!/usr/bin/env bash
# Capture the --data-binary payload to $CAPTURE_FILE, ignore the rest, succeed.
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--data-binary" ]; then printf '%s' "$2" > "$CAPTURE_FILE"; fi
  shift
done
exit 0
"""


def _run(tmp_path: Path, *args: str) -> dict:
    capture = tmp_path / "payload.json"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text(_FAKE_CURL, encoding="utf-8")
    curl.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["CAPTURE_FILE"] = str(capture)
    env["VICTORIALOGS_URL"] = "http://homelab:9428"

    res = subprocess.run(
        ["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=30
    )
    assert res.returncode == 0, f"emit failed: {res.stderr}"
    data: dict = json.loads(capture.read_text(encoding="utf-8"))
    return data


@pytest.mark.unit
def test_emit_builds_canonical_envelope_with_typed_fields(tmp_path: Path) -> None:
    obj = _run(
        tmp_path,
        "--event-type",
        "deploy",
        "--env",
        "test",
        "--msg",
        "prod deploy success",
        "--field",
        "status=success",
        "--field",
        "duration_ms=42000",
        "--field",
        "dry_run=false",
        "--field",
        "sha=abc1234",
    )
    # Stream fields + envelope.
    assert obj["schema"] == "ops_event/v1"
    assert obj["event_type"] == "deploy"
    assert obj["app"] == "podcast_scraper"
    assert obj["env"] == "test"
    assert obj["_msg"] == "prod deploy success"
    assert obj["_time"].endswith("Z")
    # Typed coercion: number stays a number, bool stays a bool, string stays a string.
    assert obj["duration_ms"] == 42000 and isinstance(obj["duration_ms"], int)
    assert obj["dry_run"] is False
    assert obj["sha"] == "abc1234"
    assert obj["status"] == "success"


@pytest.mark.unit
def test_emit_defaults_app_and_env_and_synthesizes_msg(tmp_path: Path) -> None:
    obj = _run(tmp_path, "--event-type", "backup", "--field", "status=failure")
    assert obj["app"] == "podcast_scraper"
    assert obj["env"] == "prod"  # default
    # No --msg → synthesized from event_type + status.
    assert obj["_msg"] == "backup failure"


@pytest.mark.unit
def test_emit_requires_event_type_and_url(tmp_path: Path) -> None:
    # Missing --event-type → non-zero exit.
    env = dict(os.environ, VICTORIALOGS_URL="http://homelab:9428")
    res = subprocess.run(
        ["bash", str(SCRIPT), "--env", "test"], env=env, capture_output=True, text=True, timeout=30
    )
    assert res.returncode != 0
    # Missing VICTORIALOGS_URL → non-zero exit.
    env2 = {k: v for k, v in os.environ.items() if k != "VICTORIALOGS_URL"}
    res2 = subprocess.run(
        ["bash", str(SCRIPT), "--event-type", "deploy"],
        env=env2,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res2.returncode != 0
