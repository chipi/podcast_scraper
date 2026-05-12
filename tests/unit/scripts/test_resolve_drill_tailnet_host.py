"""Tests for scripts/ops/resolve_drill_tailnet_host.sh (DR drill MagicDNS)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ops" / "resolve_drill_tailnet_host.sh"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "tailscale"


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    full_env = {**os.environ, **env}
    return subprocess.run(
        ["/usr/bin/env", "bash", str(SCRIPT)],
        cwd=str(REPO_ROOT),
        env=full_env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("fixture", "primary", "expected_stdout"),
    [
        (
            "status_drill_podcast_suffix.json",
            "dr-podcast.tail-test.ts.net",
            "dr-podcast-1.tail-test.ts.net",
        ),
        (
            "status_drill_podcast_canonical.json",
            "dr-podcast.tail-test.ts.net",
            "dr-podcast.tail-test.ts.net",
        ),
        (
            "status_drill_podcast_hostname_only.json",
            "dr-podcast.tail-test.ts.net",
            "dr-podcast.tail-test.ts.net",
        ),
        (
            "status_drill_tagged_magicdns_only.json",
            "dr-podcast.tail-test.ts.net",
            "hetzner-drill-abc.tail-test.ts.net",
        ),
    ],
)
def test_resolver_finds_online_host(fixture: str, primary: str, expected_stdout: str) -> None:
    fp = FIXTURES / fixture
    proc = _run(
        {
            "DRILL_TAILNET_FQDN": primary,
            "TAILSCALE_STATUS_JSON_PATH": str(fp),
        }
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == expected_stdout


def test_resolver_fails_when_no_drill_host_online() -> None:
    proc = _run(
        {
            "DRILL_TAILNET_FQDN": "dr-podcast.tail-test.ts.net",
            "TAILSCALE_STATUS_JSON_PATH": str(FIXTURES / "status_no_prod_podcast.json"),
        }
    )
    assert proc.returncode == 1
    assert "DRILL_TAILNET_FQDN" in proc.stderr or "No online" in proc.stderr


def test_resolver_rejects_non_drill_stem() -> None:
    proc = _run(
        {
            "DRILL_TAILNET_FQDN": "other.tail-test.ts.net",
            "TAILSCALE_STATUS_JSON_PATH": str(FIXTURES / "status_drill_podcast_canonical.json"),
        }
    )
    assert proc.returncode == 1
    assert "dr-podcast" in proc.stderr.lower()


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be executable"
