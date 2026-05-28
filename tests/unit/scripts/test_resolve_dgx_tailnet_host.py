"""Tests for scripts/ops/resolve_dgx_tailnet_host.sh (RFC-089)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ops" / "resolve_dgx_tailnet_host.sh"
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


def test_resolver_finds_online_dgx_host() -> None:
    proc = _run(
        {
            "DGX_TAILNET_FQDN": "dgx-llm-1.tail-test.ts.net",
            "TAILSCALE_STATUS_JSON_PATH": str(FIXTURES / "status_dgx_llm_suffix.json"),
        }
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "dgx-llm-1.tail-test.ts.net"


def test_resolver_warns_on_suffix_drift() -> None:
    proc = _run(
        {
            "DGX_TAILNET_FQDN": "dgx-llm-2.tail-test.ts.net",
            "TAILSCALE_STATUS_JSON_PATH": str(FIXTURES / "status_dgx_llm_drift.json"),
        }
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "dgx-llm-1.tail-test.ts.net"
    assert "suffix drift" in proc.stderr.lower()


def test_resolver_fails_when_no_dgx_online() -> None:
    proc = _run(
        {
            "DGX_TAILNET_FQDN": "dgx-llm-1.tail-test.ts.net",
            "TAILSCALE_STATUS_JSON_PATH": str(FIXTURES / "status_no_prod_podcast.json"),
        }
    )
    assert proc.returncode == 1


def test_resolver_rejects_non_dgx_stem() -> None:
    proc = _run(
        {
            "DGX_TAILNET_FQDN": "other.tail-test.ts.net",
            "TAILSCALE_STATUS_JSON_PATH": str(FIXTURES / "status_dgx_llm_suffix.json"),
        }
    )
    assert proc.returncode == 1
    assert "dgx-llm" in proc.stderr.lower()


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be executable"
