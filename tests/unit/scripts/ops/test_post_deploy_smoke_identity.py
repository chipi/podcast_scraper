"""DR-1: post_deploy_smoke.sh corpus-IDENTITY assertion (#24, the highest-leverage guard).

A stale corpus passes every subsystem probe — /api/health is green, episodes/digest/graph all
respond. The one thing that catches "deploy reported green while serving the OLD corpus" (#14) is
the identity check: served ``corpus_produced_by.produced_at`` / ``corpus_code_version`` must equal
the value we shipped. That logic (script L141-158) had no test, so a refactor comparing the wrong
field — or dropping the exit 1 — would silently reopen the #14 hole. These tests stand up a fake
/api/health and assert the smoke goes RED on a mismatch and GREEN on a match.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SMOKE = Path(__file__).resolve().parents[4] / "scripts" / "ops" / "post_deploy_smoke.sh"

_needs_tools = pytest.mark.skipif(
    any(shutil.which(t) is None for t in ("bash", "curl", "jq")),
    reason="needs bash + curl + jq",
)

_HEALTHY_FLAGS = {
    "status": "ok",
    "artifacts_api": True,
    "search_api": True,
    "explore_api": True,
    "index_routes_api": True,
    "corpus_library_api": True,
    "corpus_digest_api": True,
    "corpus_metrics_api": True,
}


class _HealthServer:
    """Threaded HTTP server that answers GET /api/health with a fixed JSON body."""

    def __init__(self, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/api/health"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *_a: object) -> None:  # silence
                pass

        self._srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)

    @property
    def base_url(self) -> str:
        # Bound to 127.0.0.1 above; server_address is (host, port) for AF_INET.
        port = self._srv.server_address[1]
        return f"http://127.0.0.1:{port}"

    def __enter__(self) -> "_HealthServer":
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._srv.shutdown()
        self._srv.server_close()


def _run_smoke(base_url: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SMOKE), "--base-url", base_url, *args],
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=60,
    )


@_needs_tools
def test_identity_match_passes() -> None:
    with _HealthServer(
        {**_HEALTHY_FLAGS, "corpus_produced_by": {"produced_at": "2026-08-08T00:00:00Z"}}
    ) as srv:
        res = _run_smoke(srv.base_url, "--expect-corpus-produced-at", "2026-08-08T00:00:00Z")
    assert res.returncode == 0, res.stderr
    assert "corpus identity OK" in res.stderr


@_needs_tools
def test_identity_mismatch_fails_red() -> None:
    with _HealthServer(
        {**_HEALTHY_FLAGS, "corpus_produced_by": {"produced_at": "2026-01-01T00:00:00Z"}}
    ) as srv:
        res = _run_smoke(srv.base_url, "--expect-corpus-produced-at", "2026-08-08T00:00:00Z")
    assert res.returncode == 1
    assert "STALE/WRONG corpus" in res.stderr


@_needs_tools
def test_identity_missing_field_fails_red() -> None:
    """Served corpus omits produced_at entirely → jq yields 'null' != expected → RED."""
    with _HealthServer(_HEALTHY_FLAGS) as srv:
        res = _run_smoke(srv.base_url, "--expect-corpus-produced-at", "2026-08-08T00:00:00Z")
    assert res.returncode == 1
    assert "STALE/WRONG corpus" in res.stderr


@_needs_tools
def test_code_version_mismatch_fails_red() -> None:
    with _HealthServer({**_HEALTHY_FLAGS, "corpus_code_version": "abc123"}) as srv:
        res = _run_smoke(srv.base_url, "--expect-corpus-code-version", "deadbee")
    assert res.returncode == 1
    assert "STALE/WRONG corpus" in res.stderr


@_needs_tools
def test_no_expectation_skips_identity_check() -> None:
    """Back-compat: with neither expectation set, the identity block is skipped (health-only)."""
    with _HealthServer(_HEALTHY_FLAGS) as srv:
        res = _run_smoke(srv.base_url)
    assert res.returncode == 0, res.stderr
    assert "health-only smoke complete" in res.stderr
