"""Infra config validation gate (#1163 / #1160 / ADR-114 (not in this repository)..111).

The edge/deploy configs were only ever hand-verified by one-off boots. These turn the
safety-relevant properties into automated checks:

* cheap content assertions (no tooling) — catch the regressions that matter (the viewer
  nginx preserves the forwarded scheme and rate-limits the API);
* tool validations (skip when the tool is absent) — ``caddy validate``, ``nginx -t``,
  ``shellcheck`` on the shipped scripts.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

REPO = Path(__file__).resolve().parents[3]
VIEWER_NGINX_CONF = REPO / "docker" / "viewer" / "default.conf.template"
SHELL_SCRIPTS = [
    REPO / "docker" / "secrets-shim.sh",
]


def _docker_is_usable() -> bool:
    """True when docker can actually RUN something, not merely when the binary exists.

    ``shutil.which("docker")`` was the guard here, and it answers the wrong question. The client
    binary is installed on every dev box; what these two tests need is a live DAEMON, because they
    shell out to ``docker run`` to get a real ``caddy validate`` / ``nginx -t``. When the daemon is
    down the binary is still on PATH, so the guard passed and the test failed with::

        AssertionError: caddy validate failed:
          error during connect: Get "http://%2Fvar%2Frun%2Fdocker.sock/_ping": EOF

    which reads as "the Caddyfile is broken" — the precise confusion a skip guard exists to
    prevent. Measured 2026-08-18: both tests passed earlier the same day and failed after the
    socket relay stopped, with the configs untouched. This is the #1657 lesson (guard on ffmpeg
    AND ffprobe, not just ffmpeg) one layer deeper: guard on the tool being USABLE, not present.

    ``docker version`` is the cheapest call that requires a server answer — ``--format`` on
    ``.Server.Version`` fails when only the client responds. CI has a working daemon, so this
    stays False only where the daemon genuinely cannot be reached, and the tests still run there.
    """
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


_HAS_DOCKER = _docker_is_usable()
_HAS_SHELLCHECK = shutil.which("shellcheck") is not None


# --------------------------------------------------------------------------- #
# cheap content assertions — always run
# --------------------------------------------------------------------------- #


def test_viewer_nginx_preserves_forwarded_proto() -> None:
    # RFC-108 (not in this repository): the operator-public viewer sits behind Caddy (TLS) on a plain-http
    # loopback hop, so nginx's own $scheme is always "http". Forwarding that
    # verbatim makes the api build an http:// OAuth redirect_uri → Google
    # redirect_uri_mismatch. It must preserve Caddy's X-Forwarded-Proto instead.
    conf = VIEWER_NGINX_CONF.read_text()
    assert "map $http_x_forwarded_proto" in conf, "must preserve Caddy's X-Forwarded-Proto"
    assert "proxy_set_header X-Forwarded-Proto $viewer_forwarded_proto" in conf
    assert (
        "proxy_set_header X-Forwarded-Proto $scheme" not in conf
    ), "must NOT forward the loopback $scheme (http) as the edge proto"


def test_viewer_dsn_is_injected_at_container_start() -> None:
    # The deployed viewer takes its GlitchTip DSN from the container (PODCAST_SENTRY_DSN_VIEWER),
    # not from a CI build-arg: nginx's envsubst renders it into index.html, main.ts reads it.
    conf = VIEWER_NGINX_CONF.read_text()
    assert 'window.__PODCAST_SENTRY_DSN_VIEWER__="${PODCAST_SENTRY_DSN_VIEWER}"' in conf
    dockerfile = (REPO / "docker" / "viewer" / "Dockerfile").read_text()
    assert "ENV PODCAST_SENTRY_DSN_VIEWER=" in dockerfile, "empty default, so envsubst renders ''"
    assert (
        "ENV NGINX_ENVSUBST_FILTER=^PODCAST_" in dockerfile
    ), "the PODCAST_ var must be substituted"
    main_ts = (REPO / "web" / "gi-kg-viewer" / "src" / "main.ts").read_text()
    assert "__PODCAST_SENTRY_DSN_VIEWER__" in main_ts
    stack_test = (REPO / ".github" / "workflows" / "stack-test.yml").read_text()
    assert "VITE_SENTRY_DSN_VIEWER" not in stack_test, "CI must not bake a viewer DSN"


def test_viewer_nginx_rate_limits_api() -> None:
    # RFC-108 (not in this repository) / T-06: the operator-public viewer must rate-limit like the player nginx,
    # keyed on the REAL client IP (real_ip recovers it from XFF), with a tighter zone on
    # the auth endpoints. Without this the operator API has no per-IP origin throttle.
    conf = VIEWER_NGINX_CONF.read_text()
    assert (
        "limit_req_zone" in conf and "limit_req zone=" in conf
    ), "operator viewer API must be rate limited"
    assert "real_ip_header X-Forwarded-For" in conf, "must rate-limit by real client IP, not Caddy"
    assert "zone=op_auth" in conf, "auth endpoints need a tighter rate zone"


@pytest.mark.skipif(not _HAS_SHELLCHECK, reason="shellcheck not installed")
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_shellcheck_clean(script: Path) -> None:
    proc = subprocess.run(  # noqa: S603
        ["shellcheck", str(script)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, f"shellcheck {script.name} failed:\n{proc.stdout}"
