"""Infra config validation gate (#1163 / #1160 / ADR-114..111).

The edge/deploy configs were only ever hand-verified by one-off boots. These turn the
safety-relevant properties into automated checks:

* cheap content assertions (no tooling) — catch the regressions that matter (nginx only
  proxies the consumer plane; preserves the forwarded scheme; Caddy admin API off);
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
NGINX_CONF = REPO / "web" / "learning-player" / "nginx.conf"
VIEWER_NGINX_CONF = REPO / "docker" / "viewer" / "default.conf.template"
CADDYFILE = REPO / "infra" / "cloud-init" / "Caddyfile"
PLAYER_CADDY = REPO / "infra" / "caddy" / "player.caddy"
OPERATOR_CADDY = REPO / "infra" / "caddy" / "operator.caddy"
PROD_USER_DATA = REPO / "infra" / "cloud-init" / "prod.user-data"
SHELL_SCRIPTS = [
    REPO / "infra" / "cloud-init" / "decrypt-secrets.sh",
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


def test_player_nginx_only_proxies_consumer_plane() -> None:
    conf = NGINX_CONF.read_text()
    assert "location /api/app/" in conf, "player nginx must proxy the consumer plane"
    # Must NOT blanket-proxy all of /api/ (that would forward /api/jobs to the backend).
    assert "location /api/ {" not in conf, "player nginx must NOT proxy all of /api/"


def test_player_nginx_preserves_forwarded_proto() -> None:
    conf = NGINX_CONF.read_text()
    assert "map $http_x_forwarded_proto" in conf, "must preserve Caddy's X-Forwarded-Proto"
    assert "proxy_set_header X-Forwarded-Proto $lp_forwarded_proto" in conf


def test_viewer_nginx_preserves_forwarded_proto() -> None:
    # RFC-108: the operator-public viewer sits behind Caddy (TLS) on a plain-http
    # loopback hop, so nginx's own $scheme is always "http". Forwarding that
    # verbatim makes the api build an http:// OAuth redirect_uri → Google
    # redirect_uri_mismatch. It must preserve Caddy's X-Forwarded-Proto instead.
    conf = VIEWER_NGINX_CONF.read_text()
    assert "map $http_x_forwarded_proto" in conf, "must preserve Caddy's X-Forwarded-Proto"
    assert "proxy_set_header X-Forwarded-Proto $viewer_forwarded_proto" in conf
    assert (
        "proxy_set_header X-Forwarded-Proto $scheme" not in conf
    ), "must NOT forward the loopback $scheme (http) as the edge proto"


def test_viewer_nginx_rate_limits_api() -> None:
    # RFC-108 / T-06: the operator-public viewer must rate-limit like the player nginx,
    # keyed on the REAL client IP (real_ip recovers it from XFF), with a tighter zone on
    # the auth endpoints. Without this the operator API has no per-IP origin throttle.
    conf = VIEWER_NGINX_CONF.read_text()
    assert (
        "limit_req_zone" in conf and "limit_req zone=" in conf
    ), "operator viewer API must be rate limited"
    assert "real_ip_header X-Forwarded-For" in conf, "must rate-limit by real client IP, not Caddy"
    assert "zone=op_auth" in conf, "auth endpoints need a tighter rate zone"


def test_player_nginx_rate_limits_api() -> None:
    conf = NGINX_CONF.read_text()
    # Rate limiting (T-06) keyed on the REAL client IP (real_ip recovers it from XFF).
    assert (
        "limit_req_zone" in conf and "limit_req zone=" in conf
    ), "consumer API must be rate limited"
    assert "real_ip_header X-Forwarded-For" in conf, "must rate-limit by real client IP, not Caddy"
    assert "zone=lp_auth" in conf, "auth endpoints need a tighter rate zone"


def test_caddy_admin_api_disabled() -> None:
    assert "admin off" in CADDYFILE.read_text(), "Caddy admin API must be off (T-02)"


def test_caddy_hardened_security_headers() -> None:
    # Fleet-wide baseline (ADR-114 §3 hardened snippet): clickjacking + MIME-sniff +
    # referrer, and HSTS ramped to 1y (T-12). Reverting any of these must fail here.
    conf = CADDYFILE.read_text()
    assert 'X-Frame-Options "DENY"' in conf, "clickjacking protection (X-Frame-Options DENY)"
    assert 'X-Content-Type-Options "nosniff"' in conf, "MIME-sniffing protection"
    assert "Referrer-Policy" in conf, "referrer policy header"
    assert "max-age=31536000" in conf, "HSTS must be ramped to 1y (T-12)"


def test_caddy_access_log_format_pinned() -> None:
    # The fail2ban caddy-access filter parses JSON with an ISO8601 ts. If the log
    # format drifts (e.g. Caddy default flips), the filter silently stops banning —
    # this couples the two so a format change fails here (T-05 / T-11).
    conf = CADDYFILE.read_text()
    assert "format json" in conf, "Caddy access log must be pinned to JSON for the f2b filter"
    assert "time_format iso8601" in conf, "log ts must be ISO8601 for fail2ban date parsing"


def test_fail2ban_caddy_access_jail_present() -> None:
    # Edge scanner-ban jail (T-05 stopgap). The filter must key on client_ip (NOT
    # remote_ip — ADR-118: post-Cloudflare remote_ip is a CF edge, banning it =
    # global outage) + 4xx, and point at the access log Caddy actually writes.
    data = PROD_USER_DATA.read_text()
    assert "[caddy-access]" in data, "caddy-access fail2ban jail must be defined"
    assert (
        'failregex = "client_ip":"<HOST>".*"status":4\\d\\d' in data
    ), "filter must ban 4xx bursts keyed on client_ip (CF-safe, ADR-118)"
    assert "logpath = /var/log/caddy/access.log" in data, "jail must watch the Caddy access log"


def test_caddy_trusts_cloudflare_for_real_ip() -> None:
    # ADR-118: CF ranges must be trusted so client_ip is the real visitor, and the
    # CF real-client header must be honored. Keep in sync with cloudflare_ip_ranges.
    conf = CADDYFILE.read_text()
    assert "trusted_proxies static" in conf, "must trust CF proxy ranges for real client_ip"
    assert "client_ip_headers Cf-Connecting-Ip" in conf, "must read CF-Connecting-IP"
    assert (
        "104.16.0.0/13" in conf
    ), "CF range list must be present (refresh from cloudflare.com/ips)"


# --------------------------------------------------------------------------- #
# tool validations — skip when the tool isn't available
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _HAS_DOCKER, reason="docker daemon not reachable (client-only or socket down)"
)
def test_caddy_config_validates(tmp_path: Path) -> None:
    sites = tmp_path / "sites"
    sites.mkdir()
    (sites / "player.caddy").write_text(PLAYER_CADDY.read_text())
    # RFC-108: the operator-public vhost is validated alongside the player (distinct host,
    # so they coexist; the __*_PREVIEW_COOKIE__ placeholders are valid literal values).
    (sites / "operator.caddy").write_text(OPERATOR_CADDY.read_text())
    proc = subprocess.run(  # noqa: S603
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{CADDYFILE}:/etc/caddy/Caddyfile:ro",
            "-v",
            f"{sites}:/etc/caddy/sites:ro",
            "caddy:2",
            "caddy",
            "validate",
            "--config",
            "/etc/caddy/Caddyfile",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"caddy validate failed:\n{proc.stderr}"


@pytest.mark.skipif(
    not _HAS_DOCKER, reason="docker daemon not reachable (client-only or socket down)"
)
def test_player_nginx_syntax_ok() -> None:
    proc = subprocess.run(  # noqa: S603
        [
            "docker",
            "run",
            "--rm",
            "--add-host",
            "api:127.0.0.1",
            "-v",
            f"{NGINX_CONF}:/etc/nginx/conf.d/default.conf:ro",
            "nginx:1.27-alpine",
            "nginx",
            "-t",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"nginx -t failed:\n{proc.stderr}"


@pytest.mark.skipif(not _HAS_SHELLCHECK, reason="shellcheck not installed")
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_shellcheck_clean(script: Path) -> None:
    proc = subprocess.run(  # noqa: S603
        ["shellcheck", str(script)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, f"shellcheck {script.name} failed:\n{proc.stdout}"
