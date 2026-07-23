"""Runtime behavior of the player nginx — the public deployment shape (#1163).

Boots the REAL learning-app image against a stub upstream (`api:8000` -> host gateway) and
proves, at runtime, the two safety properties the config-content + app-only-backend tests
can only assert structurally:

* only `/api/app/*` is proxied to the backend — an operator path (`/api/jobs`) never
  reaches the upstream (nginx serves the SPA fallback instead);
* the consumer API is rate limited (a burst returns 429).

Complements the existing full-app playwright e2e (which exercises app logic against the
real backend) by covering the containerized nginx path-filter + rate-limit.
"""

from __future__ import annotations

import http.server
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

IMAGE = "podcast-scraper-learning-app:latest"
NGINX_HOST_PORT = 18092


def _pick_free_port() -> int:
    """Ask the kernel for an ephemeral port that's currently free, then close
    the probe socket so the caller can bind it.

    Race-prone in principle (another process could grab it before we bind),
    but on a test machine it's stable enough. Alternative would be to bind
    the actual stub with port 0 and read back the assigned port before
    docker-run — one more change, saved for later if this races in practice.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _render_nginx_conf(source: Path, upstream_port: int) -> str:
    """Rewrite the shipped nginx.conf to point every `api:8000` upstream at
    `api:<upstream_port>`. Kept as a text-level replace to preserve every
    other property (rate limits, headers, SPA fallback) verbatim.
    """
    original = source.read_text(encoding="utf-8")
    # Match "proxy_pass http://api:8000" — the two locations in the shipped
    # config; brittle to whitespace but the config is committed so any drift
    # is caught by the test authors, not silently by the runtime.
    replaced = original.replace("http://api:8000", f"http://api:{upstream_port}")
    if replaced == original:
        raise AssertionError(
            "test_player_nginx_runtime: expected to rewrite http://api:8000 in nginx.conf; "
            "if the upstream form changed, update _render_nginx_conf.",
        )
    return replaced


class _StubUpstream(http.server.BaseHTTPRequestHandler):
    """Answers ANY path with a distinctive marker so tests can tell 'reached backend'
    from 'nginx served the SPA fallback'."""

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"upstream":"backend-reached"}')

    def log_message(self, *_a) -> None:  # noqa: ANN002
        pass


@pytest.fixture(scope="module")
def player_base() -> Iterator[str]:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not on PATH")
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        pytest.skip(f"{IMAGE} not built (docker build web/learning-player -t {IMAGE})")

    # #1261 fix: nginx.conf hard-codes `api:8000`, but on Docker Desktop for
    # macOS the host-gateway route often reaches whatever process is on
    # localhost:8000 — which locally is a leftover `make serve` FastAPI, not
    # our stub, so the stub loses even after binding. Pick a random free port
    # for the stub AND render a nginx.conf that proxies at that port. Both
    # ends match by construction; no host-machine port conflicts.
    stub_port = _pick_free_port()
    conf_source = Path(__file__).resolve().parents[3] / "web/learning-player/nginx.conf"
    rendered_conf = _render_nginx_conf(conf_source, stub_port)
    conf_dir = Path(tempfile.mkdtemp(prefix="lp-nginx-runtime-"))
    conf_path = conf_dir / "default.conf"
    conf_path.write_text(rendered_conf, encoding="utf-8")

    # Stub upstream binds all interfaces so the container-side nginx can reach
    # it via host-gateway. Test-local only; test doesn't run in CI without an
    # explicit slow-integration profile.
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", stub_port), _StubUpstream)  # nosec B104
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--add-host",
            "api:host-gateway",
            "-p",
            f"{NGINX_HOST_PORT}:80",
            "-v",
            f"{conf_path}:/etc/nginx/conf.d/default.conf:ro",
            IMAGE,
        ],
        capture_output=True,
        text=True,
    )
    cid = run.stdout.strip()
    if not cid:
        srv.shutdown()
        pytest.fail(f"failed to start nginx container: {run.stderr}")
    # Wait until the container's nginx actually accepts a connection — a fixed
    # sleep raced on Docker Desktop for macOS where port publishing lags
    # container start by ~1-3 seconds (#1261 followup).
    base = f"http://127.0.0.1:{NGINX_HOST_PORT}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            urllib.request.urlopen(base + "/", timeout=1)  # noqa: S310
            break
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    else:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        srv.shutdown()
        pytest.fail("nginx container did not accept connections within 30s")
    try:
        yield base
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        srv.shutdown()
        try:
            conf_path.unlink(missing_ok=True)
            conf_dir.rmdir()
        except OSError:
            # Best-effort cleanup — the tempdir prefix means orphans are
            # harmless and rare enough not to fail the test on this.
            pass


def _get(url: str) -> tuple[int, str]:
    try:
        r = urllib.request.urlopen(url, timeout=5)  # noqa: S310 - localhost only
        return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def test_api_app_is_proxied_to_backend(player_base: str) -> None:
    code, body = _get(f"{player_base}/api/app/anything")
    assert code == 200 and "backend-reached" in body, "/api/app/* must reach the backend"


def test_operator_api_never_reaches_backend(player_base: str) -> None:
    # /api/jobs is not an /api/app/ location -> nginx serves the SPA fallback, never the
    # operator API. The upstream marker must be ABSENT.
    _code, body = _get(f"{player_base}/api/jobs")
    assert "backend-reached" not in body, "/api/jobs must NOT be proxied to the backend"


def test_consumer_api_is_rate_limited(player_base: str) -> None:
    codes = [_get(f"{player_base}/api/app/auth/login")[0] for _ in range(60)]
    assert 429 in codes, "a burst on the auth endpoint must hit the rate limit (429)"


# --------------------------------------------------------------------------- #
# #1261-9 — SPA fallback for the new listener routes (topic / person / browse)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    [
        "/topic/topic:ai",
        "/person/person:jane-doe",
        "/browse/topics",
        "/browse/people",
    ],
)
def test_new_listener_routes_fall_through_to_spa(player_base: str, path: str) -> None:
    # A direct-load of a client-side route must not 404 — nginx should serve the
    # SPA (index.html) so vue-router can render the matching view.
    code, body = _get(f"{player_base}{path}")
    assert code == 200, f"{path} returned {code}, expected 200 (SPA fallback)"
    assert '<div id="app">' in body, f"{path} did not serve the SPA index.html"
