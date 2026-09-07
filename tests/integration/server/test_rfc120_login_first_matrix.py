"""RFC-120 (#2009) login-first release gate.

Auto-enumerates the consumer app's routes and asserts every ``/api/app`` GET requires a
session EXCEPT the explicit anonymous allow-list (the teaser lure + protocol-anonymous
OAuth/unsubscribe/health). Because it walks ``app.routes`` rather than a hand-list, a NEW
endpoint added without auth fails this test — that is the point: this is the gate that must be
green before the edge ``@bearer`` rule deploys.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

_FIXTURE_CORPUS = Path(__file__).resolve().parents[3] / "tests/fixtures/app-validation-corpus/v3"

# The ONLY content reachable without a free account: the curated teaser lure, plus endpoints
# that are anonymous by protocol (OAuth discovery/flow, one-click unsubscribe, health).
_ANON_ALLOW = {
    "/api/app/discover",
    "/api/app/corpus/trending-topics",
    "/api/app/artwork",
    "/api/app/comms/unsubscribe",
    "/api/app/auth/login",
    "/api/app/auth/callback",
    "/api/app/auth/logout",
    "/api/app/auth/dev-users",
    "/api/app/auth/status",
    "/api/app/mcp/oauth/register",
    "/api/app/mcp/oauth/token",
    "/.well-known/oauth-authorization-server",
    "/api/health",
}
# GET routes that require auth but answer via an OAuth redirect, not a 401 — excluded from the
# strict 401 assertion (still NOT anonymous access).
_REDIRECTING = {"/api/app/mcp/oauth/authorize"}


def _concrete(path: str) -> str:
    """Fill ``{param}`` placeholders with a dummy so the path is callable."""
    return re.sub(r"\{[^}]+\}", "x", path)


def _make_app():
    return create_app(_FIXTURE_CORPUS, static_dir=False)


def test_every_app_route_requires_auth_except_allow_list() -> None:
    """Every method (GET/POST/PUT/DELETE/PATCH) on every /api/app route must 401 anonymously,
    except the allow-list. `@bearer` lets any Authorization:Bearer past the edge on every
    /api/app/* path, so the backend is the ONLY enforcement layer — this must cover writes too."""
    app = _make_app()
    client = TestClient(app)  # anonymous — no cookie, no bearer
    checked = 0
    for route in app.routes:
        methods: set[str] = getattr(route, "methods", set()) or set()
        path = getattr(route, "path", "")
        if not (path.startswith("/api/app") or path == "/.well-known/oauth-authorization-server"):
            continue
        concrete = _concrete(path)
        for method in sorted(methods - {"HEAD", "OPTIONS"}):
            if concrete in _ANON_ALLOW:
                # Allow-list: must NOT be auth-blocked (200/400/404/422/503 fine; 401 is failure).
                resp = client.request(method, concrete, json={})
                assert resp.status_code != 401, f"allow-list {method} {path} unexpectedly 401"
                checked += 1
                continue
            if concrete in _REDIRECTING:
                continue
            resp = client.request(method, concrete, json={})
            assert resp.status_code == 401, (
                f"{method} {path} must require a session (login-first) but was reachable "
                f"anonymously (got {resp.status_code})"
            )
            checked += 1
    assert checked > 25, f"route enumeration found too few app routes ({checked})"


@pytest.mark.parametrize(
    "path,key",
    [
        ("/api/app/discover?limit=50", "items"),
        ("/api/app/corpus/trending-topics?limit=50", "topics"),
    ],
)
def test_teaser_clamps_anonymous_callers(path: str, key: str) -> None:
    """Teaser endpoints stay anonymous but clamp an anon caller to <=8 regardless of ?limit."""
    client = TestClient(_make_app())
    resp = client.get(path)
    assert resp.status_code == 200, f"{path} anon expected 200, got {resp.status_code}"
    assert len(resp.json().get(key, [])) <= 8, f"{path} did not clamp anonymous callers"


def test_teaser_trending_ignores_filter_params_for_anon() -> None:
    """M1: an anon caller must not be able to sweep min_velocity/min_total to enumerate different
    8-topic slices past the clamp — the filter params are locked to defaults for anon."""
    client = TestClient(_make_app())
    base = client.get("/api/app/corpus/trending-topics").json().get("topics", [])
    swept = (
        client.get("/api/app/corpus/trending-topics?min_velocity=999&min_total=999")
        .json()
        .get("topics", [])
    )
    assert swept == base, "anon trending must ignore filter params (Fable-5 M1)"
