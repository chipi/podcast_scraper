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
    # Avatar serve is deliberately open (#2109, operator 2026-09-16): an `<img src=…>` cannot send
    # an Authorization header, and the native shell carries its session in exactly that header — so
    # a session-gated avatar 401'd on device and every user silently fell back to initials. The id
    # is an opaque 24-hex token that is never displayed, and show/episode artwork is already served
    # unauthenticated by this same API.
    #
    # #2109 opened the route and did not update this list, so the matrix has been red on main since.
    # Listed here rather than skipped: the allow-list branch still asserts the route is not
    # auth-BLOCKED, and any NEW unauthenticated route still fails until someone writes its reason
    # down here.
    "/api/app/profile/x/avatar",
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


def _iter_routes(routes, prefix: str = ""):
    """Yield ``(path, methods)`` for every route, walking INTO included routers.

    ``app.routes`` is not a flat list. Since fastapi 0.141 (pinned at pyproject.toml:236, raised
    in #1029 / Dependabot #1375) ``include_router()`` stores an ``_IncludedRouter`` wrapper that
    holds the child router plus its prefix, instead of splicing the child's routes into the parent.
    Those wrappers have no ``.path``.

    The previous version of this test read ``route.path`` straight off ``app.routes``, so after
    that bump it saw 4 routes — ``/openapi.json``, ``/docs``, ``/docs/oauth2-redirect``,
    ``/redoc`` — and **zero** ``/api/app`` paths. It was enumerating nothing, which means the
    login-first assertion below was running against nothing: a new unauthenticated ``/api/app``
    route would NOT have been caught. Walking the tree recovers all 124 of them.
    """
    for route in routes:
        included = getattr(route, "original_router", None)
        if included is not None:
            ctx = getattr(route, "include_context", None)
            yield from _iter_routes(included.routes, prefix + (getattr(ctx, "prefix", "") or ""))
            continue
        path = getattr(route, "path", None)
        if path is not None:
            yield prefix + path, (getattr(route, "methods", set()) or set())


def test_every_app_route_requires_auth_except_allow_list() -> None:
    """Every method (GET/POST/PUT/DELETE/PATCH) on every /api/app route must 401 anonymously,
    except the allow-list. `@bearer` lets any Authorization:Bearer past the edge on every
    /api/app/* path, so the backend is the ONLY enforcement layer — this must cover writes too."""
    app = _make_app()
    client = TestClient(app)  # anonymous — no cookie, no bearer
    checked = 0
    for path, methods in _iter_routes(app.routes):
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
    # Two separate failures, because they mean different things and one is far more dangerous.
    #
    # ZERO means enumeration itself broke — the guard is inert and every assertion above was
    # vacuous. That is how this test spent the time since the fastapi 0.141 bump asserting nothing,
    # and it must never again read as "a bit of coverage went missing".
    assert checked, (
        "route enumeration found NO routes — this guard is inert, not merely thin. `app.routes` "
        "is not flat (fastapi stores _IncludedRouter wrappers); see _iter_routes. Every assertion "
        "in this test just passed without executing."
    )
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
