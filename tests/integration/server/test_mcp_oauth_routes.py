"""Integration tests for the MCP OAuth flow through the routes (RFC-112 slice 3, #1471)."""

from __future__ import annotations

import base64
import hashlib
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_rate_limit, app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user, set_mcp_access


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    # The limiter is a process-global; reset it so /register + /token calls don't leak across tests.
    app_rate_limit.reset()
    yield
    app_rate_limit.reset()


_ISSUER = "https://app.example.com"
_RESOURCE_AUD = "https://mcp.example.com"
_REDIRECT = "https://claude.ai/api/mcp/callback"
_INTERNAL = "internal-mcp-tok"


def _pkce() -> tuple[str, str]:
    verifier = "v-" + "a" * 60
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    return verifier, challenge


def _app(tmp_path: Path, monkeypatch, *, mcp_access: bool = True):
    monkeypatch.setenv("APP_MCP_ISSUER_URL", _ISSUER)
    monkeypatch.setenv("APP_MCP_RESOURCE_URL", _RESOURCE_AUD)  # tokens are aud-bound to this
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "app"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    app.state.internal_mcp_token = _INTERNAL
    user = get_or_create_user(data_dir, provider="google", subject="s", email="u@g.com", name="U")
    if mcp_access:
        set_mcp_access(data_dir, user.user_id, True)
    client = TestClient(app)
    tok = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, tok)
    return client, data_dir, user.user_id


def test_metadata_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _app(tmp_path, monkeypatch)
    meta = client.get("/.well-known/oauth-authorization-server").json()
    assert meta["issuer"] == _ISSUER
    assert meta["authorization_endpoint"].endswith("/api/app/mcp/oauth/authorize")
    assert meta["code_challenge_methods_supported"] == ["S256"]


def test_authorize_unauthenticated_redirects_to_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remote client's /authorize with no player session bounces to Google sign-in (not 401).

    Regression for the 2026-08-08 connector dead-end: claude.ai opened /authorize as a top-level
    navigation, got a hard 401 ('Not authenticated'), and never showed a login prompt. It must
    302 to /auth/login with a guarded return_to back to /authorize so the flow completes.
    """
    client, _data_dir, _uid = _app(tmp_path, monkeypatch)
    client.cookies.clear()  # fresh browser: no player session
    _v, challenge = _pkce()
    params = {
        "response_type": "code",
        "client_id": "mcpc_dummy",
        "redirect_uri": _REDIRECT,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "mcp:read",
        "state": "st",
    }
    r = client.get("/api/app/mcp/oauth/authorize", params=params, follow_redirects=False)
    assert r.status_code == 302, r.text
    loc = r.headers["location"]
    assert loc.startswith("/api/app/auth/login?return_to="), loc
    from urllib.parse import parse_qs, unquote, urlsplit

    rt = unquote(parse_qs(urlsplit(loc).query)["return_to"][0])
    assert rt.startswith("/api/app/mcp/oauth/authorize"), rt
    assert "client_id=mcpc_dummy" in rt  # original params preserved for the return trip


def test_full_authorization_code_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, data_dir, uid = _app(tmp_path, monkeypatch)
    verifier, challenge = _pkce()

    # 1. DCR
    reg = client.post(
        "/api/app/mcp/oauth/register",
        json={"redirect_uris": [_REDIRECT], "client_name": "claude.ai"},
    )
    assert reg.status_code == 201
    cid = reg.json()["client_id"]

    # 2. authorize: GET renders consent (no code issued yet)
    params = {
        "response_type": "code",
        "client_id": cid,
        "redirect_uri": _REDIRECT,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": "xyz",
        "scope": "mcp:read",
    }
    page = client.get("/api/app/mcp/oauth/authorize", params=params)
    assert page.status_code == 200 and "<form method=post" in page.text

    # 3. approve → 302 redirect with a code + state
    approve = client.post("/api/app/mcp/oauth/authorize", data=params, follow_redirects=False)
    assert approve.status_code == 302
    location = approve.headers["location"]
    assert location.startswith(_REDIRECT) and "state=xyz" in location
    code = location.split("code=")[1].split("&")[0]

    # 4. token exchange (PKCE)
    tok = client.post(
        "/api/app/mcp/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": cid,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": _REDIRECT,
        },
    )
    assert tok.status_code == 200
    access = tok.json()["access_token"]
    assert access.startswith("clp_mcpat_")

    # 5. the OAuth access token authenticates against the internal verify seam
    verify = client.post(
        "/internal/mcp/verify", json={"token": access}, headers={"X-Internal-Token": _INTERNAL}
    )
    assert verify.json() == {
        "authenticated": True,
        "user_id": uid,
        "mcp_access": True,
        "scope": "mcp:read",
        "aud": _RESOURCE_AUD,
        # The verify seam surfaces the user's role so a rank-scoped MCP server (e.g. the obs MCP,
        # admin-only #56) can gate on it; the content MCP ignores it. A fresh user defaults to
        # listener (internal_mcp.py verify → user.role).
        "role": "listener",
    }


def test_second_authorize_is_silent_after_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _data_dir, _uid = _app(tmp_path, monkeypatch)
    _v, challenge = _pkce()
    reg = client.post(
        "/api/app/mcp/oauth/register", json={"redirect_uris": [_REDIRECT], "client_name": "c"}
    )
    cid = reg.json()["client_id"]
    params = {
        "response_type": "code",
        "client_id": cid,
        "redirect_uri": _REDIRECT,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": "s1",
        "scope": "mcp:read",
    }
    # first GET renders the consent page (not yet remembered)
    first = client.get("/api/app/mcp/oauth/authorize", params=params)
    assert first.status_code == 200 and "<form method=post" in first.text
    # approve → remembers consent
    approve = client.post("/api/app/mcp/oauth/authorize", data=params, follow_redirects=False)
    assert approve.status_code == 302
    # second GET is now SILENT: direct 302 with a fresh code, no consent page re-prompt
    second = client.get("/api/app/mcp/oauth/authorize", params=params, follow_redirects=False)
    assert second.status_code == 302
    loc = second.headers["location"]
    assert loc.startswith(_REDIRECT) and "code=" in loc and "state=s1" in loc


def test_consent_page_discloses_redirect_and_offers_deny(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _data_dir, _uid = _app(tmp_path, monkeypatch)
    _v, challenge = _pkce()
    reg = client.post(
        "/api/app/mcp/oauth/register",
        json={"redirect_uris": [_REDIRECT], "client_name": "Definitely Claude"},
    )
    cid = reg.json()["client_id"]
    page = client.get(
        "/api/app/mcp/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": cid,
            "redirect_uri": _REDIRECT,
            "code_challenge": challenge,
        },
    )
    # The consent screen must disclose WHERE the code goes so a look-alike client_name can't trick
    # the user (DCR is open) — and must offer a way to decline. We extract the disclosed origin and
    # compare by EQUALITY (not a URL-substring `in` check, which trips CodeQL's
    # py/incomplete-url-substring-sanitization heuristic — `==` is exactly the fix that rule wants).
    import re

    assert "redirected to" in page.text  # the "where the code goes" disclosure renders
    disclosed = re.search(r"redirected to <b>([^<]+)</b>", page.text)
    assert disclosed is not None and disclosed.group(1) == "https://claude.ai"  # the real origin
    assert "Deny" in page.text
    assert page.headers.get("X-Frame-Options") == "DENY"


def test_authorize_rejects_unsupported_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _data_dir, _uid = _app(tmp_path, monkeypatch)
    _v, challenge = _pkce()
    reg = client.post(
        "/api/app/mcp/oauth/register", json={"redirect_uris": [_REDIRECT], "client_name": "c"}
    )
    cid = reg.json()["client_id"]
    resp = client.get(
        "/api/app/mcp/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": cid,
            "redirect_uri": _REDIRECT,
            "code_challenge": challenge,
            "scope": "mcp:admin",  # not supported → must be refused, not silently minted
        },
    )
    assert resp.status_code == 400


def test_authorize_requires_entitlement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, data_dir, _ = _app(tmp_path, monkeypatch, mcp_access=False)
    reg = client.post(
        "/api/app/mcp/oauth/register", json={"redirect_uris": [_REDIRECT], "client_name": "c"}
    )
    cid = reg.json()["client_id"]
    _v, challenge = _pkce()
    resp = client.get(
        "/api/app/mcp/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": cid,
            "redirect_uri": _REDIRECT,
            "code_challenge": challenge,
        },
    )
    assert resp.status_code == 403


def test_authorize_rejects_unregistered_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _ = _app(tmp_path, monkeypatch)
    reg = client.post(
        "/api/app/mcp/oauth/register", json={"redirect_uris": [_REDIRECT], "client_name": "c"}
    )
    cid = reg.json()["client_id"]
    _v, challenge = _pkce()
    resp = client.get(
        "/api/app/mcp/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": cid,
            "redirect_uri": "https://evil.example/cb",
            "code_challenge": challenge,
        },
    )
    assert resp.status_code == 400  # anti open-redirect


def test_bad_token_grant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _app(tmp_path, monkeypatch)
    reg = client.post(
        "/api/app/mcp/oauth/register", json={"redirect_uris": [_REDIRECT], "client_name": "c"}
    )
    cid = reg.json()["client_id"]
    resp = client.post(
        "/api/app/mcp/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": cid,
            "code": "nope",
            "code_verifier": "x",
            "redirect_uri": _REDIRECT,
        },
    )
    assert resp.status_code == 400 and resp.json()["error"] == "invalid_grant"


def test_register_rate_limited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _data_dir, _uid = _app(tmp_path, monkeypatch)
    codes = [
        client.post(
            "/api/app/mcp/oauth/register", json={"redirect_uris": [_REDIRECT], "client_name": "c"}
        ).status_code
        for _ in range(7)  # limit is 5 / 60s per IP
    ]
    assert codes[:5] == [201] * 5
    assert 429 in codes[5:]  # excess registrations are throttled


def test_oauth_disabled_without_issuer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_MCP_ISSUER_URL", raising=False)
    app = create_app(tmp_path, static_dir=False)
    app.state.app_data_dir = tmp_path / "app"
    app.state.session_secret = "s"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/.well-known/oauth-authorization-server").status_code == 503
