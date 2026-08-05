"""Integration tests for the MCP OAuth flow through the routes (RFC-112 slice 3, #1471)."""

from __future__ import annotations

import base64
import hashlib
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user, set_mcp_access

_ISSUER = "https://app.example.com"
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
    assert verify.json() == {"authenticated": True, "user_id": uid, "mcp_access": True}


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


def test_oauth_disabled_without_issuer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_MCP_ISSUER_URL", raising=False)
    app = create_app(tmp_path, static_dir=False)
    app.state.app_data_dir = tmp_path / "app"
    app.state.session_secret = "s"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/.well-known/oauth-authorization-server").status_code == 503
