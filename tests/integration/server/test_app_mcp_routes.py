"""Route tests for MCP token management + the internal verify seam (RFC-112 slice 1, #1471)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user, set_mcp_access

_TOKEN = "internal-mcp-tok"


def _app(tmp_path: Path, *, mcp_access: bool, internal_token: str = _TOKEN):
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "app"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    app.state.internal_mcp_token = internal_token
    user = get_or_create_user(data_dir, provider="google", subject="s", email="u@g.com", name="U")
    if mcp_access:
        set_mcp_access(data_dir, user.user_id, True)
    client = TestClient(app)
    tok = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, tok)
    return client, data_dir, user.user_id


def test_management_requires_entitlement(tmp_path: Path) -> None:
    client, _, _ = _app(tmp_path, mcp_access=False)
    assert client.get("/api/app/mcp/tokens").status_code == 403


def test_create_list_revoke(tmp_path: Path) -> None:
    client, _, _ = _app(tmp_path, mcp_access=True)
    assert client.get("/api/app/mcp/tokens").json()["items"] == []
    created = client.post("/api/app/mcp/tokens", json={"label": "Claude Code"})
    assert created.status_code == 201
    body = created.json()
    assert body["token"].startswith("clp_mcp_")  # plaintext shown once
    tid = body["meta"]["id"]

    listed = client.get("/api/app/mcp/tokens").json()["items"]
    assert [t["id"] for t in listed] == [tid]
    assert "token" not in listed[0] and "hash" not in listed[0]

    remaining = client.delete(f"/api/app/mcp/tokens/{tid}")
    assert remaining.status_code == 200 and remaining.json()["items"] == []


def test_config_requires_entitlement(tmp_path: Path) -> None:
    client, _, _ = _app(tmp_path, mcp_access=False)
    assert client.get("/api/app/mcp/config").status_code == 403


def test_config_reports_connector_and_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_MCP_RESOURCE_URL", "https://mcp.example.com/")
    monkeypatch.setenv("APP_MCP_ISSUER_URL", "https://app.example.com")
    client, _, _ = _app(tmp_path, mcp_access=True)
    cfg = client.get("/api/app/mcp/config").json()
    # origin (trailing slash trimmed) + the /mcp endpoint path = the URL a client pastes.
    assert cfg["connector_url"] == "https://mcp.example.com/mcp"
    assert cfg["authorization_server"] == "https://app.example.com"
    assert cfg["oauth_enabled"] is True


def test_config_null_when_unconfigured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_MCP_RESOURCE_URL", raising=False)
    monkeypatch.delenv("APP_MCP_ISSUER_URL", raising=False)
    client, _, _ = _app(tmp_path, mcp_access=True)
    cfg = client.get("/api/app/mcp/config").json()
    assert cfg["connector_url"] is None
    assert cfg["oauth_enabled"] is False


def test_connections_list_and_revoke(tmp_path: Path) -> None:
    from podcast_scraper.server import app_oauth_server as oa

    client, data_dir, uid = _app(tmp_path, mcp_access=True)
    # Seed a connected OAuth client (a remembered consent) directly in the store.
    reg = oa.register_client(
        data_dir, redirect_uris=["https://claude.ai/cb"], client_name="claude.ai"
    )
    cid = reg["client_id"]
    oa.remember_consent(data_dir, user_id=uid, client_id=cid, scope="mcp:read")

    listed = client.get("/api/app/mcp/connections").json()["items"]
    assert [c["client_id"] for c in listed] == [cid]
    assert listed[0]["client_name"] == "claude.ai"

    remaining = client.delete(f"/api/app/mcp/connections/{cid}")
    assert remaining.status_code == 200 and remaining.json()["items"] == []
    # consent is forgotten → a later authorize would re-prompt (not silent)
    assert oa.has_consent(data_dir, user_id=uid, client_id=cid, scope="mcp:read") is False


def test_connections_require_entitlement(tmp_path: Path) -> None:
    client, _, _ = _app(tmp_path, mcp_access=False)
    assert client.get("/api/app/mcp/connections").status_code == 403


def test_internal_verify_flow(tmp_path: Path) -> None:
    client, data_dir, uid = _app(tmp_path, mcp_access=True)
    token = client.post("/api/app/mcp/tokens", json={"label": "a"}).json()["token"]
    h = {"X-Internal-Token": _TOKEN}

    ok = client.post("/internal/mcp/verify", json={"token": token}, headers=h)
    assert ok.status_code == 200
    assert ok.json() == {
        "authenticated": True,
        "user_id": uid,
        "mcp_access": True,
        "scope": "mcp:read",
        "aud": "",  # a PAT carries no audience
        "role": "listener",  # surfaced for rank-scoped servers (obs MCP is admin-only, #56)
    }

    bad = client.post("/internal/mcp/verify", json={"token": "clp_mcp_nope"}, headers=h)
    assert bad.json()["authenticated"] is False


def test_internal_verify_reflects_revoked_entitlement(tmp_path: Path) -> None:
    client, data_dir, uid = _app(tmp_path, mcp_access=True)
    token = client.post("/api/app/mcp/tokens", json={"label": "a"}).json()["token"]
    set_mcp_access(data_dir, uid, False)  # entitlement revoked after the token was minted
    h = {"X-Internal-Token": _TOKEN}
    resp = client.post("/internal/mcp/verify", json={"token": token}, headers=h)
    assert resp.json()["authenticated"] is False  # denied at connect time


def test_internal_verify_auth_gate(tmp_path: Path) -> None:
    client, _, _ = _app(tmp_path, mcp_access=True)
    # wrong internal token → 401
    assert (
        client.post(
            "/internal/mcp/verify", json={"token": "x"}, headers={"X-Internal-Token": "wrong"}
        ).status_code
        == 401
    )
    # unconfigured → 503
    client2, _, _ = _app(tmp_path, mcp_access=True, internal_token="")
    assert (
        client2.post(
            "/internal/mcp/verify", json={"token": "x"}, headers={"X-Internal-Token": "x"}
        ).status_code
        == 503
    )


def _admin_client(tmp_path: Path):
    from podcast_scraper.server import app_roles

    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "app"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    admin = get_or_create_user(
        data_dir,
        provider="google",
        subject="admin",
        email="a@g.com",
        name="A",
        role=app_roles.ADMIN,
    )
    target = get_or_create_user(data_dir, provider="google", subject="t", email="t@g.com", name="T")
    client = TestClient(app)
    tok = app_sessions.sign({"user_id": admin.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, tok)
    return client, target.user_id


def test_admin_grants_mcp_access(tmp_path: Path) -> None:
    client, target = _admin_client(tmp_path)
    resp = client.patch(f"/api/app/admin/users/{target}", json={"mcp_access": True})
    assert resp.status_code == 200 and resp.json()["mcp_access"] is True
    # revoke
    resp2 = client.patch(f"/api/app/admin/users/{target}", json={"mcp_access": False})
    assert resp2.json()["mcp_access"] is False
