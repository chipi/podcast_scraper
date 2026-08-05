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


def test_internal_verify_flow(tmp_path: Path) -> None:
    client, data_dir, uid = _app(tmp_path, mcp_access=True)
    token = client.post("/api/app/mcp/tokens", json={"label": "a"}).json()["token"]
    h = {"X-Internal-Token": _TOKEN}

    ok = client.post("/internal/mcp/verify", json={"token": token}, headers=h)
    assert ok.status_code == 200
    assert ok.json() == {"authenticated": True, "user_id": uid, "mcp_access": True}

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
