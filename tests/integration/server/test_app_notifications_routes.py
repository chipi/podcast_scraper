"""Route tests for the in-app notification inbox ``/api/app/notifications`` (wave-I)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_notifications_store, app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]


def _authed(tmp_path: Path) -> tuple[TestClient, Path, str]:
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "appdata"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client, data_dir, user.user_id


def test_list_empty(tmp_path: Path) -> None:
    client, _, _ = _authed(tmp_path)
    body = client.get("/api/app/notifications").json()
    assert body == {"items": [], "unread": 0}


def test_list_returns_seeded_newest_first(tmp_path: Path) -> None:
    client, data_dir, uid = _authed(tmp_path)
    app_notifications_store.add_notification(data_dir, uid, ntype="product", title="Old", now=1000)
    app_notifications_store.add_notification(
        data_dir, uid, ntype="new_episodes", title="New", deep_link="/player/x", now=2000
    )
    body = client.get("/api/app/notifications").json()
    assert [i["title"] for i in body["items"]] == ["New", "Old"]
    assert body["unread"] == 2


def test_mark_one_read(tmp_path: Path) -> None:
    client, data_dir, uid = _authed(tmp_path)
    rec = app_notifications_store.add_notification(data_dir, uid, ntype="product", title="A")
    assert rec is not None
    resp = client.post(f"/api/app/notifications/{rec['id']}/read")
    assert resp.status_code == 200 and resp.json()["unread"] == 0


def test_mark_all_read(tmp_path: Path) -> None:
    client, data_dir, uid = _authed(tmp_path)
    app_notifications_store.add_notification(data_dir, uid, ntype="product", title="A")
    app_notifications_store.add_notification(data_dir, uid, ntype="product", title="B")
    resp = client.post("/api/app/notifications/read-all")
    assert resp.status_code == 200 and resp.json()["unread"] == 0


def test_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.session_secret = "test-secret"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/api/app/notifications").status_code in (401, 403)
