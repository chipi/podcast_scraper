"""Integration tests for per-user state routes — playback, queue, library (#1065).

Auth is established by forging a signed session cookie (the secret is known in-test),
avoiding the full OAuth dance.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]


def _authed_client(tmp_path: Path) -> TestClient:
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "appdata"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def test_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    client = TestClient(app)
    assert client.get("/api/app/queue").status_code == 401
    assert client.get("/api/app/playback/ep").status_code == 401
    assert client.get("/api/app/library").status_code == 401


def test_playback_save_and_resume(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    assert client.get("/api/app/playback/ep").json()["position_seconds"] == 0.0
    put = client.put("/api/app/playback/ep", json={"position_seconds": 42.5})
    assert put.status_code == 200, put.text
    assert put.json()["position_seconds"] == 42.5
    assert client.get("/api/app/playback/ep").json()["position_seconds"] == 42.5


def test_queue_roundtrip(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    assert client.get("/api/app/queue").json()["items"] == []
    assert client.put("/api/app/queue", json={"items": ["a", "b"]}).status_code == 200
    assert client.get("/api/app/queue").json()["items"] == ["a", "b"]


def test_library_subscribe_list_unsubscribe(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    assert client.get("/api/app/library").json()["items"] == []
    added = client.post("/api/app/library", json={"feed_id": "f1", "title": "Show One"})
    assert [i["feed_id"] for i in added.json()["items"]] == ["f1"]
    client.post("/api/app/library", json={"feed_id": "f2"})
    listed = client.get("/api/app/library").json()["items"]
    assert {i["feed_id"] for i in listed} == {"f1", "f2"}
    removed = client.delete("/api/app/library/f1")
    assert [i["feed_id"] for i in removed.json()["items"]] == ["f2"]
