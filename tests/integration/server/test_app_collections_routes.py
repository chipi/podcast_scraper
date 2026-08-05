"""Route tests for the collections surface ``/api/app/collections`` (#1417)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions, app_user_state
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


def test_create_list_add_detail_delete(tmp_path: Path) -> None:
    client, data_dir, uid = _authed(tmp_path)
    # a highlight to add
    app_user_state.add_highlight(
        data_dir, uid, {"id": "h1", "episode_slug": "ep", "kind": "span", "created_at": 1}
    )

    assert client.get("/api/app/collections").json()["items"] == []
    cid = client.post("/api/app/collections", json={"name": "AI takes"}).json()["id"]

    added = client.post(f"/api/app/collections/{cid}/items", json={"highlight_id": "h1"})
    assert added.status_code == 200 and added.json()["count"] == 1

    detail = client.get(f"/api/app/collections/{cid}").json()
    assert detail["collection"]["name"] == "AI takes"
    assert [h["id"] for h in detail["highlights"]] == ["h1"]

    # delete the collection
    remaining = client.delete(f"/api/app/collections/{cid}")
    assert remaining.status_code == 200 and remaining.json()["items"] == []


def test_add_item_to_unknown_collection_404(tmp_path: Path) -> None:
    client, _, _ = _authed(tmp_path)
    resp = client.post("/api/app/collections/col_missing/items", json={"highlight_id": "h1"})
    assert resp.status_code == 404


def test_detail_drops_deleted_highlight(tmp_path: Path) -> None:
    client, data_dir, uid = _authed(tmp_path)
    cid = client.post("/api/app/collections", json={"name": "c"}).json()["id"]
    client.post(f"/api/app/collections/{cid}/items", json={"highlight_id": "ghost"})
    # 'ghost' has no backing highlight → hydrated detail drops it
    assert client.get(f"/api/app/collections/{cid}").json()["highlights"] == []


def test_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.session_secret = "s"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/api/app/collections").status_code in (401, 403)
