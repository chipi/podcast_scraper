"""Integration tests for the USERPREFS-1 preferences routes.

Auth is established by forging a signed session cookie (same as the sibling
test_app_user_state_routes suite), avoiding the full OAuth dance.
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


def test_preferences_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    client = TestClient(app)
    assert client.get("/api/app/preferences").status_code == 401
    assert client.put("/api/app/preferences", json={"preferences": {}}).status_code == 401
    assert client.patch("/api/app/preferences", json={"preferences": {}}).status_code == 401


def test_preferences_get_default_is_empty(tmp_path: Path) -> None:
    """First read for a new user returns an empty payload — the client falls back to
    its local defaults for every unset key."""
    client = _authed_client(tmp_path)
    r = client.get("/api/app/preferences")
    assert r.status_code == 200, r.text
    assert r.json() == {"preferences": {}}


def test_preferences_put_replaces_entire_payload(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    body = {"preferences": {"theme": "dark", "graphLenses": {"velocityHalo": True}}}
    put = client.put("/api/app/preferences", json=body)
    assert put.status_code == 200, put.text
    assert put.json()["preferences"] == body["preferences"]
    # A subsequent PUT with a different shape wipes the old keys.
    put2 = client.put("/api/app/preferences", json={"preferences": {"theme": "light"}})
    assert put2.json()["preferences"] == {"theme": "light"}


def test_preferences_patch_shallow_merges(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    client.put("/api/app/preferences", json={"preferences": {"theme": "dark", "corpus": "/a"}})
    patch = client.patch("/api/app/preferences", json={"preferences": {"theme": "light"}})
    assert patch.status_code == 200, patch.text
    # theme updated, corpus preserved.
    assert patch.json()["preferences"] == {"theme": "light", "corpus": "/a"}


def test_preferences_patch_with_null_deletes_key(tmp_path: Path) -> None:
    """USERPREFS-1 delete semantics: PATCH with a `null` value removes the key.

    Lets the client reset a specific preference to its local default without
    touching the rest of the payload — cleaner than PUT'ing the whole minus-one.
    """
    client = _authed_client(tmp_path)
    client.put(
        "/api/app/preferences",
        json={"preferences": {"theme": "dark", "corpus": "/a", "leftPanel": True}},
    )
    patch = client.patch("/api/app/preferences", json={"preferences": {"corpus": None}})
    assert patch.status_code == 200
    assert "corpus" not in patch.json()["preferences"]
    assert patch.json()["preferences"] == {"theme": "dark", "leftPanel": True}


def test_preferences_survives_across_requests_per_user(tmp_path: Path) -> None:
    """The file lands under <data_dir>/users/<user_id>/preferences.json and is
    scoped to the signed-in user — a second request reads the same bytes."""
    client = _authed_client(tmp_path)
    client.patch("/api/app/preferences", json={"preferences": {"theme": "dark"}})
    # New client instance (same signed cookie / same user) sees the persisted state.
    client2 = _authed_client(tmp_path)
    # Different token but same user_id in tmp_path/appdata/users/… — but note:
    # get_or_create_user for provider=stub / subject=s1 returns the SAME user_id.
    r = client2.get("/api/app/preferences")
    assert r.json()["preferences"] == {"theme": "dark"}


def test_preferences_patch_stores_arbitrary_json_shapes(tmp_path: Path) -> None:
    """Server round-trips without interpretation — bool, string, number, nested
    object, array, all stored as-is. Client owns the shape."""
    client = _authed_client(tmp_path)
    payload = {
        "boolFlag": True,
        "stringSetting": "value",
        "numberSetting": 42,
        "nested": {"deep": {"key": [1, 2, 3]}},
        "array": [{"id": "a"}, {"id": "b"}],
    }
    r = client.patch("/api/app/preferences", json={"preferences": payload})
    assert r.status_code == 200
    assert r.json()["preferences"] == payload
