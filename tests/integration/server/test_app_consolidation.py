"""Integration tests for the P3 Consolidation resurfacing routes (#1123).

Pacing settings need only a signed-in user + the app data dir. The resurfacing
FEED now also needs a corpus root: since #38 it resolves each due highlight's
graph refs so that it withholds exactly what Your Week and the digest email
withhold. Before that it answered from per-user files alone, and could list
captures the other two surfaces silently dropped.
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


def _signed_in_client(root: Path) -> TestClient:
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = root / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    user = get_or_create_user(
        root / "appdata", provider="stub", subject="u", email="u@x.com", name="U"
    )
    signed = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, signed)
    return client


def test_resurfacing_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/api/app/resurfacing").status_code in (401, 403)


def test_resurfacing_settings_pause_and_empty_feed(tmp_path: Path) -> None:
    client = _signed_in_client(tmp_path)

    # Default pacing settings.
    got = client.get("/api/app/resurfacing/settings")
    assert got.status_code == 200
    settings = got.json()
    assert settings["paused"] is False

    # Pause via PUT (echo the current settings back with the flag flipped so the
    # body always satisfies the schema).
    settings["paused"] = True
    put = client.put("/api/app/resurfacing/settings", json=settings)
    assert put.status_code == 200
    assert put.json()["paused"] is True

    # The feed honours the pause and, with no captured highlights, is empty.
    feed = client.get("/api/app/resurfacing")
    assert feed.status_code == 200
    body = feed.json()
    assert body["paused"] is True
    assert body["items"] == []

    # Marking an absent highlight is a 404, NOT the "idempotent 204 no-op" this used to assert.
    #
    # That phrasing dressed up a gap as a feature: the route wrote whatever key it was handed, with
    # no existence or ownership check, so resurfacing.json accumulated an entry per call, for ever
    # (#39). Nothing read those keys back — select_due iterates highlights — so it was unbounded
    # growth rather than a wrong answer, and a 204 made it look deliberate. It stopped being merely
    # untidy when #35 wired the mark to a `?revisit=` query parameter, i.e. to any string a user can
    # type into their address bar.
    assert client.post("/api/app/resurfacing/h_absent/surfaced").status_code == 404
