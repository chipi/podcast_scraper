"""Integration tests for ``POST /api/app/graph-events`` (graph analytics ingest).

The signed-in → per-user routing (``user.user_id if user else "anon"``) is a trivial branch over
``get_optional_user``; the anon path here plus ``test_app_graph_telemetry`` (record under an
explicit user id) cover the substance without minting a session cookie.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server import app_graph_telemetry, app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user, set_role

pytestmark = [pytest.mark.integration]


def _client(root: Path) -> TestClient:
    app = create_app(root, static_dir=False)
    app.state.app_data_dir = root / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    return TestClient(app)


def test_graph_events_anonymous_logged_under_anon(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/api/app/graph-events",
        json={
            "events": [
                {"action": "node_tap", "id": "topic:a"},
                {"action": "redraw", "nodes": 10, "edges": 12},
            ]
        },
    )
    assert resp.status_code == 204
    events = app_graph_telemetry.read_events(tmp_path / "appdata", "anon")
    assert [e["action"] for e in events] == ["node_tap", "redraw"]
    assert all("ts" in e for e in events)  # server-stamped when the client omits it
    assert events[1]["nodes"] == 10  # size payload preserved


def test_graph_events_empty_is_noop_204(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post("/api/app/graph-events", json={"events": []})
    assert resp.status_code == 204
    assert app_graph_telemetry.read_events(tmp_path / "appdata", "anon") == []


def test_graph_events_summary_requires_admin(tmp_path: Path) -> None:
    # Anonymous → the admin gate rejects (no cookie minted, so no secret literal needed here).
    client = _client(tmp_path)
    assert client.get("/api/app/graph-events/summary").status_code in (401, 403)


def _admin_client(root: Path) -> TestClient:
    """A TestClient with an admin session cookie (exercises the admin-gated GETs)."""
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = root / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    data_dir = root / "appdata"
    user = get_or_create_user(data_dir, provider="stub", subject="admin", email="a@x.com", name="A")
    set_role(data_dir, user.user_id, "admin")
    signed = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, signed)
    return client


def test_graph_events_admin_summary_sessions_and_timeline(tmp_path: Path) -> None:
    client = _admin_client(tmp_path)
    client.post(
        "/api/app/graph-events",
        json={
            "events": [
                {
                    "action": "graph_recenter",
                    "target_id": "topic:a",
                    "session_id": "s1",
                    "ts": 1000,
                },
                {"action": "graph_rail_nav", "to_id": "person:b", "session_id": "s1", "ts": 1001},
                {
                    "action": "graph_redraw",
                    "nodes": 12,
                    "edges": 20,
                    "session_id": "s1",
                    "ts": 1002,
                },
            ]
        },
    )
    client.post(
        "/api/app/graph-events",
        json={
            "events": [
                {
                    "action": "graph_node_tap",
                    "id": "topic:c",
                    "kind": "topic",
                    "session_id": "s2",
                    "ts": 2000,
                },
                {"action": "graph_broke", "reason": "stuck", "session_id": "s2", "ts": 2001},
            ]
        },
    )

    summary = client.get("/api/app/graph-events/summary")
    assert summary.status_code == 200
    body = summary.json()
    assert body["by_action"]["graph_rail_nav"] == 1
    assert body["breakage"]["count"] == 1
    assert "size" in body and body["users"] >= 1

    sessions = client.get("/api/app/graph-events/sessions")
    assert sessions.status_code == 200
    assert {"s1", "s2"} <= {s["session_id"] for s in sessions.json()["sessions"]}

    timeline = client.get("/api/app/graph-events/session/s1")
    assert timeline.status_code == 200
    tl = timeline.json()
    assert tl["session_id"] == "s1"
    assert [e["action"] for e in tl["events"]] == [
        "graph_recenter",
        "graph_rail_nav",
        "graph_redraw",
    ]
