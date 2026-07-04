"""Integration tests for ``POST /api/app/graph-events`` (graph analytics ingest).

The signed-in → per-user routing (``user.user_id if user else "anon"``) is a trivial branch over
``get_optional_user``; the anon path here plus ``test_app_graph_telemetry`` (record under an
explicit user id) cover the substance without minting a session cookie.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server import app_graph_telemetry
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy

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
