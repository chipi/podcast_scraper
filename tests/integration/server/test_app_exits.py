"""Integration tests for ``POST /api/app/app-exits`` (#2279) — native exit reasons to the log sink."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration, pytest.mark.app]


def _client(root: Path, signed_in: bool) -> TestClient:
    data_dir = root / "appdata"
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    if signed_in:
        user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
        token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
        client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def _exit_events(caplog: pytest.LogCaptureFixture) -> list[dict]:
    out = []
    for rec in caplog.records:
        try:
            payload = json.loads(rec.getMessage())
        except ValueError:
            continue
        if payload.get("event_type") == "app_exit":
            out.append(payload)
    return out


BODY = {
    "platform": "ios",
    "app_version": "1.0.0 (3)",
    "entries": [
        {
            "source": "metrickit",
            "reason": "bg_memory_pressure",
            "count": 4,
            "at": "2026-10-04T00:00:00Z",
        },
        {"source": "webview_terminated", "reason": "webcontent_terminated"},
    ],
}


def test_each_entry_becomes_one_app_exit_event(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    resp = _client(tmp_path, signed_in=True).post("/api/app/app-exits", json=BODY)
    assert resp.status_code == 204
    events = _exit_events(caplog)
    assert [(e["source"], e["reason"], e["count"]) for e in events] == [
        ("metrickit", "bg_memory_pressure", 4),
        ("webview_terminated", "webcontent_terminated", 1),
    ]
    assert all(e["platform"] == "ios" and e["signed_in"] is True for e in events)


def test_signed_out_is_accepted_and_carries_no_identity(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A signed-out app is evicted just the same; the event says only that nobody was signed in.
    caplog.set_level(logging.INFO)
    resp = _client(tmp_path, signed_in=False).post("/api/app/app-exits", json=BODY)
    assert resp.status_code == 204
    events = _exit_events(caplog)
    assert len(events) == 2
    assert all(e["signed_in"] is False and "user_id" not in e for e in events)


@pytest.mark.parametrize(
    "bad",
    [
        {**BODY, "platform": "windows"},
        {**BODY, "entries": [{"source": "metrickit", "reason": "Bad Reason!"}]},
        {**BODY, "entries": [{"source": "rumour", "reason": "x"}]},
        {**BODY, "entries": [{"source": "metrickit", "reason": "x", "count": -1}]},
        {**BODY, "entries": [{"source": "metrickit", "reason": "x"}] * 51},
    ],
)
def test_rejects_anything_outside_the_contract(tmp_path: Path, bad: dict) -> None:
    assert (
        _client(tmp_path, signed_in=False).post("/api/app/app-exits", json=bad).status_code == 422
    )
