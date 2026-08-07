"""Route tests for the in-app "Your Week" surface ``/api/app/your-week`` (#1412).

The route mirrors ``app_digest_personal.assemble_digest_payload`` (the same rollup the email
sends) but is DECOUPLED from email consent — a user's own data is always visible in-app.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_comms_store, app_digest_personal, app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]


def _app(tmp_path: Path):
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    return app


def _authed(app, tmp_path: Path):
    data_dir = tmp_path / "appdata"
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client, user


def test_your_week_401_when_unauthenticated(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path))
    assert client.get("/api/app/your-week").status_code == 401


def test_your_week_empty_for_new_user(tmp_path: Path) -> None:
    client, _ = _authed(_app(tmp_path), tmp_path)
    resp = client.get("/api/app/your-week")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sections"] == []
    # "Aug 1 – 7" (same month) or "Jul 28 – Aug 3" (spanning) — day numbers, portable format.
    assert re.match(r"^[A-Z][a-z]{2} \d{1,2} – (?:[A-Z][a-z]{2} )?\d{1,2}$", body["period_label"])
    assert body["generated_at"].endswith("Z")


def test_your_week_decoupled_from_email_consent(tmp_path: Path, monkeypatch) -> None:
    """The in-app view returns content even when the email digest is OFF (consent-decoupled)."""
    client, user = _authed(_app(tmp_path), tmp_path)
    app_comms_store.set_comms(tmp_path / "appdata", user.user_id, digest={"enabled": False})
    payload = {"sections": [{"kind": "revisit", "items": [{"episode_slug": "x"}]}]}
    monkeypatch.setattr(app_digest_personal, "assemble_digest_payload", lambda *a, **k: payload)
    resp = client.get("/api/app/your-week")
    assert resp.status_code == 200
    assert resp.json()["sections"] == payload["sections"]


def test_your_week_passes_through_none_as_empty(tmp_path: Path, monkeypatch) -> None:
    client, _ = _authed(_app(tmp_path), tmp_path)
    monkeypatch.setattr(app_digest_personal, "assemble_digest_payload", lambda *a, **k: None)
    resp = client.get("/api/app/your-week")
    assert resp.status_code == 200 and resp.json()["sections"] == []
