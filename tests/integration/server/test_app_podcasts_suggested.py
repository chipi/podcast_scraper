"""GET /api/app/podcasts/suggested — the guided start's shows (operator 2026-10-08).

Active in the last month first, then the most loved across listeners; shows the listener already
follows are left out. The clock is pinned with ``APP_TRENDING_NOW``, as Trends pins it.
"""

from __future__ import annotations

import json
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


def _episode(root: Path, *, stem: str, feed: str, published: str) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {
            "feed_id": feed,
            "title": f"Show {feed}",
            "url": f"https://pod.example/{feed}.xml",
        },
        "episode": {
            "episode_id": stem,
            "title": f"Episode {stem}",
            "published_date": published,
            "duration_seconds": 1000,
        },
        "summary": {"title": "Sum", "bullets": ["a"]},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello", encoding="utf-8")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("APP_TRENDING_NOW", "2026-07-20T00:00:00Z")
    # QUIET stopped a year ago but is the most followed; LOVED and FRESH are both active.
    _episode(tmp_path, stem="q1", feed="QUIET", published="2025-06-01T00:00:00")
    _episode(tmp_path, stem="l1", feed="LOVED", published="2026-07-01T00:00:00")
    _episode(tmp_path, stem="f1", feed="FRESH", published="2026-07-18T00:00:00")
    _episode(tmp_path, stem="m1", feed="MINE", published="2026-07-19T00:00:00")
    data_dir = tmp_path / "appdata"
    for uid in ("o1", "o2", "o3"):
        app_user_state.add_subscription(data_dir, uid, {"feed_id": "QUIET"})
    app_user_state.add_subscription(data_dir, "o1", {"feed_id": "LOVED"})
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    app_user_state.add_subscription(data_dir, user.user_id, {"feed_id": "MINE"})
    c = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    c.cookies.set(app_sessions.SESSION_COOKIE, token)
    return c


def test_active_and_loved_first_followed_left_out(client: TestClient) -> None:
    body = client.get("/api/app/podcasts/suggested").json()
    assert [it["feed_id"] for it in body["items"]] == ["LOVED", "FRESH", "QUIET"]
    # The same tile shape as /podcasts, so the client renders it with ShowTile unchanged.
    assert body["items"][0]["title"] == "Show LOVED"
    assert body["items"][0]["episode_count"] == 1


def test_limit(client: TestClient) -> None:
    assert len(client.get("/api/app/podcasts/suggested?limit=1").json()["items"]) == 1


def test_signed_out_is_refused(client: TestClient) -> None:
    client.cookies.clear()
    assert client.get("/api/app/podcasts/suggested").status_code == 401
