"""Integration tests for per-user state routes — playback, queue, library (#1065).

Auth is established by forging a signed session cookie (the secret is known in-test),
avoiding the full OAuth dance.
"""

from __future__ import annotations

import json
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
    assert client.get("/api/app/me/stats").status_code == 401
    assert client.post("/api/app/listen/ep").status_code == 401


def test_listen_then_my_stats(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    # Empty to start.
    empty = client.get("/api/app/me/stats")
    assert empty.status_code == 200, empty.text
    assert empty.json()["episodes"] == 0 and empty.json()["day_streak"] == 0
    # Record an open (no corpus in-test → feed_id resolves to None, but the event still logs).
    assert client.post("/api/app/listen/ep1").status_code == 204
    stats = client.get("/api/app/me/stats").json()
    assert stats["episodes"] == 1
    assert stats["day_streak"] == 1
    assert stats["active_days"] == 1
    assert stats["daily"][-1]["count"] == 1  # today's bucket


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


def test_interests_roundtrip_and_dedup(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    assert client.get("/api/app/interests").json()["items"] == []
    put = client.put("/api/app/interests", json={"items": ["tc:ai", "tc:health", "tc:ai", ""]})
    assert put.status_code == 200, put.text
    # Dedup + blank-drop, order preserved.
    assert put.json()["items"] == ["tc:ai", "tc:health"]
    assert client.get("/api/app/interests").json()["items"] == ["tc:ai", "tc:health"]


def test_follow_unfollow_interest_token(tmp_path: Path) -> None:
    client = _authed_client(tmp_path)
    # Follow a topic and a person from an entity card (URL-encoded `:`); idempotent.
    client.post("/api/app/interests/topic%3Aai")
    body = client.post("/api/app/interests/person%3Ajane").json()
    assert body["items"] == ["topic:ai", "person:jane"]
    assert client.post("/api/app/interests/topic%3Aai").json()["items"] == [
        "topic:ai",
        "person:jane",
    ]
    # Unfollow.
    assert client.delete("/api/app/interests/topic%3Aai").json()["items"] == ["person:jane"]


def _write_kg_episode(root: Path, *, stem: str, episode_id: str) -> None:
    """Minimal episode so favorite-episode hydration resolves a real slug."""
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "f", "title": "Show", "url": "https://p.example/f.xml"},
        "episode": {"episode_id": episode_id, "title": "Hello", "published_date": "2024-01-01"},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hi", encoding="utf-8")


def test_favorites_roundtrip_grouped_and_hydrated(tmp_path: Path) -> None:
    from podcast_scraper.server.app_slugs import slug_for_row
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    _write_kg_episode(tmp_path, stem="0001-hello", episode_id="ep1")
    slug = slug_for_row(build_catalog_rows_cumulative(tmp_path)[0])
    client = _authed_client(tmp_path)

    assert client.get("/api/app/favorites").json() == {"episodes": [], "insights": []}
    # save an episode (hydrated from catalog) + an insight (from snapshot)
    client.put("/api/app/favorites", json={"kind": "episode", "ref": slug, "label": "Hello"})
    body = client.put(
        "/api/app/favorites",
        json={
            "kind": "insight",
            "ref": f"{slug}#i1",
            "label": "A claim",
            "slug": slug,
            "start_ms": 5000,
        },
    ).json()
    assert [e["slug"] for e in body["episodes"]] == [slug]
    assert body["insights"][0] == {
        "ref": f"{slug}#i1",
        "text": "A claim",
        "episode_slug": slug,
        "podcast_title": None,
        "start_ms": 5000,
    }
    # remove the episode (url-encoded ref); insight remains
    after = client.delete(f"/api/app/favorites/episode/{slug}").json()
    assert after["episodes"] == [] and len(after["insights"]) == 1


def test_favorites_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    client = TestClient(app)
    assert client.get("/api/app/favorites").status_code == 401
    assert client.put("/api/app/favorites", json={"kind": "episode", "ref": "x"}).status_code == 401


def test_interests_requires_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path / "appdata"
    client = TestClient(app)
    assert client.get("/api/app/interests").status_code == 401
    assert client.put("/api/app/interests", json={"items": ["tc:x"]}).status_code == 401


def test_disabled_user_blocked_on_state_route(tmp_path: Path) -> None:
    from podcast_scraper.server.app_user_store import set_disabled, user_id_for

    client = _authed_client(tmp_path)
    assert client.get("/api/app/queue").status_code == 200  # works while enabled
    set_disabled(tmp_path / "appdata", user_id_for("stub", "s1"), True)
    assert client.get("/api/app/queue").status_code == 401  # disabled → locked out


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
