"""`POST /api/app/me/moments-heard` and the stats built on it (operator 2026-10-12).

A short listen saves no position and accrues no listening time, so until this route it left no
trace and stats could say nothing about it. Locks: a clip and a "keep" are recorded, the body is
checked at the edge, and both show up in `/me/stats` and `/me/recap`.
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


def _client(tmp_path: Path) -> TestClient:
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "appdata"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="m1", email="m@x.com", name="M")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def test_a_heard_clip_and_a_keep_reach_stats_and_the_recap(tmp_path: Path) -> None:
    client = _client(tmp_path)
    now = int(time.time())
    clip = {
        "kind": "clip",
        "slug": "ep1",
        "insight_id": "i1",
        "seconds": 42.5,
        "subject_id": "topic:risk",
        "subject_kind": "topic",
        "subject_label": "risk management",
        "client_ts": now,
    }
    assert client.post("/api/app/me/moments-heard", json=clip).status_code == 204
    keep = {"kind": "keep", "slug": "ep1", "client_ts": now}
    assert client.post("/api/app/me/moments-heard", json=keep).status_code == 204

    stats = client.get("/api/app/me/stats").json()
    assert (stats["moments_heard_total"], stats["moments_seconds_total"]) == (1, 42.5)
    recap = client.get("/api/app/me/recap", params={"window": "week"}).json()
    assert (recap["moments_heard"], recap["moment_episodes"], recap["moment_to_full"]) == (1, 1, 1)
    assert recap["subjects"][0]["label"] == "risk management"
    assert recap["subjects"][0]["kind"] == "topic"


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "played", "slug": "ep1"},  # not a kind we record
        {"kind": "clip", "slug": ""},
        {"kind": "clip", "slug": "ep1", "seconds": -1},
        {"kind": "clip", "slug": "ep1", "subject_kind": "show"},
    ],
)
def test_a_malformed_event_is_refused_at_the_edge(tmp_path: Path, body: dict) -> None:
    assert _client(tmp_path).post("/api/app/me/moments-heard", json=body).status_code == 422


def test_a_highlight_keeps_where_it_was_saved(tmp_path: Path) -> None:
    client = _client(tmp_path)
    body = {"episode_slug": "ep1", "kind": "moment", "start_ms": 1000, "source": "moment"}
    saved = client.post("/api/app/highlights", json=body)
    assert saved.status_code == 201
    assert saved.json()["source"] == "moment"
    assert client.get("/api/app/me/stats").json()["capture_from_moment"] == 1


def test_an_episode_says_which_of_its_moments_were_heard(tmp_path: Path) -> None:
    """Operator 2026-10-14: the player ticks the moments heard to their end, ever. Only that
    episode's, only clips (not "Keep listening"), each once."""
    client = _client(tmp_path)
    for body in (
        {"kind": "clip", "slug": "ep1", "insight_id": "b", "seconds": 6},
        {"kind": "clip", "slug": "ep1", "insight_id": "a", "seconds": 6},
        {"kind": "clip", "slug": "ep1", "insight_id": "a", "seconds": 6},
        {"kind": "clip", "slug": "ep2", "insight_id": "c", "seconds": 6},
        {"kind": "keep", "slug": "ep1"},
    ):
        assert client.post("/api/app/me/moments-heard", json=body).status_code == 204
    got = client.get("/api/app/me/moments-heard", params={"slug": "ep1"}).json()
    assert got == {"slug": "ep1", "insight_ids": ["a", "b"]}
    assert (
        client.get("/api/app/me/moments-heard", params={"slug": "ep3"}).json()["insight_ids"] == []
    )
    assert client.get("/api/app/me/moments-heard").status_code == 422
