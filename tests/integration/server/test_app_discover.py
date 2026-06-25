"""Integration tests for personalized discovery (#1098).

GET /api/app/clusters (interests picker) and GET /api/app/discover (flag-gated ranking):
- flag OFF (default) → recency, identical to the catalog;
- flag ON + signed-in user with interests → significance × interest-affinity re-ranking.
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


def _write_episode(
    root: Path,
    *,
    stem: str,
    episode_id: str,
    topics: list[tuple[str, str]],
    published: str,
    with_gi: bool = False,
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show", "url": "https://pod.example/feed.xml"},
        "episode": {
            "episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "published_date": published,
            "duration_seconds": 1000,
        },
        "summary": {"title": "Sum", "bullets": ["a"]},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello", encoding="utf-8")
    nodes = [{"id": tid, "type": "Topic", "properties": {"label": label}} for tid, label in topics]
    (root / "metadata" / f"{stem}.kg.json").write_text(
        json.dumps({"episode_id": episode_id, "nodes": nodes}), encoding="utf-8"
    )
    if with_gi:
        gi = {"episode_id": episode_id, "nodes": [], "edges": []}
        (root / "metadata" / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")


def _corpus(root: Path) -> None:
    # epOld is older but about AI; epNew is newer but about Health.
    _write_episode(
        root,
        stem="0001-old",
        episode_id="old",
        topics=[("topic:ai", "AI")],
        published="2024-01-01T00:00:00",
        with_gi=True,
    )
    _write_episode(
        root,
        stem="0002-new",
        episode_id="new",
        topics=[("topic:health", "Health")],
        published="2024-06-01T00:00:00",
    )
    (root / "search").mkdir(parents=True, exist_ok=True)
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ai",
                "canonical_label": "AI",
                "member_count": 3,
                "members": [{"topic_id": "topic:ai", "label": "AI"}],
            },
            {
                "graph_compound_parent_id": "tc:health",
                "canonical_label": "Health",
                "member_count": 1,
                "members": [{"topic_id": "topic:health", "label": "Health"}],
            },
        ]
    }
    (root / "search" / "topic_clusters.json").write_text(json.dumps(payload), encoding="utf-8")


def _client(root: Path, *, personalized: bool) -> TestClient:
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = root / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    app.state.personalized_ranking = personalized
    return TestClient(app)


def _sign_in(client: TestClient, root: Path, interests: list[str]) -> None:
    data_dir = root / "appdata"
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    app_user_state.set_interests(data_dir, user.user_id, interests)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)


def test_clusters_endpoint_returns_top_by_prevalence(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = (
        _client(tmp_path, personalized=False).get("/api/app/clusters", params={"limit": 5}).json()
    )
    ids = [c["id"] for c in body["items"]]
    assert ids == ["tc:ai", "tc:health"]  # ranked by member_count desc
    assert body["items"][0] == {"id": "tc:ai", "label": "AI", "size": 3}


def test_discover_recency_when_flag_off(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=False)
    _sign_in(client, tmp_path, ["tc:ai"])  # interests present but flag off → still recency
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode new", "Episode old"]  # newest-first


def test_discover_personalizes_when_flag_on_and_interests_set(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["tc:ai"])
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    # epOld (about AI, the user's interest) now leads despite being older.
    assert titles == ["Episode old", "Episode new"]


def test_discover_recency_when_flag_on_but_anonymous(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)  # no sign-in → no interests → recency
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode new", "Episode old"]
