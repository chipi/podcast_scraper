"""GET /api/app/whats-new — Home's What's new from the listener's own world (operator 2026-10-07).

Newest first, from the shows they follow plus episodes carrying a followed topic / person / theme /
storyline. With nothing followed, or nothing matching, it is the newest across every show and
``scope`` says ``all`` so the section can label itself honestly.
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
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

pytestmark = [pytest.mark.integration]


def _episode(root: Path, *, stem: str, eid: str, feed: str, published: str) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {
            "feed_id": feed,
            "title": f"Show {feed}",
            "url": f"https://pod.example/{feed}.xml",
        },
        "episode": {
            "episode_id": eid,
            "title": f"Episode {eid}",
            "published_date": published,
            "duration_seconds": 1000,
        },
        "summary": {"title": "Sum", "bullets": ["a"]},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello", encoding="utf-8")


def _corpus(root: Path) -> None:
    # Newest last in time: c (feed B, about AI) < b (feed A) < a (feed B, nothing followed-able).
    _episode(root, stem="0001-c", eid="c", feed="B", published="2024-01-01T00:00:00")
    _episode(root, stem="0002-b", eid="b", feed="A", published="2024-03-01T00:00:00")
    _episode(root, stem="0003-a", eid="a", feed="B", published="2024-06-01T00:00:00")
    rel = {r.episode_id: r.metadata_relative_path for r in build_catalog_rows_cumulative(root)}
    (root / "search").mkdir(parents=True, exist_ok=True)
    # The two-tier indexer's sidecar: one KG row per (token, episode).
    sidecar = {
        "1": {
            "doc_type": "kg_topic",
            "source_id": "topic:ai",
            "source_metadata_relative_path": rel["c"],
        }
    }
    (root / "search" / "metadata.json").write_text(json.dumps(sidecar), encoding="utf-8")
    themes = {
        "clusters": [
            {
                "graph_compound_parent_id": "tc:machines",
                "canonical_label": "Machines",
                "member_count": 2,
                "members": [{"topic_id": "topic:ai", "label": "AI"}],
            }
        ]
    }
    (root / "search" / "topic_clusters.json").write_text(json.dumps(themes), encoding="utf-8")


def _client(root: Path, *, interests: list[str], follows: list[str]) -> TestClient:
    data_dir = root / "appdata"
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    app_user_state.set_interests(data_dir, user.user_id, interests)
    for feed in follows:
        app_user_state.add_subscription(data_dir, user.user_id, {"feed_id": feed})
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def _ids(body: dict) -> list[str]:
    return [it["title"].removeprefix("Episode ") for it in body["items"]]


def test_nothing_followed_is_every_show_and_says_so(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = _client(tmp_path, interests=[], follows=[]).get("/api/app/whats-new").json()
    assert body["scope"] == "all"
    assert _ids(body) == ["a", "b", "c"]


def test_a_followed_show_brings_only_its_episodes_newest_first(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = _client(tmp_path, interests=[], follows=["B"]).get("/api/app/whats-new").json()
    assert body["scope"] == "yours"
    assert _ids(body) == ["a", "c"]


def test_a_followed_topic_brings_the_episodes_that_carry_it(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = _client(tmp_path, interests=["topic:ai"], follows=[]).get("/api/app/whats-new").json()
    assert body["scope"] == "yours"
    assert _ids(body) == ["c"]


def test_a_followed_theme_reaches_episodes_through_its_member_topics(tmp_path: Path) -> None:
    # `tc:` is not an episode token; it has to expand to its topics or a theme follow matches nothing.
    _corpus(tmp_path)
    body = _client(tmp_path, interests=["tc:machines"], follows=[]).get("/api/app/whats-new").json()
    assert body["scope"] == "yours"
    assert _ids(body) == ["c"]


def test_follows_and_topics_combine_without_duplicates(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = _client(tmp_path, interests=["topic:ai"], follows=["A"]).get("/api/app/whats-new").json()
    assert _ids(body) == ["b", "c"]


def test_follows_that_match_nothing_fall_back_to_every_show(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = (
        _client(tmp_path, interests=["topic:cooking"], follows=["Z"])
        .get("/api/app/whats-new")
        .json()
    )
    assert body["scope"] == "all"
    assert _ids(body) == ["a", "b", "c"]


def test_the_limit_is_honoured(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = _client(tmp_path, interests=[], follows=[]).get("/api/app/whats-new?limit=2").json()
    assert _ids(body) == ["a", "b"]


def test_requires_a_signed_in_listener(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, interests=[], follows=[])
    client.cookies.clear()
    assert client.get("/api/app/whats-new").status_code == 401
