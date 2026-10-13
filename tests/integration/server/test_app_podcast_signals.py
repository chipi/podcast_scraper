"""Integration test for the consumer show-signals route.

GET /api/app/podcasts/{feed_id}/signals — a listener-shaped projection over the same
feed-signals aggregation the operator Show rail uses (grounding dropped).
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


def _authed_client(root: Path) -> TestClient:
    data_dir = root / "_appdata"
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def _doc(feed_id: str, episode_id: str, title: str) -> dict:
    return {
        "feed": {"feed_id": feed_id, "title": "My Show"},
        "episode": {"episode_id": episode_id, "title": title, "published_date": "2024-03-10"},
        "summary": {"title": "Sum", "bullets": ["a", "b"]},
    }


def _topic(node_id: str, label: str) -> dict:
    return {"id": node_id, "type": "Topic", "properties": {"label": label}}


def _person(node_id: str, name: str, kind: str = "person") -> dict:
    return {
        "id": node_id,
        "type": "Entity",
        "properties": {"label": name, "name": name, "kind": kind},
    }


def _write_ep(meta: Path, stem: str, feed_id: str, episode_id: str, nodes: list[dict]) -> None:
    (meta / f"{stem}.metadata.json").write_text(
        json.dumps(_doc(feed_id, episode_id, stem)), "utf-8"
    )
    (meta / f"{stem}.kg.json").write_text(json.dumps({"nodes": nodes}), "utf-8")


def _enrich(root: Path, enricher_id: str, data: dict) -> None:
    d = root / "enrichments"
    d.mkdir(exist_ok=True)
    (d / f"{enricher_id}.json").write_text(json.dumps({"status": "ok", "data": data}), "utf-8")


def test_consumer_podcast_signals(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    _write_ep(
        meta,
        "x1",
        "showx",
        "x1",
        [_topic("topic:ai", "AI"), _topic("topic:ml", "ML"), _person("person:jane", "Jane Doe")],
    )
    _write_ep(
        meta,
        "x2",
        "showx",
        "x2",
        [
            _topic("topic:ai", "AI"),
            _person("person:jane", "Jane Doe"),
            _person("org:acme", "Acme", "org"),
            # A publisher mislabelled as a person on older data — must be stripped (#2).
            _person("person:the-new-york-times", "The New York Times"),
        ],
    )
    _enrich(
        tmp_path,
        "topic_theme_clusters",
        {
            "clusters": [
                {
                    "graph_compound_parent_id": "thc:ai",
                    "canonical_label": "AI stuff",
                    "members": [{"topic_id": "topic:ai"}, {"topic_id": "topic:ml"}],
                }
            ]
        },
    )
    _enrich(
        tmp_path,
        "temporal_velocity",
        {"topics": [{"topic_id": "topic:ai", "velocity_last_over_6mo": 2.0, "total": 5}]},
    )

    client = _authed_client(tmp_path)
    r = client.get("/api/app/podcasts/showx/signals")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["feed_id"] == "showx"
    assert body["episode_count"] == 2
    assert {t["topic_id"] for t in body["top_topics"]} >= {"topic:ai", "topic:ml"}
    # top_topics carry velocity (for bubble sizing) from temporal_velocity.
    ai = next(t for t in body["top_topics"] if t["topic_id"] == "topic:ai")
    assert ai["velocity"] == 2.0
    people = {p["person_id"]: p for p in body["key_people"]}
    assert people["person:jane"]["episode_count"] == 2
    assert "org:acme" not in people  # org excluded
    assert "person:the-new-york-times" not in people  # publisher stripped from people (#2)
    assert {p["person_id"] for p in body["recurring_guests"]} == {"person:jane"}
    assert {t["storyline_id"] for t in body["dominant_storylines"]} == {"thc:ai"}
    assert {t["topic_id"] for t in body["trending_topics"]} == {"topic:ai"}
    # Consumer projection drops the operator-only grounding score.
    assert "grounding" not in body


def test_consumer_podcast_signals_unknown_feed_is_empty(tmp_path: Path) -> None:
    (tmp_path / "metadata").mkdir()
    client = _authed_client(tmp_path)
    r = client.get("/api/app/podcasts/ghost/signals")
    assert r.status_code == 200
    body = r.json()
    assert body["episode_count"] == 0
    assert body["top_topics"] == [] and body["key_people"] == []


def test_themes_and_storylines_carry_how_distinctive_they_are(tmp_path: Path) -> None:
    """Distinctive themes and storylines, not only topics (operator 2026-10-12).

    Two shows; "finance" topics run through every episode of show F and none of show G, while
    "learning" runs through every episode of both. The finance theme and storyline are twice as
    common on F as across the library (lift 2.0); the learning one is wallpaper (lift 1.0).
    """
    meta = tmp_path / "metadata"
    meta.mkdir()
    for i in range(2):
        _write_ep(
            meta,
            f"f{i}",
            "showf",
            f"f{i}",
            [_topic("topic:bonds", "bonds"), _topic("topic:learning", "learning")],
        )
        _write_ep(meta, f"g{i}", "showg", f"g{i}", [_topic("topic:learning", "learning")])
    finance = [{"topic_id": "topic:bonds"}, {"topic_id": "topic:stocks"}]
    learning = [{"topic_id": "topic:learning"}]
    _enrich(
        tmp_path,
        "topic_theme_clusters",
        {
            "clusters": [
                {
                    "graph_compound_parent_id": "thc:fin",
                    "canonical_label": "Money",
                    "members": finance,
                },
                {
                    "graph_compound_parent_id": "thc:learn",
                    "canonical_label": "Study",
                    "members": learning,
                },
            ]
        },
    )
    search = tmp_path / "search"
    search.mkdir()
    (search / "topic_clusters.json").write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "graph_compound_parent_id": "tc:fin",
                        "canonical_label": "investing",
                        "members": finance,
                    },
                    {
                        "graph_compound_parent_id": "tc:learn",
                        "canonical_label": "learning",
                        "members": learning,
                    },
                ]
            }
        ),
        "utf-8",
    )
    body = _authed_client(tmp_path).get("/api/app/podcasts/showf/signals").json()
    themes = {t["theme_id"]: t for t in body["dominant_themes"]}
    assert themes["tc:fin"] == {
        "theme_id": "tc:fin",
        "label": "investing",
        "topic_count": 1,
        "episode_count": 2,
        "lift": 2.0,
    }
    assert themes["tc:learn"]["lift"] == 1.0
    storylines = {s["storyline_id"]: s for s in body["dominant_storylines"]}
    assert (storylines["thc:fin"]["episode_count"], storylines["thc:fin"]["lift"]) == (2, 2.0)
    assert storylines["thc:learn"]["lift"] == 1.0
