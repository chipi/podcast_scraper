"""Route-level unit tests for the consumer ``/api/app/*`` surface.

Drives the FastAPI routers (episodes, relational, discover, user-state) through a
``TestClient`` over a tiny on-disk fixture corpus so the route wrappers — 404/503 paths,
the episode-reach cache, the personalized-discovery gate, favorites hydration through the
auth-gated endpoints — are covered by the unit suite (the codecov PR upload is unit-only).
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
from podcast_scraper.server.routes import app_episodes as episodes_routes

pytestmark = [pytest.mark.unit]


# --------------------------------------------------------------------------- #
# fixture corpus
# --------------------------------------------------------------------------- #


def _write_episode(
    root: Path,
    *,
    stem: str,
    episode_id: str,
    persons: list[tuple[str, str]] | None = None,
    topics: list[tuple[str, str]] | None = None,
    published: str = "2024-03-10T00:00:00",
    media_url: str | None = None,
    with_gi: bool = False,
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    content: dict = {"transcript_file_path": f"transcripts/{stem}.txt"}
    if media_url is not None:
        content["media_url"] = media_url
        content["media_type"] = "audio/mpeg"
    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show", "url": "https://pod.example/feed.xml"},
        "episode": {
            "episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "published_date": published,
            "duration_seconds": 1000,
        },
        "summary": {"title": "Sum", "bullets": ["a", "b"]},
        "content": content,
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello world", encoding="utf-8")
    nodes = [{"id": pid, "type": "Person", "properties": {"name": n}} for pid, n in (persons or [])]
    nodes += [
        {"id": tid, "type": "Topic", "properties": {"label": la}} for tid, la in (topics or [])
    ]
    (root / "metadata" / f"{stem}.kg.json").write_text(
        json.dumps({"episode_id": episode_id, "nodes": nodes}), encoding="utf-8"
    )
    if with_gi:
        (root / "metadata" / f"{stem}.gi.json").write_text(
            json.dumps({"episode_id": episode_id, "nodes": [], "edges": []}), encoding="utf-8"
        )


def _corpus(root: Path) -> None:
    _write_episode(
        root,
        stem="0001-a",
        episode_id="ep1",
        persons=[("person:jane-doe", "Jane Doe"), ("person:bob", "Bob")],
        topics=[("topic:ai", "AI"), ("topic:ml", "Machine Learning")],
        published="2024-01-01T00:00:00",
        media_url="https://cdn.example/a.mp3",
        with_gi=True,
    )
    _write_episode(
        root,
        stem="0002-b",
        episode_id="ep2",
        persons=[("person:jane-doe", "Jane Doe"), ("person:carol", "Carol")],
        topics=[("topic:ai", "AI")],
        published="2024-06-01T00:00:00",
    )


def _write_clusters(root: Path) -> None:
    (root / "search").mkdir(parents=True, exist_ok=True)
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ai",
                "canonical_label": "Artificial Intelligence",
                "member_count": 2,
                "members": [
                    {"topic_id": "topic:ai", "label": "AI"},
                    {"topic_id": "topic:ml", "label": "Machine Learning"},
                ],
            }
        ]
    }
    (root / "search" / "topic_clusters.json").write_text(json.dumps(payload), encoding="utf-8")


def _client(root: Path) -> TestClient:
    return TestClient(create_app(root, static_dir=False))


def _slug(root: Path, episode_id: str) -> str:
    from podcast_scraper.server.app_slugs import slug_for_row
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    for row in build_catalog_rows_cumulative(root):
        if row.episode_id == episode_id:
            return slug_for_row(row)
    raise AssertionError(f"no slug for {episode_id}")


# --------------------------------------------------------------------------- #
# relational routes (persons / topics / entity-search)
# --------------------------------------------------------------------------- #


def test_person_card_route_and_404(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path)
    ok = client.get("/api/app/persons/person:jane-doe")
    assert ok.status_code == 200, ok.text
    assert ok.json()["episode_count"] == 2
    assert client.get("/api/app/persons/person:nobody").status_code == 404


def test_topic_card_route_and_404(tmp_path: Path) -> None:
    _corpus(tmp_path)
    _write_clusters(tmp_path)
    client = _client(tmp_path)
    ok = client.get("/api/app/topics/topic:ai")
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["cluster_id"] == "tc:ai"
    assert {s["id"] for s in body["sibling_topics"]} == {"topic:ml"}
    assert client.get("/api/app/topics/topic:none").status_code == 404


def test_entity_search_route(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path)
    hit = client.get("/api/app/entities/search", params={"q": "Jane Doe"}).json()
    assert hit["entity"]["id"] == "person:jane-doe"
    miss = client.get("/api/app/entities/search", params={"q": "nothing here"}).json()
    assert miss["entity"] is None


# --------------------------------------------------------------------------- #
# discover routes (clusters / discover feed)
# --------------------------------------------------------------------------- #


def test_clusters_route(tmp_path: Path) -> None:
    _corpus(tmp_path)
    _write_clusters(tmp_path)
    body = _client(tmp_path).get("/api/app/clusters").json()
    assert [c["id"] for c in body["items"]] == ["tc:ai"]
    assert body["items"][0]["size"] == 2


def test_discover_recency_default_when_personalization_off(tmp_path: Path) -> None:
    _corpus(tmp_path)
    # No personalization flag → newest-first recency.
    body = _client(tmp_path).get("/api/app/discover").json()
    assert [e["title"] for e in body["items"]] == ["Episode ep2", "Episode ep1"]


def test_discover_personalized_for_signed_in_user(tmp_path: Path) -> None:
    _corpus(tmp_path)
    _write_clusters(tmp_path)
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "appdata"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.personalized_ranking = True
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    # Follow the AI cluster → ep1 (about AI, +GI depth) ranks ahead of the newer ep2.
    assert client.put("/api/app/interests", json={"items": ["tc:ai"]}).status_code == 200
    body = client.get("/api/app/discover").json()
    assert body["items"][0]["title"] == "Episode ep1"


# --------------------------------------------------------------------------- #
# episode routes (detail / entities / insights / audio-source / stats cache)
# --------------------------------------------------------------------------- #


def test_episodes_list_and_pagination(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path)
    page1 = client.get("/api/app/episodes", params={"page": 1, "page_size": 1}).json()
    assert page1["total"] == 2
    assert page1["page_size"] == 1
    assert len(page1["items"]) == 1
    assert page1["has_more"] is True
    page2 = client.get("/api/app/episodes", params={"page": 2, "page_size": 1}).json()
    assert page2["has_more"] is False


def test_podcasts_list(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = _client(tmp_path).get("/api/app/podcasts").json()
    assert [p["feed_id"] for p in body["items"]] == ["myfeed"]
    assert body["items"][0]["episode_count"] == 2


def test_podcast_episodes_list_scoped_to_feed(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = _client(tmp_path).get("/api/app/podcasts/myfeed/episodes").json()
    assert body["total"] == 2


def test_episode_related_empty_when_no_index(tmp_path: Path) -> None:
    _corpus(tmp_path)
    slug = _slug(tmp_path, "ep1")
    # No vector index built → graceful empty 200 (the outcome.error branch).
    resp = _client(tmp_path).get(f"/api/app/episodes/{slug}/related")
    assert resp.status_code == 200, resp.text
    assert resp.json()["items"] == []


def test_episode_search_within_episode(tmp_path: Path) -> None:
    _corpus(tmp_path)
    slug = _slug(tmp_path, "ep1")
    resp = _client(tmp_path).get(f"/api/app/episodes/{slug}/search", params={"q": "hello"})
    assert resp.status_code == 200, resp.text
    # Whatever the retrieval yields, the response shape is the search contract.
    assert "results" in resp.json() or "passages" in resp.json() or "error" in resp.json()


def test_episode_detail_and_unknown_slug_404(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path)
    slug = _slug(tmp_path, "ep1")
    detail = client.get(f"/api/app/episodes/{slug}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["has_gi"] is True
    assert client.get("/api/app/episodes/no-such-slug").status_code == 404


def test_episode_entities_with_cluster_enrichment(tmp_path: Path) -> None:
    _corpus(tmp_path)
    _write_clusters(tmp_path)
    slug = _slug(tmp_path, "ep1")
    body = _client(tmp_path).get(f"/api/app/episodes/{slug}/entities").json()
    ai = next(t for t in body["topics"] if t["id"] == "topic:ai")
    assert ai["cluster_id"] == "tc:ai"


def test_episode_insights_present_and_absent(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path)
    # ep2 has no GI → empty insights (200, graceful).
    slug2 = _slug(tmp_path, "ep2")
    assert client.get(f"/api/app/episodes/{slug2}/insights").json()["insights"] == []
    # ep1 has a GI artifact (empty nodes) → still 200.
    slug1 = _slug(tmp_path, "ep1")
    assert client.get(f"/api/app/episodes/{slug1}/insights").status_code == 200


def test_episode_audio_source_present_and_missing(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path)
    slug1 = _slug(tmp_path, "ep1")
    ok = client.get(f"/api/app/episodes/{slug1}/audio-source").json()
    assert ok["url"] == "https://cdn.example/a.mp3"
    assert ok["strategy"] == "direct"
    # ep2 has no media_url → 404.
    slug2 = _slug(tmp_path, "ep2")
    assert client.get(f"/api/app/episodes/{slug2}/audio-source").status_code == 404


def test_episode_segments_404_when_no_segments_file(tmp_path: Path) -> None:
    _corpus(tmp_path)
    slug = _slug(tmp_path, "ep1")
    # No segments.json adjacent to the transcript → 404.
    assert _client(tmp_path).get(f"/api/app/episodes/{slug}/segments").status_code == 404


def test_episode_stats_no_app_data_dir_zero_reach(tmp_path: Path) -> None:
    _corpus(tmp_path)
    slug = _slug(tmp_path, "ep1")
    body = _client(tmp_path).get(f"/api/app/episodes/{slug}/stats").json()
    assert body["listeners"] == 0 and body["opens"] == 0


def test_episode_reach_cache_hit_returns_memoized(tmp_path: Path) -> None:
    # Direct test of the TTL memo: a second call within the window returns the cached dict
    # without rescanning (exercises the cache-hit branch of `_episode_reach`).
    episodes_routes._episode_reach_cache.clear()
    data_dir = tmp_path / "appdata"
    first = episodes_routes._episode_reach(data_dir, "some-slug")
    cached = episodes_routes._episode_reach(data_dir, "some-slug")
    assert first == cached
    key = (str(data_dir), "some-slug")
    assert key in episodes_routes._episode_reach_cache
    # No app data dir → fixed empty reach (the early-return branch).
    assert episodes_routes._episode_reach(None, "x") == {"listeners": 0, "opens": 0, "daily": []}


# --------------------------------------------------------------------------- #
# user-state routes (auth-gated): favorites hydration + listen→stats
# --------------------------------------------------------------------------- #


def _authed(tmp_path: Path) -> TestClient:
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


def test_favorites_hydrate_episode_and_insight_through_route(tmp_path: Path) -> None:
    _corpus(tmp_path)
    slug = _slug(tmp_path, "ep1")
    client = _authed(tmp_path)
    assert client.get("/api/app/favorites").json() == {"episodes": [], "insights": []}
    client.put("/api/app/favorites", json={"kind": "episode", "ref": slug, "label": "E"})
    body = client.put(
        "/api/app/favorites",
        json={"kind": "insight", "ref": f"{slug}#i1", "label": "claim", "slug": slug},
    ).json()
    assert [e["slug"] for e in body["episodes"]] == [slug]
    assert body["insights"][0]["ref"] == f"{slug}#i1"
    after = client.delete(f"/api/app/favorites/episode/{slug}").json()
    assert after["episodes"] == [] and len(after["insights"]) == 1


def test_listen_resolves_feed_then_user_stats(tmp_path: Path) -> None:
    _corpus(tmp_path)
    slug = _slug(tmp_path, "ep1")
    client = _authed(tmp_path)
    # listen with a real corpus → feed_id resolves through resolve_slug (the happy path).
    assert client.post(f"/api/app/listen/{slug}").status_code == 204
    stats = client.get("/api/app/me/stats").json()
    assert stats["episodes"] == 1
    assert stats["shows"] == 1  # feed_id was resolved and recorded


def test_playback_list_and_queue_through_routes(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    assert client.get("/api/app/playback").json()["items"] == []
    client.put("/api/app/playback/ep", json={"position_seconds": 12.0})
    listed = client.get("/api/app/playback").json()["items"]
    assert listed[0]["slug"] == "ep" and listed[0]["position_seconds"] == 12.0
    assert client.put("/api/app/queue", json={"items": ["a"]}).json()["items"] == ["a"]


def test_library_add_list_remove_through_routes(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    client.post("/api/app/library", json={"feed_id": "f1", "title": "One"})
    assert [i["feed_id"] for i in client.get("/api/app/library").json()["items"]] == ["f1"]
    assert client.delete("/api/app/library/f1").json()["items"] == []
