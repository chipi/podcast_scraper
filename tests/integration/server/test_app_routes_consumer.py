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

pytestmark = [pytest.mark.integration]


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


def _real_slug(root: Path, episode_id: str = "ep1") -> str:
    """The slug of an episode that actually exists in the fixture corpus — and so has a KG.

    The resurfacing tests used to capture against a made-up ``show-ep01``. That is an episode with
    no KG, which #38 now treats as the pipeline defect it is: every revisit surface withholds a
    capture that cannot carry the graph. Testing the ladder against a non-existent episode meant
    testing it in a state the product is not supposed to reach.
    """
    from podcast_scraper.server.app_slugs import slug_for_row
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    for row in build_catalog_rows_cumulative(root):
        if row.episode_id == episode_id:
            return slug_for_row(row)
    raise AssertionError(f"{episode_id} not in the fixture corpus")


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


def test_infinite_position_is_rejected_instead_of_poisoning_every_later_read(
    tmp_path: Path,
) -> None:
    """``Infinity`` is a bare JSON token Python's parser accepts, and ``ge=0`` is true of ``inf``.

    Stored, it made the response layer un-renderable: Starlette's ``JSONResponse.render`` uses
    ``allow_nan=False``, so GET /playback 500ed on EVERY record, and /me/stats 500ed on a
    ``listening_seconds`` that summed to ``inf``. Both stayed broken until that one record was
    overwritten — a user could brick two of their own endpoints with a single request.
    """
    client = _authed(tmp_path)
    client.put("/api/app/playback/ep", json={"position_seconds": 12.0})

    bad = client.put(
        "/api/app/playback/ep",
        content=b'{"position_seconds": Infinity}',
        headers={"content-type": "application/json"},
    )
    assert bad.status_code == 422, bad.text

    # The good record is intact and both endpoints still render.
    assert client.get("/api/app/playback").json()["items"][0]["position_seconds"] == 12.0
    assert client.get("/api/app/me/stats").status_code == 200


def test_library_add_list_remove_through_routes(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    client.post("/api/app/library", json={"feed_id": "f1", "title": "One"})
    assert [i["feed_id"] for i in client.get("/api/app/library").json()["items"]] == ["f1"]
    assert client.delete("/api/app/library/f1").json()["items"] == []


# --------------------------------------------------------------------------- #
# capture routes (auth-gated): highlights + notes + Markdown export (#1115)
# --------------------------------------------------------------------------- #


def test_capture_routes_require_auth(tmp_path: Path) -> None:
    client = _client(tmp_path)  # signed-out
    assert client.get("/api/app/highlights").status_code == 401
    assert client.post("/api/app/notes", json={"target": "episode", "target_id": "x", "text": "n"})
    assert client.get("/api/app/highlights/export.md").status_code == 401


def test_highlight_create_list_patch_delete(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    assert client.get("/api/app/highlights").json()["items"] == []
    created = client.post(
        "/api/app/highlights",
        json={
            "episode_slug": "show-ep01",
            "kind": "span",
            "start_ms": 10_000,
            "end_ms": 14_000,
            "quote_text": "the anchor is the timestamp",
            "color": "amber",
        },
    )
    assert created.status_code == 201, created.text
    hid = created.json()["id"]
    assert hid.startswith("h_") and created.json()["created_at"] > 0
    # scoped list
    assert [
        h["id"] for h in client.get("/api/app/highlights?episode=show-ep01").json()["items"]
    ] == [hid]
    assert client.get("/api/app/highlights?episode=other").json()["items"] == []
    # patch colour, immutable slug preserved
    patched = client.patch(f"/api/app/highlights/{hid}", json={"color": "rose"}).json()
    assert patched["color"] == "rose" and patched["episode_slug"] == "show-ep01"
    assert client.patch("/api/app/highlights/ghost", json={"color": "x"}).status_code == 404
    # an explicit null clears the colour (exclude_unset); an omitted field is untouched
    assert client.patch(f"/api/app/highlights/{hid}", json={"color": None}).json()["color"] is None
    assert (
        client.patch(f"/api/app/highlights/{hid}", json={"quote_text": "edited"}).json()["color"]
        is None
    )
    # delete
    assert client.delete(f"/api/app/highlights/{hid}").json()["items"] == []


def test_highlight_capture_persists_graph_refs(tmp_path: Path) -> None:
    # #1419: a captured highlight resolves + stores its episode's canonical person/topic refs.
    _corpus(tmp_path)
    client = _authed(tmp_path)
    slug = _slug(tmp_path, "ep1")
    created = client.post(
        "/api/app/highlights",
        json={"episode_slug": slug, "kind": "span", "start_ms": 1000, "quote_text": "q"},
    )
    assert created.status_code == 201, created.text
    refs = created.json()["graph_refs"]
    assert refs, "expected resolved graph refs for an episode with KG"
    assert all(r["kind"] in ("person", "topic") for r in refs)
    assert all(r["id"].startswith(("person:", "topic:")) for r in refs)
    # persisted: shows up on the subsequent list too
    listed = client.get(f"/api/app/highlights?episode={slug}").json()["items"][0]
    assert listed["graph_refs"] == refs


def test_note_create_list_patch_delete(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    created = client.post(
        "/api/app/notes",
        json={"target": "highlight", "target_id": "h_abc", "text": "reframed my thinking"},
    )
    assert created.status_code == 201, created.text
    nid = created.json()["id"]
    assert nid.startswith("n_")
    assert created.json()["created_at"] == created.json()["updated_at"]
    # scoped list
    scoped = client.get("/api/app/notes?target=highlight&target_id=h_abc").json()["items"]
    assert [n["id"] for n in scoped] == [nid]
    # patch bumps updated_at text
    patched = client.patch(f"/api/app/notes/{nid}", json={"text": "second thoughts"}).json()
    assert patched["text"] == "second thoughts"
    assert client.patch("/api/app/notes/ghost", json={"text": "x"}).status_code == 404
    # empty text rejected by the schema (min_length=1)
    assert (
        client.post(
            "/api/app/notes", json={"target": "episode", "target_id": "e", "text": ""}
        ).status_code
        == 422
    )
    assert client.delete(f"/api/app/notes/{nid}").json()["items"] == []


def test_highlights_markdown_export_groups_and_resolves_titles(tmp_path: Path) -> None:
    _corpus(tmp_path)
    slug = _slug(tmp_path, "ep1")
    client = _authed(tmp_path)
    h = client.post(
        "/api/app/highlights",
        json={
            "episode_slug": slug,
            "kind": "span",
            "start_ms": 90_000,
            "quote_text": "deep sleep consolidates memory",
        },
    ).json()
    client.post(
        "/api/app/notes",
        json={"target": "highlight", "target_id": h["id"], "text": "remember this"},
    )
    resp = client.get("/api/app/highlights/export.md")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    assert "attachment" in resp.headers.get("content-disposition", "")
    body = resp.text
    assert "# My Highlights" in body
    assert "Episode ep1" in body  # episode_title resolved through the corpus
    assert '"deep sleep consolidates memory"' in body
    assert "_note:_ remember this" in body


def test_highlights_markdown_export_empty(tmp_path: Path) -> None:
    client = _authed(tmp_path)
    body = client.get("/api/app/highlights/export.md").text
    assert "_No highlights captured yet._" in body


# --------------------------------------------------------------------------- #
# consumer enrichment read surface (#1121, RFC-088 envelopes)
# --------------------------------------------------------------------------- #


def _write_envelope(path: Path, enricher_id: str, data: object, status: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "enricher_id": enricher_id,
                "enricher_version": "1.0",
                "schema_version": "1.0",
                "status": status,
                "data": data,
            }
        ),
        encoding="utf-8",
    )


def test_episode_enrichment_surfaces_ok_envelopes_and_skips_failed(tmp_path: Path) -> None:
    _corpus(tmp_path)
    slug = _slug(tmp_path, "ep1")
    enrich_dir = tmp_path / "metadata" / "enrichments"
    _write_envelope(
        enrich_dir / "0001-a.topic_cooccurrence.json", "topic_cooccurrence", {"pairs": 3}
    )
    _write_envelope(enrich_dir / "0001-a.grounding_rate.json", "grounding_rate", {"rate": 0.9})
    # a failed envelope is present but must NOT surface
    _write_envelope(
        enrich_dir / "0001-a.topic_consensus.json", "topic_consensus", None, status="failed"
    )
    body = _client(tmp_path).get(f"/api/app/episodes/{slug}/enrichment").json()
    assert body["slug"] == slug
    assert body["signals"]["topic_cooccurrence"] == {"pairs": 3}
    assert body["signals"]["grounding_rate"] == {"rate": 0.9}
    assert "topic_consensus" not in body["signals"]


def test_episode_enrichment_unknown_slug_404(tmp_path: Path) -> None:
    _corpus(tmp_path)
    assert _client(tmp_path).get("/api/app/episodes/ghost-404/enrichment").status_code == 404


def test_episode_enrichment_empty_when_no_envelopes(tmp_path: Path) -> None:
    _corpus(tmp_path)
    slug = _slug(tmp_path, "ep1")
    assert _client(tmp_path).get(f"/api/app/episodes/{slug}/enrichment").json()["signals"] == {}


def test_corpus_enrichment_surfaces_corpus_scope_envelopes(tmp_path: Path) -> None:
    _corpus(tmp_path)
    _write_envelope(
        tmp_path / "enrichments" / "temporal_velocity.json", "temporal_velocity", {"trend": ["ai"]}
    )
    _write_envelope(
        tmp_path / "enrichments" / "run_summary.json", "run_summary", {"x": 1}
    )  # bookkeeping → skipped
    _write_envelope(
        tmp_path / "enrichments" / "topic_similarity.json",
        "topic_similarity",
        None,
        status="failed",
    )  # a failed corpus enricher must not surface
    body = _client(tmp_path).get("/api/app/corpus/enrichment").json()
    assert body["signals"]["temporal_velocity"] == {"trend": ["ai"]}
    assert "run_summary" not in body["signals"]
    assert "topic_similarity" not in body["signals"]


# --------------------------------------------------------------------------- #
# your-corpus lens on person/topic cards (#1122)
# --------------------------------------------------------------------------- #


def test_person_card_scope_mine_requires_auth(tmp_path: Path) -> None:
    _corpus(tmp_path)
    resp = _client(tmp_path).get("/api/app/persons/person:jane-doe", params={"scope": "mine"})
    assert resp.status_code == 401


def test_person_card_scope_mine_filters_to_heard_corpus(tmp_path: Path) -> None:
    _corpus(tmp_path)  # jane-doe appears in ep1 AND ep2
    client = _authed(tmp_path)
    ep1 = _slug(tmp_path, "ep1")
    # scope=all → both episodes; scope=mine (nothing captured yet) → zero, honest empty card
    assert client.get("/api/app/persons/person:jane-doe").json()["episode_count"] == 2
    empty = client.get("/api/app/persons/person:jane-doe", params={"scope": "mine"}).json()
    assert empty["episode_count"] == 0 and empty["episodes"] == []
    # capture ep1 → the lens now shows just that episode ("you heard her in …")
    client.post("/api/app/highlights", json={"episode_slug": ep1, "kind": "moment", "start_ms": 0})
    mine = client.get("/api/app/persons/person:jane-doe", params={"scope": "mine"}).json()
    assert mine["episode_count"] == 1
    assert [e["slug"] for e in mine["episodes"]] == [ep1]


def test_topic_card_scope_mine_filters_to_heard_corpus(tmp_path: Path) -> None:
    _corpus(tmp_path)  # topic:ai appears in ep1 AND ep2
    _write_clusters(tmp_path)
    client = _authed(tmp_path)
    ep2 = _slug(tmp_path, "ep2")
    client.post("/api/app/highlights", json={"episode_slug": ep2, "kind": "moment", "start_ms": 0})
    mine = client.get("/api/app/topics/topic:ai", params={"scope": "mine"}).json()
    assert [e["slug"] for e in mine["episodes"]] == [ep2]
    assert mine["episode_count"] == 1


# --------------------------------------------------------------------------- #
# topic perspectives (#1146) + corpus scope (#1149)
# --------------------------------------------------------------------------- #


def test_topic_perspectives_scope_mine_requires_auth(tmp_path: Path) -> None:
    """The #1149 scope=mine lens on perspectives is auth-gated (was untested — R3-B1).

    Guards the ``_user_set`` 401 on ``topic_perspectives_route``; every sibling
    scope=mine route has this test, perspectives did not. The scope *filtering* itself
    is code-verified leak-safe (server-side additive heard-set filter) and exercised by
    the Playwright e2e; a route-level filter test would need a perspectives-shaped GI
    fixture the shared ``_corpus`` helper does not build.
    """
    _corpus(tmp_path)
    resp = _client(tmp_path).get("/api/app/topics/topic:ai/perspectives", params={"scope": "mine"})
    assert resp.status_code == 401


# --------------------------------------------------------------------------- #
# resurfacing + derived interests (#1123)
# --------------------------------------------------------------------------- #


def test_resurfacing_requires_auth(tmp_path: Path) -> None:
    assert _client(tmp_path).get("/api/app/resurfacing").status_code == 401
    assert _client(tmp_path).get("/api/app/interests/derived").status_code == 401


def test_resurfacing_due_then_pause_then_mark_surfaced(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _authed(tmp_path)
    # An old highlight (created well in the past) is due; a brand-new one is not.
    slug = _real_slug(tmp_path)
    old = client.post(
        "/api/app/highlights", json={"episode_slug": slug, "kind": "moment", "start_ms": 0}
    ).json()
    # Backdate the highlight's created_at so it clears the 2-day first step.
    import json as _json

    hpath = tmp_path / "appdata" / "users"
    user_dir = next(hpath.iterdir())
    hl_file = user_dir / "highlights.json"
    rows = _json.loads(hl_file.read_text())
    rows[0]["created_at"] = 1  # epoch → far past
    hl_file.write_text(_json.dumps(rows))

    due = client.get("/api/app/resurfacing").json()
    assert [i["highlight"]["id"] for i in due["items"]] == [old["id"]]
    assert due["items"][0]["reflection_prompt"]  # a prompt is attached

    # Pausing hides everything.
    assert (
        client.put("/api/app/resurfacing/settings", json={"paused": True}).json()["paused"] is True
    )
    paused = client.get("/api/app/resurfacing").json()
    assert paused["items"] == [] and paused["paused"] is True

    # Resume + mark surfaced advances the ladder (count becomes 1).
    client.put("/api/app/resurfacing/settings", json={"paused": False})
    assert client.post(f"/api/app/resurfacing/{old['id']}/surfaced").status_code == 204
    state = _json.loads((user_dir / "resurfacing.json").read_text())
    assert state[old["id"]]["count"] == 1


def test_marking_an_id_you_do_not_own_is_a_404(tmp_path: Path) -> None:
    """The route wrote whatever key it was handed — no existence check, no ownership check (#39).

    Junk keys are never READ back (select_due iterates highlights), so this was unbounded growth
    rather than a wrong answer: resurfacing.json accumulated one entry per call, for ever, at the
    caller's discretion. It stopped being merely untidy with #35, which made the mark fire from a
    `?revisit=` query parameter — i.e. from any string a user can type into the address bar.
    """
    import json as _json

    _corpus(tmp_path)
    client = _authed(tmp_path)
    real = client.post(
        "/api/app/highlights",
        json={"episode_slug": _real_slug(tmp_path), "kind": "moment", "start_ms": 0},
    ).json()

    assert client.post("/api/app/resurfacing/h_not_mine/surfaced").status_code == 404
    assert client.post("/api/app/resurfacing/..%2Fetc%2Fpasswd/surfaced").status_code == 404

    user_dir = next((tmp_path / "appdata" / "users").iterdir())
    state_file = user_dir / "resurfacing.json"
    # Nothing was written at all — not even an empty file.
    assert not state_file.exists() or _json.loads(state_file.read_text()) == {}

    # The real one still works, so the check gates rather than blocks.
    assert client.post(f"/api/app/resurfacing/{real['id']}/surfaced").status_code == 204
    assert list(_json.loads(state_file.read_text())) == [real["id"]]


def test_deleting_a_highlight_takes_its_schedule_with_it(tmp_path: Path) -> None:
    """The delete cascade (#39). resurfacing.json was the one per-user file where a deleted
    capture still left a trace — one dead key each, growing without bound."""
    import json as _json

    _corpus(tmp_path)
    client = _authed(tmp_path)
    slug = _real_slug(tmp_path)
    keep = client.post(
        "/api/app/highlights", json={"episode_slug": slug, "kind": "moment", "start_ms": 0}
    ).json()
    doomed = client.post(
        "/api/app/highlights", json={"episode_slug": slug, "kind": "moment", "start_ms": 60}
    ).json()
    for hid in (keep["id"], doomed["id"]):
        assert client.post(f"/api/app/resurfacing/{hid}/surfaced").status_code == 204

    user_dir = next((tmp_path / "appdata" / "users").iterdir())
    state_file = user_dir / "resurfacing.json"
    assert set(_json.loads(state_file.read_text())) == {keep["id"], doomed["id"]}

    assert client.delete(f"/api/app/highlights/{doomed['id']}").status_code == 200
    # The deleted one's entry is gone; the survivor's is untouched, not collaterally wiped.
    remaining = _json.loads(state_file.read_text())
    assert set(remaining) == {keep["id"]}
    assert remaining[keep["id"]]["count"] == 1


def test_derived_interests_rank_corpus_entities(tmp_path: Path) -> None:
    _corpus(tmp_path)  # ep1 + ep2 both feature person:jane-doe; ep1 also bob; topics ai/ml
    client = _authed(tmp_path)
    for eid in ("ep1", "ep2"):
        client.post(
            "/api/app/highlights",
            json={"episode_slug": _slug(tmp_path, eid), "kind": "moment", "start_ms": 0},
        )
    items = client.get("/api/app/interests/derived").json()["items"]
    by_token = {i["token"]: i for i in items}
    # jane-doe occurs in both captured episodes → count 2, ranked first.
    #
    # These previously asserted "person:person:jane-doe" / "topic:topic:ai" — the double-prefixed
    # tokens derive_interest_signals emitted because the KG ids it is handed already carry their
    # prefix. That encoded the defect as the contract. A derived token is supposed to be a USABLE
    # interest token; a doubled one matches nothing the ranker compares against and nothing
    # POST /interests/{token} can act on.
    assert by_token["person:jane-doe"]["count"] == 2
    assert items[0]["token"] == "person:jane-doe"
    assert "topic:ai" in by_token
    assert not [t for t in by_token if t.startswith(("person:person:", "topic:topic:"))]


def test_highlights_export_falls_back_to_slug_when_episode_unknown(tmp_path: Path) -> None:
    # A highlight on a slug that resolves to no corpus episode → the export still renders, using the
    # bare slug as the heading (title hydration is best-effort, never breaks export).
    _corpus(tmp_path)
    client = _authed(tmp_path)
    client.post(
        "/api/app/highlights",
        json={"episode_slug": "ghost-ep-404", "kind": "moment", "start_ms": 1000},
    )
    body = client.get("/api/app/highlights/export.md").text
    assert "ghost-ep-404" in body  # heading is the slug; no title resolved


# --- re-anchoring actually runs (RFC-098 / PRD-040 FR3.1a) --------------------------------------
#
# reanchor_highlight existed but had NO production caller: list_highlights returned stored rows
# verbatim, so anchor_status was never set, the client's drift badge could never appear, and after
# a re-scrape the stored segment_ids — which are POSITIONAL (seg_{index}) — made the transcript
# highlight the wrong paragraph as saved.


def _write_segments(root: Path, stem: str, segments: list[dict]) -> None:
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    (root / "transcripts" / f"{stem}.segments.json").write_text(
        json.dumps(segments), encoding="utf-8"
    )


_QUOTE = "the part worth keeping"


def _seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


def _capture_span(client: TestClient, slug: str) -> str:
    resp = client.post(
        "/api/app/highlights",
        json={
            "episode_slug": slug,
            "kind": "span",
            "start_ms": 5_000,
            "end_ms": 9_000,
            "quote_text": _QUOTE,
        },
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def test_highlights_are_reanchored_against_the_current_transcript(tmp_path: Path) -> None:
    _corpus(tmp_path)
    _write_segments(
        tmp_path,
        "0001-a",
        [_seg(0.0, 5.0, "Intro chatter. "), _seg(5.0, 9.0, _QUOTE), _seg(9.0, 20.0, " Outro.")],
    )
    client = _authed(tmp_path)
    _capture_span(client, _slug(tmp_path, "ep1"))

    item = client.get("/api/app/highlights").json()["items"][0]
    assert item["anchor_status"] == "anchored"
    assert item["segment_ids"] == ["seg_0001"]


def test_a_shifted_transcript_reanchors_to_the_new_segment_ids(tmp_path: Path) -> None:
    """An ad inserted at the head renumbers every later segment; the quote is still there."""
    _corpus(tmp_path)
    _write_segments(
        tmp_path,
        "0001-a",
        [_seg(0.0, 5.0, "Intro chatter. "), _seg(5.0, 9.0, _QUOTE), _seg(9.0, 20.0, " Outro.")],
    )
    client = _authed(tmp_path)
    _capture_span(client, _slug(tmp_path, "ep1"))
    before = client.get("/api/app/highlights").json()["items"][0]["segment_ids"]

    # Re-scrape: a sponsor read is now segment 0, so the quote lives at a different INDEX while
    # keeping its timestamps.
    _write_segments(
        tmp_path,
        "0001-a",
        [
            _seg(0.0, 2.0, "A word from our sponsor. "),
            _seg(2.0, 5.0, "Intro chatter. "),
            _seg(5.0, 9.0, _QUOTE),
            _seg(9.0, 20.0, " Outro."),
        ],
    )
    after = client.get("/api/app/highlights").json()["items"][0]
    assert after["anchor_status"] == "anchored"
    assert after["segment_ids"] == ["seg_0002"], after["segment_ids"]
    assert after["segment_ids"] != before, "the stored (stale) ids were served unchanged"


def test_a_transcript_that_lost_the_quote_is_marked_drifted_not_mis_anchored(
    tmp_path: Path,
) -> None:
    """The window still exists but the passage moved — "anchored" must not be claimed."""
    _corpus(tmp_path)
    _write_segments(
        tmp_path,
        "0001-a",
        [_seg(0.0, 5.0, "Intro chatter. "), _seg(5.0, 9.0, _QUOTE), _seg(9.0, 20.0, " Outro.")],
    )
    client = _authed(tmp_path)
    _capture_span(client, _slug(tmp_path, "ep1"))

    _write_segments(
        tmp_path,
        "0001-a",
        [_seg(0.0, 5.0, "Totally new intro. "), _seg(5.0, 9.0, "An unrelated advert. ")],
    )
    item = client.get("/api/app/highlights").json()["items"][0]
    assert item["anchor_status"] == "drifted"
    assert item["quote_text"] == _QUOTE  # never dropped
    assert item["id"]  # still returned


def test_reanchoring_survives_a_corpus_without_segments(tmp_path: Path) -> None:
    """No segments file → serve what we stored, never a 500."""
    _corpus(tmp_path)  # no segments written
    client = _authed(tmp_path)
    _capture_span(client, _slug(tmp_path, "ep1"))
    resp = client.get("/api/app/highlights")
    assert resp.status_code == 200
    assert resp.json()["items"][0]["quote_text"] == _QUOTE


def test_a_finished_episode_is_recorded_as_finished(tmp_path: Path) -> None:
    """Nothing used to mark an episode finished, so it never left "Continue listening".

    The last cadence save left it parked seconds from the end, and re-opening it resumed at
    end-epsilon and instantly re-triggered auto-advance. Kept as a flag rather than by clearing the
    record, so "I finished this" survives — the client sets it on `ended` or at the completion
    threshold (skipping the outro is a normal way to finish, and `ended` never fires for it).
    """
    client = _authed(tmp_path)
    client.put("/api/app/playback/ep", json={"position_seconds": 12.0})
    assert client.get("/api/app/playback/ep").json()["finished"] is False

    client.put("/api/app/playback/ep", json={"position_seconds": 1790.0, "finished": True})
    assert client.get("/api/app/playback/ep").json()["finished"] is True
    listed = client.get("/api/app/playback").json()["items"]
    assert next(i for i in listed if i["slug"] == "ep")["finished"] is True


def test_a_record_written_before_the_flag_existed_reads_as_unfinished(tmp_path: Path) -> None:
    """Absent is not the same as false only if we say so — an old record must not read as finished."""
    import json as _json

    client = _authed(tmp_path)
    client.put("/api/app/playback/ep", json={"position_seconds": 12.0})
    path = tmp_path / "appdata" / "users"
    rec_file = next(path.glob("*/playback.json"))
    rec_file.write_text(_json.dumps({"ep": {"position_seconds": 12.0, "updated_at": 1}}))

    assert client.get("/api/app/playback/ep").json()["finished"] is False
    assert client.get("/api/app/playback").json()["items"][0]["finished"] is False


# --- one gate for all three revisit surfaces (#38) ------------------------------------------------
#
# The email is a REMINDER of the page you would see anyway (product call, 2026-08-17), so Your Week
# and the digest must agree by construction — they do, both built by assemble_digest_payload. The
# Revisit tab was the odd one out: a different code path (select_due) with NO graph requirement, so
# it listed captures the other two silently withheld. Same user, same highlight, two answers, and
# an empty Your Week beside a populated Revisit tab that read as a bug.


def _due_capture(client, tmp_path: Path, slug: str) -> dict:
    """Capture a moment on ``slug`` and backdate it past the ladder's 2-day first rung."""
    import json as _json

    created: dict = client.post(
        "/api/app/highlights", json={"episode_slug": slug, "kind": "moment", "start_ms": 0}
    ).json()
    hl_file = next((tmp_path / "appdata" / "users").iterdir()) / "highlights.json"
    rows = _json.loads(hl_file.read_text())
    for row in rows:
        if row["id"] == created["id"]:
            row["created_at"] = 1  # epoch → long overdue
    hl_file.write_text(_json.dumps(rows))
    return created


def test_a_capture_with_a_graph_appears_on_every_revisit_surface(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _authed(tmp_path)
    created = _due_capture(client, tmp_path, _real_slug(tmp_path, "ep1"))

    tab = client.get("/api/app/resurfacing").json()
    assert [i["highlight"]["id"] for i in tab["items"]] == [created["id"]]

    week = client.get("/api/app/your-week").json()
    revisit = next((s for s in week["sections"] if s["kind"] == "revisit"), None)
    assert revisit is not None
    assert created["id"] in [i.get("highlight_id") for i in revisit["items"]]


def test_a_capture_with_no_graph_appears_on_NONE_of_them(tmp_path: Path) -> None:
    """The divergence this closes: the tab used to list it while Your Week withheld it.

    An episode with no KG is a pipeline defect — corpus validation now fails the build on it — but
    while one exists, the three surfaces must at least agree about it. Disagreeing is what made an
    empty Your Week impossible to explain from the app.
    """
    _write_episode(tmp_path, stem="0003-c", episode_id="ep3", persons=[], topics=[])
    client = _authed(tmp_path)
    created = _due_capture(client, tmp_path, _real_slug(tmp_path, "ep3"))

    tab = client.get("/api/app/resurfacing").json()
    assert tab["items"] == [], f"the Revisit tab still lists a graphless capture: {tab}"

    week = client.get("/api/app/your-week").json()
    assert not any(s["kind"] == "revisit" for s in week["sections"])

    # It is withheld, not deleted — the capture is still the user's, and still listed as a highlight.
    assert created["id"] in [h["id"] for h in client.get("/api/app/highlights").json()["items"]]


def test_an_inverted_window_is_rejected_at_the_edge(tmp_path: Path) -> None:
    """`end_ms >= start_ms` (#34.8). Each field was bounded alone; the PAIR was not.

    An inverted window makes the re-anchor overlap test near-vacuous — the span matches no segment,
    or matches by accident — and the highlight then re-anchors to nothing while reporting success.
    `reanchor_highlight` swaps lo/hi defensively, but a value that cannot mean anything should not
    reach storage in the first place: rejecting at the edge beats reasoning about it at every
    consumer downstream.
    """
    _corpus(tmp_path)
    client = _authed(tmp_path)
    slug = _real_slug(tmp_path)
    bad = client.post(
        "/api/app/highlights",
        json={"episode_slug": slug, "kind": "span", "start_ms": 9000, "end_ms": 1000},
    )
    assert bad.status_code == 422, bad.text
    assert "end_ms" in bad.text and "start_ms" in bad.text

    # The well-formed neighbours still work, so the guard gates rather than blocks.
    for payload in (
        {"episode_slug": slug, "kind": "span", "start_ms": 1000, "end_ms": 9000},
        {"episode_slug": slug, "kind": "moment", "start_ms": 5000},
        # A zero-length window is not inverted — a span that starts and ends together is legal.
        {"episode_slug": slug, "kind": "span", "start_ms": 5000, "end_ms": 5000},
    ):
        ok = client.post("/api/app/highlights", json=payload)
        assert ok.status_code in (200, 201), (payload, ok.text)


def test_a_cross_field_rejection_renders_its_own_422(tmp_path: Path) -> None:
    """The 422 handler must survive the error shapes a model_validator produces.

    Pydantic puts the raised exception ITSELF into an error's ``ctx``
    (``{'error': ValueError(...)}``), which no JSON encoder can render. The handler added for
    `Infinity` (#46) knew about non-finite floats and passed every other object through untouched —
    so the moment this codebase gained its first cross-field validator, the 422 path started
    raising while reporting a 422: input rejected correctly, then 500 on the way out.

    Found by adding the validator, not by reading the handler. The same shape of bug as the one the
    handler was written for, one type further along.
    """
    _corpus(tmp_path)
    client = _authed(tmp_path)
    resp = client.post(
        "/api/app/highlights",
        json={
            "episode_slug": _real_slug(tmp_path),
            "kind": "span",
            "start_ms": 9000,
            "end_ms": 1000,
        },
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()  # must PARSE — the point is that the report itself is renderable
    assert isinstance(body["detail"], list) and body["detail"], body
    # The offending context survived as text rather than exploding the encoder.
    assert "end_ms" in resp.text


# --- generous caps on per-user state (#51) ---------------------------------------------------------
#
# Nothing was capped except a collection NAME, so a runaway client loop could grow one account's
# files until that account's own reads degraded. Self-inflicted only — no cross-user blast radius —
# so the numbers are set high enough that no real user reaches one. Each cap REJECTS at the
# boundary rather than truncating: silently trimming text somebody wrote is the worse failure.


def test_an_over_long_note_is_rejected_not_truncated(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _authed(tmp_path)
    hl = client.post(
        "/api/app/highlights",
        json={"episode_slug": _real_slug(tmp_path), "kind": "moment", "start_ms": 0},
    ).json()

    ok = client.post(
        "/api/app/notes", json={"target": "highlight", "target_id": hl["id"], "text": "x" * 32_000}
    )
    assert ok.status_code in (200, 201), ok.text  # exactly at the cap is fine

    too_long = client.post(
        "/api/app/notes",
        json={"target": "highlight", "target_id": hl["id"], "text": "x" * 32_001},
    )
    assert too_long.status_code == 422, too_long.text
    # Nothing was stored under a truncated body — the user keeps their text, we keep our word.
    bodies = [n["text"] for n in client.get("/api/app/notes").json()["items"]]
    assert all(len(b) <= 32_000 for b in bodies)
    assert len(bodies) == 1


def test_an_over_long_quote_is_rejected(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _authed(tmp_path)
    slug = _real_slug(tmp_path)
    over = client.post(
        "/api/app/highlights",
        json={"episode_slug": slug, "kind": "span", "start_ms": 0, "quote_text": "q" * 8_001},
    )
    assert over.status_code == 422, over.text
    at_cap = client.post(
        "/api/app/highlights",
        json={"episode_slug": slug, "kind": "span", "start_ms": 0, "quote_text": "q" * 8_000},
    )
    assert at_cap.status_code in (200, 201), at_cap.text


def test_an_over_long_favorite_label_is_rejected(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _authed(tmp_path)
    over = client.put(
        "/api/app/favorites",
        json={"kind": "episode", "ref": "ep-1", "label": "L" * 501},
    )
    assert over.status_code == 422, over.text


def test_an_over_long_queue_is_rejected(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _authed(tmp_path)
    assert client.put(
        "/api/app/queue", json={"items": [f"s{i}" for i in range(500)]}
    ).status_code in (
        200,
        201,
    )
    over = client.put("/api/app/queue", json={"items": [f"s{i}" for i in range(501)]})
    assert over.status_code == 422, over.text
