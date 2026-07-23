"""Integration tests for GET /api/app/search — library-wide grounded search (#1068).

Mocks the inner ``run_corpus_search`` so no LanceDB index / ML embedding runs in CI.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.search.corpus_search import CorpusSearchOutcome
from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.app_user_store import get_or_create_user
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

pytestmark = [pytest.mark.integration]


def _client(root: Path) -> TestClient:
    return TestClient(create_app(root, static_dir=False))


def _authed(root: Path, subject: str = "s1") -> TestClient:
    app = create_app(root, static_dir=False)
    data_dir = root / "appdata"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(
        data_dir, provider="stub", subject=subject, email=f"{subject}@x.com", name=subject
    )
    client = TestClient(app)
    signed_cookie = app_sessions.sign(
        {"user_id": user.user_id, "iat": int(time.time())}, "test-secret"
    )
    client.cookies.set(app_sessions.SESSION_COOKIE, signed_cookie)
    return client


def _slug(root: Path, episode_id: str) -> str:
    for row in build_catalog_rows_cumulative(root):
        if row.episode_id == episode_id:
            return slug_for_row(row)
    raise AssertionError(f"no slug for {episode_id}")


def test_app_search_maps_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(output_dir: Path, query: str, **kwargs: Any) -> CorpusSearchOutcome:
        assert query == "transformers"
        assert kwargs.get("top_k") == 5
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "insight:1",
                    "score": 0.9,
                    "metadata": {"doc_type": "insight", "episode_id": "ep1"},
                    "text": "hi",
                }
            ],
            lift_stats={"transcript_hits_returned": 0, "lift_applied": 0},
        )

    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", fake_run)
    resp = _client(tmp_path).get("/api/app/search", params={"q": "transformers", "top_k": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["error"] is None
    assert body["query"] == "transformers"
    assert [r["doc_id"] for r in body["results"]] == ["insight:1"]
    # source_tier is stamped by structured_corpus_search from metadata.doc_type
    assert body["results"][0]["source_tier"] == "insight"


def test_app_search_no_index_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(output_dir: Path, query: str, **kwargs: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(error="no_index", detail="no LanceDB index")

    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", fake_run)
    body = _client(tmp_path).get("/api/app/search", params={"q": "x"}).json()
    assert body["error"] == "no_index"
    assert body["results"] == []


def test_app_search_requires_query(tmp_path: Path) -> None:
    assert _client(tmp_path).get("/api/app/search").status_code == 422


def _write_episode_meta(
    root: Path, *, feed_id: str, episode_id: str, title: str, feed_image: str
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {
            "feed_id": feed_id,
            "title": "Show A",
            "url": f"https://{feed_id}.ex/f.xml",
            "image_url": feed_image,
        },
        "episode": {
            "episode_id": episode_id,
            "title": title,
            "published_date": "2024-03-01T00:00:00",
        },
        "content": {},
    }
    (root / "metadata" / f"{episode_id}.metadata.json").write_text(
        json.dumps(doc), encoding="utf-8"
    )


def test_app_search_enriches_hits_with_slug_titles_and_artwork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real catalog row lets _attach_consumer_slugs join the hit → slug, titles, artwork.
    _write_episode_meta(
        tmp_path,
        feed_id="showa",
        episode_id="ep1",
        title="Ep One",
        feed_image="https://img/feed.jpg",
    )

    def fake_run(output_dir: Path, query: str, **kwargs: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "insight:1",
                    "score": 0.9,
                    "metadata": {"doc_type": "insight", "feed_id": "showa", "episode_id": "ep1"},
                    "text": "hi",
                }
            ],
            lift_stats={"transcript_hits_returned": 0, "lift_applied": 0},
        )

    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", fake_run)
    body = _client(tmp_path).get("/api/app/search", params={"q": "x"}).json()
    md = body["results"][0]["metadata"]
    assert md["episode_title"] == "Ep One"
    assert md["podcast_title"] == "Show A"
    assert md["episode_slug"]  # deterministic, non-empty
    # No locally-stored art in this fixture → falls back to the remote feed image.
    assert md["episode_artwork"] == "https://img/feed.jpg"


# --------------------------------------------------------------------------- #
# scope=mine — grounded recall over the user's heard∪captured set (P3 #1120)
# --------------------------------------------------------------------------- #


def _two_episode_corpus(root: Path) -> None:
    for fid, eid, title in [("showa", "ep1", "Captured One"), ("showb", "ep2", "Unheard Two")]:
        _write_episode_meta(root, feed_id=fid, episode_id=eid, title=title, feed_image="x")


def _both_hits_run(output_dir: Path, query: str, **kwargs: Any) -> CorpusSearchOutcome:
    return CorpusSearchOutcome(
        results=[
            {
                "doc_id": "i1",
                "score": 0.9,
                "text": "a",
                "metadata": {"doc_type": "insight", "feed_id": "showa", "episode_id": "ep1"},
            },
            {
                "doc_id": "i2",
                "score": 0.8,
                "text": "b",
                "metadata": {"doc_type": "insight", "feed_id": "showb", "episode_id": "ep2"},
            },
        ],
        lift_stats={"transcript_hits_returned": 0, "lift_applied": 0},
    )


def test_scope_mine_requires_auth(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    assert (
        _client(tmp_path).get("/api/app/search", params={"q": "x", "scope": "mine"}).status_code
        == 401
    )


def test_scope_mine_empty_corpus_is_zero_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_episode_corpus(tmp_path)
    called = {"n": 0}

    def spy(output_dir: Path, query: str, **kwargs: Any) -> CorpusSearchOutcome:
        called["n"] += 1
        return _both_hits_run(output_dir, query, **kwargs)

    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", spy)
    # The user has heard/captured nothing → empty results, and search isn't even run.
    body = _authed(tmp_path).get("/api/app/search", params={"q": "x", "scope": "mine"}).json()
    assert body["results"] == []
    assert called["n"] == 0


def test_scope_mine_filters_to_captured_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_episode_corpus(tmp_path)
    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", _both_hits_run)
    client = _authed(tmp_path)
    # Capture ep1 (a highlight) → it joins the user's corpus; ep2 stays outside.
    slug1 = _slug(tmp_path, "ep1")
    client.post(
        "/api/app/highlights", json={"episode_slug": slug1, "kind": "moment", "start_ms": 0}
    )
    # scope=all returns both; scope=mine keeps only the captured episode.
    all_body = client.get("/api/app/search", params={"q": "x", "scope": "all"}).json()
    assert {r["doc_id"] for r in all_body["results"]} == {"i1", "i2"}
    mine_body = client.get("/api/app/search", params={"q": "x", "scope": "mine"}).json()
    assert [r["doc_id"] for r in mine_body["results"]] == ["i1"]
    assert mine_body["results"][0]["metadata"]["episode_slug"] == slug1


def test_scope_mine_is_isolated_between_users(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_episode_corpus(tmp_path)
    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", _both_hits_run)
    alice, bob = _authed(tmp_path, "alice"), _authed(tmp_path, "bob")
    alice.post(
        "/api/app/highlights",
        json={"episode_slug": _slug(tmp_path, "ep1"), "kind": "moment", "start_ms": 0},
    )
    bob.post(
        "/api/app/highlights",
        json={"episode_slug": _slug(tmp_path, "ep2"), "kind": "moment", "start_ms": 0},
    )
    # Each user's scope=mine reflects only their OWN corpus.
    a = alice.get("/api/app/search", params={"q": "x", "scope": "mine"}).json()
    b = bob.get("/api/app/search", params={"q": "x", "scope": "mine"}).json()
    assert [r["doc_id"] for r in a["results"]] == ["i1"]
    assert [r["doc_id"] for r in b["results"]] == ["i2"]


# --------------------------------------------------------------------------- #
# enrich_results — RFC-088 QueryEnricher chain over consumer search (#1261)
# --------------------------------------------------------------------------- #


def _seed_topic_similarity(root: Path, topic_id: str, siblings: list[tuple[str, float]]) -> None:
    """Write the enricher input the chunk-3 query_topic_relatedness reads."""
    out = root / "enrichments"
    out.mkdir(parents=True, exist_ok=True)
    (out / "topic_similarity.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "enricher_id": "topic_similarity",
                "data": {
                    "topic_count": 1,
                    "topics": [
                        {
                            "topic_id": topic_id,
                            "top_k": [
                                {"topic_id": sid, "similarity": sim} for sid, sim in siblings
                            ],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_app_search_enrich_results_decorates_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When enrich_results=true and topic_similarity.json exists, a kg_topic hit
    gains ``metadata.query_enrichments.related_topics``."""
    _seed_topic_similarity(tmp_path, "topic:ai", [("topic:ml", 0.91), ("topic:safety", 0.83)])

    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "kg_topic:ai",
                    "score": 0.9,
                    "text": "AI",
                    "metadata": {"doc_type": "kg_topic", "topic_id": "topic:ai"},
                }
            ]
        )

    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", fake_run)
    body = (
        _client(tmp_path)
        .get("/api/app/search", params={"q": "ai", "enrich_results": "true"})
        .json()
    )
    related = body["results"][0]["metadata"].get("query_enrichments", {}).get("related_topics")
    assert related is not None
    assert related[0]["topic_id"] == "topic:ml"


def test_app_search_enrich_results_false_by_default_leaves_hits_bare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default enrich_results=false — no query_enrichments field written."""
    _seed_topic_similarity(tmp_path, "topic:ai", [("topic:ml", 0.91)])

    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "kg_topic:ai",
                    "score": 0.9,
                    "text": "AI",
                    "metadata": {"doc_type": "kg_topic", "topic_id": "topic:ai"},
                }
            ]
        )

    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", fake_run)
    body = _client(tmp_path).get("/api/app/search", params={"q": "ai"}).json()
    assert "query_enrichments" not in body["results"][0]["metadata"]


def test_app_search_enrich_results_empty_hits_does_not_break_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """enrich_results=true with zero hits → skip the chain, return 200 + []."""

    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(results=[])

    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", fake_run)
    r = _client(tmp_path).get("/api/app/search", params={"q": "x", "enrich_results": "true"})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_app_search_enrich_results_missing_topic_similarity_passes_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No topic_similarity.json under enrichments/ → chain runs but decorates
    nothing; hits pass through unchanged, no error surfaced."""

    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "kg_topic:ai",
                    "score": 0.9,
                    "text": "AI",
                    "metadata": {"doc_type": "kg_topic", "topic_id": "topic:ai"},
                }
            ]
        )

    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", fake_run)
    body = (
        _client(tmp_path)
        .get("/api/app/search", params={"q": "ai", "enrich_results": "true"})
        .json()
    )
    assert body["error"] is None
    assert "query_enrichments" not in body["results"][0]["metadata"]
