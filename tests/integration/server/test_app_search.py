"""Integration tests for GET /api/app/search — library-wide grounded search (#1068).

Mocks the inner ``run_corpus_search`` so no LanceDB index / ML embedding runs in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.search.corpus_search import CorpusSearchOutcome
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _client(root: Path) -> TestClient:
    return TestClient(create_app(root, static_dir=False))


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
