"""Integration: digest semantic topic bands when search returns hits."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.search.corpus_search import CorpusSearchOutcome
from podcast_scraper.server.app import create_app

pytestmark = pytest.mark.integration


def _row(published: str, *, eid: str = "ep1", feed: str = "feed_a") -> dict:
    return {
        "feed": {"feed_id": feed, "title": "Show"},
        "episode": {
            "episode_id": eid,
            "title": "Hello",
            "published_date": published,
        },
        "summary": {"title": "Sum", "bullets": ["a", "b", "c", "d", "e"]},
    }


def test_digest_include_topics_builds_bands_when_search_hits_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = date.today().isoformat()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "one.metadata.json").write_text(
        json.dumps(_row(f"{today}T12:00:00Z")),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "podcast_scraper.server.routes.corpus_digest.load_digest_topics",
        lambda: [
            {
                "id": "climate",
                "label": "Climate",
                "query": "climate science",
            },
        ],
    )

    def fake_run(
        output_dir: Path,
        query: str,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        del output_dir
        if query == "digest":
            return CorpusSearchOutcome(
                results=[{"score": 1.0, "metadata": {"episode_id": "ep1", "feed_id": "feed_a"}}],
            )
        return CorpusSearchOutcome(
            results=[
                {
                    "score": 0.92,
                    "text": "climate",
                    "metadata": {
                        "doc_type": "summary",
                        "episode_id": "ep1",
                        "feed_id": "feed_a",
                    },
                },
            ],
        )

    monkeypatch.setattr(
        "podcast_scraper.server.routes.corpus_digest.run_corpus_search",
        fake_run,
    )

    client = TestClient(create_app(tmp_path, static_dir=False))
    r = client.get(
        "/api/corpus/digest",
        params={"path": str(tmp_path), "window": "7d", "include_topics": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["topics_unavailable_reason"] is None
    assert len(body["topics"]) == 1
    assert body["topics"][0]["topic_id"] == "climate"
    assert len(body["topics"][0]["hits"]) >= 1


def test_digest_max_rows_clamp_and_probe_no_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = date.today().isoformat()
    meta = tmp_path / "metadata"
    meta.mkdir()
    for i in range(5):
        (meta / f"e{i}.metadata.json").write_text(
            json.dumps(_row(f"{today}T12:00:00Z", eid=f"e{i}")),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "podcast_scraper.server.routes.corpus_digest.load_digest_topics",
        lambda: [],
    )

    def fake_run(
        output_dir: Path,
        query: str,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        del output_dir, kwargs
        if query == "digest":
            return CorpusSearchOutcome(error="no_index")
        return CorpusSearchOutcome(results=[])

    monkeypatch.setattr(
        "podcast_scraper.server.routes.corpus_digest.run_corpus_search",
        fake_run,
    )

    client = TestClient(create_app(tmp_path, static_dir=False))
    r = client.get(
        "/api/corpus/digest",
        params={"path": str(tmp_path), "window": "24h", "max_rows": "99"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["topics_unavailable_reason"] == "no_index"
    assert len(body["rows"]) <= 50
