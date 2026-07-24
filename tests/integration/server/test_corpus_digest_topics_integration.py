"""Integration: digest semantic topic bands when search returns hits."""

from __future__ import annotations

import json
from datetime import datetime, timezone
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
    today = datetime.now(timezone.utc).date().isoformat()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "one.metadata.json").write_text(
        json.dumps(_row(f"{today}T12:00:00Z")),
        encoding="utf-8",
    )
    # ADR-099 #995: the digest now checks for an index on disk (not a probe search) before
    # building topic bands — create a LanceDB index marker so the (mocked) band searches run.
    lance_idx = tmp_path / "search" / "lance_index"
    lance_idx.mkdir(parents=True)
    (lance_idx / "marker").write_text("x", encoding="utf-8")

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
    today = datetime.now(timezone.utc).date().isoformat()
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


def test_digest_topic_bands_cached_and_invalidate_on_reindex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The expensive topic bands are cached per process and re-used across
    requests, then recomputed after an explicit clear and after the lance index
    mtime changes (reindex). Rows stay fresh (recomputed every call)."""
    import os as _os

    from podcast_scraper.server.routes.corpus_digest import clear_digest_topics_cache

    today = datetime.now(timezone.utc).date().isoformat()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "one.metadata.json").write_text(
        json.dumps(_row(f"{today}T12:00:00Z")), encoding="utf-8"
    )
    lance_idx = tmp_path / "search" / "lance_index"
    lance_idx.mkdir(parents=True)
    (lance_idx / "marker").write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "podcast_scraper.server.routes.corpus_digest.load_digest_topics",
        lambda: [{"id": "climate", "label": "Climate", "query": "climate science"}],
    )

    band_calls = {"n": 0}

    def fake_run(output_dir: Path, query: str, **kwargs: Any) -> CorpusSearchOutcome:
        del output_dir, kwargs
        if query == "digest":
            return CorpusSearchOutcome(results=[])
        band_calls["n"] += 1  # a band (topic-query) search
        return CorpusSearchOutcome(
            results=[
                {
                    "score": 0.9,
                    "text": "climate",
                    "metadata": {"doc_type": "summary", "episode_id": "ep1", "feed_id": "feed_a"},
                }
            ]
        )

    monkeypatch.setattr("podcast_scraper.server.routes.corpus_digest.run_corpus_search", fake_run)
    clear_digest_topics_cache()  # isolate from other tests in this process

    client = TestClient(create_app(tmp_path, static_dir=False))
    params = {"path": str(tmp_path), "window": "7d", "include_topics": "true"}

    r1 = client.get("/api/corpus/digest", params=params)
    assert r1.status_code == 200 and len(r1.json()["topics"]) == 1
    assert band_calls["n"] == 1  # computed once

    r2 = client.get("/api/corpus/digest", params=params)
    assert r2.status_code == 200 and len(r2.json()["topics"]) == 1
    assert band_calls["n"] == 1  # cache HIT — no recompute

    clear_digest_topics_cache()
    client.get("/api/corpus/digest", params=params)
    assert band_calls["n"] == 2  # explicit clear → recompute

    # Reindex signal: bump the lance index dir mtime → cache self-invalidates.
    future = _os.path.getmtime(lance_idx) + 100
    _os.utime(lance_idx, (future, future))
    client.get("/api/corpus/digest", params=params)
    assert band_calls["n"] == 3  # mtime changed → recompute
