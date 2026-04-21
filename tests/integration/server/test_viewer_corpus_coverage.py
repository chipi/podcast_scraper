"""Integration tests for GET /api/corpus/coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _episode_doc(
    *,
    feed_id: str = "feed_a",
    episode_id: str = "ep1",
    published: str = "2024-03-10T00:00:00",
) -> dict:
    return {
        "feed": {"feed_id": feed_id, "title": "Show A"},
        "episode": {
            "episode_id": episode_id,
            "title": "T",
            "published_date": published,
        },
        "summary": {"title": "S", "bullets": ["x"]},
    }


def test_corpus_coverage_counts_and_buckets(tmp_path: Path) -> None:
    meta = tmp_path / "feeds" / "a" / "metadata"
    meta.mkdir(parents=True)
    stem = meta / "one"
    (stem.with_suffix(".metadata.json")).write_text(
        json.dumps(_episode_doc(episode_id="e1", published="2024-01-05T00:00:00")),
        encoding="utf-8",
    )
    (stem.with_suffix(".gi.json")).write_text("{}", encoding="utf-8")
    (stem.with_suffix(".kg.json")).write_text("{}", encoding="utf-8")

    stem2 = meta / "two"
    (stem2.with_suffix(".metadata.json")).write_text(
        json.dumps(
            _episode_doc(
                episode_id="e2",
                feed_id="feed_b",
                published="2024-01-15T00:00:00",
            ),
        ),
        encoding="utf-8",
    )
    (stem2.with_suffix(".gi.json")).write_text("{}", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/coverage", params={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["total_episodes"] == 2
    assert body["with_gi"] == 2
    assert body["with_kg"] == 1
    assert body["with_both"] == 1
    assert body["with_neither"] == 0
    assert len(body["by_month"]) == 1
    assert body["by_month"][0]["month"] == "2024-01"
    assert body["by_month"][0]["total"] == 2
    assert body["by_month"][0]["with_both"] == 1
    feeds = {f["feed_id"]: f for f in body["by_feed"]}
    assert feeds["feed_a"]["with_gi"] == 1 and feeds["feed_a"]["with_kg"] == 1
    assert feeds["feed_b"]["with_gi"] == 1 and feeds["feed_b"]["with_kg"] == 0


def test_corpus_coverage_metadata_only_episodes_count_as_neither(tmp_path: Path) -> None:
    """Episode rows with metadata but no GI/KG siblings contribute ``with_neither``."""
    meta = tmp_path / "feeds" / "z" / "metadata"
    meta.mkdir(parents=True)
    stem = meta / "orphan"
    (stem.with_suffix(".metadata.json")).write_text(
        json.dumps(
            _episode_doc(
                episode_id="e0",
                feed_id="",
                published="bad-month",
            ),
        ),
        encoding="utf-8",
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/coverage", params={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["total_episodes"] == 1
    assert body["with_neither"] == 1
    assert body["with_gi"] == 0
    assert body["by_month"] == []
    feeds = {f["feed_id"]: f for f in body["by_feed"]}
    assert "(unknown)" in feeds


def test_corpus_coverage_empty_corpus(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/coverage", params={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["total_episodes"] == 0
    assert body["by_month"] == []
    assert body["by_feed"] == []
