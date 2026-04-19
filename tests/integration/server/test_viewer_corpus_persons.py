"""Integration tests for GET /api/corpus/persons/top."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _episode_doc(*, episode_id: str = "ep99", published: str = "2024-04-01T00:00:00") -> dict:
    return {
        "feed": {"feed_id": "f1", "title": "F"},
        "episode": {
            "episode_id": episode_id,
            "title": "Ep",
            "published_date": published,
        },
        "summary": {"title": "S", "bullets": ["a"]},
    }


def _minimal_gi() -> dict:
    return {
        "episode_id": "ep99",
        "nodes": [
            {"id": "person:alice", "type": "Person", "properties": {"name": "Alice"}},
            {"id": "q1", "type": "Quote", "properties": {"text": "hi"}},
            {"id": "i1", "type": "Insight", "properties": {"text": "thought"}},
            {"id": "topic:tax", "type": "Topic", "properties": {}},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": "q1", "to": "person:alice"},
            {"type": "SUPPORTED_BY", "from": "i1", "to": "q1"},
            {"type": "ABOUT", "from": "i1", "to": "topic:tax"},
        ],
    }


def test_corpus_persons_top_ranking(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    stem = meta / "ep99"
    (stem.with_suffix(".metadata.json")).write_text(
        json.dumps(_episode_doc()),
        encoding="utf-8",
    )
    (stem.with_suffix(".gi.json")).write_text(json.dumps(_minimal_gi()), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/persons/top", params={"path": str(tmp_path), "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["total_persons"] == 1
    assert len(body["persons"]) == 1
    p0 = body["persons"][0]
    assert p0["person_id"] == "person:alice"
    assert p0["display_name"] == "Alice"
    assert p0["episode_count"] == 1
    assert p0["insight_count"] == 1
    assert p0["top_topics"] == ["topic:tax"]


def test_corpus_persons_top_empty(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/persons/top", params={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["persons"] == []
    assert body["total_persons"] == 0
