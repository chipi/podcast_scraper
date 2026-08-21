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


def test_corpus_persons_top_excludes_unresolved_speaker(tmp_path: Path) -> None:
    """An unnamed diarization voice must not rank as a top person (#1167)."""
    meta = tmp_path / "metadata"
    meta.mkdir()
    stem = meta / "ep99"
    (stem.with_suffix(".metadata.json")).write_text(json.dumps(_episode_doc()), encoding="utf-8")
    gi = _minimal_gi()
    # A placeholder speaker with its own grounded insight — must be dropped, not ranked.
    gi["nodes"] += [
        {"id": "person:speaker-00", "type": "Person", "properties": {"name": "SPEAKER_00"}},
        {"id": "q2", "type": "Quote", "properties": {"text": "yo"}},
        {"id": "i2", "type": "Insight", "properties": {"text": "noise"}},
    ]
    gi["edges"] += [
        {"type": "SPOKEN_BY", "from": "q2", "to": "person:speaker-00"},
        {"type": "SUPPORTED_BY", "from": "i2", "to": "q2"},
        {"type": "ABOUT", "from": "i2", "to": "topic:tax"},
    ]
    (stem.with_suffix(".gi.json")).write_text(json.dumps(gi), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/persons/top", params={"path": str(tmp_path), "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["total_persons"] == 1
    assert [p["person_id"] for p in body["persons"]] == ["person:alice"]


def test_corpus_persons_top_two_episodes_ranking(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()

    def write_ep(stem: str, person_name: str, topic: str) -> None:
        gi = {
            "episode_id": stem,
            "nodes": [
                {
                    "id": f"person:{person_name.lower()}",
                    "type": "Person",
                    "properties": {"name": person_name},
                },
                {"id": "q1", "type": "Quote", "properties": {"text": "x"}},
                {"id": "i1", "type": "Insight", "properties": {"text": "t"}},
                {"id": topic, "type": "Topic", "properties": {"label": topic}},
            ],
            "edges": [
                {"type": "SPOKEN_BY", "from": "q1", "to": f"person:{person_name.lower()}"},
                {"type": "SUPPORTED_BY", "from": "i1", "to": "q1"},
                {"type": "ABOUT", "from": "i1", "to": topic},
            ],
        }
        base = meta / stem
        (base.with_suffix(".metadata.json")).write_text(
            json.dumps(_episode_doc(episode_id=stem, published="2024-05-01T00:00:00")),
            encoding="utf-8",
        )
        (base.with_suffix(".gi.json")).write_text(json.dumps(gi), encoding="utf-8")

    write_ep("ep_a", "Alice", "topic:alpha")
    write_ep("ep_b", "Bob", "topic:beta")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/persons/top", params={"path": str(tmp_path), "limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["total_persons"] == 2
    assert len(body["persons"]) == 2
    ids = [p["person_id"] for p in body["persons"]]
    assert "person:alice" in ids and "person:bob" in ids


def test_corpus_persons_top_empty(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/persons/top", params={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["persons"] == []
    assert body["total_persons"] == 0


def _stamp_corpus(root: Path, mtime: float) -> None:
    import os

    stamp = root / "corpus_run_summary.json"
    stamp.write_text("{}", encoding="utf-8")
    os.utime(stamp, (mtime, mtime))


def test_corpus_persons_top_scans_gi_once_per_ingest(tmp_path: Path, monkeypatch) -> None:
    """The GI scan (the operator surface's worst O(corpus) cost) runs once per corpus mtime.

    Repeated calls — and calls with a different ``limit`` — must hit the cache; only an ingest
    (corpus_run_summary mtime bump) may trigger a rescan.
    """
    from podcast_scraper import perf_cache
    from podcast_scraper.server.routes import corpus_persons

    perf_cache.clear()
    meta = tmp_path / "metadata"
    meta.mkdir()
    stem = meta / "ep99"
    stem.with_suffix(".metadata.json").write_text(json.dumps(_episode_doc()), encoding="utf-8")
    stem.with_suffix(".gi.json").write_text(json.dumps(_minimal_gi()), encoding="utf-8")
    _stamp_corpus(tmp_path, 1_000_000.0)

    scans = [0]
    real = corpus_persons._rank_all_persons

    def _counting(root: Path):
        scans[0] += 1
        return real(root)

    monkeypatch.setattr(corpus_persons, "_rank_all_persons", _counting)

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    for limit in (5, 1, 1000, 5):
        r = client.get("/api/corpus/persons/top", params={"path": str(tmp_path), "limit": limit})
        assert r.status_code == 200
    assert scans[0] == 1, "the GI scan re-ran on a cache hit (or a different limit rescanned)"

    # A new ingest bumps the stamp → the next read must rescan and reflect it.
    _stamp_corpus(tmp_path, 2_000_000.0)
    r = client.get("/api/corpus/persons/top", params={"path": str(tmp_path), "limit": 5})
    assert r.status_code == 200
    assert scans[0] == 2, "invalidation did not trigger a rescan after ingest"

    perf_cache.clear()
