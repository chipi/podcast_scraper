"""Unit tests for the inverted KG entity index (relational-card perf remediation).

Guards the perf contract that makes person/topic/entity-search O(matches): the index is built once
per ingest (not per request), invalidates when the corpus changes, and its inverted maps + label
refs are correct. Card-output parity itself is covered by ``test_app_relational_view`` (which now
runs through this index on the default path).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import app_kg_index

pytestmark = [pytest.mark.unit]


def _write_episode(
    root: Path,
    *,
    stem: str,
    episode_id: str,
    persons: list[tuple[str, str]],
    topics: list[tuple[str, str]],
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "f1", "title": "Show"},
        "episode": {
            "episode_id": episode_id,
            "title": episode_id,
            "published_date": "2024-01-01T00:00:00",
        },
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    nodes = [{"id": pid, "type": "Person", "properties": {"name": n}} for pid, n in persons]
    nodes += [{"id": tid, "type": "Topic", "properties": {"label": la}} for tid, la in topics]
    (root / "metadata" / f"{stem}.kg.json").write_text(
        json.dumps({"episode_id": episode_id, "nodes": nodes}), encoding="utf-8"
    )


def _stamp(root: Path, mtime: float) -> None:
    stamp = root / "corpus_run_summary.json"
    stamp.write_text("{}", encoding="utf-8")
    import os

    os.utime(stamp, (mtime, mtime))


@pytest.fixture(autouse=True)
def _fresh_cache():
    perf_cache.clear()
    yield
    perf_cache.clear()


def _corpus(root: Path) -> None:
    _write_episode(
        root,
        stem="0001",
        episode_id="e1",
        persons=[("person:jane", "Jane Doe"), ("person:bob", "Bob")],
        topics=[("topic:ai", "AI")],
    )
    _write_episode(
        root,
        stem="0002",
        episode_id="e2",
        persons=[("person:jane", "Jane Doe")],
        topics=[("topic:ai", "AI"), ("topic:ml", "Machine Learning")],
    )


def test_inverted_maps_and_label_refs(tmp_path: Path) -> None:
    _corpus(tmp_path)
    _stamp(tmp_path, 1_000_000.0)
    idx = app_kg_index.get_kg_index(tmp_path)

    # Jane is in both episodes; Bob in one; ai in both; ml in one.
    assert len(idx.person_episodes("person:jane")) == 2
    assert len(idx.person_episodes("person:bob")) == 1
    assert len(idx.topic_episodes("topic:ai")) == 2
    assert len(idx.topic_episodes("topic:ml")) == 1
    assert idx.person_episodes("person:nobody") == []

    # Normalized-label refs resolve case/punctuation-insensitively.
    assert idx.person_ref_by_norm["jane doe"].id == "person:jane"
    assert idx.topic_ref_by_norm["machine learning"].id == "topic:ml"


def test_index_built_once_then_cached(tmp_path: Path, monkeypatch) -> None:
    _corpus(tmp_path)
    _stamp(tmp_path, 1_000_000.0)
    calls = [0]
    real = app_kg_index.build_kg_index

    def _counting(root: Path):
        calls[0] += 1
        return real(root)

    monkeypatch.setattr(app_kg_index, "build_kg_index", _counting)
    for _ in range(6):
        app_kg_index.get_kg_index(tmp_path)
    assert calls[0] == 1, "the KG index was rebuilt on a cache hit (per-request KG parse is back)"


def test_index_invalidates_on_ingest(tmp_path: Path) -> None:
    _corpus(tmp_path)
    _stamp(tmp_path, 1_000_000.0)
    assert len(app_kg_index.get_kg_index(tmp_path).topic_episodes("topic:ml")) == 1

    # A new episode about ml lands and the ingest stamp advances.
    _write_episode(
        tmp_path,
        stem="0003",
        episode_id="e3",
        persons=[("person:carol", "Carol")],
        topics=[("topic:ml", "Machine Learning")],
    )
    _stamp(tmp_path, 2_000_000.0)
    idx = app_kg_index.get_kg_index(tmp_path)
    assert len(idx.topic_episodes("topic:ml")) == 2, "a stale KG index hid the new episode"
    assert idx.person_ref_by_norm["carol"].id == "person:carol"
