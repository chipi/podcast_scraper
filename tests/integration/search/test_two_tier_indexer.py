"""Integration test for the from-corpus two-tier indexer (RFC-090 Phase 2 / B).

Stubs the (already-tested) FAISS-indexer corpus extraction and exercises this
module's own logic — embed → map → upsert → both tiers queryable — against a real
LanceDB index with real MiniLM embeddings.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.search import hybrid_search as hs, two_tier_indexer as tti  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402


def _stub_extraction(monkeypatch, tmp_path, rows):
    meta_path = tmp_path / "corpus" / "metadata" / "ep1.json"
    monkeypatch.setattr(tti, "discover_metadata_files", lambda root: [meta_path])
    monkeypatch.setattr(tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": "ep1"}})
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: tmp_path / "corpus")
    monkeypatch.setattr(tti, "_collect_docs_for_episode", lambda *a, **k: rows)


def test_builds_both_tiers_and_is_queryable(tmp_path, monkeypatch):
    rows = [
        (
            "insight:1",
            "The central bank is shifting monetary policy",
            {"doc_type": "insight", "episode_id": "ep1", "feed_id": "show1", "grounded": True},
        ),
        (
            "chunk:1",
            "Markets moved sharply as the central bank signaled a policy shift",
            {
                "doc_type": "transcript",
                "episode_id": "ep1",
                "feed_id": "show1",
                "timestamp_start_ms": 2000,
                "timestamp_end_ms": 5000,
            },
        ),
        # A non-two-tier row (quote) must be ignored.
        ("quote:1", "we are shifting", {"doc_type": "quote", "episode_id": "ep1"}),
    ]
    _stub_extraction(monkeypatch, tmp_path, rows)

    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    lance = corpus / "search" / "lance_index"
    stats = tti.build_two_tier_index(corpus, lance, allow_download=True)

    assert stats.episodes == 1
    assert stats.insights == 1 and stats.segments == 1  # quote ignored

    backend = LanceDBBackend(str(lance))
    health = backend.health()
    assert health["insights"] == 1 and health["segments"] == 1

    # Index meta records the model + the model's real dim (derived, not assumed 384).
    meta = backend.read_index_meta()
    assert meta is not None and meta["embedding_model"]
    assert meta["embed_dim"] == 384  # MiniLM

    rows_out = hs.hybrid_candidates(corpus, "central bank policy shift", top_k=5)
    assert rows_out is not None and len(rows_out) >= 1
    by_id = {r.doc_id: r for r in rows_out}
    assert by_id["chunk:1"].metadata["timestamp_start_ms"] == 2000  # ms preserved via seconds


def test_limit_episodes_caps_walk(tmp_path, monkeypatch):
    rows = [("insight:1", "x", {"doc_type": "insight", "episode_id": "ep1", "feed_id": "s"})]
    # Two metadata files, but limit_episodes=0 stops before any work.
    meta = tmp_path / "corpus" / "metadata" / "ep1.json"
    monkeypatch.setattr(tti, "discover_metadata_files", lambda root: [meta, meta])
    monkeypatch.setattr(tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": "ep1"}})
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: tmp_path / "corpus")
    monkeypatch.setattr(tti, "_collect_docs_for_episode", lambda *a, **k: rows)
    (tmp_path / "corpus" / "metadata").mkdir(parents=True)

    stats = tti.build_two_tier_index(
        tmp_path / "corpus", tmp_path / "corpus" / "search" / "li", limit_episodes=0
    )
    assert stats.episodes == 0 and stats.insights == 0
