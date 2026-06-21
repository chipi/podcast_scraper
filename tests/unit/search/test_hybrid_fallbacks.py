"""Unit tests for hybrid_candidates failure branches (RFC-090 Phase 2; FAISS retired #995).

These are the non-regression guarantees the module sells. Index-side failures (missing
dir, open error, dim mismatch, retrieve error) return None so the caller reports
``no_index`` (there is no FAISS fallback anymore). Query-embedding failures (model
missing/offline, empty vector) raise ``QueryEmbeddingError`` so the caller can report
``embed_failed`` instead of the misleading "run indexing". Faked backend + encode keep
this off a real LanceDB index.
"""

from __future__ import annotations

import pytest

from podcast_scraper.search import hybrid_search as hs

pytestmark = pytest.mark.unit


def _index(tmp_path):
    idx = tmp_path / "corpus" / "search" / "lance_index"
    idx.mkdir(parents=True)
    return tmp_path / "corpus"


class _FakeBackend:
    def __init__(self, *a, **k):
        pass

    def read_index_meta(self):
        return {"embedding_model": "m", "embed_dim": 384}


def _patch_backend(monkeypatch, backend_cls=_FakeBackend):
    monkeypatch.setattr(
        "podcast_scraper.search.backends.lancedb_backend.LanceDBBackend", backend_cls
    )


def test_missing_index_returns_none(tmp_path):
    assert hs.hybrid_candidates(tmp_path / "nope", "q", top_k=5) is None


def test_open_failure_falls_back(tmp_path, monkeypatch):
    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("cannot open")

    _patch_backend(monkeypatch, _Boom)
    assert hs.hybrid_candidates(_index(tmp_path), "q", top_k=5) is None


def test_embed_failure_raises_query_embedding_error(tmp_path, monkeypatch):
    # A query-embedding failure (model missing/offline) is NOT no_index — the index is
    # fine. Raise QueryEmbeddingError so the caller reports embed_failed.
    _patch_backend(monkeypatch)

    def _raise(*a, **k):
        raise RuntimeError("embed boom")

    monkeypatch.setattr(hs.embedding_loader, "encode", _raise)
    with pytest.raises(hs.QueryEmbeddingError):
        hs.hybrid_candidates(_index(tmp_path), "q", top_k=5)


def test_malformed_embedding_raises_query_embedding_error(tmp_path, monkeypatch):
    # An empty/invalid query vector is also an embed failure, not a missing index.
    _patch_backend(monkeypatch)
    monkeypatch.setattr(hs.embedding_loader, "encode", lambda *a, **k: [])  # empty
    with pytest.raises(hs.QueryEmbeddingError):
        hs.hybrid_candidates(_index(tmp_path), "q", top_k=5)


def test_dim_mismatch_falls_back(tmp_path, monkeypatch):
    _patch_backend(monkeypatch)  # meta says embed_dim 384
    monkeypatch.setattr(hs.embedding_loader, "encode", lambda *a, **k: [0.1, 0.2, 0.3])  # 3 != 384
    assert hs.hybrid_candidates(_index(tmp_path), "q", top_k=5) is None


def test_retrieve_failure_falls_back(tmp_path, monkeypatch):
    _patch_backend(monkeypatch)
    monkeypatch.setattr(hs.embedding_loader, "encode", lambda *a, **k: [0.0] * 384)  # matches dim

    class _BoomLayer:
        def __init__(self, *a, **k):
            pass

        def retrieve(self, *a, **k):
            raise RuntimeError("retrieve boom")

    monkeypatch.setattr("podcast_scraper.search.retrieval.RetrievalLayer", _BoomLayer)
    assert hs.hybrid_candidates(_index(tmp_path), "q", top_k=5) is None


def test_tier_for_mixed_doc_types_is_all():
    # Mixed insight+transcript request → must search both tiers.
    assert hs._tier_for(["insight", "transcript"]) == "all"
    assert hs._tier_for(None) == "all"
    assert hs._tier_for(["insight"]) == "insight"
    assert hs._tier_for(["transcript"]) == "segment"


def test_unsafe_output_dir_falls_back(tmp_path, monkeypatch):
    _patch_backend(monkeypatch)
    # A path containing '..' is rejected by safe_resolve_directory → None (caller: no_index).
    unsafe = str(tmp_path / "a" / ".." / "b")
    assert hs.hybrid_candidates(unsafe, "q", top_k=5) is None
