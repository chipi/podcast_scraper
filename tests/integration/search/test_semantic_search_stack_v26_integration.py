"""Integration: semantic search stack without E2E ML jobs.

Exercises ``corpus_search``, ``indexer`` helpers, ``gil_chunk_offset_verify``, and
optionally a tiny FAISS round-trip when ``faiss`` is importable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import pytest

from podcast_scraper import config as config_mod
from podcast_scraper.search import corpus_search, faiss_store
from podcast_scraper.search.corpus_search import dedupe_kg_surface_rows, run_corpus_search
from podcast_scraper.search.gil_chunk_offset_verify import (
    build_offset_alignment_report,
    half_open_ranges_overlap,
    load_index_metadata_map,
    overlap_width,
    transcript_chunk_spans_by_episode,
)
from podcast_scraper.search.indexer import (
    _embedding_dim,
    _filter_rows_by_doc_types,
    _kg_embed_text_entity,
    _kg_embed_text_topic,
    _kg_entity_kind_for_meta,
    _kg_vector_rows_from_path,
    _resolve_index_dir,
)
from podcast_scraper.search.protocol import IndexStats, SearchResult

pytestmark = pytest.mark.integration


def test_dedupe_kg_surface_rows_merges_same_surface_text() -> None:
    rows: List[dict[str, Any]] = [
        {
            "text": "Acme Corp",
            "score": 0.5,
            "metadata": {"doc_type": "kg_entity", "episode_id": "a"},
        },
        {
            "text": "acme  corp",
            "score": 0.9,
            "metadata": {"doc_type": "kg_entity", "episode_id": "b"},
        },
        {"text": "Other", "metadata": {"doc_type": "insight"}, "score": 1.0},
    ]
    out = dedupe_kg_surface_rows(rows)
    assert len(out) == 2
    kg_row = next(r for r in out if r["metadata"]["doc_type"] == "kg_entity")
    meta = kg_row["metadata"]
    assert meta.get("kg_surface_match_count") == 2
    assert set(meta.get("kg_surface_episode_ids") or []) == {"a", "b"}


class _FakeVectorStore:
    """Minimal stand-in for ``FaissVectorStore`` in ``run_corpus_search``."""

    ntotal = 12

    def __init__(self, hits: List[SearchResult]) -> None:
        self._hits = hits

    def stats(self) -> IndexStats:
        return IndexStats(
            total_vectors=self.ntotal,
            doc_type_counts={},
            feeds_indexed=[],
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            last_updated="",
            index_size_bytes=0,
        )

    def search(
        self,
        _query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        *,
        overfetch_factor: int = 3,
    ) -> List[SearchResult]:
        del overfetch_factor
        if filters and "doc_type" in filters:
            dt = filters["doc_type"]
            return [h for h in self._hits if h.metadata.get("doc_type") == dt][:top_k]
        return self._hits[:top_k]


class _FakeVectorStoreCapturingFilters(_FakeVectorStore):
    """Records the last ``filters`` passed to ``search`` for assertions."""

    def __init__(self, hits: List[SearchResult], seen: dict[str, Any]) -> None:
        super().__init__(hits)
        self._seen = seen

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        *,
        overfetch_factor: int = 3,
    ) -> List[SearchResult]:
        self._seen["filters"] = filters
        return super().search(
            query_embedding,
            top_k=top_k,
            filters=filters,
            overfetch_factor=overfetch_factor,
        )


def test_run_corpus_search_empty_query(tmp_path: Path) -> None:
    out = run_corpus_search(tmp_path, "   ")
    assert out.error == "empty_query"


def test_run_corpus_search_no_index(tmp_path: Path) -> None:
    out = run_corpus_search(tmp_path, "hello world")
    assert out.error == "no_index"


def test_run_corpus_search_load_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    idx = tmp_path / "search"
    idx.mkdir()
    (idx / faiss_store.VECTORS_FILE).write_text("x", encoding="utf-8")
    (idx / faiss_store.INDEX_META_FILE).write_text("{}", encoding="utf-8")

    def _bad_load(cls: type, _path: Path) -> Any:
        raise RuntimeError("bad index")

    monkeypatch.setattr(
        corpus_search.FaissVectorStore,
        "load",
        classmethod(_bad_load),
    )
    out = run_corpus_search(tmp_path, "hello")
    assert out.error == "load_failed"


def test_run_corpus_search_embed_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    idx = tmp_path / "search"
    idx.mkdir()
    (idx / faiss_store.VECTORS_FILE).write_bytes(b"x")
    (idx / faiss_store.INDEX_META_FILE).write_text(
        json.dumps(
            {
                "embedding_dim": 384,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "index_kind": faiss_store.INDEX_VARIANT_FLAT,
            },
        ),
        encoding="utf-8",
    )

    fake = _FakeVectorStore([])
    monkeypatch.setattr(
        corpus_search.FaissVectorStore,
        "load",
        classmethod(lambda cls, _p: fake),
    )
    monkeypatch.setattr(
        corpus_search.embedding_loader,
        "encode",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("no embed")),
    )
    out = run_corpus_search(tmp_path, "climate science")
    assert out.error == "embed_failed"


def test_run_corpus_search_success_multi_doc_types_and_dedupe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx_dir = tmp_path / "search"
    idx_dir.mkdir()
    (idx_dir / faiss_store.VECTORS_FILE).write_bytes(b"x")
    (idx_dir / faiss_store.INDEX_META_FILE).write_text("{}", encoding="utf-8")

    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    (meta_dir / "row.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"feed_id": "f1"},
                "episode": {"episode_id": "ep1", "title": "E"},
                "summary": {"bullets": ["a"]},
            },
        ),
        encoding="utf-8",
    )

    hits = [
        SearchResult(
            doc_id="d1",
            score=0.9,
            metadata={"doc_type": "insight", "episode_id": "ep1", "feed_id": "f1"},
        ),
        SearchResult(
            doc_id="d2",
            score=0.8,
            metadata={"doc_type": "summary", "episode_id": "ep1", "feed_id": "f1"},
        ),
        SearchResult(
            doc_id="d3",
            score=0.7,
            metadata={"doc_type": "kg_topic", "episode_id": "ep1", "feed_id": "f1", "text": "Tea"},
        ),
        SearchResult(
            doc_id="d4",
            score=0.6,
            metadata={"doc_type": "kg_topic", "episode_id": "ep1", "feed_id": "f1", "text": "tea"},
        ),
    ]
    fake = _FakeVectorStore(hits)
    monkeypatch.setattr(
        corpus_search.FaissVectorStore,
        "load",
        classmethod(lambda cls, _p: fake),
    )
    monkeypatch.setattr(
        corpus_search.embedding_loader,
        "encode",
        lambda *_a, **_k: [0.01] * 384,
    )

    out = run_corpus_search(
        tmp_path,
        "topic tea",
        doc_types=["insight", "kg_topic"],
        top_k=5,
        dedupe_kg_surfaces=True,
    )
    assert out.error is None
    assert out.lift_stats is not None
    assert len(out.results) <= 5


def test_run_corpus_search_single_doc_type_uses_faiss_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = tmp_path / "search"
    idx.mkdir()
    (idx / faiss_store.VECTORS_FILE).write_bytes(b"x")
    (idx / faiss_store.INDEX_META_FILE).write_text("{}", encoding="utf-8")
    hits = [
        SearchResult("a", 1.0, {"doc_type": "quote"}),
        SearchResult("b", 0.5, {"doc_type": "insight"}),
    ]
    seen: dict[str, Any] = {}
    fake = _FakeVectorStoreCapturingFilters(hits, seen)
    monkeypatch.setattr(
        corpus_search.FaissVectorStore,
        "load",
        classmethod(lambda cls, _p: fake),
    )
    monkeypatch.setattr(
        corpus_search.embedding_loader,
        "encode",
        lambda *_a, **_k: [0.0] * 384,
    )
    run_corpus_search(tmp_path, "q", doc_types=["insight"], top_k=3)
    assert seen.get("filters") == {"doc_type": "insight"}


def test_indexer_helpers_and_kg_vector_rows(tmp_path: Path) -> None:
    assert _embedding_dim("sentence-transformers/all-MiniLM-L6-v2") == 384
    cfg = config_mod.Config(rss_urls=["https://example.com/feed.xml"])
    p = _resolve_index_dir(str(tmp_path), cfg)
    assert p == (tmp_path / "search").resolve()

    assert _kg_embed_text_topic({"label": "  L ", "description": " D "}) == "L D"
    assert _kg_embed_text_topic({}) is None
    assert _kg_entity_kind_for_meta({"kind": "org"}) == "organization"
    assert _kg_entity_kind_for_meta({"kind": "person"}) == "person"
    assert _kg_embed_text_entity({"name": "A", "label": "a", "description": "x"}) == "A x"

    rows_in = [
        ("1", "t", {"doc_type": "insight"}),
        ("2", "t", {"doc_type": "summary"}),
    ]
    filtered = _filter_rows_by_doc_types(rows_in, {"summary"})
    assert len(filtered) == 1

    kg_path = tmp_path / "x.kg.json"
    kg_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "t1",
                        "type": "Topic",
                        "properties": {"label": "Climate"},
                    },
                    {
                        "id": "e1",
                        "type": "Entity",
                        "properties": {"name": "ACME", "kind": "org"},
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    rows = _kg_vector_rows_from_path(kg_path, "scope", "ep99", "feed1", "2024-01-01")
    assert len(rows) == 2
    assert any("kg_topic" in r[0] for r in rows)
    assert any("kg_entity" in r[0] for r in rows)


def test_gil_chunk_offset_pure_and_report(tmp_path: Path) -> None:
    assert half_open_ranges_overlap(0, 5, 3, 4) is True
    assert overlap_width(0, 5, 3, 4) == 1

    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "c1": {
                    "doc_type": "transcript",
                    "episode_id": "ep1",
                    "char_start": 0,
                    "char_end": 100,
                },
            },
        ),
        encoding="utf-8",
    )
    loaded = load_index_metadata_map(tmp_path)
    by_ep = transcript_chunk_spans_by_episode(loaded)
    assert by_ep["ep1"] == [(0, 100)]

    gi_path = tmp_path / "e.gi.json"
    gi_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "q1",
                        "type": "Quote",
                        "properties": {"char_start": 10, "char_end": 20},
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    rep = build_offset_alignment_report(
        gi_by_episode={"ep1": gi_path},
        metadata_by_doc=loaded,
        max_samples_per_episode=4,
    )
    assert rep["quotes_total"] == 1
    assert rep["verdict"] in (
        "aligned",
        "mostly_aligned",
        "divergent",
        "no_quotes",
        "no_indexed_transcript_for_quotes",
    )


def test_faiss_store_metadata_matches_and_minimal_roundtrip() -> None:
    pytest.importorskip("faiss")
    assert faiss_store._metadata_matches({"doc_type": "x"}, {"doc_type": "x"}) is True
    assert faiss_store._metadata_matches({"doc_type": "x"}, {"doc_type": "y"}) is False
    assert faiss_store._nlist_for_ntotal(1000) >= 4
    assert faiss_store._pq_m_for_dim(384) == 48

    store = faiss_store.FaissVectorStore(8, embedding_model="test-model")
    store.upsert(
        "doc-a",
        [0.1] * 8,
        {"doc_type": "summary", "episode_id": "e1"},
    )
    res = store.search([0.1] * 8, top_k=3, filters={"doc_type": "summary"})
    assert len(res) >= 1
    assert res[0].doc_id == "doc-a"
