"""Unit tests for FaissVectorStore (#484 / RFC-061 Phase 1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast, List

import pytest

from podcast_scraper.search import FaissVectorStore, VectorStore
from podcast_scraper.search.faiss_store import METADATA_FILE, VECTORS_FILE


def _unit(*xs: float) -> list[float]:
    import numpy as np

    v = np.array(xs, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("zero vector")
    return cast(List[float], (v / n).tolist())


@pytest.mark.unit
def test_doc_ids_for_episode_filters_metadata() -> None:
    store = FaissVectorStore(2)
    store.upsert("a", _unit(1, 0), {"episode_id": "ep:1", "doc_type": "insight"})
    store.upsert("b", _unit(0, 1), {"episode_id": "ep:2", "doc_type": "insight"})
    assert set(store.doc_ids_for_episode("ep:1")) == {"a"}


@pytest.mark.unit
def test_faiss_vector_store_is_vector_store_protocol() -> None:
    """FaissVectorStore satisfies VectorStore at runtime (structural typing)."""
    store = FaissVectorStore(4)
    assert isinstance(store, VectorStore)


@pytest.mark.unit
def test_faiss_ntotal_matches_index() -> None:
    store = FaissVectorStore(3)
    assert store.ntotal == 0
    store.upsert("x", _unit(1, 0, 0), {})
    assert store.ntotal == 1


@pytest.mark.unit
def test_upsert_search_returns_expected_order(tmp_path: Path) -> None:
    """Nearest neighbor by inner product (cosine on L2-normalized rows)."""
    store = FaissVectorStore(4, index_dir=tmp_path)
    e0 = _unit(1, 0, 0, 0)
    e1 = _unit(0, 1, 0, 0)
    e2 = _unit(1, 1, 0, 0)
    store.upsert("a", e0, {"doc_type": "insight", "feed_id": "f1"})
    store.upsert("b", e1, {"doc_type": "insight"})
    store.upsert("c", e2, {"doc_type": "quote"})
    q = _unit(1, 0, 0, 0)
    hits = store.search(q, top_k=2)
    assert [h.doc_id for h in hits[:2]] == ["a", "c"]


@pytest.mark.unit
def test_batch_upsert_duplicate_doc_id_last_wins() -> None:
    """Later row in the same batch overwrites the same doc_id."""
    store = FaissVectorStore(2)
    store.batch_upsert(
        ["x", "x"],
        [_unit(1, 0), _unit(0, 1)],
        [{"v": 1}, {"v": 2}],
    )
    hits = store.search(_unit(0, 1), top_k=1)
    assert hits[0].metadata.get("v") == 2
    assert hits[0].doc_id == "x"


@pytest.mark.unit
def test_delete_removes_from_search() -> None:
    store = FaissVectorStore(3)
    store.upsert("p", _unit(1, 0, 0), {})
    store.upsert("q", _unit(0, 1, 0), {})
    store.delete(["p", "q"])
    hits = store.search(_unit(1, 0, 0), top_k=5)
    assert hits == []


@pytest.mark.unit
def test_search_metadata_filter_post_fetch(tmp_path: Path) -> None:
    """Post-filter keeps only matching doc_type after over-fetch."""
    store = FaissVectorStore(4, index_dir=tmp_path)
    for i in range(6):
        store.upsert(
            f"insight-{i}",
            _unit(1, 0.01 * i, 0, 0),
            {"doc_type": "insight"},
        )
    store.upsert("quote-1", _unit(1, 0, 0, 0), {"doc_type": "quote"})
    hits = store.search(_unit(1, 0, 0, 0), top_k=3, filters={"doc_type": "insight"})
    assert len(hits) == 3
    assert all(h.metadata.get("doc_type") == "insight" for h in hits)


@pytest.mark.unit
def test_search_filter_accepts_list_of_values() -> None:
    store = FaissVectorStore(2)
    store.upsert("a", _unit(1, 0), {"doc_type": "insight"})
    store.upsert("b", _unit(0, 1), {"doc_type": "quote"})
    hits = store.search(
        _unit(1, 0),
        top_k=5,
        filters={"doc_type": ["insight", "summary"]},
    )
    assert len(hits) == 1 and hits[0].doc_id == "a"


@pytest.mark.unit
def test_wrong_embedding_dim_raises() -> None:
    store = FaissVectorStore(3)
    with pytest.raises(ValueError, match="dim"):
        store.upsert("a", [0.1, 0.2], {})
    store.upsert("a", _unit(1, 0, 0), {})
    with pytest.raises(ValueError, match="Query dim"):
        store.search([0.1, 0.2], top_k=1)


@pytest.mark.unit
def test_persist_load_roundtrip(tmp_path: Path) -> None:
    d = tmp_path / "search"
    store = FaissVectorStore(4, embedding_model="test-model", index_dir=d)
    store.upsert("z", _unit(0, 0, 0, 1), {"doc_type": "insight", "feed_id": "pod-a"})
    store.persist()
    assert (d / VECTORS_FILE).is_file()
    assert (d / METADATA_FILE).is_file()

    loaded = FaissVectorStore.load(d)
    assert loaded.embedding_dim == 4
    assert loaded.stats().embedding_model == "test-model"
    hits = loaded.search(_unit(0, 0, 0, 1), top_k=1)
    assert hits[0].doc_id == "z"
    assert hits[0].metadata["feed_id"] == "pod-a"


@pytest.mark.unit
def test_load_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FaissVectorStore.load(tmp_path)


@pytest.mark.unit
def test_stats_counts_and_size(tmp_path: Path) -> None:
    d = tmp_path / "idx"
    store = FaissVectorStore(2, index_dir=d)
    store.upsert("a", _unit(1, 0), {"doc_type": "insight", "feed_id": "F"})
    store.upsert("b", _unit(0, 1), {"doc_type": "quote", "feed_id": "F"})
    store.persist()
    st = store.stats()
    assert st.total_vectors == 2
    assert st.doc_type_counts.get("insight") == 1
    assert st.doc_type_counts.get("quote") == 1
    assert "F" in st.feeds_indexed
    assert st.embedding_dim == 2
    assert st.index_size_bytes > 0


@pytest.mark.unit
def test_persist_requires_directory() -> None:
    store = FaissVectorStore(2)
    with pytest.raises(ValueError, match="index_dir"):
        store.persist()


@pytest.mark.unit
def test_index_meta_format_version(tmp_path: Path) -> None:
    store = FaissVectorStore(2, index_dir=tmp_path)
    store.upsert("only", _unit(1, 0), {})
    store.persist()
    meta = json.loads((tmp_path / "index_meta.json").read_text(encoding="utf-8"))
    assert meta["format_version"] == 1
    assert meta["index_kind"] == "faiss_flat_ip_idmap"


@pytest.mark.unit
def test_maybe_upgrade_auto_ivf_low_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("faiss")
    from podcast_scraper.search import faiss_store as fs_mod

    monkeypatch.setattr(fs_mod, "FAISS_AUTO_IVF_MIN_VECTORS", 4)
    monkeypatch.setattr(fs_mod, "FAISS_AUTO_IVFPQ_MIN_VECTORS", 100_000)
    store = FaissVectorStore(8, index_dir=tmp_path)
    for i in range(6):
        vec = [0.0] * 8
        vec[0] = float(i + 1) / 20.0
        vec[1] = 1.0
        store.upsert(f"id{i}", _unit(*vec), {"doc_type": "insight", "episode_id": "e"})
    store.maybe_upgrade_approximate_index("auto")
    assert store.index_variant == fs_mod.INDEX_VARIANT_IVF
