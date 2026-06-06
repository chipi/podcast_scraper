"""Provider field round-trips through FaissVectorStore (ADR-098 / #897)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search.faiss_store import (
    DEFAULT_EMBEDDING_MODEL,
    FaissVectorStore,
    FORMAT_VERSION,
    INDEX_META_FILE,
    INDEX_VARIANT_FLAT,
    VECTORS_FILE,
)

pytestmark = pytest.mark.unit


def test_persist_then_load_preserves_embedding_provider(tmp_path: Path) -> None:
    """Persisted index_meta.json carries embedding_provider; load() reads it back."""
    store = FaissVectorStore(4, embedding_provider="ollama", index_dir=tmp_path)
    store.persist()

    loaded = FaissVectorStore.load(tmp_path)
    assert loaded.embedding_provider == "ollama"
    assert loaded.stats().embedding_provider == "ollama"


def test_persist_default_provider_is_sentence_transformers(tmp_path: Path) -> None:
    """When __init__ omits embedding_provider, the default lands in the persisted blob."""
    store = FaissVectorStore(4, index_dir=tmp_path)
    store.persist()

    blob = json.loads((tmp_path / INDEX_META_FILE).read_text(encoding="utf-8"))
    assert blob["embedding_provider"] == "sentence_transformers"


def test_load_backfills_legacy_meta_blob_without_provider_field(tmp_path: Path) -> None:
    """Legacy indexes (pre-#897) lack embedding_provider in their meta JSON.

    load() must default to ``sentence_transformers`` so existing indexes don't
    suddenly look like provider drift to the staleness checker.
    """
    # Seed a synthetic legacy state: write a meta_blob WITHOUT embedding_provider.
    store = FaissVectorStore(4, index_dir=tmp_path)
    store.persist()  # writes the new-shape meta with embedding_provider

    # Rewrite meta_blob to the legacy shape (no embedding_provider key).
    legacy = {
        "format_version": FORMAT_VERSION,
        "embedding_dim": 4,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "index_kind": INDEX_VARIANT_FLAT,
        "created_at": None,
        "last_updated": "",
    }
    (tmp_path / INDEX_META_FILE).write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    loaded = FaissVectorStore.load(tmp_path)
    assert loaded.embedding_provider == "sentence_transformers"
    assert loaded.stats().embedding_provider == "sentence_transformers"


def test_stats_surfaces_embedding_provider_on_in_memory_store() -> None:
    """``IndexStats`` carries the provider even without an on-disk index dir."""
    store = FaissVectorStore(4, embedding_provider="ollama")
    st = store.stats()
    assert st.embedding_provider == "ollama"
    # Sanity: existing fields still populated.
    assert st.embedding_dim == 4


def test_persist_then_load_round_trips_both_provider_and_model(tmp_path: Path) -> None:
    """Provider and model are independent fields — neither overwrites the other on round-trip."""
    store = FaissVectorStore(
        4,
        embedding_model="nomic-embed-text",
        embedding_provider="ollama",
        index_dir=tmp_path,
    )
    store.persist()

    loaded = FaissVectorStore.load(tmp_path)
    assert loaded.embedding_model == "nomic-embed-text"
    assert loaded.embedding_provider == "ollama"


def test_vectors_file_constant_exists_after_persist(tmp_path: Path) -> None:
    """Sanity: persist actually writes the on-disk artefacts the loader expects."""
    store = FaissVectorStore(4, embedding_provider="ollama", index_dir=tmp_path)
    store.persist()
    assert (tmp_path / VECTORS_FILE).is_file()
    assert (tmp_path / INDEX_META_FILE).is_file()
