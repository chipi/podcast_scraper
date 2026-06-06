"""Provider-mismatch triggers FAISS rebuild in indexer (ADR-098 / #897).

Covers the new branch in ``_open_or_rebuild_vector_store`` that tears down the
on-disk FAISS when the recorded provider differs from the configured one.
This is a destructive code path (it ``rmtree``s the index dir), so a regression
test is non-optional — a typo in the comparison would silently corrupt search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from podcast_scraper.search.faiss_store import FaissVectorStore, VECTORS_FILE
from podcast_scraper.search.indexer import _open_or_rebuild_vector_store

pytestmark = pytest.mark.unit


def _seed_existing_index(
    tmp_path: Path, *, provider: str, model: str = "minilm-l6", dim: int = 4
) -> None:
    store = FaissVectorStore(
        dim, embedding_model=model, embedding_provider=provider, index_dir=tmp_path
    )
    store.persist()
    # Sanity: persisted file is on disk.
    assert (tmp_path / VECTORS_FILE).is_file()


def test_rebuild_when_existing_provider_differs(tmp_path: Path) -> None:
    """Existing index has provider=sentence_transformers; config wants ollama → tear down."""
    _seed_existing_index(tmp_path, provider="sentence_transformers")
    fingerprints: Dict[str, str] = {"ep1": "abc"}

    result = _open_or_rebuild_vector_store(
        tmp_path,
        fingerprints,
        dim=4,
        resolved_model="minilm-l6",
        rebuild=False,
        cfg_embedding_model="minilm-l6",
        provider="ollama",
    )

    assert result.embedding_provider == "ollama"
    # Fingerprints must be cleared so the next index run actually re-embeds.
    assert fingerprints == {}


def test_no_rebuild_when_provider_matches(tmp_path: Path) -> None:
    """Same provider → existing store loads unchanged; fingerprints survive."""
    _seed_existing_index(tmp_path, provider="ollama")
    fingerprints: Dict[str, str] = {"ep1": "abc"}

    result = _open_or_rebuild_vector_store(
        tmp_path,
        fingerprints,
        dim=4,
        resolved_model="minilm-l6",
        rebuild=False,
        cfg_embedding_model="minilm-l6",
        provider="ollama",
    )

    assert result.embedding_provider == "ollama"
    # Fingerprints preserved — incremental re-index keeps its work.
    assert fingerprints == {"ep1": "abc"}


def test_legacy_index_without_provider_field_defaults_to_sentence_transformers(
    tmp_path: Path,
) -> None:
    """Pre-#897 index loads with provider=sentence_transformers; ollama config still rebuilds."""
    # Persist a legacy-shaped store (default provider = sentence_transformers).
    _seed_existing_index(tmp_path, provider="sentence_transformers")
    fingerprints: Dict[str, str] = {"ep1": "abc"}

    result = _open_or_rebuild_vector_store(
        tmp_path,
        fingerprints,
        dim=4,
        resolved_model="minilm-l6",
        rebuild=False,
        cfg_embedding_model="minilm-l6",
        provider="ollama",
    )

    assert result.embedding_provider == "ollama"
    assert fingerprints == {}


def test_explicit_rebuild_flag_bypasses_provider_check(tmp_path: Path) -> None:
    """``rebuild=True`` short-circuits everything; provider on new store reflects config."""
    _seed_existing_index(tmp_path, provider="sentence_transformers")
    fingerprints: Dict[str, str] = {"ep1": "abc"}

    result = _open_or_rebuild_vector_store(
        tmp_path,
        fingerprints,
        dim=4,
        resolved_model="minilm-l6",
        rebuild=True,
        cfg_embedding_model="minilm-l6",
        provider="ollama",
    )

    assert result.embedding_provider == "ollama"
