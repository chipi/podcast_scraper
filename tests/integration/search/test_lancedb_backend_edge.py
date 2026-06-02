"""Edge-branch coverage for LanceDBBackend (RFC-090 #855 + hardening)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.search.backend import SearchQuery  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402


def test_search_on_empty_backend_returns_nothing(tmp_path):
    # No tables created → read path skips missing tiers (no autocreate), returns [].
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    assert b.search_bm25(SearchQuery(text="x", embedding=[], tier="all")) == []
    assert b.search_vector(SearchQuery(text="x", embedding=[0.0] * 4, tier="all")) == []


def test_health_on_empty_backend(tmp_path):
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    h = b.health()
    assert h["status"] == "ok" and h["segments"] == 0 and h["insights"] == 0


def test_meta_rejects_unsafe_path(tmp_path):
    # safe_resolve_directory rejects a '..' path → read returns None, write is a no-op.
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    b.path = str(tmp_path / "a" / ".." / "b")  # contains '..' → sanitizer rejects
    assert b.read_index_meta() is None
    b.write_index_meta("some-model")  # must not raise


def test_meta_round_trip(tmp_path):
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    b.write_index_meta("sentence-transformers/all-MiniLM-L6-v2")
    meta = b.read_index_meta()
    assert meta is not None and meta["embedding_model"].endswith("MiniLM-L6-v2")
    assert meta["embed_dim"] == 4
