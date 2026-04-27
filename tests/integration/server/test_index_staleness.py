"""Unit tests for index staleness heuristics (GitHub #507)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper.server.index_staleness import (
    compute_index_staleness,
    REASON_ARTIFACTS_NEWER,
    REASON_CORPUS_SEARCH_PARENT_HINT,
    REASON_EMBEDDING_MODEL_MISMATCH,
    REASON_MULTI_FEED_BATCH_INCOMPLETE,
    REASON_NO_INDEX_BUT_METADATA,
)

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm]/[server]
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


def test_no_index_with_metadata_recommends(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    (meta / "a.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": "e"}, "feed": {}}),
        encoding="utf-8",
    )
    st = compute_index_staleness(
        tmp_path,
        index_available=False,
        index_reason="no_index",
        index_last_updated=None,
        index_embedding_model=None,
        embedding_model_query=None,
    )
    assert st.reindex_recommended is True
    assert REASON_NO_INDEX_BUT_METADATA in st.reindex_reasons


def test_faiss_unavailable_does_not_force_reindex(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    (meta / "a.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": "e"}, "feed": {}}),
        encoding="utf-8",
    )
    st = compute_index_staleness(
        tmp_path,
        index_available=False,
        index_reason="faiss_unavailable",
        index_last_updated=None,
        index_embedding_model=None,
        embedding_model_query=None,
    )
    assert st.reindex_recommended is False
    assert REASON_NO_INDEX_BUT_METADATA not in st.reindex_reasons


def test_multi_feed_summary_incomplete_reason(tmp_path: Path) -> None:
    (tmp_path / "corpus_run_summary.json").write_text(
        json.dumps({"overall_ok": False, "feeds": []}),
        encoding="utf-8",
    )
    st = compute_index_staleness(
        tmp_path,
        index_available=True,
        index_reason=None,
        index_last_updated="2099-01-01T00:00:00Z",
        index_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_model_query=None,
    )
    assert REASON_MULTI_FEED_BATCH_INCOMPLETE in st.reindex_reasons
    assert st.reindex_recommended is False


@patch(
    "podcast_scraper.server.index_staleness.newest_index_source_mtime_epoch",
    return_value=2_000_000_000.0,
)
def test_artifacts_newer_than_index_flagged(_mock: object, tmp_path: Path) -> None:
    st = compute_index_staleness(
        tmp_path,
        index_available=True,
        index_reason=None,
        index_last_updated="2020-01-01T00:00:00Z",
        index_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_model_query=None,
    )
    assert st.reindex_recommended is True
    assert REASON_ARTIFACTS_NEWER in st.reindex_reasons


@patch(
    "podcast_scraper.server.index_staleness.newest_index_source_mtime_epoch",
    return_value=946684800.0,
)
def test_index_newer_than_artifacts_not_stale(_mock: object, tmp_path: Path) -> None:
    st = compute_index_staleness(
        tmp_path,
        index_available=True,
        index_reason=None,
        index_last_updated="2020-06-01T00:00:00Z",
        index_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_model_query=None,
    )
    assert REASON_ARTIFACTS_NEWER not in st.reindex_reasons
    assert st.reindex_recommended is False


def test_embedding_model_mismatch_via_query_param(tmp_path: Path) -> None:
    st = compute_index_staleness(
        tmp_path,
        index_available=True,
        index_reason=None,
        index_last_updated="2020-01-01T00:00:00Z",
        index_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_model_query="sentence-transformers/all-mpnet-base-v2",
    )
    assert REASON_EMBEDDING_MODEL_MISMATCH in st.reindex_reasons
    assert st.reindex_recommended is True


def test_corpus_search_parent_hint_in_reasons(tmp_path: Path) -> None:
    """When corpus root is a feed subtree but parent has search/, emit hint."""
    corpus = tmp_path / "corpus"
    (corpus / "search").mkdir(parents=True)
    (corpus / "search" / "vectors.faiss").write_bytes(b"x")
    feed_root = corpus / "feeds" / "show_a"
    meta_dir = feed_root / "metadata"
    meta_dir.mkdir(parents=True)
    (meta_dir / "x.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": "e"}, "feed": {}}),
        encoding="utf-8",
    )

    st = compute_index_staleness(
        feed_root,
        index_available=True,
        index_reason=None,
        index_last_updated="2099-01-01T00:00:00Z",
        index_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_model_query=None,
    )
    assert REASON_CORPUS_SEARCH_PARENT_HINT in st.reindex_reasons
