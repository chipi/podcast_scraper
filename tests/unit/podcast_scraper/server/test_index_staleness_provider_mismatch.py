"""REASON_EMBEDDING_PROVIDER_MISMATCH detection in compute_index_staleness (#897).

Covers the new branch in ``server/index_staleness.py`` that flags a provider
drift between the persisted search index and the active profile / query.
This is the operator-visible signal that drives index rebuild — a regression
here would silently leave indexes built under one provider being queried
under another (the exact failure mode #897 was meant to prevent).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server.index_staleness import (
    compute_index_staleness,
    REASON_EMBEDDING_PROVIDER_MISMATCH,
)

pytestmark = pytest.mark.unit


def _common_args(corpus_root: Path) -> dict:
    """Fields shared by every test; only the provider pair varies."""
    return {
        "corpus_root": corpus_root,
        "index_available": True,
        "index_reason": None,
        "index_last_updated": "2026-01-01T00:00:00Z",
        "index_embedding_model": "minilm-l6",
        "embedding_model_query": "minilm-l6",
    }


def test_flags_mismatch_when_explicit_query_provider_differs(tmp_path: Path) -> None:
    fields = compute_index_staleness(
        **_common_args(tmp_path),
        index_embedding_provider="sentence_transformers",
        embedding_provider_query="ollama",
    )
    assert REASON_EMBEDDING_PROVIDER_MISMATCH in fields.reindex_reasons
    assert fields.reindex_recommended is True


def test_no_flag_when_providers_match(tmp_path: Path) -> None:
    fields = compute_index_staleness(
        **_common_args(tmp_path),
        index_embedding_provider="ollama",
        embedding_provider_query="ollama",
    )
    assert REASON_EMBEDDING_PROVIDER_MISMATCH not in fields.reindex_reasons


def test_legacy_index_without_provider_defaults_to_sentence_transformers(tmp_path: Path) -> None:
    """Older index has ``None`` for provider; backfill to sentence_transformers (Config default).

    The legacy-default backfill must NOT silently spam REASON_EMBEDDING_PROVIDER_MISMATCH for
    every operator on an existing corpus the moment they upgrade.
    """
    fields = compute_index_staleness(
        **_common_args(tmp_path),
        index_embedding_provider=None,  # legacy index, no field recorded
        embedding_provider_query="sentence_transformers",
    )
    assert REASON_EMBEDDING_PROVIDER_MISMATCH not in fields.reindex_reasons


def test_legacy_index_against_ollama_query_does_flag(tmp_path: Path) -> None:
    """Legacy index defaults to sentence_transformers; querying with ollama IS a real mismatch."""
    fields = compute_index_staleness(
        **_common_args(tmp_path),
        index_embedding_provider=None,
        embedding_provider_query="ollama",
    )
    assert REASON_EMBEDDING_PROVIDER_MISMATCH in fields.reindex_reasons
    assert fields.reindex_recommended is True


def test_no_provider_check_when_index_unavailable(tmp_path: Path) -> None:
    """When the index isn't loadable at all, provider check is moot — no false flag.

    The earlier ``if not index_available`` short-circuit returns before the provider
    block runs; this guards against a future refactor that drops that guard.
    """
    fields = compute_index_staleness(
        corpus_root=tmp_path,
        index_available=False,
        index_reason="no_index",
        index_last_updated=None,
        index_embedding_model=None,
        embedding_model_query=None,
        index_embedding_provider="sentence_transformers",
        embedding_provider_query="ollama",
    )
    assert REASON_EMBEDDING_PROVIDER_MISMATCH not in fields.reindex_reasons


def test_provider_mismatch_combines_with_other_reasons(tmp_path: Path) -> None:
    """Provider drift is one possible reason among several — order/dedup must be sane."""
    fields = compute_index_staleness(
        corpus_root=tmp_path,
        index_available=True,
        index_reason=None,
        index_last_updated="2026-01-01T00:00:00Z",
        index_embedding_model="other-model",  # ← model mismatch
        embedding_model_query="minilm-l6",
        index_embedding_provider="sentence_transformers",  # ← provider mismatch
        embedding_provider_query="ollama",
    )
    assert REASON_EMBEDDING_PROVIDER_MISMATCH in fields.reindex_reasons
    # sorted+deduped list — both reasons present, each only once.
    assert len(fields.reindex_reasons) == len(set(fields.reindex_reasons))
