"""Coverage for run_corpus_search early-return branches (RFC-090 Phase 2)."""

from __future__ import annotations

import pytest

from podcast_scraper.search.corpus_search import run_corpus_search

pytestmark = pytest.mark.unit


def test_empty_query_returns_error(tmp_path):
    assert run_corpus_search(tmp_path, "   ").error == "empty_query"


def test_no_index_returns_error(tmp_path):
    # No LanceDB index → no_index.
    assert run_corpus_search(tmp_path, "a real query").error == "no_index"
