"""Unit tests for the serving router config + corpus_search dispatch (RFC-090 / ADR-099).

No LanceDB build — the LanceDB hybrid path is the single search path (FAISS retired,
#995), so dispatch is exercised by stubbing ``hybrid_candidates``; the router block is
still read from ``config/search.yaml`` and is tested against a temp config.
"""

from __future__ import annotations

import pytest

from podcast_scraper.search import corpus_search, hybrid_search
from podcast_scraper.search.protocol import SearchResult

pytestmark = pytest.mark.unit


def test_serving_router_built_from_config(tmp_path, monkeypatch):
    from podcast_scraper.search.query_router import MLQueryRouter, RulesQueryRouter

    cfg = tmp_path / "search.yaml"
    monkeypatch.setenv("PODCAST_SEARCH_CONFIG", str(cfg))
    cfg.write_text("router:\n  mode: ml\n  model_path: ./nope.joblib\n", encoding="utf-8")
    assert isinstance(hybrid_search._serving_router(), MLQueryRouter)
    cfg.write_text("router:\n  mode: rules\n", encoding="utf-8")
    assert isinstance(hybrid_search._serving_router(), RulesQueryRouter)
    cfg.write_text("backend: lancedb\n", encoding="utf-8")  # no router block
    assert hybrid_search._serving_router() is None


def test_dispatch_returns_hybrid_rows(tmp_path, monkeypatch):
    """The LanceDB hybrid path is always-on: candidate rows flow straight to results."""
    row = SearchResult(
        doc_id="insight:1",
        score=0.9,
        metadata={"doc_type": "insight", "text": "hybrid hit", "episode_id": "ep1"},
    )
    monkeypatch.setattr(corpus_search, "hybrid_candidates", lambda *a, **k: [row])

    outcome = corpus_search.run_corpus_search(tmp_path, "AI scaling", top_k=5)
    assert outcome.error is None
    assert any(r.get("doc_id") == "insight:1" for r in outcome.results)


def test_dispatch_reports_no_index_when_candidates_none(tmp_path, monkeypatch):
    """``hybrid_candidates`` returning ``None`` means no usable index → ``no_index``
    (FAISS was retired, #995; there is no fallback)."""
    monkeypatch.setattr(corpus_search, "hybrid_candidates", lambda *a, **k: None)
    outcome = corpus_search.run_corpus_search(tmp_path, "AI scaling", top_k=5)
    assert outcome.error == "no_index"
