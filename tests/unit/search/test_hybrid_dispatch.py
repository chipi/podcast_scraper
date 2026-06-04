"""Unit tests for the hybrid flag reader + corpus_search dispatch (RFC-090 Phase 2).

No LanceDB/FAISS — the flag reader is exercised against a temp config and the
dispatch is exercised by stubbing the hybrid bridge, so the FAISS-vs-hybrid routing
logic is verified in isolation.
"""

from __future__ import annotations

import pytest

from podcast_scraper.search import corpus_search, hybrid_search
from podcast_scraper.search.protocol import SearchResult

pytestmark = pytest.mark.unit


def test_flag_reader_reads_config_via_env_path(tmp_path, monkeypatch):
    cfg = tmp_path / "search.yaml"
    monkeypatch.setenv("PODCAST_SEARCH_CONFIG", str(cfg))
    monkeypatch.delenv("PODCAST_HYBRID_SEARCH", raising=False)

    cfg.write_text("serving:\n  hybrid_enabled: true\n", encoding="utf-8")
    assert hybrid_search.hybrid_search_enabled() is True

    cfg.write_text("serving:\n  hybrid_enabled: false\n", encoding="utf-8")
    assert hybrid_search.hybrid_search_enabled() is False


def test_shipped_config_defaults_hybrid_on(monkeypatch):
    """The shipped config/search.yaml turns hybrid on by default (#874).

    Found via the robust repo-root anchor regardless of CWD, so a missing
    PODCAST_SEARCH_CONFIG falls back to the shipped default (now on).
    """
    monkeypatch.delenv("PODCAST_HYBRID_SEARCH", raising=False)
    monkeypatch.delenv("PODCAST_SEARCH_CONFIG", raising=False)
    assert hybrid_search.hybrid_search_enabled() is True


def test_env_var_overrides_config(monkeypatch):
    # The container escape hatch (audit H1): env wins, no config file needed.
    monkeypatch.setenv("PODCAST_HYBRID_SEARCH", "true")
    assert hybrid_search.hybrid_search_enabled() is True
    monkeypatch.setenv("PODCAST_HYBRID_SEARCH", "off")
    assert hybrid_search.hybrid_search_enabled() is False


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


def test_dispatch_uses_hybrid_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_search, "hybrid_search_enabled", lambda: True)
    row = SearchResult(
        doc_id="insight:1",
        score=0.9,
        metadata={"doc_type": "insight", "text": "hybrid hit", "episode_id": "ep1"},
    )
    monkeypatch.setattr(corpus_search, "hybrid_candidates", lambda *a, **k: [row])

    outcome = corpus_search.run_corpus_search(tmp_path, "AI scaling", top_k=5)
    assert outcome.error is None
    assert any(r.get("doc_id") == "insight:1" for r in outcome.results)  # hybrid path, no FAISS


def test_dispatch_falls_back_to_faiss_when_hybrid_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_search, "hybrid_search_enabled", lambda: True)
    monkeypatch.setattr(corpus_search, "hybrid_candidates", lambda *a, **k: None)  # signal fallback
    # No FAISS index in tmp_path → FAISS path reports no_index (proves we fell through).
    outcome = corpus_search.run_corpus_search(tmp_path, "AI scaling", top_k=5)
    assert outcome.error == "no_index"


def test_dispatch_skips_hybrid_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_search, "hybrid_search_enabled", lambda: False)

    def _boom(*a, **k):
        raise AssertionError("hybrid_candidates must not be called when disabled")

    monkeypatch.setattr(corpus_search, "hybrid_candidates", _boom)
    outcome = corpus_search.run_corpus_search(tmp_path, "AI scaling", top_k=5)
    assert outcome.error == "no_index"  # straight to FAISS path
