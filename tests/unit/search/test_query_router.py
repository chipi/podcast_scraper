"""Unit tests for the pluggable query router (RFC-092, #860)."""

from __future__ import annotations

import pytest

from podcast_scraper.search.backend import ScoredResult
from podcast_scraper.search.query_router import (
    get_query_router,
    MLQueryRouter,
    RulesQueryRouter,
)
from podcast_scraper.search.retrieval import RetrievalLayer

pytestmark = pytest.mark.unit


def test_rules_router_matches_classify_query():
    r = RulesQueryRouter()
    assert r.classify("exact quote from the transcript") == "raw_evidence"
    assert r.classify("Sam Altman") == "entity_lookup"
    assert r.classify("how has AI evolved over time") == "temporal_tracking"


def test_factory_returns_rules_by_default():
    assert isinstance(get_query_router("rules"), RulesQueryRouter)
    assert isinstance(get_query_router("bogus"), RulesQueryRouter)
    assert isinstance(get_query_router("ml"), MLQueryRouter)


def test_ml_router_falls_back_to_rules_when_model_absent(tmp_path):
    # No model file → must degrade to rules, not raise.
    router = MLQueryRouter(tmp_path / "missing.joblib")
    assert router.classify("Sam Altman") == "entity_lookup"
    assert router.classify("exact verbatim quote") == "raw_evidence"


def test_ml_router_uses_model_when_present():
    class _StubModel:
        def predict(self, X):
            return ["cross_show_synthesis"]

    router = MLQueryRouter()
    router._model = _StubModel()
    router._loaded = True
    # Patch the embed step so we don't load MiniLM in a unit test.
    router._embed = lambda text: [0.0, 0.0, 0.0]  # type: ignore[method-assign]
    assert router.classify("anything") == "cross_show_synthesis"


def test_ml_router_rejects_out_of_vocab_label_falls_back():
    class _BadModel:
        def predict(self, X):
            return ["not_a_real_intent"]

    router = MLQueryRouter()
    router._model = _BadModel()
    router._loaded = True
    router._embed = lambda text: [0.0]  # type: ignore[method-assign]
    # Out-of-vocab prediction → rules fallback ("Sam Altman" → entity_lookup).
    assert router.classify("Sam Altman") == "entity_lookup"


class _FakeBackend:
    def search_bm25(self, query):
        return [ScoredResult("seg1", 1.0, 1, {}, "bm25", "segment")]

    def search_vector(self, query):
        return [ScoredResult("ins0", 1.0, 1, {}, "vector", "insight")]


def test_retrieval_layer_uses_injected_router():
    class _StubRouter:
        seen = []

        def classify(self, text):
            _StubRouter.seen.append(text)
            return "semantic"

    layer = RetrievalLayer(_FakeBackend(), router=_StubRouter())
    layer.retrieve("some query", [0.1, 0.2])
    assert _StubRouter.seen == ["some query"]  # router drove classification


def test_retrieval_layer_default_router_is_rules():
    # No router → falls back to rules classify_query; behaviour unchanged.
    layer = RetrievalLayer(_FakeBackend())
    out = layer.retrieve("Sam Altman", [0.1])
    assert {r.doc_id for r in out} == {"seg1", "ins0"}


def test_corrupt_model_file_falls_back_to_rules(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_text("not a joblib file", encoding="utf-8")
    router = MLQueryRouter(bad)
    assert router.classify("Sam Altman") == "entity_lookup"  # load failed → rules


def test_predict_exception_falls_back_to_rules():
    class _Raises:
        def predict(self, X):
            raise RuntimeError("inference boom")

    router = MLQueryRouter()
    router._model = _Raises()
    router._loaded = True
    router._embed = lambda text: [0.0]  # type: ignore[method-assign]
    assert router.classify("Sam Altman") == "entity_lookup"  # predict raised → rules
