"""Unit tests for ``podcast_scraper.search.quality_metrics``.

The sibling of ``tests/unit/podcast_scraper/kg/test_quality_metrics.py``. Everything here
runs against pure functions or a stub retrieval layer — no LanceDB index, no ML extras, no
network — because a measurement tool that can only be tested against a live index is a
tool nobody runs in CI.

The distinction these tests defend hardest is ``None`` (NOT MEASURED) versus ``0.0``
(measured, and bad). Collapsing the two is how a quality report learns to say "fine" about
a corpus it could not score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.search.quality_metrics import (
    compute_search_quality_metrics,
    count_consensus_pairs_in_topk,
    enforce_rfc107_thresholds,
    hit_has_lifted,
    hit_tier,
    mrr_at_k,
    ndcg_at_k,
    QueryQualityResult,
    SearchQualityMetrics,
)

pytestmark = pytest.mark.unit


class _Hit:
    """Stand-in for a ScoredResult / CompoundResult."""

    def __init__(self, doc_id, source_tier="insight", insight=None, segment=None, payload=None):
        self.doc_id = doc_id
        self.source_tier = source_tier
        self.insight = insight
        self.segment = segment
        self.payload = payload or {}


class _Layer:
    """Retrieval layer stub: returns a canned ranking, classifies by lookup."""

    def __init__(self, hits, intents=None, fail=False):
        self._hits = hits
        self._intents = intents or {}
        self._fail = fail

    def retrieve(self, *, text, embedding, k, signals):
        if self._fail:
            raise RuntimeError("index unavailable")
        return self._hits[:k]

    def classify(self, text):
        return self._intents.get(text)


def _query(qid="q1", q="text", **kw):
    entry = {"id": qid, "q": q, "label_status": "unlabeled-seed"}
    entry.update(kw)
    return entry


# -- ranking metrics -------------------------------------------------------------------
def test_ndcg_is_one_when_every_relevant_doc_leads() -> None:
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k(["x", "y"], {"a"}, k=3) == 0.0


def test_ndcg_rewards_rank_not_just_presence() -> None:
    """The same hit lower down must score strictly worse, or ranking is unmeasured."""
    first = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
    third = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    assert first > third > 0.0


def test_mrr_is_the_reciprocal_of_the_first_relevant_rank() -> None:
    assert mrr_at_k(["x", "a"], {"a"}, k=10) == pytest.approx(0.5)
    assert mrr_at_k(["x", "y"], {"a"}, k=10) == 0.0


# -- hit accessors ---------------------------------------------------------------------
def test_hit_tier_falls_back_to_unknown_rather_than_crashing() -> None:
    assert hit_tier(_Hit("d", source_tier=None)) == "unknown"


def test_compound_lift_detected_from_either_shape() -> None:
    assert hit_has_lifted(_Hit("d", insight={"i": 1}, segment={"s": 1})) is True
    assert hit_has_lifted(_Hit("d", payload={"lifted": True})) is True
    assert hit_has_lifted(_Hit("d")) is False


# -- consensus pairs -------------------------------------------------------------------
def test_consensus_pairs_count_only_those_touching_top_k() -> None:
    pairs: list[dict[str, Any]] = [
        {"insight_a_id": "a", "insight_b_id": "z"},
        {"insight_a_id": "q", "insight_b_id": "r"},
        {"insight_a_id": "b", "insight_b_id": "z", "grounded": False},
    ]
    touching, grounded = count_consensus_pairs_in_topk(["a", "b"], pairs)
    assert (touching, grounded) == (2, 1)


# -- aggregation: None must survive as None --------------------------------------------
def test_unlabeled_queries_leave_ndcg_unmeasured_not_zero() -> None:
    metrics = compute_search_quality_metrics(
        corpus=Path("."),
        queries=[_query()],
        layer=_Layer([_Hit("a")]),
    )
    assert metrics.per_query[0].ndcg_at_k is None
    assert metrics.to_dict()["ndcg_at_k_mean"] is None


def test_labeled_queries_are_scored() -> None:
    metrics = compute_search_quality_metrics(
        corpus=Path("."),
        queries=[_query(expected_top_k_doc_ids=["a"], label_status="regression-anchor")],
        layer=_Layer([_Hit("a")]),
    )
    assert metrics.to_dict()["ndcg_at_k_mean"] == pytest.approx(1.0)


def test_retired_queries_are_skipped_and_reported() -> None:
    metrics = compute_search_quality_metrics(
        corpus=Path("."),
        queries=[_query(label_status="retired"), _query(qid="q2")],
        layer=_Layer([_Hit("a")]),
    )
    assert metrics.skipped_queries == ["q1"]
    assert len(metrics.per_query) == 1


def test_one_failing_query_does_not_lose_the_others() -> None:
    metrics = compute_search_quality_metrics(
        corpus=Path("."), queries=[_query()], layer=_Layer([], fail=True)
    )
    assert metrics.per_query == []
    assert metrics.errors and "index unavailable" in metrics.errors[0]


# -- seeding ---------------------------------------------------------------------------
def test_seeding_freezes_unlabeled_queries_only() -> None:
    unlabeled = _query(qid="new")
    audited = _query(
        qid="audited",
        q="other",
        label_status="human-audit",
        expected_top_k_doc_ids=["kept"],
    )
    compute_search_quality_metrics(
        corpus=Path("."),
        queries=[unlabeled, audited],
        layer=_Layer([_Hit("fresh")]),
        seed_labels=True,
    )
    assert unlabeled["expected_top_k_doc_ids"] == ["fresh"]
    assert unlabeled["label_status"] == "regression-anchor"
    # The human audit must survive a reseed untouched — otherwise re-running the tool
    # quietly replaces judgement with whatever search does today.
    assert audited["expected_top_k_doc_ids"] == ["kept"]
    assert audited["label_status"] == "human-audit"


# -- thresholds ------------------------------------------------------------------------
def test_thresholds_are_off_by_default_so_nothing_gates_by_accident() -> None:
    ok, failures = enforce_rfc107_thresholds(
        SearchQualityMetrics(per_query=[_result()]),
    )
    assert ok and failures == []


def test_a_floor_on_an_unmeasured_metric_fails() -> None:
    """Otherwise deleting the labels is the cheapest way to turn a gate green."""
    ok, failures = enforce_rfc107_thresholds(
        SearchQualityMetrics(per_query=[_result(ndcg=None)]), min_ndcg=0.5
    )
    assert not ok
    assert any("NOT MEASURED" in f for f in failures)


def test_a_floor_below_the_measured_value_passes_and_above_it_fails() -> None:
    metrics = SearchQualityMetrics(per_query=[_result(ndcg=0.6)])
    assert enforce_rfc107_thresholds(metrics, min_ndcg=0.5)[0] is True
    assert enforce_rfc107_thresholds(metrics, min_ndcg=0.7)[0] is False


def test_retrieval_errors_fail_enforcement() -> None:
    metrics = SearchQualityMetrics(per_query=[_result()], errors=["q1: boom"])
    ok, failures = enforce_rfc107_thresholds(metrics)
    assert not ok and any("retrieval errors" in f for f in failures)


def _result(ndcg=1.0):
    return QueryQualityResult(
        id="q1",
        q="text",
        intent_expected="semantic",
        intent_predicted="semantic",
        label_status="regression-anchor",
        ndcg_at_k=ndcg,
        mrr_at_k=ndcg,
        tier_counts={"insight": 1, "segment": 1},
        compound_lift_hits=0,
        transcript_hits=1,
        hit_count=2,
    )
