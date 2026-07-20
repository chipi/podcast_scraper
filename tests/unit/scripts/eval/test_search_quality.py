"""Pure scoring-math tests for the Search v3 quality-eval harness.

Loads the module directly (not via the package) so the tests don't drag in
sentence-transformers / lancedb / RetrievalLayer — that's covered by the
integration path when the harness runs end-to-end. Here we cover the metric
math + the labelled-query-set shape guard.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval" / "search_quality.py"
QUERIES_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "viewer-validation-corpus" / "v3" / "search-queries.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("search_quality_module", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ndcg_perfect_ranking() -> None:
    mod = _load_module()
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b", "c"}
    assert mod._ndcg_at_k(retrieved, relevant, k=10) == pytest.approx(1.0)


def test_ndcg_reverse_ranking_is_less_than_one() -> None:
    mod = _load_module()
    retrieved = ["x", "y", "z", "a", "b", "c"]  # relevants at tail
    relevant = {"a", "b", "c"}
    got = mod._ndcg_at_k(retrieved, relevant, k=10)
    assert 0 < got < 1


def test_ndcg_no_relevant_is_zero() -> None:
    mod = _load_module()
    assert mod._ndcg_at_k(["a", "b"], {"x", "y"}, k=10) == 0.0


def test_ndcg_empty_relevant_is_zero() -> None:
    mod = _load_module()
    assert mod._ndcg_at_k(["a", "b"], set(), k=10) == 0.0


def test_mrr_first_position() -> None:
    mod = _load_module()
    assert mod._mrr_at_k(["a", "b"], {"a"}, k=10) == pytest.approx(1.0)


def test_mrr_third_position() -> None:
    mod = _load_module()
    assert mod._mrr_at_k(["a", "b", "c"], {"c"}, k=10) == pytest.approx(1.0 / 3)


def test_mrr_none_in_topk_is_zero() -> None:
    mod = _load_module()
    assert mod._mrr_at_k(["a", "b"], {"x"}, k=10) == 0.0


# ---- Labelled query set shape guard --------------------------------------

_EXPECTED_INTENTS = {
    "entity_lookup",
    "raw_evidence",
    "temporal_tracking",
    "cross_show_synthesis",
    "semantic",
}


def _load_queries() -> dict:
    data: dict = json.loads(QUERIES_PATH.read_text())
    return data


def test_query_set_has_at_least_25_queries() -> None:
    """RFC-107 §T2 asks for ≥25 queries covering all 5 intents."""
    data = _load_queries()
    assert len(data["queries"]) >= 25


def test_query_set_covers_all_five_intents() -> None:
    data = _load_queries()
    got = {q.get("intent_expected") for q in data["queries"]}
    missing = _EXPECTED_INTENTS - got
    assert not missing, f"missing intent classes: {sorted(missing)}"


def test_query_set_has_at_least_five_queries_per_intent() -> None:
    data = _load_queries()
    counts: dict[str, int] = {}
    for q in data["queries"]:
        intent = q.get("intent_expected")
        counts[intent] = counts.get(intent, 0) + 1
    low = {k: v for k, v in counts.items() if v < 5}
    assert not low, f"intent classes with <5 queries: {low}"


def test_every_query_carries_required_fields() -> None:
    data = _load_queries()
    required = {
        "id",
        "q",
        "intent_expected",
        "expected_top_k_doc_ids",
        "min_ndcg_at_10",
        "label_status",
    }
    for q in data["queries"]:
        missing = required - set(q.keys())
        assert not missing, f"query {q.get('id')} missing fields: {sorted(missing)}"


def test_query_ids_are_unique() -> None:
    data = _load_queries()
    ids = [q["id"] for q in data["queries"]]
    assert len(ids) == len(set(ids)), "duplicate query ids in search-queries.json"


def test_query_label_statuses_are_recognized() -> None:
    """label_status must be one of the documented lifecycle values."""
    allowed = {"unlabeled-seed", "regression-anchor", "human-audit", "retired"}
    data = _load_queries()
    for q in data["queries"]:
        assert (
            q["label_status"] in allowed
        ), f"query {q['id']} has unknown label_status: {q['label_status']!r}"


# ---- --seed-labels mode contract (Stabilization pass, 2026-07-20) --------


# ---- Enriched-answer + topic_consensus metric helpers -------------------


def test_count_consensus_pairs_no_pairs() -> None:
    mod = _load_module()
    assert mod._count_consensus_pairs_in_topk(["a", "b"], []) == (0, 0)


def test_count_consensus_pairs_no_topk() -> None:
    mod = _load_module()
    pairs = [{"insight_a_id": "x", "insight_b_id": "y"}]
    assert mod._count_consensus_pairs_in_topk([], pairs) == (0, 0)


def test_count_consensus_pairs_touch_and_grounded_flag() -> None:
    mod = _load_module()
    pairs = [
        {"insight_a_id": "a1", "insight_b_id": "z1"},  # touches (a1 in top-K)
        {"insight_a_id": "gone", "insight_b_id": "a2"},  # touches (a2 in top-K)
        {"insight_a_id": "gone1", "insight_b_id": "gone2"},  # miss
        {"insight_a_id": "a1", "insight_b_id": "z2", "grounded": False},  # touches but not grounded
    ]
    touch, grounded = mod._count_consensus_pairs_in_topk(["a1", "a2"], pairs)
    assert touch == 3
    assert grounded == 2


def test_count_enriched_sources_empty_and_populated() -> None:
    mod = _load_module()
    assert mod._count_enriched_sources(None) == (None, None)
    assert mod._count_enriched_sources({}) == (0, 0)
    e = {
        "sources": [
            {"grounded": True},
            {"grounded": False},
            {"grounded": True},
            {},  # missing key → not grounded
        ]
    }
    assert mod._count_enriched_sources(e) == (4, 2)


def test_enriched_groundedness_rate_null_when_no_calls() -> None:
    mod = _load_module()
    # Build QueryResult-like objects with None enriched counts
    Q = mod.QueryResult
    per = [
        Q(
            id="a",
            q="q",
            intent_expected=None,
            intent_predicted=None,
            label_status="regression-anchor",
            ndcg_at_10=None,
            mrr_at_10=None,
            tier_counts={},
            compound_lift_hits=0,
            transcript_hits=0,
            hit_count=0,
            top_doc_ids=[],
        )
    ]
    assert mod._enriched_groundedness_rate(per) is None


def test_enriched_groundedness_rate_populated() -> None:
    mod = _load_module()
    Q = mod.QueryResult
    per = [
        Q(
            id="a",
            q="q",
            intent_expected=None,
            intent_predicted=None,
            label_status="regression-anchor",
            ndcg_at_10=None,
            mrr_at_10=None,
            tier_counts={},
            compound_lift_hits=0,
            transcript_hits=0,
            hit_count=0,
            top_doc_ids=[],
            enriched_source_count=3,
            enriched_grounded_source_count=3,
        ),
        Q(
            id="b",
            q="q",
            intent_expected=None,
            intent_predicted=None,
            label_status="regression-anchor",
            ndcg_at_10=None,
            mrr_at_10=None,
            tier_counts={},
            compound_lift_hits=0,
            transcript_hits=0,
            hit_count=0,
            top_doc_ids=[],
            enriched_source_count=2,
            enriched_grounded_source_count=1,
        ),
    ]
    # 4/5 = 0.8
    assert mod._enriched_groundedness_rate(per) == pytest.approx(0.8)


def test_topic_consensus_precision_null_when_no_pairs_seen() -> None:
    mod = _load_module()
    Q = mod.QueryResult
    per = [
        Q(
            id="a",
            q="q",
            intent_expected=None,
            intent_predicted=None,
            label_status="regression-anchor",
            ndcg_at_10=None,
            mrr_at_10=None,
            tier_counts={},
            compound_lift_hits=0,
            transcript_hits=0,
            hit_count=0,
            top_doc_ids=[],
            consensus_pairs_in_topk=None,
            consensus_pairs_grounded=None,
        )
    ]
    assert mod._topic_consensus_precision(per) is None


def test_topic_consensus_precision_populated() -> None:
    mod = _load_module()
    Q = mod.QueryResult
    per = [
        Q(
            id="a",
            q="q",
            intent_expected=None,
            intent_predicted=None,
            label_status="regression-anchor",
            ndcg_at_10=None,
            mrr_at_10=None,
            tier_counts={},
            compound_lift_hits=0,
            transcript_hits=0,
            hit_count=0,
            top_doc_ids=[],
            consensus_pairs_in_topk=3,
            consensus_pairs_grounded=2,
        )
    ]
    # 2/3 ≈ 0.667
    assert mod._topic_consensus_precision(per) == pytest.approx(2 / 3)


def test_seed_labels_ships_shipped_query_set_carries_regression_anchor_labels() -> None:
    """The checked-in query set was seeded during S0 stabilization — every
    query must be regression-anchor with a non-empty expected_top_k_doc_ids.
    A future human-audit pass may flip individual queries to `human-audit`;
    this test still passes for those. But we never want to ship all-null
    labels again — the whole point of --seed-labels was to close that gap."""
    data = _load_queries()
    unlabeled = [q for q in data["queries"] if q["label_status"] == "unlabeled-seed"]
    assert not unlabeled, (
        f"queries reverted to unlabeled-seed: {[q['id'] for q in unlabeled]}. "
        "Re-seed with `python scripts/eval/search_quality.py --seed-labels`."
    )
    empty_labels = [
        q
        for q in data["queries"]
        if q["label_status"] in ("regression-anchor", "human-audit")
        and not q.get("expected_top_k_doc_ids")
    ]
    assert not empty_labels, (
        f"queries claim a label_status but have empty expected_top_k_doc_ids: "
        f"{[q['id'] for q in empty_labels]}"
    )
