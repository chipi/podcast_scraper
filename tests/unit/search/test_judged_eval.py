"""Unit tests for the judged hybrid-vs-FAISS eval harness (RFC-057 / C)."""

from __future__ import annotations

import pytest

from podcast_scraper.search.judged_eval import (
    build_judgment_template,
    JudgmentRecord,
    score_from_judgments,
)

pytestmark = pytest.mark.unit


def test_build_template_unions_candidates_with_per_backend_ranks():
    faiss = {"q": [("a", "text-a"), ("b", "text-b")]}
    hybrid = {"q": [("b", "text-b"), ("c", "text-c")]}
    records = build_judgment_template(
        ["q"],
        faiss_ranks=lambda q: faiss[q],
        hybrid_ranks=lambda q: hybrid[q],
        intent_of=lambda q: "semantic",
        k=10,
    )
    rec = records[0]
    assert rec.intent == "semantic"
    assert rec.faiss_ranking == ["a", "b"] and rec.hybrid_ranking == ["b", "c"]
    cand = {c["doc_id"]: c for c in rec.candidates}
    assert set(cand) == {"a", "b", "c"}  # unioned
    assert cand["a"]["faiss_rank"] == 1 and cand["a"]["hybrid_rank"] is None
    assert cand["b"]["faiss_rank"] == 2 and cand["b"]["hybrid_rank"] == 1


def test_score_rewards_ranking_relevant_doc_higher():
    # Same relevant doc "good" (grade 2); hybrid ranks it #1, FAISS ranks it #3.
    rec = JudgmentRecord(
        query="q",
        intent="semantic",
        faiss_ranking=["x", "y", "good"],
        hybrid_ranking=["good", "x", "y"],
        relevance={"good": 2, "x": 0, "y": 0},
    )
    scores = score_from_judgments([rec], k=10)
    assert scores["hybrid"]["ndcg"] > scores["faiss"]["ndcg"]
    assert scores["hybrid"]["ndcg"] == pytest.approx(1.0)  # relevant doc at rank 1 → ideal
    assert scores["_judged_count"]["n"] == 1.0


def test_recall_counts_relevant_in_top_k():
    rec = JudgmentRecord(
        query="q",
        intent="semantic",
        faiss_ranking=["a", "b", "c", "d"],
        hybrid_ranking=["a", "b", "c", "d"],
        relevance={"a": 1, "d": 1},  # two relevant
    )
    scores = score_from_judgments([rec], k=2)  # top-2 contains only "a"
    assert scores["faiss"]["recall"] == pytest.approx(0.5)


def test_unjudged_records_are_skipped():
    rec = JudgmentRecord(
        query="q", intent="semantic", faiss_ranking=["a"], hybrid_ranking=["a"], relevance={"a": 0}
    )
    scores = score_from_judgments([rec], k=10)
    assert scores["_judged_count"]["n"] == 0.0  # no positive grade → no signal
    assert scores["hybrid"]["ndcg"] == 0.0
