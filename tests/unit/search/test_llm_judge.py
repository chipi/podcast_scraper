"""Unit tests for the LLM-as-judge grader (RFC-057 / Step 2)."""

from __future__ import annotations

import pytest

from podcast_scraper.search.judged_eval import JudgmentRecord
from podcast_scraper.search.llm_judge import (
    build_grading_prompt,
    grade_record,
    grade_records,
    parse_grades,
)

pytestmark = pytest.mark.unit


def _rec():
    return JudgmentRecord(
        query="oil prices",
        intent="semantic",
        candidates=[
            {"doc_id": "a", "text": "crude oil rallied"},
            {"doc_id": "b", "text": "unrelated cooking show"},
        ],
        baseline_ranking=["a", "b"],
        hybrid_ranking=["b", "a"],
    )


def test_build_prompt_lists_query_and_candidates():
    p = build_grading_prompt(_rec())
    assert 'Query: "oil prices"' in p
    assert "id=a:" in p and "id=b:" in p


def test_parse_grades_extracts_clamps_and_filters():
    resp = 'Sure! Here you go: {"a": 2, "b": 0, "c": 5, "x": "nope"}'
    grades = parse_grades(resp, ["a", "b", "c"])
    assert grades["a"] == 2 and grades["b"] == 0
    assert grades["c"] == 2  # 5 clamped to 2
    assert "x" not in grades  # not a candidate id / unparseable


def test_parse_grades_handles_garbage():
    assert parse_grades("no json here", ["a"]) == {}
    assert parse_grades("{not valid json", ["a"]) == {}
    assert parse_grades("[1,2,3]", ["a"]) == {}  # not a dict


def test_grade_record_fills_relevance():
    rec = grade_record(_rec(), lambda prompt: '{"a": 2, "b": 0}')
    assert rec.relevance == {"a": 2, "b": 0}


def test_grade_record_survives_model_error():
    def _boom(prompt):
        raise RuntimeError("api down")

    rec = grade_record(_rec(), _boom)
    assert rec.relevance == {}  # failed grade → unjudged, not a crash


def test_grade_records_batch():
    recs = grade_records([_rec(), _rec()], lambda p: '{"a": 1, "b": 1}')
    assert all(r.relevance == {"a": 1, "b": 1} for r in recs)


def test_parse_grades_skips_non_int_grade_for_valid_id():
    # A valid candidate id with a non-integer grade is skipped (not crashed).
    assert parse_grades('{"a": "high", "b": 1}', ["a", "b"]) == {"b": 1}
