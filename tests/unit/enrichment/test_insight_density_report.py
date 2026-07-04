"""Unit tests for the insight-density eval report (#1140)."""

from __future__ import annotations

import pytest

from scripts.eval.insight_density_report import aggregate, classify, counts_of, render

pytestmark = pytest.mark.unit


def test_aggregate_sums_corpus_and_shows_ignoring_untimed() -> None:
    items = [
        ("Show A", {"early": 5, "mid": 3, "late": 2}),
        ("Show A", {"early": 1, "mid": 1, "late": 0}),
        ("Show B", {"early": 0, "mid": 0, "late": 0}),  # no timed insights → ignored
    ]
    agg = aggregate(items)
    assert agg["episodes"] == 2  # the all-zero episode doesn't count
    assert agg["corpus"] == {"early": 6, "mid": 4, "late": 2}
    assert agg["total_insights"] == 12
    assert agg["shows"]["Show A"]["episodes"] == 2
    assert "Show B" not in agg["shows"]


def test_classify_front_even_back() -> None:
    assert classify({"early": 8, "mid": 1, "late": 1}) == "front-loaded"
    assert classify({"early": 1, "mid": 1, "late": 8}) == "back-loaded"
    assert classify({"early": 3, "mid": 4, "late": 3}) == "even"


def test_counts_of_reads_data_counts_and_tolerates_junk() -> None:
    assert counts_of({"data": {"counts": {"early": 2, "mid": 1, "late": 0}}}) == {
        "early": 2,
        "mid": 1,
        "late": 0,
    }
    assert counts_of({}) == {"early": 0, "mid": 0, "late": 0}
    assert counts_of("not a dict") == {"early": 0, "mid": 0, "late": 0}


def test_render_includes_distribution_and_show_classification() -> None:
    out = render(aggregate([("Show A", {"early": 8, "mid": 1, "late": 1})]))
    assert "Corpus distribution" in out
    assert "Show A" in out
    assert "front-loaded" in out
    assert "1 episodes" in out or "1 ep" in out
