"""Unit tests for #698 grounding-quality metrics in ``gi_scorer``."""

from __future__ import annotations

from typing import Any, Dict, List

from podcast_scraper.evaluation.gi_scorer import compute_gil_prediction_stats


def _gil_payload(
    *,
    insights: int,
    quotes: int,
    grounded_insight_ids: List[str] | None = None,
    nli_scores: List[float] | None = None,
) -> Dict[str, Any]:
    """Build a minimal ``output.gil`` payload for tests."""
    nodes: List[Dict[str, Any]] = [
        {"id": f"i{i}", "type": "Insight", "text": f"insight {i}"} for i in range(insights)
    ] + [{"id": f"q{j}", "type": "Quote", "text": f"quote {j}"} for j in range(quotes)]
    edges: List[Dict[str, Any]] = []
    grounded_ids = grounded_insight_ids or []
    scores = nli_scores or []
    for k, iid in enumerate(grounded_ids):
        edge: Dict[str, Any] = {
            "type": "SUPPORTED_BY",
            "source": iid,
            "target": f"q{k % max(quotes, 1)}",
        }
        if k < len(scores):
            edge["nli_score"] = scores[k]
        edges.append(edge)
    return {"nodes": nodes, "edges": edges}


class TestGroundingRate:
    def test_zero_insights_returns_zero(self) -> None:
        out = compute_gil_prediction_stats([])
        assert out["grounding_rate"] == 0.0
        assert out["quotes_per_insight_mean"] == 0.0
        assert out["mean_nli_score"] == 0.0

    def test_all_insights_grounded(self) -> None:
        gil = _gil_payload(
            insights=3,
            quotes=3,
            grounded_insight_ids=["i0", "i1", "i2"],
            nli_scores=[0.9, 0.8, 0.7],
        )
        out = compute_gil_prediction_stats([{"output": {"gil": gil}}])
        assert out["grounding_rate"] == 1.0
        assert abs(out["quotes_per_insight_mean"] - 1.0) < 1e-9
        assert abs(out["mean_nli_score"] - 0.8) < 1e-9

    def test_partial_grounding(self) -> None:
        gil = _gil_payload(
            insights=4,
            quotes=2,
            grounded_insight_ids=["i0", "i1"],
            nli_scores=[0.9, 0.7],
        )
        out = compute_gil_prediction_stats([{"output": {"gil": gil}}])
        assert out["grounding_rate"] == 0.5
        assert abs(out["quotes_per_insight_mean"] - 0.5) < 1e-9

    def test_no_grounding(self) -> None:
        gil = _gil_payload(insights=3, quotes=0)
        out = compute_gil_prediction_stats([{"output": {"gil": gil}}])
        assert out["grounding_rate"] == 0.0
        assert out["quotes_per_insight_mean"] == 0.0
        assert out["mean_nli_score"] == 0.0

    def test_aggregates_across_episodes(self) -> None:
        # Episode A: 4 insights, 2 grounded → 50%.
        # Episode B: 2 insights, 2 grounded → 100%.
        # Combined: 6 insights, 4 grounded → 66.67%.
        ep_a = _gil_payload(
            insights=4,
            quotes=2,
            grounded_insight_ids=["i0", "i1"],
            nli_scores=[0.9, 0.8],
        )
        ep_b = _gil_payload(
            insights=2,
            quotes=2,
            grounded_insight_ids=["i0", "i1"],
            nli_scores=[0.7, 0.6],
        )
        out = compute_gil_prediction_stats(
            [{"output": {"gil": ep_a}}, {"output": {"gil": ep_b}}],
        )
        assert abs(out["grounding_rate"] - (4 / 6)) < 1e-9
        # quotes_per_insight averaged per episode: (2/4 + 2/2) / 2 = 0.75
        assert abs(out["quotes_per_insight_mean"] - 0.75) < 1e-9

    def test_supported_by_lowercase_edge_type_recognised(self) -> None:
        """Edge type ``supported_by`` (lowercase) also counts as grounding."""
        gil: Dict[str, Any] = {
            "nodes": [
                {"id": "i0", "type": "Insight"},
                {"id": "q0", "type": "Quote"},
            ],
            "edges": [{"type": "supported_by", "source": "i0", "target": "q0", "nli_score": 0.85}],
        }
        out = compute_gil_prediction_stats([{"output": {"gil": gil}}])
        assert out["grounding_rate"] == 1.0
        assert abs(out["mean_nli_score"] - 0.85) < 1e-9

    def test_nli_score_missing_excluded_from_mean(self) -> None:
        """Edges without ``nli_score`` are still grounding edges but don't pollute mean."""
        gil: Dict[str, Any] = {
            "nodes": [
                {"id": "i0", "type": "Insight"},
                {"id": "q0", "type": "Quote"},
            ],
            "edges": [{"type": "SUPPORTED_BY", "source": "i0", "target": "q0"}],
        }
        out = compute_gil_prediction_stats([{"output": {"gil": gil}}])
        # Insight is grounded.
        assert out["grounding_rate"] == 1.0
        # No score → mean stays 0.
        assert out["mean_nli_score"] == 0.0

    def test_empty_predictions_dont_pollute(self) -> None:
        out = compute_gil_prediction_stats(
            [
                {"output": {}},  # no gil key
                {"output": {"gil": "not a dict"}},  # invalid type
            ],
        )
        assert out["episodes_with_gil"] == 0
        assert out["grounding_rate"] == 0.0

    def test_legacy_fields_still_present(self) -> None:
        """Existing callers depend on legacy ``avg_*`` count fields staying available."""
        gil = _gil_payload(insights=2, quotes=4, grounded_insight_ids=["i0"])
        out = compute_gil_prediction_stats([{"output": {"gil": gil}}])
        assert "avg_insight_nodes" in out
        assert "avg_quote_nodes" in out
        assert "avg_edges" in out
        assert "episodes_with_gil" in out
        assert out["episodes_with_gil"] == 1
