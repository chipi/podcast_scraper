"""Scoring helpers for grounded insights (GIL) experiment predictions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _count_gil_nodes(gil: Dict[str, Any]) -> Tuple[int, int, int]:
    """Return counts of Insight nodes, Quote nodes, and edges."""
    raw_n = gil.get("nodes")
    nodes: List[Any] = raw_n if isinstance(raw_n, list) else []
    raw_e = gil.get("edges")
    edges: List[Any] = raw_e if isinstance(raw_e, list) else []
    insights = sum(1 for n in nodes if isinstance(n, dict) and n.get("type") == "Insight")
    quotes = sum(1 for n in nodes if isinstance(n, dict) and n.get("type") == "Quote")
    return insights, quotes, len(edges)


def compute_gil_prediction_stats(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate intrinsic-style stats from ``output.gil`` payloads."""
    insight_counts: List[int] = []
    quote_counts: List[int] = []
    edge_counts: List[int] = []
    with_payload = 0
    for pred in predictions:
        gil = pred.get("output", {}).get("gil")
        if not isinstance(gil, dict):
            continue
        with_payload += 1
        i, q, e = _count_gil_nodes(gil)
        insight_counts.append(i)
        quote_counts.append(q)
        edge_counts.append(e)

    def _avg(vals: List[int]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    return {
        "episodes_with_gil": with_payload,
        "avg_insight_nodes": _avg(insight_counts),
        "avg_quote_nodes": _avg(quote_counts),
        "avg_edges": _avg(edge_counts),
    }


def compute_gil_vs_reference_metrics(
    predictions: List[Dict[str, Any]],
    reference_id: str,
    reference_path: Path,
    *,
    dataset_id: str,
) -> Dict[str, Any]:
    """Compare predictions to per-episode gold JSON files under ``reference_path``.

    Gold files: ``{episode_id}.json`` with the same shape as ``output.gil`` (full GIL dict).

    Args:
        predictions: Loaded prediction records.
        reference_id: Reference set identifier.
        reference_path: Directory containing gold JSON files.
        dataset_id: Dataset identifier for reporting.

    Returns:
        Metrics dict stored under ``vs_reference[reference_id]``.
    """
    pred_by_id = {str(p.get("episode_id")): p for p in predictions if p.get("episode_id")}
    gold_files = sorted(reference_path.glob("*.json"))
    gold_files = [f for f in gold_files if f.name != "index.json"]

    scored = 0
    exact_triple_matches = 0
    missing_pred = 0
    missing_gold_read = 0

    for gf in gold_files:
        eid = gf.stem
        if eid not in pred_by_id:
            continue
        try:
            gold = json.loads(gf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping gold %s: %s", gf, exc)
            missing_gold_read += 1
            continue
        if not isinstance(gold, dict):
            missing_gold_read += 1
            continue

        pred_gil = pred_by_id[eid].get("output", {}).get("gil")
        if not isinstance(pred_gil, dict):
            missing_pred += 1
            continue

        gi, gq, ge = _count_gil_nodes(gold)
        pi, pq, pe = _count_gil_nodes(pred_gil)
        scored += 1
        if (gi, gq, ge) == (pi, pq, pe):
            exact_triple_matches += 1

    total = len(pred_by_id)
    return {
        "schema": "metrics_gil_v1",
        "task": "grounded_insights",
        "reference_id": reference_id,
        "dataset_id": dataset_id,
        "scored_episodes": {"scored": scored, "total": total},
        "insight_quote_edge_count_exact_match_rate": (
            float(exact_triple_matches / scored) if scored else 0.0
        ),
        "counts": {
            "missing_pred_gil": missing_pred,
            "gold_read_errors": missing_gold_read,
            "exact_triple_matches": exact_triple_matches,
        },
    }
