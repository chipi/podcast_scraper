"""Scoring helpers for grounded insights (GIL) experiment predictions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from podcast_scraper.utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)


def _load_gil_reference_map(reference_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load per-episode reference GIL payloads from silver ``predictions.jsonl`` or gold JSON."""
    silver = reference_path / "predictions.jsonl"
    if silver.is_file():
        by_id: Dict[str, Dict[str, Any]] = {}
        try:
            raw = silver.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Could not read silver reference %s: %s",
                silver,
                format_exception_for_log(exc),
            )
            return {}
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = rec.get("episode_id")
            gil = rec.get("output", {}).get("gil")
            if eid is not None and isinstance(gil, dict):
                by_id[str(eid)] = gil
        return by_id

    by_file: Dict[str, Dict[str, Any]] = {}
    for gf in sorted(reference_path.glob("*.json")):
        if gf.name == "index.json":
            continue
        eid = gf.stem
        try:
            data = json.loads(gf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Skipping reference GIL file %s: %s",
                gf,
                format_exception_for_log(exc),
            )
            continue
        if isinstance(data, dict):
            by_file[eid] = data
    return by_file


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
    ref_map = _load_gil_reference_map(reference_path)

    scored = 0
    exact_triple_matches = 0
    missing_pred = 0
    missing_ref = 0

    for eid, gold in ref_map.items():
        if eid not in pred_by_id:
            continue
        if not isinstance(gold, dict):
            missing_ref += 1
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
            "gold_read_errors": missing_ref,
            "exact_triple_matches": exact_triple_matches,
        },
    }
