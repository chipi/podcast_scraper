"""Scoring helpers for knowledge graph (KG) experiment predictions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from podcast_scraper.utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)


def _load_kg_reference_map(reference_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load per-episode reference KG payloads from silver ``predictions.jsonl`` or gold JSON."""
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
            kg = rec.get("output", {}).get("kg")
            if eid is not None and isinstance(kg, dict):
                by_id[str(eid)] = kg
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
                "Skipping reference KG file %s: %s",
                gf,
                format_exception_for_log(exc),
            )
            continue
        if isinstance(data, dict):
            by_file[eid] = data
    return by_file


def _count_kg_nodes_edges(kg: Dict[str, Any]) -> Tuple[int, int]:
    """Return node count and edge count."""
    raw_n = kg.get("nodes")
    nodes: List[Any] = raw_n if isinstance(raw_n, list) else []
    raw_e = kg.get("edges")
    edges: List[Any] = raw_e if isinstance(raw_e, list) else []
    return len(nodes), len(edges)


def compute_kg_prediction_stats(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate intrinsic-style stats from ``output.kg`` payloads."""
    node_counts: List[int] = []
    edge_counts: List[int] = []
    with_payload = 0
    for pred in predictions:
        kg = pred.get("output", {}).get("kg")
        if not isinstance(kg, dict):
            continue
        with_payload += 1
        n, e = _count_kg_nodes_edges(kg)
        node_counts.append(n)
        edge_counts.append(e)

    def _avg(vals: List[int]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    return {
        "episodes_with_kg": with_payload,
        "avg_nodes": _avg(node_counts),
        "avg_edges": _avg(edge_counts),
    }


def compute_kg_vs_reference_metrics(
    predictions: List[Dict[str, Any]],
    reference_id: str,
    reference_path: Path,
    *,
    dataset_id: str,
) -> Dict[str, Any]:
    """Compare predictions to per-episode gold JSON under ``reference_path``.

    Gold files: ``{episode_id}.json`` with the same shape as ``output.kg``.

    Args:
        predictions: Loaded prediction records.
        reference_id: Reference set identifier.
        reference_path: Directory containing gold JSON files.
        dataset_id: Dataset identifier for reporting.

    Returns:
        Metrics dict stored under ``vs_reference[reference_id]``.
    """
    pred_by_id = {str(p.get("episode_id")): p for p in predictions if p.get("episode_id")}
    ref_map = _load_kg_reference_map(reference_path)

    scored = 0
    exact_matches = 0
    missing_pred = 0
    missing_gold_read = 0

    for eid, gold in ref_map.items():
        if eid not in pred_by_id:
            continue
        if not isinstance(gold, dict):
            missing_gold_read += 1
            continue

        pred_kg = pred_by_id[eid].get("output", {}).get("kg")
        if not isinstance(pred_kg, dict):
            missing_pred += 1
            continue

        gn, ge = _count_kg_nodes_edges(gold)
        pn, pe = _count_kg_nodes_edges(pred_kg)
        scored += 1
        if (gn, ge) == (pn, pe):
            exact_matches += 1

    total = len(pred_by_id)
    return {
        "schema": "metrics_kg_v1",
        "task": "knowledge_graph",
        "reference_id": reference_id,
        "dataset_id": dataset_id,
        "scored_episodes": {"scored": scored, "total": total},
        "node_edge_count_exact_match_rate": float(exact_matches / scored) if scored else 0.0,
        "counts": {
            "missing_pred_kg": missing_pred,
            "gold_read_errors": missing_gold_read,
            "exact_count_matches": exact_matches,
        },
    }
