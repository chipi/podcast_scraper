"""Shared relational-capability core (RFC-095 §3).

Lifts the hybrid re-rank out of the HTTP route so **both** the ``/api/relational/*``
handlers and the MCP relational tools layer the same entity-scoped hybrid ranking over
the structural graph order (PRD-033: structure via graph, ranking via hybrid). Best-effort
— degrades to structural order when no index is available. No HTTP / MCP coupling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from . import relational_queries as rq
from .corpus_search import run_corpus_search


def hybrid_insight_scores(root: Path, label: str) -> Dict[str, float]:
    """Hybrid insight scores keyed by GIL node id (``metadata.source_id``), best-effort.

    Runs one insight-scoped search for *label* and maps each hit's source node id to its
    score. Any failure (no index, error) yields ``{}`` → the caller keeps structural order.
    Never raises.
    """
    if not label.strip():
        return {}
    try:
        outcome = run_corpus_search(root, label, doc_types=["insight"], top_k=50)
    except Exception:  # noqa: BLE001 - re-rank is best-effort; degrade to structural order
        return {}
    if outcome.error:
        return {}
    scores: Dict[str, float] = {}
    for row in outcome.results:
        source_id = str((row.get("metadata") or {}).get("source_id") or "")
        if source_id:
            scores[source_id] = float(row.get("score") or 0.0)
    return scores


def rerank_relational_insights(
    root: Path,
    graph: rq.GraphLike,
    subject_id: str,
    nodes: List[rq.RelatedNode],
) -> List[rq.RelatedNode]:
    """Best-effort hybrid re-rank of insight *nodes* by relevance to the subject label.

    No-op (structural order) when there is nothing to re-rank or the index yields no
    matching scores.
    """
    if len(nodes) <= 1:
        return nodes
    scores = hybrid_insight_scores(root, rq.node_label(graph, subject_id))
    return rq.rerank_by_relevance(nodes, scores)
