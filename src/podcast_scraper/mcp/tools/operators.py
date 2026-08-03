"""Search result-set operators — cluster / consensus over a query's hits.

Parity with ``GET /api/search?operator=cluster|consensus``: run the same hybrid search,
then group the hits by cluster or filter cross-speaker consensus pairs to the surfaced
topics. Query in, operated result-set out (the agent doesn't hand-roll hit dicts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..context import CorpusContext


def _search_hits(
    ctx: CorpusContext, query: str, tier: str, top_k: int
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the shared search; return (error_envelope_or_None, hit dicts for the operators)."""
    from ...search.capability import doc_types_for_tier, structured_corpus_search

    cleaned = (query or "").strip()
    if not cleaned:
        return {"query": cleaned, "error": "empty_query"}, []
    out = structured_corpus_search(
        ctx.corpus_dir,
        cleaned,
        doc_types=doc_types_for_tier(tier),
        top_k=max(1, min(100, int(top_k))),
    )
    if out.get("error"):
        return {"query": cleaned, "error": out["error"], "detail": out.get("detail")}, []
    hits = [
        {"doc_id": r.get("doc_id", ""), "metadata": r.get("metadata") or {}}
        for r in (out.get("results") or [])
    ]
    return None, hits


def cluster_search(
    ctx: CorpusContext, query: str, tier: str = "both", top_k: int = 20
) -> Dict[str, Any]:
    """Search, then group the hits by topic/theme cluster (``operator=cluster`` parity).

    Returns clustered ``groups`` (largest first, ``ungrouped`` last) — a structured view of
    "what themes does this query surface", instead of a flat ranked list.
    """
    from ...search.operators import cluster_hits

    err, hits = _search_hits(ctx, query, tier, top_k)
    if err is not None:
        return err
    groups = cluster_hits(hits, Path(ctx.corpus_dir))
    return {"query": query.strip(), "groups": groups, "hit_count": len(hits)}


def consensus_search(
    ctx: CorpusContext, query: str, tier: str = "both", top_k: int = 20, max_pairs: int = 20
) -> Dict[str, Any]:
    """Search, then surface cross-speaker consensus pairs among the hit topics.

    Filters the ``topic_consensus`` enricher to topics this query surfaced — "where do
    speakers agree on what this query is about". Empty when the enricher output is absent.
    """
    from ...search.operators import consensus_pairs_for_hits

    err, hits = _search_hits(ctx, query, tier, top_k)
    if err is not None:
        return err
    pairs = consensus_pairs_for_hits(hits, Path(ctx.corpus_dir), max_pairs=max(1, int(max_pairs)))
    return {"query": query.strip(), "consensus_pairs": pairs, "hit_count": len(hits)}
