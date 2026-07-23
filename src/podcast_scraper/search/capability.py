"""Shared search-capability core (RFC-095 §3).

Lifts the structured-search assembly out of the HTTP route so **both** the
``GET /api/search`` handler and the MCP ``search_corpus`` tool call one tested function.
No HTTP / Pydantic / MCP coupling — returns plain dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .corpus_search import run_corpus_search
from .router import classify_query, tier_for_doc_type

# Agent-friendly evidence tiers → the indexer doc_types they map to (PRD-033 FR1.3).
_TIER_DOC_TYPES: Dict[str, Optional[List[str]]] = {
    "insight": ["insight"],
    "segment": ["transcript"],
    "both": None,
}


def doc_types_for_tier(tier: Optional[str]) -> Optional[List[str]]:
    """Map an evidence tier (``insight`` / ``segment`` / ``both``) to indexer doc_types."""
    if tier is None:
        return None
    return _TIER_DOC_TYPES.get(tier, None)


def structured_corpus_search(
    root: Path,
    query: str,
    *,
    doc_types: Optional[Sequence[str]] = None,
    feed: Optional[str] = None,
    since: Optional[str] = None,
    speaker: Optional[str] = None,
    topic: Optional[str] = None,
    episode_id: Optional[str] = None,
    grounded_only: bool = False,
    top_k: int = 10,
    embedding_model: Optional[str] = None,
    dedupe_kg_surfaces: bool = True,
) -> Dict[str, Any]:
    """Run hybrid corpus search and assemble the structured result.

    Single source of truth shared by ``GET /api/search`` and the MCP ``search_corpus``
    tool: runs :func:`run_corpus_search`, stamps each hit with its ``source_tier``
    (PRD-033 FR1.1) and the response with the detected ``query_type`` (FR1.4). Returns a
    plain dict: ``{query_type, results: [{doc_id, score, metadata, text, source_tier,
    supporting_quotes, lifted}], error, detail, lift_stats}``.
    """
    query_type = classify_query(query)
    outcome = run_corpus_search(
        root,
        query,
        doc_types=list(doc_types) if doc_types else None,
        feed=feed,
        since=since,
        speaker=speaker,
        topic=topic,
        episode_id=episode_id,
        grounded_only=grounded_only,
        top_k=top_k,
        index_path=None,
        embedding_model=embedding_model,
        dedupe_kg_surfaces=dedupe_kg_surfaces,
    )
    if outcome.error:
        return {
            "query_type": query_type,
            "results": [],
            "error": outcome.error,
            "detail": outcome.detail,
            "lift_stats": None,
        }
    results: List[Dict[str, Any]] = []
    for row in outcome.results:
        meta = dict(row.get("metadata") or {})
        lifted = row.get("lifted")
        results.append(
            {
                "doc_id": str(row.get("doc_id", "")),
                "score": float(row.get("score", 0.0)),
                "metadata": meta,
                "text": str(row.get("text") or ""),
                "source_tier": tier_for_doc_type(str(meta.get("doc_type") or "")),
                "supporting_quotes": row.get("supporting_quotes"),
                "lifted": lifted if isinstance(lifted, dict) else None,
            }
        )
    return {
        "query_type": query_type,
        "results": results,
        "error": None,
        "detail": None,
        "lift_stats": outcome.lift_stats,
    }
