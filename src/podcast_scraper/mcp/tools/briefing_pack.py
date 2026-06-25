"""``corpus_briefing_pack`` MCP tool (RFC-093 / #861).

The "one opinionated synthesized tool" the RFC describes — packages
``structured_corpus_search`` output through the existing
``build_briefing_pack`` (LITM-positioned: critical at top, supporting
in the middle, caveats at the end) so agents on Claude Desktop /
Cursor / autoresearch get a ready-to-paste-into-context brief
instead of raw search hits.

Lives next to the other RFC-095 tools; registered in
``mcp/server.py``. Returns the same MCP envelope shape as the rest
(uniform ``{ok, data, note}`` after the server wrapper).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..context import CorpusContext

DEFAULT_MAX_TOKENS = 8000
DEFAULT_TOP_K = 10


def _dict_to_scored_result(row: Dict[str, Any]) -> Any:
    """Adapter — search-tool dict shape -> ScoredResult that
    ``build_briefing_pack`` expects.

    The structured_corpus_search response is plain JSON. The pack
    builder consumes the typed Result dataclasses. Lazy-import
    so the MCP tool module loads without the search backend on the
    import path.
    """
    from ...search.backend import ScoredResult

    metadata = dict(row.get("metadata") or {})
    payload: Dict[str, Any] = {
        "text": row.get("text") or "",
        **metadata,
    }
    if row.get("supporting_quotes"):
        payload["supporting_quotes"] = row["supporting_quotes"]
    if row.get("lifted"):
        payload["lifted"] = row["lifted"]
    return ScoredResult(
        doc_id=str(row.get("doc_id", "")),
        score=float(row.get("score") or 0.0),
        rank=int(row.get("rank") or 0),
        payload=payload,
        signal=str(row.get("signal", "rrf")),
        source_tier=str(row.get("source_tier", "segment")),
    )


def corpus_briefing_pack(
    ctx: CorpusContext,
    query: str,
    *,
    tier: str = "both",
    grounded_only: bool = False,
    feed: Optional[str] = None,
    since: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """Build a LITM-positioned briefing pack for *query* over the corpus.

    Runs ``structured_corpus_search`` (same retrieval the
    ``search_corpus`` tool uses), feeds the results through
    ``build_briefing_pack``, returns:

    ``{
        query, query_type,
        rendered: <LITM-ordered text the agent can paste into context>,
        token_count: int, max_tokens: int,
        top_insight_id: str | null,
        supporting_segment_ids: [str],
        coverage_summary: {show_ids, episode_count, date_range},
        confidence_p50: float,
        result_count: int,
    }``

    Empty query returns ``{error: "empty_query"}`` — matches
    ``search_corpus`` semantics.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return {
            "query": cleaned,
            "query_type": "semantic",
            "rendered": "",
            "error": "empty_query",
        }

    # Lazy imports — keep the tool module's load cost down for
    # ``mcp list-tools`` style introspection.
    from ...search.capability import doc_types_for_tier, structured_corpus_search
    from ...search.context_pack import build_briefing_pack

    search = structured_corpus_search(
        ctx.corpus_dir,
        cleaned,
        doc_types=doc_types_for_tier(tier),
        grounded_only=grounded_only,
        feed=feed,
        since=since,
        top_k=max(1, min(100, int(top_k))),
    )
    if search.get("error"):
        return {
            "query": cleaned,
            "query_type": search.get("query_type", "semantic"),
            "rendered": "",
            "error": search["error"],
            "detail": search.get("detail"),
        }

    rows: List[Dict[str, Any]] = search.get("results") or []
    typed_results = [_dict_to_scored_result(r) for r in rows]
    pack = build_briefing_pack(
        cleaned,
        search.get("query_type") or "semantic",
        typed_results,
        max_tokens=int(max_tokens),
    )

    return {
        "query": cleaned,
        "query_type": pack.query_type,
        "rendered": pack.render(),
        "token_count": pack.token_count,
        "max_tokens": int(max_tokens),
        "top_insight_id": pack.top_insight.doc_id if pack.top_insight else None,
        "supporting_segment_ids": [s.doc_id for s in pack.supporting_segments],
        "coverage_summary": pack.coverage_summary,
        "confidence_p50": pack.confidence_p50,
        "result_count": len(typed_results),
    }
