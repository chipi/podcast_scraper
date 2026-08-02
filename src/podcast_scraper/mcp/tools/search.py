"""``search_corpus`` tool (RFC-095 slice 1) — hybrid two-tier corpus search."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..context import CorpusContext


def search_corpus(
    ctx: CorpusContext,
    query: str,
    *,
    tier: str = "both",
    grounded_only: bool = False,
    feed: Optional[str] = None,
    since: Optional[str] = None,
    speaker: Optional[str] = None,
    topic: Optional[str] = None,
    episode_id: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Hybrid two-tier corpus search returning grounded evidence.

    ``tier`` is the evidence tier: ``insight`` (synthesized), ``segment`` (raw transcript),
    or ``both``. ``speaker``/``topic``/``episode_id`` scope the search (parity with
    ``GET /api/search``): pass a resolved ``person:``/``topic:`` id (see ``resolve_entity``)
    or an episode id to restrict hits. Returns ``{query_type, results: [{doc_id,
    source_tier, score, text, metadata, supporting_quotes?, lifted?}], error, lift_stats}``
    — the same structured shape the viewer's ``/api/search`` produces. Empty query →
    ``error: "empty_query"``.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return {
            "query_type": "semantic",
            "results": [],
            "error": "empty_query",
            "detail": None,
            "lift_stats": None,
        }
    from ...search.capability import doc_types_for_tier, structured_corpus_search

    return structured_corpus_search(
        ctx.corpus_dir,
        cleaned,
        doc_types=doc_types_for_tier(tier),
        grounded_only=grounded_only,
        feed=feed,
        since=since,
        speaker=speaker,
        topic=topic,
        episode_id=episode_id,
        top_k=max(1, min(100, int(top_k))),
    )
