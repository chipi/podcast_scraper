"""``search_corpus`` tool (RFC-095 slice 1) — hybrid two-tier corpus search."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..context import CorpusContext

# Referential parity (cross-surface pivot): map a hit's ``doc_type`` to the canonical id it
# carries + the tools that consume that id, so an agent can chain a search result straight
# into the graph/insight tools instead of the surfaces being islands.
_PIVOT_BY_DOC_TYPE: Dict[str, tuple[str, tuple[str, ...]]] = {
    "insight": ("insight", ("insight_detail", "related_insights")),
    "kg_entity": (
        "entity",
        ("entity_neighborhood", "insights_about_entity", "co_occurring_entities"),
    ),
    "kg_topic": (
        "topic",
        ("topic_entities", "who_said_about_topic", "related_topics", "topic_timeline"),
    ),
}
# Everything transcript/quote/summary-ish pivots on its episode.
_EPISODE_PIVOT = ("episode", ("episode_detail", "episode_insights", "episode_speaker_roster"))


def _pivot_for(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The pivot handle for one hit: the canonical id + which tools expand it."""
    doc_type = str(meta.get("doc_type") or "")
    kind, tools = _PIVOT_BY_DOC_TYPE.get(doc_type, _EPISODE_PIVOT)
    pivot_id = (
        str(meta.get("source_id") or "")
        if doc_type in _PIVOT_BY_DOC_TYPE
        else str(meta.get("episode_id") or "")
    )
    if not pivot_id:
        return None
    return {"id": pivot_id, "kind": kind, "expand_with": list(tools)}


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

    out = structured_corpus_search(
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
    # Stamp each hit with its cross-surface pivot handle (referential parity): the id +
    # the tools that expand it, so an agent can chain search -> graph/insight in one hop.
    results: List[Dict[str, Any]] = out.get("results") or []
    for row in results:
        row["pivot"] = _pivot_for(row.get("metadata") or {})
    return out
