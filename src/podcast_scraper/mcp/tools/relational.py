"""Relational MCP tools (RFC-095 slice 2) — the seven RFC-094 traversals.

Wrap :mod:`podcast_scraper.search.relational_queries` over the cross-layer
``CorpusGraph``. Flat insight lists (positions / insights-about / related-insights) are
hybrid-re-ranked via the shared :mod:`relational_capability`; grouped results
(who-said / cross-show) keep structural order. All take canonical ids — call
``resolve_entity`` first to turn names into ids.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ...enrichment.enrichers._loaders import is_unresolved_speaker_placeholder
from ..context import CorpusContext


def _graph(ctx: CorpusContext) -> Any:
    from ...search.corpus_graph import get_corpus_graph

    # reconcile_hosts (#1056): names recurring network-feed hosts across a show.
    return get_corpus_graph(ctx.corpus_dir, derive_speaker_links=True, reconcile_hosts=True)


def _node(node: Any) -> Dict[str, Any]:
    return {
        "id": node.id,
        "type": node.type,
        "text": node.text,
        "show_id": node.show_id,
        "episode_id": node.episode_id,
    }


def _nodes(nodes: Any) -> List[Dict[str, Any]]:
    # #1193: drop unresolved diarization placeholders (``person:speaker-NN``) — the MCP read
    # surface had no guard, unlike the HTTP person surfaces, so ``Speaker 02`` leaked as a Person.
    # Only person placeholders match the pattern, so insight/topic/org nodes pass through untouched.
    return [_node(n) for n in nodes if not is_unresolved_speaker_placeholder(n.id, n.text)]


def _groups(groups: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    # Drop placeholder-person group keys (who-said groups by ``person:`` id); non-person keys
    # (e.g. ``show_id`` in cross-show synthesis) never match the pattern (#1193).
    return {
        key: _nodes(value)
        for key, value in groups.items()
        if not is_unresolved_speaker_placeholder(key)
    }


def _reranked(ctx: CorpusContext, graph: Any, subject_id: str, nodes: Any) -> List[Dict[str, Any]]:
    from ...search.relational_capability import rerank_relational_insights

    return _nodes(rerank_relational_insights(ctx.corpus_dir, graph, subject_id, nodes))


def person_positions(ctx: CorpusContext, person_id: str, k: int = 20) -> Dict[str, Any]:
    """Insights a person stated (``person:`` id), hybrid-re-ranked by relevance."""
    from ...search import relational_queries as rq

    if is_unresolved_speaker_placeholder(person_id):
        return {"subject": person_id, "results": []}  # #1193: never serve a placeholder's data
    graph = _graph(ctx)
    return {
        "subject": person_id,
        "results": _reranked(ctx, graph, person_id, rq.positions_of(graph, person_id, k=k)),
    }


def insights_about_entity(ctx: CorpusContext, entity_id: str, k: int = 20) -> Dict[str, Any]:
    """Insights that mention a person/org (``person:`` / ``org:`` id), hybrid-re-ranked."""
    from ...search import relational_queries as rq

    if is_unresolved_speaker_placeholder(entity_id):
        return {"subject": entity_id, "results": []}  # #1193
    graph = _graph(ctx)
    return {
        "subject": entity_id,
        "results": _reranked(ctx, graph, entity_id, rq.insights_about(graph, entity_id, k=k)),
    }


def related_insights(ctx: CorpusContext, insight_id: str, k: int = 20) -> Dict[str, Any]:
    """Sibling insights sharing a topic or mentioned entity with *insight_id*, hybrid-re-ranked."""
    from ...search import relational_queries as rq

    graph = _graph(ctx)
    return {
        "subject": insight_id,
        "results": _reranked(ctx, graph, insight_id, rq.related_insights(graph, insight_id, k=k)),
    }


def topic_entities(ctx: CorpusContext, topic_id: str, k: int = 20) -> Dict[str, Any]:
    """Entities a topic's insights mention (``topic:`` id), ranked by mention frequency."""
    from ...search import relational_queries as rq

    graph = _graph(ctx)
    return {
        "subject": topic_id,
        "results": _nodes(rq.entities_in_topic(graph, topic_id, k=k)),
    }


def show_episodes(ctx: CorpusContext, podcast_id: str, k: int = 20) -> Dict[str, Any]:
    """A show's episodes (``podcast:`` id; HAS_EPISODE)."""
    from ...search import relational_queries as rq

    graph = _graph(ctx)
    return {
        "subject": podcast_id,
        "results": _nodes(rq.episodes_of(graph, podcast_id, k=k)),
    }


def who_said_about_topic(ctx: CorpusContext, topic_id: str, k: int = 20) -> Dict[str, Any]:
    """Per-person insights about a topic (``topic:`` id), grouped by ``person:`` id."""
    from ...search import relational_queries as rq

    graph = _graph(ctx)
    return {"subject": topic_id, "groups": _groups(rq.who_said(graph, topic_id, k=k))}


def cross_show_synthesis(ctx: CorpusContext, topic_id: str, per_show: int = 1) -> Dict[str, Any]:
    """Top insight from each distinct show covering a topic (``topic:`` id)."""
    from ...search import relational_queries as rq

    graph = _graph(ctx)
    return {
        "subject": topic_id,
        "groups": _groups(rq.cross_show_synthesis(graph, topic_id, per_show=per_show)),
    }
