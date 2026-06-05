"""Relational MCP tools (RFC-095 slice 2) — the seven RFC-094 traversals.

Wrap :mod:`podcast_scraper.search.relational_queries` over the cross-layer
``CorpusGraph``. Flat insight lists (positions / insights-about / related-insights) are
hybrid-re-ranked via the shared :mod:`relational_capability`; grouped results
(who-said / cross-show) keep structural order. All take canonical ids — call
``resolve_entity`` first to turn names into ids.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..context import CorpusContext


def _graph(ctx: CorpusContext) -> Any:
    from ...search.corpus_graph import get_corpus_graph

    return get_corpus_graph(ctx.corpus_dir, derive_speaker_links=True)


def _node(node: Any) -> Dict[str, Any]:
    return {
        "id": node.id,
        "type": node.type,
        "text": node.text,
        "show_id": node.show_id,
        "episode_id": node.episode_id,
    }


def _nodes(nodes: Any) -> List[Dict[str, Any]]:
    return [_node(n) for n in nodes]


def _groups(groups: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    return {key: _nodes(value) for key, value in groups.items()}


def _reranked(ctx: CorpusContext, graph: Any, subject_id: str, nodes: Any) -> List[Dict[str, Any]]:
    from ...search.relational_capability import rerank_relational_insights

    return _nodes(rerank_relational_insights(ctx.corpus_dir, graph, subject_id, nodes))


def person_positions(ctx: CorpusContext, person_id: str, k: int = 20) -> Dict[str, Any]:
    """Insights a person stated (``person:`` id), hybrid-re-ranked by relevance."""
    from ...search import relational_queries as rq

    graph = _graph(ctx)
    return {
        "subject": person_id,
        "results": _reranked(ctx, graph, person_id, rq.positions_of(graph, person_id, k=k)),
    }


def insights_about_entity(ctx: CorpusContext, entity_id: str, k: int = 20) -> Dict[str, Any]:
    """Insights that mention a person/org (``person:`` / ``org:`` id), hybrid-re-ranked."""
    from ...search import relational_queries as rq

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
