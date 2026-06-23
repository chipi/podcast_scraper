"""GET /api/relational/* — relational corpus queries over the typed graph (RFC-094 §2, #882).

Thin FastAPI routes wrapping ``search.relational_queries``, one per query. Each traverses
the typed meaning-bearing edges of the cross-layer ``CorpusGraph`` (#874) to answer a
PRD-033 surface question — a person's stated positions, who said what about a topic, the
entities an insight mentions, a show's episodes, and the cross-show synthesis of a topic.

Corpus resolution and the ``path`` override mirror ``/api/search``. The graph is built
with ``derive_speaker_links=True`` so the synthesized ``STATES`` edges (Person→Insight)
exist for ``positions`` / ``who-said``; it is process-cached by ``get_corpus_graph``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query, Request

from podcast_scraper.search import relational_queries as rq
from podcast_scraper.search.corpus_graph import get_corpus_graph
from podcast_scraper.search.relational_capability import rerank_relational_insights
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import (
    RelatedNodeModel,
    RelationalGroupedResponse,
    RelationalListResponse,
)

router = APIRouter(tags=["relational"])


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path | None:
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, fallback)
    return fallback


def _root_or_none(request: Request, path: str | None) -> Path | None:
    return _resolve_corpus_root(path, getattr(request.app.state, "output_dir", None))


def _graph_or_none(request: Request, path: str | None):
    """Resolve the corpus root and return its cached graph, or ``None`` if unconfigured."""
    root = _root_or_none(request, path)
    if root is None:
        return None
    # reconcile_hosts (#1056): names recurring network-feed hosts across a show so the
    # viewer's connectivity surfaces match the MCP's (one relational layer, two front-ends).
    return get_corpus_graph(root, derive_speaker_links=True, reconcile_hosts=True)


def _maybe_rerank(
    request: Request,
    path: str | None,
    graph,
    subject_id: str,
    nodes: list[rq.RelatedNode],
) -> list[rq.RelatedNode]:
    """Best-effort hybrid re-rank (shared capability), no-op when no corpus root resolves."""
    root = _root_or_none(request, path)
    if root is None:
        return nodes
    return rerank_relational_insights(root, graph, subject_id, nodes)


def _node(n: rq.RelatedNode) -> RelatedNodeModel:
    return RelatedNodeModel(
        id=n.id, type=n.type, text=n.text, show_id=n.show_id, episode_id=n.episode_id
    )


def _group(groups: dict[str, list[rq.RelatedNode]]) -> dict[str, list[RelatedNodeModel]]:
    return {key: [_node(n) for n in nodes] for key, nodes in groups.items()}


@router.get("/relational/positions", response_model=RelationalListResponse)
async def positions(
    request: Request,
    person: str = Query(min_length=1, description="Canonical person id, e.g. person:jane-doe."),
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
    k: int = Query(default=20, ge=1, le=200),
) -> RelationalListResponse:
    """Insights a person *stated* (Person Landing, PRD-033 FR4.1)."""
    graph = _graph_or_none(request, path)
    if graph is None:
        return RelationalListResponse(subject=person, error="no_corpus_path")
    nodes = _maybe_rerank(request, path, graph, person, rq.positions_of(graph, person, k=k))
    return RelationalListResponse(subject=person, results=[_node(n) for n in nodes])


@router.get("/relational/insights-about", response_model=RelationalListResponse)
async def insights_about(
    request: Request,
    entity: str = Query(min_length=1, description="Canonical person/org id the insights mention."),
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
    k: int = Query(default=20, ge=1, le=200),
) -> RelationalListResponse:
    """Insights that *mention* a person/org (entity grounding)."""
    graph = _graph_or_none(request, path)
    if graph is None:
        return RelationalListResponse(subject=entity, error="no_corpus_path")
    nodes = _maybe_rerank(request, path, graph, entity, rq.insights_about(graph, entity, k=k))
    return RelationalListResponse(subject=entity, results=[_node(n) for n in nodes])


@router.get("/relational/topic-entities", response_model=RelationalListResponse)
async def topic_entities(
    request: Request,
    topic: str = Query(min_length=1, description="Canonical topic id, e.g. topic:inflation."),
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
    k: int = Query(default=20, ge=1, le=200),
) -> RelationalListResponse:
    """Entities involved in a topic, ranked by mention frequency (Topic Entity View, FR4.2)."""
    graph = _graph_or_none(request, path)
    if graph is None:
        return RelationalListResponse(subject=topic, error="no_corpus_path")
    results = [_node(n) for n in rq.entities_in_topic(graph, topic, k=k)]
    return RelationalListResponse(subject=topic, results=results)


@router.get("/relational/entities-in", response_model=RelationalListResponse)
async def entities_in(
    request: Request,
    insight: str = Query(min_length=1, description="Canonical insight id."),
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
) -> RelationalListResponse:
    """People/orgs an insight mentions (not its speaker)."""
    graph = _graph_or_none(request, path)
    if graph is None:
        return RelationalListResponse(subject=insight, error="no_corpus_path")
    results = [_node(n) for n in rq.entities_in(graph, insight)]
    return RelationalListResponse(subject=insight, results=results)


@router.get("/relational/episodes", response_model=RelationalListResponse)
async def episodes(
    request: Request,
    podcast: str = Query(min_length=1, description="Canonical podcast id, e.g. podcast:my-show."),
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
    k: int | None = Query(default=None, ge=1, le=1000),
) -> RelationalListResponse:
    """A show's episodes (show navigation, PRD-033 FR3.3 / FR2.3)."""
    graph = _graph_or_none(request, path)
    if graph is None:
        return RelationalListResponse(subject=podcast, error="no_corpus_path")
    results = [_node(n) for n in rq.episodes_of(graph, podcast, k=k)]
    return RelationalListResponse(subject=podcast, results=results)


@router.get("/relational/related-insights", response_model=RelationalListResponse)
async def related_insights(
    request: Request,
    insight: str = Query(min_length=1, description="Canonical insight id (the seed)."),
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
    k: int = Query(default=20, ge=1, le=200),
) -> RelationalListResponse:
    """Sibling insights sharing a topic or mentioned entity (Detail / Graph, FR4.3 / FR5)."""
    graph = _graph_or_none(request, path)
    if graph is None:
        return RelationalListResponse(subject=insight, error="no_corpus_path")
    nodes = _maybe_rerank(request, path, graph, insight, rq.related_insights(graph, insight, k=k))
    return RelationalListResponse(subject=insight, results=[_node(n) for n in nodes])


@router.get("/relational/episode-insights", response_model=RelationalListResponse)
async def episode_insights(
    request: Request,
    episode: str = Query(min_length=1, description="Episode id (bare or episode: node id)."),
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
    k: int = Query(default=20, ge=1, le=200),
) -> RelationalListResponse:
    """Insights related to an episode's own insights (Episode Detail, FR4.3)."""
    graph = _graph_or_none(request, path)
    if graph is None:
        return RelationalListResponse(subject=episode, error="no_corpus_path")
    results = [_node(n) for n in rq.episode_related_insights(graph, episode, k=k)]
    return RelationalListResponse(subject=episode, results=results)


@router.get("/relational/who-said", response_model=RelationalGroupedResponse)
async def who_said(
    request: Request,
    topic: str = Query(min_length=1, description="Canonical topic id, e.g. topic:inflation."),
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
    k: int = Query(default=20, ge=1, le=200),
) -> RelationalGroupedResponse:
    """Per-person insights about a topic, keyed by person id (Topic Entity View, FR4.2)."""
    graph = _graph_or_none(request, path)
    if graph is None:
        return RelationalGroupedResponse(subject=topic, error="no_corpus_path")
    return RelationalGroupedResponse(subject=topic, groups=_group(rq.who_said(graph, topic, k=k)))


@router.get("/relational/cross-show", response_model=RelationalGroupedResponse)
async def cross_show(
    request: Request,
    topic: str = Query(min_length=1, description="Canonical topic id, e.g. topic:inflation."),
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
    per_show: int = Query(default=1, ge=1, le=50),
) -> RelationalGroupedResponse:
    """Top insight(s) per distinct show covering a topic — the corpus differentiator (FR3.2)."""
    graph = _graph_or_none(request, path)
    if graph is None:
        return RelationalGroupedResponse(subject=topic, error="no_corpus_path")
    groups = rq.cross_show_synthesis(graph, topic, per_show=per_show)
    return RelationalGroupedResponse(subject=topic, groups=_group(groups))


@router.get("/relational/topics", response_model=RelationalListResponse)
async def topics(
    request: Request,
    person: str = Query(min_length=1, description="Canonical person id, e.g. person:jane-doe."),
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
    k: int = Query(default=20, ge=1, le=200),
) -> RelationalListResponse:
    """Topics a person engages — the **structural** lens (#1055).

    person→STATES→insight→ABOUT→topic over the typed graph. NOTE: the **grounded**
    (quote-backed) lens is ``GET /api/cil/persons/{id}/topics`` — the two can differ
    (relational/* is graph traversal; cil/* is grounded GI). Choose by intent.
    """
    graph = _graph_or_none(request, path)
    if graph is None:
        return RelationalListResponse(subject=person, error="no_corpus_path")
    return RelationalListResponse(
        subject=person, results=[_node(n) for n in rq.topics_of(graph, person, k=k)]
    )


@router.get("/relational/co-speakers", response_model=RelationalListResponse)
async def co_speakers(
    request: Request,
    person: str = Query(min_length=1, description="Canonical person id."),
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
    k: int = Query(default=20, ge=1, le=200),
) -> RelationalListResponse:
    """People who speak on the same topics as a person — the social graph (#1055)."""
    graph = _graph_or_none(request, path)
    if graph is None:
        return RelationalListResponse(subject=person, error="no_corpus_path")
    return RelationalListResponse(
        subject=person, results=[_node(n) for n in rq.co_speakers(graph, person, k=k)]
    )


@router.get("/relational/related-topics", response_model=RelationalListResponse)
async def related_topics(
    request: Request,
    topic: str = Query(min_length=1, description="Canonical topic id, e.g. topic:inflation."),
    path: str | None = Query(default=None, description="Corpus output dir; omit for default."),
    k: int = Query(default=20, ge=1, le=200),
) -> RelationalListResponse:
    """Topics that share insights with a topic — topic↔topic connectivity (#1055)."""
    graph = _graph_or_none(request, path)
    if graph is None:
        return RelationalListResponse(subject=topic, error="no_corpus_path")
    return RelationalListResponse(
        subject=topic, results=[_node(n) for n in rq.related_topics(graph, topic, k=k)]
    )
