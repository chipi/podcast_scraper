"""Relational query layer over the cross-layer CorpusGraph (RFC-094, #882).

Read-only queries that traverse the **typed** meaning-bearing edges (#874) to answer
the surface questions PRD-033 needs:

- `positions_of(person)` — the insights a person *stated* (Person Landing, FR4.1).
- `who_said(topic)` — per-person insights on a topic (Topic Entity View, FR4.2).
- `insights_about(entity)` — insights that *mention* a person/org (entity grounding).
- `entities_in(insight)` — the people/orgs an insight mentions.
- `episodes_of(show)` — a show's episodes (show navigation, FR3.3 / FR2.3).
- `cross_show_synthesis(topic)` — the top insight per distinct show covering a topic
  (the corpus differentiator, FR3.2 / FR4.2).
- `related_insights(insight)` — sibling insights sharing a topic or mentioned entity
  (Detail panel "related", Graph neighbourhood; FR4.3 / FR5).

These rely on **edge types** (via ``CorpusGraph.typed_neighbors``): a person↔insight
pair can be ``STATES`` (the person stated it) or ``MENTIONS`` (the insight mentions the
person), and the queries must not conflate them. Structure comes from the graph; these
are intentionally read-only and never raise (empty result on a missing/unknown id).
Supporting-evidence ranking via hybrid (RFC-090) is layered at the call site.

The functions accept any object exposing ``get_node(id) -> Node | None`` and
``typed_neighbors(id, edge_type) -> list[str]`` (i.e. ``CorpusGraph``). For
``positions_of`` / ``who_said`` the graph must be built with
``derive_speaker_links=True`` (which synthesizes the ``STATES`` edges).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

from .corpus_graph import Node

# Edge types in the cross-layer graph (RFC-094 / #874).
_STATES = "STATES"  # Person -> Insight (derived; the person stated it)
_MENTIONS = "MENTIONS"  # Insight -> Entity (the insight concerns a person/org)
_ABOUT = "ABOUT"  # Insight -> Topic
_HAS_INSIGHT = "HAS_INSIGHT"  # Episode -> Insight
_HAS_EPISODE = "HAS_EPISODE"  # Podcast -> Episode

_INSIGHT = ("insight",)
_ENTITY = ("person", "org")
_PERSON = ("person",)
_EPISODE = ("episode",)
_PODCAST = ("podcast",)


class GraphLike(Protocol):
    """The subset of ``CorpusGraph`` the queries use (keeps them unit-testable)."""

    def get_node(self, node_id: Optional[str]) -> Optional[Node]:
        """Return the node for *node_id*, or ``None`` if absent."""
        ...

    def typed_neighbors(self, node_id: str, edge_type: str) -> List[str]:
        """Return the neighbors of *node_id* reached by an edge of *edge_type*."""
        ...


@dataclass
class RelatedNode:
    """A graph node projected for a relational-query result."""

    id: str
    type: str
    text: str = ""
    show_id: str = ""
    episode_id: str = ""


def _project(node: Node) -> RelatedNode:
    props = node.payload or {}
    text = props.get("text") or props.get("label") or props.get("name") or ""
    return RelatedNode(
        id=node.id,
        type=node.type,
        text=str(text)[:500],
        show_id=str(props.get("show_id") or props.get("podcast_id") or props.get("feed_id") or ""),
        episode_id=str(props.get("episode_id") or ""),
    )


def _via(
    graph: GraphLike,
    node_id: Optional[str],
    edge_type: str,
    node_types: Sequence[str],
    *,
    limit: Optional[int] = None,
) -> List[Node]:
    """Neighbors of *node_id* via *edge_type* whose node type is in *node_types*."""
    if not node_id:
        return []
    out: List[Node] = []
    for neighbor_id in graph.typed_neighbors(node_id, edge_type):
        node = graph.get_node(neighbor_id)
        if node is not None and node.type in node_types:
            out.append(node)
            if limit is not None and len(out) >= limit:
                break
    return out


def positions_of(graph: GraphLike, person_id: str, *, k: int = 20) -> List[RelatedNode]:
    """Insights a person *stated* — the `STATES` (Person→Insight) edge (#874)."""
    return [_project(n) for n in _via(graph, person_id, _STATES, _INSIGHT, limit=k)]


def insights_about(graph: GraphLike, entity_id: str, *, k: int = 20) -> List[RelatedNode]:
    """Insights that *mention* a person/org — `MENTIONS` (Insight→Entity, #874)."""
    return [_project(n) for n in _via(graph, entity_id, _MENTIONS, _INSIGHT, limit=k)]


def entities_in(graph: GraphLike, insight_id: str) -> List[RelatedNode]:
    """People/orgs an insight mentions — `MENTIONS` (not the speaker, which is `STATES`)."""
    return [_project(n) for n in _via(graph, insight_id, _MENTIONS, _ENTITY)]


def episodes_of(graph: GraphLike, podcast_id: str, *, k: Optional[int] = None) -> List[RelatedNode]:
    """A show's episodes — `HAS_EPISODE` (Podcast→Episode)."""
    return [_project(n) for n in _via(graph, podcast_id, _HAS_EPISODE, _EPISODE, limit=k)]


def _show_id_for_insight(graph: GraphLike, insight: Node) -> str:
    """Resolve an insight's show via insight→Episode→Podcast, falling back to payload."""
    direct = (insight.payload or {}).get("show_id") or (insight.payload or {}).get("feed_id")
    if direct:
        return str(direct)
    for episode in _via(graph, insight.id, _HAS_INSIGHT, _EPISODE):
        for podcast in _via(graph, episode.id, _HAS_EPISODE, _PODCAST, limit=1):
            return podcast.id
    return ""


def who_said(graph: GraphLike, topic_id: str, *, k: int = 20) -> Dict[str, List[RelatedNode]]:
    """Per-person insights about a topic — `ABOUT` (Insight→Topic) + `STATES`.

    Returns ``{person_id: [insights]}`` (each list capped at *k*). Insights with no
    attributed speaker are omitted (attribution is diarization-gated, #876).
    """
    out: Dict[str, List[RelatedNode]] = {}
    for insight in _via(graph, topic_id, _ABOUT, _INSIGHT):
        for person in _via(graph, insight.id, _STATES, _PERSON):
            bucket = out.setdefault(person.id, [])
            if len(bucket) < k:
                bucket.append(_project(insight))
    return out


def related_insights(graph: GraphLike, insight_id: str, *, k: int = 20) -> List[RelatedNode]:
    """Sibling insights sharing a topic (`ABOUT`) or mentioned entity (`MENTIONS`).

    The structural "related" set for the Detail panel / Graph neighbourhood: walk
    insight→topic→insight and insight→entity→insight (2 hops), excluding the seed.
    Deterministic graph order, de-duplicated; a hybrid-scored re-rank (RFC-090) by
    relevance to the seed is layered at the call site when an index is available.
    """
    seed = graph.get_node(insight_id)
    if seed is None:
        return []
    out: List[RelatedNode] = []
    seen = {insight_id}
    for edge_type, hub_types in ((_ABOUT, ("topic",)), (_MENTIONS, _ENTITY)):
        for hub in _via(graph, insight_id, edge_type, hub_types):
            for sibling in _via(graph, hub.id, edge_type, _INSIGHT):
                if sibling.id in seen:
                    continue
                seen.add(sibling.id)
                out.append(_project(sibling))
                if len(out) >= k:
                    return out
    return out


def episode_related_insights(
    graph: GraphLike, episode_id: str, *, k: int = 20
) -> List[RelatedNode]:
    """Insights related to an episode — siblings of the episode's own insights (FR4.3).

    Walk episode→`HAS_INSIGHT`→insights (this episode's), then each insight's topic /
    entity siblings (via `related_insights`), excluding insights that belong to this
    episode. Deterministic graph order, de-duplicated, capped at *k*. ``episode_id`` may
    be the bare id or the canonical ``episode:`` node id.
    """
    node = graph.get_node(episode_id) or graph.get_node(f"episode:{episode_id}")
    if node is None:
        return []
    own = _via(graph, node.id, _HAS_INSIGHT, _INSIGHT)
    own_ids = {ins.id for ins in own}
    out: List[RelatedNode] = []
    seen = set(own_ids)
    for insight in own:
        for sibling in related_insights(graph, insight.id, k=k):
            if sibling.id in seen:
                continue
            seen.add(sibling.id)
            out.append(sibling)
            if len(out) >= k:
                return out
    return out


def cross_show_synthesis(
    graph: GraphLike, topic_id: str, *, per_show: int = 1
) -> Dict[str, List[RelatedNode]]:
    """Top insight(s) **per distinct show** covering a topic — the corpus differentiator.

    Returns ``{show_id: [insights]}`` (each capped at *per_show*). Shows that cannot be
    resolved are dropped. Ranking within a show is graph order here; a hybrid-scored
    re-rank (RFC-090) is layered at the call site when an index is available.
    """
    out: Dict[str, List[RelatedNode]] = {}
    for insight in _via(graph, topic_id, _ABOUT, _INSIGHT):
        show = _show_id_for_insight(graph, insight)
        if not show:
            continue
        bucket = out.setdefault(show, [])
        if len(bucket) < per_show:
            bucket.append(_project(insight))
    return out
