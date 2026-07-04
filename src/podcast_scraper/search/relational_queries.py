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
from typing import Dict, List, Optional, Protocol, Sequence, Union

from .corpus_graph import Node

# Edge types in the cross-layer graph (RFC-094 / #874).
_STATES = "STATES"  # Person -> Insight (derived; the person stated it)
# Insight -> Entity (person/org). RFC-097 v3.0 split the legacy generic
# ``MENTIONS`` into typed ``MENTIONS_PERSON`` / ``MENTIONS_ORG`` so the
# viewer and queries can style/filter descriptive edges without re-reading
# node types. Queries traverse the full family — new artifacts emit only
# the typed variants, but pre-v3 corpora may still carry the legacy generic.
_MENTIONS_FAMILY: tuple = ("MENTIONS", "MENTIONS_PERSON", "MENTIONS_ORG")
_ABOUT = "ABOUT"  # Insight -> Topic
_HAS_INSIGHT = "HAS_INSIGHT"  # Episode -> Insight
_HAS_EPISODE = "HAS_EPISODE"  # Podcast -> Episode
_SUPPORTED_BY = "SUPPORTED_BY"  # Quote <-> Insight (the quotes grounding an insight)

_INSIGHT = ("insight",)
_ENTITY = ("person", "org")
_PERSON = ("person",)
_EPISODE = ("episode",)
_PODCAST = ("podcast",)
_QUOTE = ("quote",)


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
    note: str = ""  # #1056: e.g. "recurring host of <show> — not auto-named"


def _project(node: Node) -> RelatedNode:
    from .corpus_graph import _looks_like_speaker_label

    props = node.payload or {}
    text = str(props.get("text") or props.get("label") or props.get("name") or "")[:500]
    # #1056 Opt1: a recurring host the roster couldn't name surfaces as a bare
    # "SPEAKER_03". When reconciliation tagged it, show the honest note instead of the
    # raw diarization id so the surface reads "recurring host of <show>", not noise.
    note = str(props.get("recurring_host_note") or "")
    if note and _looks_like_speaker_label(text):
        text = note
    return RelatedNode(
        id=node.id,
        type=node.type,
        text=text,
        show_id=str(props.get("show_id") or props.get("podcast_id") or props.get("feed_id") or ""),
        episode_id=str(props.get("episode_id") or ""),
        note=note,
    )


def node_label(graph: GraphLike, node_id: str) -> str:
    """Display text for a subject node (name/label/text), or '' if absent.

    Used to build an entity/topic-scoped hybrid query for re-ranking (PRD-033 FR4.x:
    structure via graph, ranking via hybrid).
    """
    node = graph.get_node(node_id)
    return _project(node).text if node is not None else ""


def rerank_by_relevance(
    results: List[RelatedNode], score_by_id: Dict[str, float]
) -> List[RelatedNode]:
    """Stable re-rank of insight results by an external relevance score (RFC-090 hybrid).

    Results carrying a score sort by score **descending**; the rest keep their original
    relative order, **after** the scored ones. Empty map → unchanged. This layers hybrid
    relevance on top of the deterministic structural graph order without dropping any
    result (a node missing from the index simply stays in place rather than vanishing).
    """
    if not score_by_id:
        return list(results)
    indexed = list(enumerate(results))
    scored = [(i, r) for i, r in indexed if r.id in score_by_id]
    unscored = [r for _, r in indexed if r.id not in score_by_id]
    scored.sort(key=lambda pair: (-score_by_id[pair[1].id], pair[0]))
    return [r for _, r in scored] + unscored


def _via(
    graph: GraphLike,
    node_id: Optional[str],
    edge_type: Union[str, Sequence[str]],
    node_types: Sequence[str],
    *,
    limit: Optional[int] = None,
) -> List[Node]:
    """Neighbors of *node_id* via *edge_type* whose node type is in *node_types*.

    *edge_type* accepts a single string (legacy) or a sequence of strings
    that represent the same semantic family (e.g. the MENTIONS family —
    ``MENTIONS`` + ``MENTIONS_PERSON`` + ``MENTIONS_ORG`` per RFC-097 v3.0).
    Results from multiple edge types are deduplicated by node id with
    graph order preserved.
    """
    if not node_id:
        return []
    edge_types: Sequence[str] = (edge_type,) if isinstance(edge_type, str) else edge_type
    seen_ids: set = set()
    out: List[Node] = []
    for et in edge_types:
        for neighbor_id in graph.typed_neighbors(node_id, et):
            if neighbor_id in seen_ids:
                continue
            node = graph.get_node(neighbor_id)
            if node is not None and node.type in node_types:
                seen_ids.add(neighbor_id)
                out.append(node)
                if limit is not None and len(out) >= limit:
                    return out
    return out


def positions_of(graph: GraphLike, person_id: str, *, k: int = 20) -> List[RelatedNode]:
    """Insights a person *stated* — the `STATES` (Person→Insight) edge (#874)."""
    return [_project(n) for n in _via(graph, person_id, _STATES, _INSIGHT, limit=k)]


def insights_about(graph: GraphLike, entity_id: str, *, k: int = 20) -> List[RelatedNode]:
    """Insights that *mention* a person/org — `MENTIONS` (Insight→Entity, #874)."""
    return [_project(n) for n in _via(graph, entity_id, _MENTIONS_FAMILY, _INSIGHT, limit=k)]


def entities_in(graph: GraphLike, insight_id: str) -> List[RelatedNode]:
    """People/orgs an insight mentions — `MENTIONS` (not the speaker, which is `STATES`)."""
    return [_project(n) for n in _via(graph, insight_id, _MENTIONS_FAMILY, _ENTITY)]


def entities_in_topic(graph: GraphLike, topic_id: str, *, k: int = 20) -> List[RelatedNode]:
    """The entities *involved* in a topic — people/orgs its insights mention (FR4.2).

    Walk topic→`ABOUT`→insights→`MENTIONS`→entities and rank each entity by how many of
    the topic's insights mention it (most-mentioned first; id as a stable tiebreak).
    Deterministic, de-duplicated, capped at *k*.
    """
    counts: Dict[str, int] = {}
    nodes: Dict[str, Node] = {}
    for insight in _via(graph, topic_id, _ABOUT, _INSIGHT):
        for entity in _via(graph, insight.id, _MENTIONS_FAMILY, _ENTITY):
            counts[entity.id] = counts.get(entity.id, 0) + 1
            nodes[entity.id] = entity
    ranked = sorted(counts, key=lambda eid: (-counts[eid], eid))[:k]
    return [_project(nodes[eid]) for eid in ranked]


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
    for edge_type, hub_types in ((_ABOUT, ("topic",)), (_MENTIONS_FAMILY, _ENTITY)):
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


# --- connectivity traversals (#1054): close the person↔topic / entity↔entity gaps ---

_TOPIC = ("topic",)


def topics_of(graph: GraphLike, person_id: str, *, k: int = 20) -> List[RelatedNode]:
    """Topics a person engages — person→`STATES`→insight→`ABOUT`→topic (#1054).

    Closes the person→topic dead-end. Ranked by how many of the person's insights touch
    each topic (most-engaged first; id as a stable tiebreak).
    """
    counts: Dict[str, int] = {}
    nodes: Dict[str, Node] = {}
    for insight in _via(graph, person_id, _STATES, _INSIGHT):
        for topic in _via(graph, insight.id, _ABOUT, _TOPIC):
            counts[topic.id] = counts.get(topic.id, 0) + 1
            nodes[topic.id] = topic
    ranked = sorted(counts, key=lambda tid: (-counts[tid], tid))[:k]
    return [_project(nodes[tid]) for tid in ranked]


def co_speakers(graph: GraphLike, person_id: str, *, k: int = 20) -> List[RelatedNode]:
    """People who speak on the same topics as *person_id* — the social graph (#1054).

    person→topics, then who else `STATES` insights `ABOUT` those topics. Ranked by the
    number of shared topics (most overlap first), excluding the person themselves.
    """
    topic_ids = set()
    for insight in _via(graph, person_id, _STATES, _INSIGHT):
        for topic in _via(graph, insight.id, _ABOUT, _TOPIC):
            topic_ids.add(topic.id)
    counts: Dict[str, int] = {}
    nodes: Dict[str, Node] = {}
    for tid in topic_ids:
        seen_for_topic = set()
        for insight in _via(graph, tid, _ABOUT, _INSIGHT):
            for person in _via(graph, insight.id, _STATES, _PERSON):
                if person.id == person_id or person.id in seen_for_topic:
                    continue
                seen_for_topic.add(person.id)
                counts[person.id] = counts.get(person.id, 0) + 1
                nodes[person.id] = person
    ranked = sorted(counts, key=lambda pid: (-counts[pid], pid))[:k]
    return [_project(nodes[pid]) for pid in ranked]


def topics_of_insight(graph: GraphLike, insight_id: str) -> List[RelatedNode]:
    """Topics an insight is `ABOUT` — lets results surface their topic links (#1054)."""
    return [_project(n) for n in _via(graph, insight_id, _ABOUT, _TOPIC)]


@dataclass
class InsightDetail:
    """An insight's *own* content, resolved from the full corpus graph.

    Unlike ``related_insights`` (structural neighbours), this is the insight itself:
    its text + type + grounded flag, the quotes that ``SUPPORTED_BY`` it, the topics
    it is ``ABOUT``, and the entities it ``MENTIONS``. Lets a viewer render an
    out-of-slice insight (e.g. a corpus-wide timeline-mention drill) without the node
    being in the loaded artifact.
    """

    id: str
    type: str
    text: str
    insight_type: str
    grounded: bool
    episode_id: str
    show_id: str
    quotes: List[RelatedNode]
    topics: List[RelatedNode]
    entities: List[RelatedNode]


def insight_detail(graph: GraphLike, insight_id: str) -> Optional[InsightDetail]:
    """Resolve an insight's own text + supporting quotes + topics + mentions.

    Returns ``None`` when the id isn't an insight node in the corpus graph.
    """
    node = graph.get_node(insight_id)
    if node is None or node.type != "insight":
        return None
    props = node.payload or {}
    return InsightDetail(
        id=node.id,
        type=node.type,
        text=str(props.get("text") or "")[:2000],
        insight_type=str(props.get("insight_type") or ""),
        grounded=bool(props.get("grounded", False)),
        episode_id=str(props.get("episode_id") or ""),
        show_id=str(props.get("show_id") or props.get("podcast_id") or props.get("feed_id") or ""),
        quotes=[_project(n) for n in _via(graph, insight_id, _SUPPORTED_BY, _QUOTE)],
        topics=topics_of_insight(graph, insight_id),
        entities=entities_in(graph, insight_id),
    )


def related_topics(graph: GraphLike, topic_id: str, *, k: int = 20) -> List[RelatedNode]:
    """Topics that share insights with *topic_id* — topic↔topic connectivity (#1054).

    topic→`ABOUT`→insight→`ABOUT`→other-topic, ranked by how many insights they co-occur in.
    """
    counts: Dict[str, int] = {}
    nodes: Dict[str, Node] = {}
    for insight in _via(graph, topic_id, _ABOUT, _INSIGHT):
        for topic in _via(graph, insight.id, _ABOUT, _TOPIC):
            if topic.id == topic_id:
                continue
            counts[topic.id] = counts.get(topic.id, 0) + 1
            nodes[topic.id] = topic
    ranked = sorted(counts, key=lambda tid: (-counts[tid], tid))[:k]
    return [_project(nodes[tid]) for tid in ranked]


def shared_topics(graph: GraphLike, person_a: str, person_b: str) -> List[RelatedNode]:
    """Topics BOTH people engage — the bridge between two voices (#1054)."""
    a_topics = {n.id: n for n in topics_of(graph, person_a, k=10_000)}
    b_ids = {n.id for n in topics_of(graph, person_b, k=10_000)}
    return [a_topics[tid] for tid in a_topics if tid in b_ids]
