"""Connectivity / neighborhood tools (#1054) — one-call multi-faceted exploration.

Applies the o11y-MCP lesson (#1052/#1053): a *join* primitive beats N query tools.
``entity_neighborhood`` returns the connected subgraph for an entity in ONE call — so an
agent doesn't chain 4–5 tools to understand a person/topic — and closes the connectivity
dead-ends the dogfood found (person→topic, person→co-people).

Uniform envelope on every tool here: ``{ok, kind, subject, data, note}`` — ``note`` says
*why* a result is empty/sparse, so an agent never confuses "no data" with "feature off".
"""

from __future__ import annotations

from typing import Any, Dict

from ..context import CorpusContext


def _ok(kind: str, subject: Dict[str, Any], data: Dict[str, Any], note: str = "") -> Dict[str, Any]:
    return {"ok": True, "kind": kind, "subject": subject, "data": data, "note": note}


def _err(kind: str, subject_id: str, note: str) -> Dict[str, Any]:
    return {"ok": False, "kind": kind, "subject": {"id": subject_id}, "data": {}, "note": note}


def _kind_of(entity_id: str) -> str:
    return entity_id.split(":", 1)[0] if ":" in entity_id else ""


def _rel(node: Any) -> Dict[str, Any]:
    return {
        "id": node.id,
        "type": node.type,
        "text": node.text,
        "show_id": node.show_id,
        "episode_id": node.episode_id,
    }


def _graph(ctx: CorpusContext) -> Any:
    from ...search.corpus_graph import get_corpus_graph

    return get_corpus_graph(ctx.corpus_dir, derive_speaker_links=True)


def entity_neighborhood(ctx: CorpusContext, entity_id: str, k: int = 8) -> Dict[str, Any]:
    """The connected subgraph for an entity, in one call (#1054).

    person → what they *stated*, what's *said about* them, their *topics*, *co-speakers*,
    and *shows*. topic → *entities*, *speakers*, *cross-show* synthesis. org → *mentioned
    in*. podcast → *episodes*. ``entity_id`` is a canonical id (``resolve_entity`` first).
    """
    from ...search import relational_queries as rq

    kind = _kind_of(entity_id)
    graph = _graph(ctx)
    if graph.get_node(entity_id) is None:
        return _err(kind, entity_id, f"no entity {entity_id!r} — resolve_entity first?")
    subject = {"id": entity_id, "label": rq.node_label(graph, entity_id)}

    if kind == "person":
        stated = rq.positions_of(graph, entity_id, k=k)
        about = rq.insights_about(graph, entity_id, k=k)
        shows = sorted({n.show_id for n in stated if n.show_id})
        data = {
            "stated": [_rel(n) for n in stated],
            "mentioned_in": [_rel(n) for n in about],
            "topics": [_rel(n) for n in rq.topics_of(graph, entity_id, k=k)],
            "co_speakers": [_rel(n) for n in rq.co_speakers(graph, entity_id, k=k)],
            "shows": shows,
        }
        note = "" if (stated or about) else "no insights — likely an unnamed/low-signal speaker"
        return _ok(kind, subject, data, note)

    if kind == "org":
        about = rq.insights_about(graph, entity_id, k=k)
        data = {
            "mentioned_in": [_rel(n) for n in about],
            "shows": sorted({n.show_id for n in about if n.show_id}),
        }
        return _ok(kind, subject, data)

    if kind == "topic":
        who = rq.who_said(graph, entity_id, k=k)
        cross = rq.cross_show_synthesis(graph, entity_id, per_show=1)
        data = {
            "entities": [_rel(n) for n in rq.entities_in_topic(graph, entity_id, k=k)],
            "speakers": sorted(who.keys()),
            "cross_show": {sid: [_rel(n) for n in v] for sid, v in cross.items()},
        }
        note = "" if data["entities"] or data["cross_show"] else "no insights link to this topic"
        return _ok(kind, subject, data, note)

    if kind == "podcast":
        episodes = [_rel(n) for n in rq.episodes_of(graph, entity_id, k=k)]
        return _ok(kind, subject, {"episodes": episodes})

    return _err(kind, entity_id, f"neighborhood not supported for id kind {kind!r}")


def person_topics(ctx: CorpusContext, person_id: str, k: int = 20) -> Dict[str, Any]:
    """The topics a person engages, ranked by how much of their output touches each (#1054).

    Closes the person→topic dead-end (person→STATES→insight→ABOUT→topic).
    """
    from ...search import relational_queries as rq

    graph = _graph(ctx)
    topics = rq.topics_of(graph, person_id, k=k)
    note = "" if topics else "no topics — the person stated no topic-linked insights"
    return _ok(
        "person",
        {"id": person_id, "label": rq.node_label(graph, person_id)},
        {"topics": [_rel(n) for n in topics]},
        note,
    )


def co_occurring_entities(ctx: CorpusContext, entity_id: str, k: int = 20) -> Dict[str, Any]:
    """Who is discussed *alongside* an entity — the social graph (#1054).

    For a ``person:`` id: people who speak on the same topics, ranked by shared-topic count.
    """
    from ...search import relational_queries as rq

    kind = _kind_of(entity_id)
    if kind != "person":
        return _err(kind, entity_id, "co_occurring_entities currently supports person: ids")
    graph = _graph(ctx)
    people = rq.co_speakers(graph, entity_id, k=k)
    note = "" if people else "no co-speakers share a topic with this person"
    return _ok(
        kind,
        {"id": entity_id, "label": rq.node_label(graph, entity_id)},
        {"co_occurring": [_rel(n) for n in people]},
        note,
    )
