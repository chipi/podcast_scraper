"""Objects belong in the connected / related entity lists (#2057).

`Object` became a first-class KG entity kind in schema 2.1 — a named thing that is neither a
person nor a body of people (an event, place, creative work or product). It is indexed for
search, drawn in the graph viewer, and served on the episode entity card.

The relational queries were still `("person", "org")`, so a walk from an insight to "the
entities it mentions" silently skipped every Object. On a freshly ingested episode that hides
`Project Catalyst`, `Novastar Fund 3` and `Fahrenheit 451` from the connected/related lists and
from the entities-involved-in-a-topic ranking.

The MENTIONS family needed the same treatment: `MENTIONS_OBJECT` is the typed edge those objects
arrive on, so a traversal listing only the person/org edge types cannot reach them however the
node filter is widened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest

from podcast_scraper.search.relational_queries import entities_in, entities_in_topic

pytestmark = pytest.mark.unit


@dataclass
class _Node:
    id: str
    type: str
    payload: Dict[str, object] = field(default_factory=dict)


@dataclass
class _Edge:
    from_id: str
    to_id: str
    type: str


class _Graph:
    """The slice of CorpusGraph the queries use."""

    def __init__(self, nodes: List[_Node], edges: List[_Edge]) -> None:
        self._nodes = {n.id: n for n in nodes}
        self._edges = edges

    def get_node(self, node_id: Optional[str]):
        return self._nodes.get(node_id or "")

    def typed_neighbors(self, node_id: str, edge_type: str):
        return [e.to_id for e in self._edges if e.from_id == node_id and e.type == edge_type]


def _graph() -> _Graph:
    nodes = [
        _Node("insight:1", "insight", {"text": "They reread Fahrenheit 451 at Novastar."}),
        _Node("topic:books", "topic", {"label": "books"}),
        _Node("person:ray-bradbury", "person", {"name": "Ray Bradbury"}),
        _Node("org:novastar", "org", {"name": "Novastar"}),
        _Node("object:fahrenheit-451", "object", {"name": "Fahrenheit 451"}),
    ]
    edges = [
        _Edge("topic:books", "insight:1", "ABOUT"),
        _Edge("insight:1", "person:ray-bradbury", "MENTIONS_PERSON"),
        _Edge("insight:1", "org:novastar", "MENTIONS_ORG"),
        _Edge("insight:1", "object:fahrenheit-451", "MENTIONS_OBJECT"),
    ]
    return _Graph(nodes, edges)


class TestEntitiesInAnInsight:
    def test_the_object_is_listed(self) -> None:
        got = {e.id for e in entities_in(_graph(), "insight:1")}
        assert "object:fahrenheit-451" in got, got

    def test_people_and_orgs_are_unaffected(self) -> None:
        got = {e.id for e in entities_in(_graph(), "insight:1")}
        assert {"person:ray-bradbury", "org:novastar"} <= got


class TestEntitiesInvolvedInATopic:
    def test_the_object_is_ranked_alongside_people(self) -> None:
        got = {e.id for e in entities_in_topic(_graph(), "topic:books")}
        assert "object:fahrenheit-451" in got, got

    def test_people_and_orgs_are_unaffected(self) -> None:
        got = {e.id for e in entities_in_topic(_graph(), "topic:books")}
        assert {"person:ray-bradbury", "org:novastar"} <= got
