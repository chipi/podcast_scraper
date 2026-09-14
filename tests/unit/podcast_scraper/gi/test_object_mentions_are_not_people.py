"""An Object mentioned in an insight must not be materialised as a Person (#2057).

`add_insight_entity_edges` creates the mention target node inside gi.json so the edge resolves
without a cross-layer join. It chose that node's type with:

    node_type = "Organization" if kind == "organization" else "Person"

`kg_entity_index` feeds it `normalized_entity_kind_from_node`, which since schema 2.1 returns
``"object"`` — and ``"object" != "organization"``, so every Object mentioned in an insight was
written into gi.json as a **Person**, with a ``MENTIONS_PERSON`` edge.

That is the same defect #2057 was filed for, in a different function: the KG stopped defaulting
unknown entity kinds to person, the viewer stopped drawing them as humans, and this pass quietly
re-created them as people one layer down. "Fahrenheit 451" and "Project Panama" are real Objects in
a freshly ingested corpus; an enrich-edges run turns them into people who can be followed, ranked
among speakers, and counted in person metrics.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.gi.relational_edges import add_insight_entity_edges, kg_entity_index

pytestmark = pytest.mark.unit

TEXT = (
    "Ray Bradbury wrote Fahrenheit 451, and the crew reread it while "
    "Acme Corp funded Project Panama."
)


def _kg() -> Dict[str, Any]:
    return {
        "schema_version": "2.1",
        "nodes": [
            {"id": "person:ray-bradbury", "type": "Person", "properties": {"name": "Ray Bradbury"}},
            {"id": "org:acme-corp", "type": "Organization", "properties": {"name": "Acme Corp"}},
            {
                "id": "object:fahrenheit-451",
                "type": "Object",
                "properties": {"name": "Fahrenheit 451"},
            },
            {
                "id": "object:project-panama",
                "type": "Object",
                "properties": {"name": "Project Panama"},
            },
        ],
    }


def _gi() -> Dict[str, Any]:
    return {
        "nodes": [{"id": "insight:1", "type": "Insight", "properties": {"text": TEXT}}],
        "edges": [],
    }


def _nodes_by_id(gi: Dict[str, Any]) -> Dict[str, dict]:
    return {n["id"]: n for n in gi["nodes"]}


class TestTheIndexReportsTheRealKind:
    def test_an_object_is_indexed_as_an_object(self) -> None:
        idx = kg_entity_index(_kg())
        assert idx["object:fahrenheit-451"][1] == "object"

    def test_person_and_org_are_unchanged(self) -> None:
        idx = kg_entity_index(_kg())
        assert idx["person:ray-bradbury"][1] == "person"
        assert idx["org:acme-corp"][1] == "organization"


class TestAnObjectIsNeverWrittenAsAPerson:
    def test_the_created_node_is_an_object(self) -> None:
        gi = _gi()
        add_insight_entity_edges(gi, kg_entity_index(_kg()))
        node = _nodes_by_id(gi).get("object:fahrenheit-451")
        assert node is not None, "the Object was never linked at all"
        assert node["type"] == "Object", f"materialised as {node['type']!r}"

    def test_no_object_leaks_into_the_person_population(self) -> None:
        # The metric that made #2057 visible: an 11th-century event ranked as the corpus's #1 voice.
        gi = _gi()
        add_insight_entity_edges(gi, kg_entity_index(_kg()))
        people = {n["id"] for n in gi["nodes"] if n.get("type") == "Person"}
        assert "object:fahrenheit-451" not in people
        assert "object:project-panama" not in people

    def test_the_edge_is_typed_for_objects(self) -> None:
        gi = _gi()
        add_insight_entity_edges(gi, kg_entity_index(_kg()))
        by_target = {e["to"]: e["type"] for e in gi["edges"]}
        assert by_target["object:fahrenheit-451"] == "MENTIONS_OBJECT"

    def test_person_and_org_edges_are_unchanged(self) -> None:
        gi = _gi()
        add_insight_entity_edges(gi, kg_entity_index(_kg()))
        by_target = {e["to"]: e["type"] for e in gi["edges"]}
        assert by_target["person:ray-bradbury"] == "MENTIONS_PERSON"
        assert by_target["org:acme-corp"] == "MENTIONS_ORG"

    def test_the_pass_is_idempotent_for_objects(self) -> None:
        gi = _gi()
        first = add_insight_entity_edges(gi, kg_entity_index(_kg()))
        before = (len(gi["nodes"]), len(gi["edges"]))
        second = add_insight_entity_edges(gi, kg_entity_index(_kg()))
        assert second == 0, f"re-run added {second} edges (first run added {first})"
        assert (len(gi["nodes"]), len(gi["edges"])) == before
