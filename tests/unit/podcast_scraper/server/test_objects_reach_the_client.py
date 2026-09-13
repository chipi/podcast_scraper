"""An `Object` entity must not be silently dropped before the client (#2057, review LOW).

`entities_from_kg` matches Person or Organization and falls through everything else, so an Object
— extracted by the KG pipeline, typed in the schema, carried through migration m0008 and written
to the search index — was discarded at the last projection before the app. Nothing logged it.

Deciding not to SHOW something is a client choice. Dropping it in the projection is an accident,
and it is the shape of every other defect in this branch: a value that exists everywhere except
where somebody looks.
"""

from __future__ import annotations

import pytest

from podcast_scraper.server.app_kg_view import entities_from_kg, objects_from_kg

pytestmark = pytest.mark.unit


def _kg(*nodes):
    return {"nodes": list(nodes)}


_OBJECT = {
    "id": "object:the-norman-conquest",
    "type": "Object",
    "properties": {"name": "The Norman Conquest", "role": "mentioned"},
}
_PERSON = {"id": "person:aaron-levie", "type": "Person", "properties": {"name": "Aaron Levie"}}
_ORG = {"id": "org:box", "type": "Organization", "properties": {"name": "Box"}}


class TestObjectsAreReachable:
    def test_an_object_node_is_projected(self) -> None:
        got = objects_from_kg(_kg(_OBJECT))
        assert [(e.id, e.name, e.kind) for e in got] == [
            ("object:the-norman-conquest", "The Norman Conquest", "object")
        ]

    def test_objects_are_deduplicated_by_id(self) -> None:
        assert len(objects_from_kg(_kg(_OBJECT, dict(_OBJECT)))) == 1

    def test_a_legacy_artifact_with_an_object_prefix_id_still_projects(self) -> None:
        # Pre-2.1 shapes carry the kind in properties rather than the node type.
        legacy = {"id": "object:x", "type": "Entity", "properties": {"name": "X", "kind": "object"}}
        assert len(objects_from_kg(_kg(legacy))) == 1


class TestTheKindsStayInTheirOwnLanes:
    """The whole point of Object is that it does not pollute people or organisations."""

    def test_objects_do_not_appear_as_people_or_orgs(self) -> None:
        persons, orgs, _topics = entities_from_kg(_kg(_OBJECT, _PERSON, _ORG))
        assert [p.id for p in persons] == ["person:aaron-levie"]
        assert [o.id for o in orgs] == ["org:box"]

    def test_people_and_orgs_do_not_appear_as_objects(self) -> None:
        assert objects_from_kg(_kg(_PERSON, _ORG)) == []


class TestDegenerateInputs:
    @pytest.mark.parametrize("bad", [None, {}, {"nodes": "nope"}, {"nodes": [None, 3, "x"]}])
    def test_malformed_input_yields_no_objects_rather_than_raising(self, bad) -> None:
        assert objects_from_kg(bad) == []

    def test_a_node_without_an_id_is_skipped(self) -> None:
        assert objects_from_kg(_kg({"type": "Object", "properties": {"name": "X"}})) == []
