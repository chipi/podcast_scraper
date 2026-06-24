"""Unit tests for the KG → consumer entities projection (#1068)."""

from __future__ import annotations

from podcast_scraper.server.app_kg_view import entities_from_kg


def test_maps_typed_and_legacy_nodes() -> None:
    kg = {
        "nodes": [
            {"id": "person:jane-doe", "type": "Person", "properties": {"name": "Jane Doe"}},
            {"id": "org:acme", "type": "Entity", "properties": {"kind": "org", "name": "Acme"}},
            {"id": "topic:ai-policy", "type": "Topic", "properties": {"label": "AI Policy"}},
            {"id": "person:no-name", "type": "Entity", "properties": {"kind": "person"}},
            {"type": "Topic", "properties": {"label": "no id"}},  # missing id → skipped
        ]
    }
    persons, orgs, topics = entities_from_kg(kg)
    assert {(p.id, p.name, p.kind) for p in persons} == {
        ("person:jane-doe", "Jane Doe", "person"),
        ("person:no-name", "no-name", "person"),  # name falls back to slug
    }
    assert [(o.id, o.name, o.kind) for o in orgs] == [("org:acme", "Acme", "org")]
    assert [(t.id, t.label) for t in topics] == [("topic:ai-policy", "AI Policy")]


def test_dedupes_by_id() -> None:
    kg = {
        "nodes": [
            {"id": "person:x", "type": "Person", "properties": {"name": "X"}},
            {"id": "person:x", "type": "Person", "properties": {"name": "X again"}},
        ]
    }
    persons, _, _ = entities_from_kg(kg)
    assert len(persons) == 1


def test_malformed_inputs_return_empty() -> None:
    assert entities_from_kg(None) == ([], [], [])
    assert entities_from_kg({"nodes": "nope"}) == ([], [], [])
    assert entities_from_kg({}) == ([], [], [])
