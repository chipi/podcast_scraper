"""Unit tests for derived relational edges (Insight→Entity, Podcast→HAS_EPISODE)."""

from __future__ import annotations

import pytest

from podcast_scraper.gi.relational_edges import (
    add_episode_show_edges,
    add_insight_entity_edges,
    kg_entity_index,
    kg_entity_names,
)

pytestmark = pytest.mark.unit


def _artifact():
    return {
        "schema_version": "2.0",
        "nodes": [
            {"id": "episode:e1", "type": "Episode", "properties": {}},
            {
                "id": "insight:1",
                "type": "Insight",
                "properties": {"text": "Elon Musk plans to list SpaceX."},
            },
            {
                "id": "insight:2",
                "type": "Insight",
                "properties": {"text": "Markets recovered this quarter."},
            },
        ],
        "edges": [],
    }


def test_insight_entity_matches_whole_word_name():
    """RFC-097 v3.0: emits typed MENTIONS_PERSON / MENTIONS_ORG based on kind."""
    art = _artifact()
    added = add_insight_entity_edges(
        art,
        {
            "person:elon-musk": ("Elon Musk", "person"),
            "org:spacex": ("SpaceX", "organization"),
        },
    )
    assert added == 2  # both names appear in insight:1
    typed = {(e["from"], e["to"], e["type"]) for e in art["edges"]}
    assert ("insight:1", "person:elon-musk", "MENTIONS_PERSON") in typed
    assert ("insight:1", "org:spacex", "MENTIONS_ORG") in typed
    # insight:2 mentions neither
    assert all(t[0] == "insight:1" for t in typed)
    # Schema bumped to v3.0 on first typed-edge addition.
    assert art["schema_version"] == "3.0"


def test_insight_entity_no_substring_false_positive():
    art = _artifact()
    # "Musk" alone should not match inside another word; and an absent entity adds nothing
    added = add_insight_entity_edges(art, {"person:cathie-wood": ("Cathie Wood", "person")})
    assert added == 0
    # No typed edges added -> schema_version unchanged.
    assert art["schema_version"] == "2.0"


def test_insight_entity_idempotent():
    """Repeated calls don't double-add (dedup by (from, to, type))."""
    art = _artifact()
    index = {"person:elon-musk": ("Elon Musk", "person")}
    assert add_insight_entity_edges(art, index) == 1
    assert add_insight_entity_edges(art, index) == 0


def test_insight_entity_skips_when_legacy_mentions_already_present():
    """Permissive: legacy generic MENTIONS suppresses re-emission of typed edge."""
    art = _artifact()
    # Pre-existing legacy edge from an old artifact
    art["edges"].append({"type": "MENTIONS", "from": "insight:1", "to": "person:elon-musk"})
    added = add_insight_entity_edges(art, {"person:elon-musk": ("Elon Musk", "person")})
    assert added == 0  # legacy edge dedup-blocks the typed one
    # schema_version not bumped because no new edge landed
    assert art["schema_version"] == "2.0"


def test_episode_show_adds_podcast_node_and_edge():
    art = _artifact()
    added = add_episode_show_edges(art, "Odd Lots")
    assert added == 1
    assert {"type": "HAS_EPISODE", "from": "podcast:odd-lots", "to": "episode:e1"} in art["edges"]
    pod = next(n for n in art["nodes"] if n["id"] == "podcast:odd-lots" and n["type"] == "Podcast")
    # RFC-097 chunk-4 retroactive sweep: schema requires "title" (not "name").
    assert pod["properties"] == {"title": "Odd Lots"}
    # idempotent
    assert add_episode_show_edges(art, "Odd Lots") == 0


def test_insight_entity_edges_add_missing_target_nodes():
    """RFC-097 chunk-4 retroactive: target Person/Organization nodes are added
    when they don't already exist in the artifact (otherwise the edge would
    dangle and the viewer would have to cross-join with kg.json)."""
    art = _artifact()
    assert not any(n["type"] in ("Person", "Organization") for n in art["nodes"])
    added = add_insight_entity_edges(
        art,
        {
            "person:elon-musk": ("Elon Musk", "person"),
            "org:spacex": ("SpaceX", "organization"),
        },
    )
    assert added == 2
    persons = [n for n in art["nodes"] if n["type"] == "Person"]
    orgs = [n for n in art["nodes"] if n["type"] == "Organization"]
    assert len(persons) == 1 and persons[0]["id"] == "person:elon-musk"
    assert persons[0]["properties"] == {"name": "Elon Musk"}
    assert len(orgs) == 1 and orgs[0]["id"] == "org:spacex"
    assert orgs[0]["properties"] == {"name": "SpaceX"}


def test_insight_entity_edges_do_not_duplicate_existing_target_nodes():
    """If the target Person already exists in the artifact (e.g. from
    SPOKEN_BY emission), the helper does NOT add a duplicate."""
    art = _artifact()
    art["nodes"].append(
        {"id": "person:elon-musk", "type": "Person", "properties": {"name": "Elon Musk"}}
    )
    pre_count = sum(1 for n in art["nodes"] if n["id"] == "person:elon-musk")
    added = add_insight_entity_edges(art, {"person:elon-musk": ("Elon Musk", "person")})
    assert added == 1  # edge added
    post_count = sum(1 for n in art["nodes"] if n["id"] == "person:elon-musk")
    assert post_count == pre_count  # node count unchanged


def test_episode_show_empty_title_noop():
    art = _artifact()
    assert add_episode_show_edges(art, "") == 0


def test_kg_entity_names_extracts_id_to_name():
    """Backward-compat: kg_entity_names returns plain {id: name} (no kind)."""
    kg = {
        "nodes": [
            {
                "id": "person:gillian-tett",
                "type": "Entity",
                "properties": {"name": "Gillian Tett", "kind": "person"},
            },
            {
                "id": "org:financial-times",
                "type": "Entity",
                "properties": {"name": "Financial Times"},
            },
            {"id": "topic:debt", "type": "Topic", "properties": {}},
        ]
    }
    assert kg_entity_names(kg) == {
        "person:gillian-tett": "Gillian Tett",
        "org:financial-times": "Financial Times",
    }


def test_kg_entity_index_carries_kind_for_typed_mentions():
    """RFC-097: kg_entity_index returns (name, kind) so typed MENTIONS_* edges can be emitted."""
    kg = {
        "nodes": [
            # v2.0 typed Person node (RFC-097)
            {
                "id": "person:gillian-tett",
                "type": "Person",
                "properties": {"name": "Gillian Tett"},
            },
            # v2.0 typed Organization node
            {
                "id": "org:financial-times",
                "type": "Organization",
                "properties": {"name": "Financial Times"},
            },
            # Legacy v1.2 Entity with `kind`
            {
                "id": "person:martin-wolf",
                "type": "Entity",
                "properties": {"name": "Martin Wolf", "kind": "person"},
            },
            # Legacy v1.0/1.1 Entity with `entity_kind`
            {
                "id": "org:lse",
                "type": "Entity",
                "properties": {"name": "LSE", "entity_kind": "organization"},
            },
            {"id": "topic:debt", "type": "Topic", "properties": {}},
        ]
    }
    assert kg_entity_index(kg) == {
        "person:gillian-tett": ("Gillian Tett", "person"),
        "org:financial-times": ("Financial Times", "organization"),
        "person:martin-wolf": ("Martin Wolf", "person"),
        "org:lse": ("LSE", "organization"),
    }
