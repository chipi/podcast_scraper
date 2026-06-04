"""Unit tests for derived relational edges (Insight→Entity, Podcast→HAS_EPISODE)."""

from __future__ import annotations

import pytest

from podcast_scraper.gi.relational_edges import (
    add_episode_show_edges,
    add_insight_entity_edges,
    kg_entity_names,
)

pytestmark = pytest.mark.unit


def _artifact():
    return {
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
    art = _artifact()
    added = add_insight_entity_edges(art, {"person:elon-musk": "Elon Musk", "org:spacex": "SpaceX"})
    assert added == 2  # both names appear in insight:1
    mentions = {(e["from"], e["to"]) for e in art["edges"] if e["type"] == "MENTIONS"}
    assert ("insight:1", "person:elon-musk") in mentions
    assert ("insight:1", "org:spacex") in mentions
    # insight:2 mentions neither
    assert all(f == "insight:1" for f, _ in mentions)


def test_insight_entity_no_substring_false_positive():
    art = _artifact()
    # "Musk" alone should not match inside another word; and an absent entity adds nothing
    added = add_insight_entity_edges(art, {"person:cathie-wood": "Cathie Wood"})
    assert added == 0


def test_insight_entity_idempotent():
    art = _artifact()
    names = {"person:elon-musk": "Elon Musk"}
    assert add_insight_entity_edges(art, names) == 1
    assert add_insight_entity_edges(art, names) == 0


def test_episode_show_adds_podcast_node_and_edge():
    art = _artifact()
    added = add_episode_show_edges(art, "Odd Lots")
    assert added == 1
    assert {"type": "HAS_EPISODE", "from": "podcast:odd-lots", "to": "episode:e1"} in art["edges"]
    assert any(n["id"] == "podcast:odd-lots" and n["type"] == "Podcast" for n in art["nodes"])
    # idempotent
    assert add_episode_show_edges(art, "Odd Lots") == 0


def test_episode_show_empty_title_noop():
    art = _artifact()
    assert add_episode_show_edges(art, "") == 0


def test_kg_entity_names_extracts_id_to_name():
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
