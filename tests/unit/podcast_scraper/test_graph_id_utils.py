"""Unit tests for global graph id helpers."""

import pytest

from podcast_scraper.graph_id_utils import (
    entity_node_id,
    episode_node_id,
    gil_insight_node_id,
    gil_quote_node_id,
    is_person_or_org_node,
    normalized_entity_kind_from_node,
    person_node_id,
    PERSON_ORG_NODE_TYPES,
    slugify_label,
    topic_node_id_from_slug,
)

pytestmark = [pytest.mark.unit]


def test_episode_and_topic_ids() -> None:
    assert episode_node_id("abc") == "episode:abc"
    assert topic_node_id_from_slug("inflation") == "topic:inflation"


def test_entity_and_person_ids() -> None:
    assert entity_node_id("person", "Jane Doe") == "person:jane-doe"
    assert entity_node_id("organization", "Acme") == "org:acme"
    assert person_node_id("Sam Altman") == "person:sam-altman"


def test_gil_hashes_stable() -> None:
    i1 = gil_insight_node_id("ep1", 0, "Hello")
    i2 = gil_insight_node_id("ep1", 0, "Hello")
    i3 = gil_insight_node_id("ep1", 1, "Hello")
    assert i1 == i2
    assert i1 != i3
    assert i1.startswith("insight:")
    q1 = gil_quote_node_id("ep1", 0, "text", 0, 4)
    q2 = gil_quote_node_id("ep1", 0, "text", 0, 5)
    assert q1 != q2
    assert q1.startswith("quote:")


def test_slugify_label() -> None:
    assert slugify_label("  Car Loans!! ") == "car-loans"


def test_slugify_label_max_len_truncates() -> None:
    long_word = "x" * 100
    assert len(slugify_label(long_word, max_len=40)) == 40


def test_person_node_id_diacritic_normalizes_like_cil() -> None:
    assert person_node_id("José") == "person:jose"


# RFC-097 v2.0 helpers — typed Person/Organization replace the legacy Entity discriminator.


def test_person_org_node_types_constant() -> None:
    assert PERSON_ORG_NODE_TYPES == frozenset({"Entity", "Person", "Organization"})


def test_is_person_or_org_node_recognizes_v2_typed_nodes() -> None:
    assert is_person_or_org_node("Person") is True
    assert is_person_or_org_node("Organization") is True
    assert is_person_or_org_node("Entity") is True  # legacy v1.x still recognized
    assert is_person_or_org_node("Topic") is False
    assert is_person_or_org_node("Episode") is False
    assert is_person_or_org_node(None) is False
    assert is_person_or_org_node(123) is False


def test_normalized_entity_kind_v2_typed_node_takes_precedence() -> None:
    # v2.0 typed nodes carry the kind in the node type itself.
    assert (
        normalized_entity_kind_from_node({"type": "Person", "properties": {"name": "x"}})
        == "person"
    )
    assert (
        normalized_entity_kind_from_node({"type": "Organization", "properties": {"name": "y"}})
        == "organization"
    )


def test_normalized_entity_kind_v1_2_kind_property() -> None:
    assert (
        normalized_entity_kind_from_node(
            {"type": "Entity", "properties": {"name": "x", "kind": "person"}}
        )
        == "person"
    )
    assert (
        normalized_entity_kind_from_node(
            {"type": "Entity", "properties": {"name": "x", "kind": "org"}}
        )
        == "organization"
    )


def test_normalized_entity_kind_legacy_entity_kind_fallback() -> None:
    assert (
        normalized_entity_kind_from_node(
            {"type": "Entity", "properties": {"name": "x", "entity_kind": "organization"}}
        )
        == "organization"
    )
    assert (
        normalized_entity_kind_from_node(
            {"type": "Entity", "properties": {"name": "x", "entity_kind": "person"}}
        )
        == "person"
    )


def test_normalized_entity_kind_defaults_to_person_when_unknown() -> None:
    # Properties missing or unrecognized → conservative default.
    assert normalized_entity_kind_from_node({"type": "Entity"}) == "person"
    assert normalized_entity_kind_from_node({"type": "Entity", "properties": {}}) == "person"
    assert normalized_entity_kind_from_node({"type": "Entity", "properties": None}) == "person"
    assert (
        normalized_entity_kind_from_node({"type": "Entity", "properties": {"kind": "bogus"}})
        == "person"
    )
