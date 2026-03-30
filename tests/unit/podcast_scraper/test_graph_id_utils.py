"""Unit tests for global graph id helpers."""

import pytest

from podcast_scraper.graph_id_utils import (
    entity_node_id,
    episode_node_id,
    gil_insight_node_id,
    gil_quote_node_id,
    slugify_label,
    speaker_node_id,
    topic_node_id_from_slug,
)

pytestmark = [pytest.mark.unit]


def test_episode_and_topic_ids() -> None:
    assert episode_node_id("abc") == "episode:abc"
    assert topic_node_id_from_slug("inflation") == "topic:inflation"


def test_entity_and_speaker_ids() -> None:
    assert entity_node_id("person", "Jane Doe") == "entity:person:jane-doe"
    assert speaker_node_id("Sam Altman") == "speaker:sam-altman"


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
