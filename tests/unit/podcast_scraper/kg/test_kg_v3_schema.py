"""KG 3.0 (ontology v3): valid artifacts for a podcast item and a test-type item, and the shapes
the design rules out (docs/architecture/corpus/ontology-v3.md, ADR-178)."""

from __future__ import annotations

import copy

import pytest

from podcast_scraper.kg.schema import validate_artifact

pytestmark = pytest.mark.unit


def _podcast() -> dict:
    return {
        "schema_version": "3.0",
        "item_id": "ep-1",
        "content_type": "podcast_episode",
        "extraction": {
            "model_version": "provider:test",
            "extracted_at": "2026-11-11T00:00:00Z",
            "document_ref": "transcripts/0001 - Ep.txt",
            "variant": "adfree",
        },
        "nodes": [
            {
                "id": "podcast:show",
                "type": "Source",
                "properties": {"title": "Show", "source_kind": "podcast", "transport": "rss"},
            },
            {
                "id": "episode:ep-1",
                "type": "ContentItem",
                "properties": {
                    "title": "Ep",
                    "content_type": "podcast_episode",
                    "source_id": "podcast:show",
                    "published_at": "2026-11-01T00:00:00Z",
                    "duration_ms": 1000,
                },
            },
            {"id": "person:ann", "type": "Person", "properties": {"name": "Ann", "role": "host"}},
            {"id": "topic:risk", "type": "Topic", "properties": {"label": "Risk", "slug": "risk"}},
        ],
        "edges": [
            {"type": "HAS_ITEM", "from": "podcast:show", "to": "episode:ep-1"},
            {
                "type": "PARTICIPATES_IN",
                "from": "person:ann",
                "to": "episode:ep-1",
                "properties": {"role": "host"},
            },
            {"type": "MENTIONS", "from": "topic:risk", "to": "episode:ep-1"},
        ],
    }


def _text_item() -> dict:
    return {
        "schema_version": "3.0",
        "item_id": "item:test_text_item:0123456789abcdef0123",
        "content_type": "test_text_item",
        "extraction": {
            "model_version": "metadata_only",
            "extracted_at": "2026-11-11T00:00:00Z",
            "document_ref": "transcripts/0001 - Text.txt",
            "variant": "raw",
        },
        "nodes": [
            {
                "id": "source:texts",
                "type": "Source",
                "properties": {"title": "Texts", "source_kind": "news", "transport": "rss"},
            },
            {
                "id": "item:test_text_item:0123456789abcdef0123",
                "type": "ContentItem",
                "properties": {
                    "title": "A text",
                    "content_type": "test_text_item",
                    "source_id": "source:texts",
                    "url": "https://example.test/a",
                    "published_at": "2026-11-10T09:00:00Z",
                },
            },
            {
                "id": "person:bo",
                "type": "Person",
                "properties": {"name": "Bo", "role": "author"},
            },
            {
                "id": "org:example",
                "type": "Organization",
                "properties": {
                    "name": "Example",
                    "role": "publisher",
                    "identifiers": [{"scheme": "domain", "value": "example.test"}],
                },
            },
        ],
        "edges": [
            {
                "type": "HAS_ITEM",
                "from": "source:texts",
                "to": "item:test_text_item:0123456789abcdef0123",
            },
            {"type": "PUBLISHED_BY", "from": "source:texts", "to": "org:example"},
            {
                "type": "PARTICIPATES_IN",
                "from": "person:bo",
                "to": "item:test_text_item:0123456789abcdef0123",
                "properties": {"role": "author"},
            },
        ],
    }


@pytest.mark.parametrize("make", [_podcast, _text_item])
def test_a_valid_v3_artifact_passes_strict_validation(make) -> None:
    validate_artifact(make(), strict=True)


def test_a_later_role_needs_no_schema_change() -> None:
    art = _text_item()
    art["nodes"][2]["properties"]["role"] = "analyst"
    validate_artifact(art, strict=True)


def _mutated(path: list, value=None, delete: bool = False) -> dict:
    art = copy.deepcopy(_text_item())
    target = art
    for key in path[:-1]:
        target = target[key]
    if delete:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return art


@pytest.mark.parametrize(
    "art",
    [
        _mutated(["item_id"], delete=True),
        _mutated(["content_type"], delete=True),
        _mutated(["extraction", "document_ref"], delete=True),
        _mutated(["extraction", "variant"], delete=True),
        _mutated(["episode_id"], "ep-1"),  # a v2 root field
        _mutated(["nodes", 1, "type"], "Episode"),  # a v2 node type
        _mutated(["nodes", 1, "properties", "content_type"], delete=True),
        _mutated(["edges", 0, "type"], "HAS_EPISODE"),  # a v2 edge
        _mutated(["nodes", 2, "properties", "role"], ""),
        _mutated(["nodes", 3, "properties", "identifiers"], [{"scheme": "cik"}]),
        _mutated(["nodes", 1, "properties", "publish_date"], "2026-11-10"),  # the v2 time field
    ],
)
def test_shapes_the_design_rules_out_are_invalid(art: dict) -> None:
    with pytest.raises(ValueError):
        validate_artifact(art, strict=True)


def test_a_v2_artifact_still_validates_against_the_v2_schema() -> None:
    v2 = {
        "schema_version": "2.1",
        "episode_id": "ep-1",
        "extraction": {
            "model_version": "metadata_only",
            "extracted_at": "2026-11-11T00:00:00Z",
            "transcript_ref": "t.txt",
        },
        "nodes": [],
        "edges": [],
    }
    validate_artifact(v2, strict=True)
    v2_as_v3 = dict(v2, schema_version="3.0")
    with pytest.raises(ValueError):
        validate_artifact(v2_as_v3)
