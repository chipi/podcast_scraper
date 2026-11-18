"""KG 3.0 as stored (``podcast_scraper.kg.v3_file``, R3b's KG half): a KG 2.1 file becomes a valid
KG 3.0 file, hosts and guests become ``CONTRIBUTES_TO`` with their role, and ``PUBLISHED_BY`` comes
only from the publisher registry."""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from podcast_scraper.kg.schema import validate_artifact
from podcast_scraper.kg.v3_file import to_v3_file
from podcast_scraper.publishers import Publisher, Registry

pytestmark = pytest.mark.unit


def _kg21() -> Dict[str, Any]:
    return {
        "schema_version": "2.1",
        "episode_id": "ep",
        "extraction": {
            "model_version": "provider:openai:gpt-x",
            "extracted_at": "2026-11-01T00:00:00Z",
            "transcript_ref": "../run_x/transcripts/ep.adfree.txt",
        },
        "nodes": [
            {
                "id": "episode:ep",
                "type": "Episode",
                "properties": {
                    "title": "Ep",
                    "podcast_id": "rss_feed_1",
                    "feed_id": "rss_feed_1",
                    "publish_date": "2026-11-01",
                },
            },
            {"id": "podcast:rss_feed_1", "type": "Podcast", "properties": {"title": "Show"}},
            {
                "id": "person:ann",
                "type": "Person",
                "properties": {
                    "name": "Ann",
                    "label": "Ann",
                    "role": "host",
                    "respelled_from": "Anne",
                },
            },
            {
                "id": "org:the-new-york-times",
                "type": "Organization",
                "properties": {"name": "The NYT", "label": "The NYT", "role": "mentioned"},
            },
        ],
        "edges": [
            {"type": "HAS_EPISODE", "from": "podcast:rss_feed_1", "to": "episode:ep"},
            {"type": "HOSTS", "from": "person:ann", "to": "podcast:rss_feed_1"},
            {"type": "MENTIONS", "from": "org:the-new-york-times", "to": "episode:ep"},
        ],
    }


def test_a_2_1_file_becomes_valid_kg_3_0() -> None:
    v3 = to_v3_file(_kg21(), registry=Registry())
    validate_artifact(v3, strict=True)
    text = json.dumps(v3)
    for name in ("episode_id", "adapted_from", "HAS_EPISODE", "HOSTS", "transcript_ref"):
        assert f'"{name}"' not in text, name
    edges = {(e["type"], e["from"], e["to"]): e.get("properties") for e in v3["edges"]}
    assert edges[("CONTRIBUTES_TO", "person:ann", "podcast:rss_feed_1")] == {"role": "host"}
    assert not any(e["type"] == "PUBLISHED_BY" for e in v3["edges"])
    ann = next(n for n in v3["nodes"] if n["id"] == "person:ann")
    assert ann["properties"]["respelled_from"] == "Anne", "the ledger's evidence stays"


def test_published_by_comes_from_the_registry_and_reuses_the_org_node() -> None:
    nyt = Publisher(
        id="org:the-new-york-times", name="The New York Times", sources=("podcast:rss_feed_1",)
    )
    v3 = to_v3_file(_kg21(), registry=Registry((nyt,)))
    validate_artifact(v3, strict=True)
    assert {
        "type": "PUBLISHED_BY",
        "from": "podcast:rss_feed_1",
        "to": "org:the-new-york-times",
    } in v3["edges"]
    orgs = [n for n in v3["nodes"] if n["id"] == "org:the-new-york-times"]
    assert len(orgs) == 1, "an identical id is one organisation (ADR-180 rule 6 as amended)"


def test_the_registry_adds_a_publisher_node_the_file_lacks() -> None:
    bbc = Publisher(id="org:bbc", name="BBC", sources=("podcast:rss_feed_1",))
    v3 = to_v3_file(_kg21(), registry=Registry((bbc,)))
    validate_artifact(v3, strict=True)
    node = next(n for n in v3["nodes"] if n["id"] == "org:bbc")
    assert node["properties"]["role"] == "publisher"


def test_a_3_0_file_comes_back_unchanged() -> None:
    v3 = to_v3_file(_kg21(), registry=Registry())
    assert to_v3_file(v3, registry=Registry()) == v3
