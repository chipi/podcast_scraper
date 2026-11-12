"""The one adapter that reads KG 2.x as ontology v3 (``kg/as_v3.py``; ontology-v3.md "Reading v2
artifacts as v3"): each mapping rule, and every KG fixture in the repo read through it."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.kg.as_v3 import document_variant, is_view, kg_as_v3
from podcast_scraper.kg.schema import validate_artifact

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).resolve().parents[4] / "tests" / "fixtures"
_V3_NODES = {"Source", "ContentItem", "Person", "Organization", "Object", "Topic"}
_V3_EDGES = {"HAS_ITEM", "PARTICIPATES_IN", "CONTRIBUTES_TO", "PUBLISHED_BY", "MENTIONS"}
_V3_EDGES |= {"RELATED_TO"}


def _v21() -> Dict[str, Any]:
    return {
        "schema_version": "2.1",
        "episode_id": "ep-1",
        "extraction": {
            "model_version": "metadata_only",
            "extracted_at": "2026-01-01T00:00:00Z",
            "transcript_ref": "transcripts/0001 - Ep.cleaned.txt",
        },
        "nodes": [
            {"id": "podcast:show", "type": "Podcast", "properties": {"title": "Show"}},
            {
                "id": "episode:ep-1",
                "type": "Episode",
                "properties": {
                    "title": "Ep",
                    "podcast_id": "show",
                    "publish_date": "2026-01-01T00:00:00",
                },
            },
            {"id": "person:ann", "type": "Person", "properties": {"name": "Ann"}},
            {"id": "person:bo", "type": "Person", "properties": {"name": "Bo", "x": 1}},
            {"id": "topic:t", "type": "Topic", "properties": {"label": "T", "topic_source": "x"}},
        ],
        "edges": [
            {"type": "HAS_EPISODE", "from": "podcast:show", "to": "episode:ep-1"},
            {"type": "HOSTS", "from": "person:ann", "to": "episode:ep-1"},
            {"type": "GUESTS_ON", "from": "person:bo", "to": "episode:ep-1"},
            {"type": "MENTIONS", "source": "episode:ep-1", "target": "topic:t"},
        ],
    }


def test_each_mapping_rule() -> None:
    v2 = _v21()
    before = copy.deepcopy(v2)
    view = kg_as_v3(v2)
    assert v2 == before, "the input is never modified"

    assert view["schema_version"] == "3.0" and view["adapted_from"] == "2.1" and is_view(view)
    assert view["item_id"] == "ep-1" and view["content_type"] == "podcast_episode"
    nodes = {n["id"]: n for n in view["nodes"]}
    assert set(nodes) == {n["id"] for n in v2["nodes"]}, "node ids never change (T-8)"
    assert nodes["podcast:show"]["type"] == "Source"
    assert nodes["podcast:show"]["properties"]["source_kind"] == "podcast"
    item = nodes["episode:ep-1"]
    assert item["type"] == "ContentItem"
    assert item["properties"] == {
        "title": "Ep",
        "content_type": "podcast_episode",
        "source_id": "podcast:show",
        "feed_id": "show",
        "published_at": "2026-01-01T00:00:00",
    }
    assert nodes["person:bo"]["properties"]["x"] == 1, "properties v3 does not define are kept"
    assert nodes["topic:t"]["properties"]["topic_source"] == "x"

    edges = [
        (e["type"], e["from"], e["to"], (e.get("properties") or {}).get("role"))
        for e in view["edges"]
    ]
    assert edges == [
        ("HAS_ITEM", "podcast:show", "episode:ep-1", None),
        ("CONTRIBUTES_TO", "person:ann", "episode:ep-1", "host"),
        ("CONTRIBUTES_TO", "person:bo", "episode:ep-1", "guest"),
        ("MENTIONS", "episode:ep-1", "topic:t", None),
    ]
    assert view["extraction"]["document_ref"] == "transcripts/0001 - Ep.cleaned.txt"
    assert view["extraction"]["variant"] == "cleaned"
    assert "transcript_ref" not in view["extraction"]
    validate_artifact(view, strict=False)


def test_a_kg_without_a_podcast_node_takes_its_source_from_the_feed() -> None:
    v2 = _v21()
    v2["nodes"] = [n for n in v2["nodes"] if n["type"] != "Podcast"]
    item = next(n for n in kg_as_v3(v2)["nodes"] if n["type"] == "ContentItem")
    assert item["properties"]["source_id"] == "podcast:show"


def test_the_item_id_comes_from_the_episode_node_or_is_absent() -> None:
    v2 = _v21()
    del v2["episode_id"]
    assert kg_as_v3(v2)["item_id"] == "ep-1", "the Episode node names it"
    v2["nodes"] = [n for n in v2["nodes"] if n["type"] != "Episode"]
    assert "item_id" not in kg_as_v3(v2), "never an empty item_id"


def test_a_v3_artifact_comes_back_as_it_is() -> None:
    v3 = {"schema_version": "3.0", "item_id": "item:t:1", "content_type": "t", "nodes": []}
    assert kg_as_v3(v3) == v3 and not is_view(kg_as_v3(v3))


@pytest.mark.parametrize(
    "ref, language, variant",
    [
        ("transcripts/0001 - Ep.txt", None, "raw"),
        ("transcripts/0001 - Ep.txt", "en-US", "raw"),
        ("transcripts/0001 - Ep.txt", "es", "en"),  # the English render of a Spanish episode
        ("transcripts/0001 - Ep.es.txt", "es", "raw"),  # its source-language original
        ("transcripts/0001 - Ep.cleaned.txt", "es", "cleaned"),
        ("transcripts/0001 - Ep.adfree.txt", None, "adfree"),
    ],
)
def test_the_document_variant(ref: str, language: str | None, variant: str) -> None:
    assert document_variant(ref, language) == variant


def _fixture_kgs() -> List[Path]:
    return sorted(_FIXTURES.rglob("*.kg.json"))


def test_every_fixture_kg_reads_through_the_adapter() -> None:
    paths = _fixture_kgs()
    assert len(paths) > 100, len(paths)
    checked_v2 = 0
    for path in paths:
        kg = json.loads(path.read_text(encoding="utf-8"))
        before = copy.deepcopy(kg)
        view = kg_as_v3(kg)  # never raises, whatever the version
        assert kg == before, path
        assert [n.get("id") for n in view.get("nodes") or []] == [
            n.get("id") for n in kg.get("nodes") or []
        ], path
        if str(kg.get("schema_version")) not in ("2.0", "2.1"):
            continue
        checked_v2 += 1
        assert {n.get("type") for n in view["nodes"]} <= _V3_NODES, path
        assert all(e["type"] in _V3_EDGES and e["from"] and e["to"] for e in view["edges"]), path
        if "extraction" in kg:
            validate_artifact(view, strict=False)
    assert checked_v2 > 50, checked_v2
