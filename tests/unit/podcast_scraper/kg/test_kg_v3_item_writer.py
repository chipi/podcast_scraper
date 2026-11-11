"""A non-podcast item's KG 3.0 artifact, written from the KG 2.x graph its extraction built
(kg/v3.py; ontology v3, ADR-178). Every artifact here must pass kg.v3.schema.json strictly."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.kg import v3
from podcast_scraper.kg.pipeline import build_artifact
from podcast_scraper.kg.schema import validate_artifact

pytestmark = pytest.mark.unit

ITEM = "item:test_text_item:one"
SOURCE = "source:0123456789abcdef0123"


def _v2(**kwargs: Any) -> Dict[str, Any]:
    cfg = type("Cfg", (), {"kg_extraction_source": "metadata_only"})()
    return build_artifact(
        ITEM,
        "The body.",
        podcast_id="Texts",
        episode_title="A text",
        cfg=cfg,
        **kwargs,
    )


def _v3(v2_payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    args: Dict[str, Any] = dict(
        item_id=ITEM,
        content_type="test_text_item",
        item_title="A text",
        source_id=SOURCE,
        source_title="Texts",
        source_kind="test",
        transport="fixture_feed",
        document_ref="transcripts/0000 - A text.txt",
        item_url="https://example.com/one",
        source_url="https://example.com/fixture-source",
        times={"published_at": "2026-11-01T09:00:00Z", "filed_at": None},
    )
    args.update(kwargs)
    return v3.item_artifact(v2_payload, **args)


def test_the_item_artifact_is_valid_kg_3() -> None:
    art = _v3(_v2())
    validate_artifact(art, strict=True)
    assert art["schema_version"] == "3.0"
    assert (art["item_id"], art["content_type"]) == (ITEM, "test_text_item")
    assert art["extraction"]["variant"] == "raw"
    by_type = {n["type"]: n for n in art["nodes"]}
    assert set(by_type) >= {"Source", "ContentItem"}
    assert "Podcast" not in by_type and "Episode" not in by_type
    item = by_type["ContentItem"]
    assert item["id"] == ITEM
    assert item["properties"]["source_id"] == SOURCE
    assert item["properties"]["published_at"] == "2026-11-01T09:00:00Z"
    assert "filed_at" not in item["properties"]  # unknown times are absent, never guessed
    assert by_type["Source"]["properties"]["source_kind"] == "test"
    assert {"type": "HAS_ITEM", "from": SOURCE, "to": ITEM} in art["edges"]


def test_hosts_and_guests_become_contributes_to_with_the_role() -> None:
    art = _v3(_v2(detected_hosts=["Ann Host"], detected_guests=["Bo Guest"]))
    validate_artifact(art, strict=True)
    roles = {
        (e["from"], e["properties"]["role"])
        for e in art["edges"]
        if e["type"] == "CONTRIBUTES_TO" and e["to"] == SOURCE
    }
    assert roles == {("person:ann-host", "host"), ("person:bo-guest", "guest")}
    ends = {e["to"] for e in art["edges"]} | {e["from"] for e in art["edges"]}
    assert not any(x.startswith(("episode:", "podcast:")) for x in ends)


def test_a_v2_shape_v3_has_no_place_for_is_refused() -> None:
    v2_payload = _v2()
    v2_payload["nodes"].append({"id": "x:1", "type": "Insight", "properties": {}})
    with pytest.raises(v3.V3MappingError, match="Insight"):
        _v3(v2_payload)
    v2_payload = _v2()
    v2_payload["edges"].append({"type": "SPOKE_IN", "from": "person:a", "to": ITEM})
    with pytest.raises(v3.V3MappingError, match="SPOKE_IN"):
        _v3(v2_payload)
