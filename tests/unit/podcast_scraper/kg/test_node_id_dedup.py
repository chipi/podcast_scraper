"""Two nodes must never share an id in a KG artifact (2026-08-26 reindex abort).

The per-site dedup keys on the raw NAME while node ids key on the SLUG, so near-identical
labels ("David Clark" / "David Clark)") collide on id — and hosts vs mentioned-entities
append from different sites with no cross-site check at all. LanceDB then refuses the
ambiguous merge and the WHOLE corpus reindex aborts. The artifact-level pass is the fix.
"""

from __future__ import annotations

from collections import Counter

import pytest

from podcast_scraper.kg.pipeline import _dedupe_nodes_by_id, build_artifact

pytestmark = [pytest.mark.unit]


def test_paren_variant_host_and_mentioned_entity_become_one_node() -> None:
    """The live shape: entity list says 'David Clark' (mentioned), hosts say 'David Clark)'."""
    payload = build_artifact(
        episode_id="ep-1",
        transcript_text="a transcript",
        podcast_id="show-1",
        episode_title="Ep",
        detected_hosts=["David Clark)"],
        prefilled_partial={
            "topics": [{"label": "venture capital"}],
            "entities": [
                {
                    "name": "David Clark",
                    "entity_kind": "person",
                    "description": "CIO of VenCap",
                }
            ],
        },
    )
    ids = [n["id"] for n in payload["nodes"]]
    dupes = {i: c for i, c in Counter(ids).items() if c > 1}
    assert not dupes, f"duplicate node ids survived to the artifact: {dupes}"
    clark = [n for n in payload["nodes"] if n["id"] == "person:david-clark"]
    assert len(clark) == 1
    props = clark[0]["properties"]
    assert props["role"] == "host", "the explicit host role must beat the default 'mentioned'"
    # NOTE deliberately NOT asserted: description survival through the prefilled-partial path —
    # that normalization drops descriptions before the dedup ever runs (pre-existing shape).
    # Property-merge semantics are pinned directly on the helper below.


def test_dedupe_helper_merges_and_preserves_order() -> None:
    nodes = [
        {"id": "person:a", "type": "Person", "properties": {"role": "mentioned", "x": 1}},
        {"id": "topic:t", "type": "Topic", "properties": {}},
        {"id": "person:a", "type": "Person", "properties": {"role": "host", "y": 2}},
    ]
    out = _dedupe_nodes_by_id(nodes)
    assert [n["id"] for n in out] == ["person:a", "topic:t"]
    assert out[0]["properties"]["role"] == "host"
    assert out[0]["properties"]["x"] == 1 and out[0]["properties"]["y"] == 2
