"""RFC-097 chunk 8 follow-up: indexer covers v2.0 typed Person + Organization nodes.

The v2.0 KG pipeline emits typed `Person` and `Organization` nodes; the indexer
must add them to the `kg_entity` vector docs alongside legacy `Entity` nodes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search.indexer import _kg_vector_rows_from_path

pytestmark = pytest.mark.unit


def _write_kg(path: Path, nodes: list[dict]) -> None:
    """Write a minimal v2.0 kg.json artifact with the given nodes."""
    path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "episode_id": "p01_e01",
                "extraction": {
                    "model_version": "test",
                    "extracted_at": "2026-06-22T00:00:00Z",
                    "transcript_ref": "x",
                },
                "nodes": nodes,
                "edges": [],
            }
        ),
        encoding="utf-8",
    )


def test_person_node_indexed_as_kg_entity_with_person_kind(tmp_path: Path) -> None:
    kg = tmp_path / "ep.kg.json"
    _write_kg(
        kg,
        [
            {
                "id": "person:maya-singletrack",
                "type": "Person",
                "properties": {"name": "Maya Singletrack", "role": "guest"},
            }
        ],
    )

    rows = _kg_vector_rows_from_path(
        kg, scope_tag="dev", episode_id="p01_e01", feed_id="feed_a", published="2026-01-01"
    )

    assert len(rows) == 1
    row_id, text, meta = rows[0]
    assert row_id == "kg_entity:dev:person:maya-singletrack"
    assert "Maya Singletrack" in text
    assert meta["doc_type"] == "kg_entity"
    assert meta["source_id"] == "person:maya-singletrack"
    assert meta["entity_kind"] == "person"


def test_organization_node_indexed_as_kg_entity_with_organization_kind(
    tmp_path: Path,
) -> None:
    kg = tmp_path / "ep.kg.json"
    _write_kg(
        kg,
        [
            {
                "id": "org:singletrack-sessions",
                "type": "Organization",
                "properties": {"name": "Singletrack Sessions"},
            }
        ],
    )

    rows = _kg_vector_rows_from_path(
        kg, scope_tag="dev", episode_id="p01_e01", feed_id="feed_a", published="2026-01-01"
    )

    assert len(rows) == 1
    _row_id, _text, meta = rows[0]
    assert meta["entity_kind"] == "organization"


def test_legacy_entity_node_still_indexes_via_kind_helper(tmp_path: Path) -> None:
    kg = tmp_path / "ep.kg.json"
    _write_kg(
        kg,
        [
            {
                "id": "entity:legacy-person",
                "type": "Entity",
                "properties": {"name": "Legacy Person", "kind": "person"},
            }
        ],
    )

    rows = _kg_vector_rows_from_path(
        kg, scope_tag="dev", episode_id="p01_e01", feed_id="feed_a", published="2026-01-01"
    )

    assert len(rows) == 1
    _row_id, _text, meta = rows[0]
    # Legacy branch: kind is resolved by _kg_entity_kind_for_meta from props.
    assert meta["entity_kind"] in ("person", None)


def test_mixed_typed_and_legacy_nodes_all_indexed(tmp_path: Path) -> None:
    kg = tmp_path / "ep.kg.json"
    _write_kg(
        kg,
        [
            {
                "id": "person:a",
                "type": "Person",
                "properties": {"name": "Person A"},
            },
            {
                "id": "org:b",
                "type": "Organization",
                "properties": {"name": "Org B"},
            },
            {
                "id": "entity:c",
                "type": "Entity",
                "properties": {"name": "Legacy C", "kind": "person"},
            },
            # Should not produce kg_entity row (Topic is its own branch).
            {
                "id": "topic:d",
                "type": "Topic",
                "properties": {"label": "Some Topic"},
            },
        ],
    )

    rows = _kg_vector_rows_from_path(
        kg, scope_tag="dev", episode_id="p01_e01", feed_id="feed_a", published="2026-01-01"
    )

    # 3 kg_entity (Person + Organization + Entity) + 1 kg_topic
    kinds = sorted({meta["doc_type"] for _, _, meta in rows})
    assert kinds == ["kg_entity", "kg_topic"]
    entity_ids = sorted(row_id for row_id, _, meta in rows if meta["doc_type"] == "kg_entity")
    assert len(entity_ids) == 3
