"""Unit tests for GI/KG identity JSON migrations."""

import pytest

from podcast_scraper.migrations.gil_kg_identity_migrations import (
    compute_position_hints_for_document,
    migrate_gi_document_v3,
    migrate_gil_document,
    migrate_kg_document,
    migrate_kg_document_v2,
)

pytestmark = pytest.mark.unit


def test_migrate_gil_speaker_to_person_and_ids() -> None:
    data = {
        "schema_version": "1.0",
        "episode_id": "e1",
        "model_version": "m",
        "prompt_version": "v1",
        "nodes": [
            {
                "id": "episode:e1",
                "type": "Episode",
                "properties": {
                    "podcast_id": "p",
                    "title": "T",
                    "publish_date": "2020-01-01T00:00:00Z",
                },
            },
            {"id": "speaker:alice", "type": "Speaker", "properties": {"name": "Alice"}},
            {
                "id": "quote:x",
                "type": "Quote",
                "properties": {
                    "text": "hi",
                    "episode_id": "e1",
                    "speaker_id": "speaker:alice",
                    "char_start": 0,
                    "char_end": 2,
                    "timestamp_start_ms": 0,
                    "timestamp_end_ms": 0,
                    "transcript_ref": "t.txt",
                },
            },
        ],
        "edges": [{"type": "SPOKEN_BY", "from": "quote:x", "to": "speaker:alice"}],
    }
    out = migrate_gil_document(data)
    assert out["schema_version"] == "2.0"
    persons = [n for n in out["nodes"] if n["type"] == "Person"]
    assert len(persons) == 1
    assert persons[0]["id"] == "person:alice"
    q = next(n for n in out["nodes"] if n["type"] == "Quote")
    assert q["properties"]["speaker_id"] == "person:alice"
    e = out["edges"][0]
    assert e["to"] == "person:alice"


def test_migrate_gil_idempotent() -> None:
    data = {
        "schema_version": "2.0",
        "episode_id": "e1",
        "nodes": [
            {"id": "person:bob", "type": "Person", "properties": {"name": "Bob"}},
        ],
        "edges": [],
    }
    out = migrate_gil_document(migrate_gil_document(data))
    assert out == migrate_gil_document(data)


def test_migrate_kg_entity_ids_and_kind() -> None:
    data = {
        "schema_version": "1.1",
        "episode_id": "e1",
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2020-01-01T00:00:00Z",
            "transcript_ref": "t.txt",
        },
        "nodes": [
            {
                "id": "entity:person:pat",
                "type": "Entity",
                "properties": {"name": "Pat", "entity_kind": "person", "role": "host"},
            },
            {
                "id": "entity:organization:acme",
                "type": "Entity",
                "properties": {"name": "Acme", "entity_kind": "organization", "role": "mentioned"},
            },
        ],
        "edges": [
            {"type": "MENTIONS", "from": "entity:person:pat", "to": "episode:e1"},
        ],
    }
    out = migrate_kg_document(data)
    assert out["schema_version"] == "1.2"
    ids = {n["id"] for n in out["nodes"]}
    assert "person:pat" in ids
    assert "org:acme" in ids
    pat = next(n for n in out["nodes"] if n["properties"]["name"] == "Pat")
    assert pat["properties"]["kind"] == "person"
    assert "entity_kind" not in pat["properties"]
    acme = next(n for n in out["nodes"] if n["properties"]["name"] == "Acme")
    assert acme["properties"]["kind"] == "org"
    assert out["edges"][0]["from"] == "person:pat"


def test_migrate_gil_nodes_not_list_unchanged() -> None:
    data = {"schema_version": "1.0", "nodes": {"not": "a list"}}
    out = migrate_gil_document(data)
    assert out["nodes"] == {"not": "a list"}


def test_migrate_kg_nodes_not_list_unchanged() -> None:
    data = {"schema_version": "1.1", "nodes": "x"}
    out = migrate_kg_document(data)
    assert out["nodes"] == "x"


def test_migrate_kg_idempotent() -> None:
    data = {
        "schema_version": "1.2",
        "episode_id": "e1",
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2020-01-01T00:00:00Z",
            "transcript_ref": "t.txt",
        },
        "nodes": [
            {
                "id": "person:x",
                "type": "Entity",
                "properties": {"name": "X", "kind": "person", "role": "host"},
            },
        ],
        "edges": [],
    }
    once = migrate_kg_document(data)
    twice = migrate_kg_document(once)
    assert once == twice


# ─── RFC-097 chunk 6: v2/v3 migrations + position_hint backfill ───


def test_migrate_kg_document_v2_entity_to_typed_nodes() -> None:
    """v2.0: legacy Entity(kind=person|org) → typed Person / Organization nodes."""
    data = {
        "schema_version": "1.2",
        "episode_id": "e1",
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2024-01-01T00:00:00Z",
            "transcript_ref": "t.txt",
        },
        "nodes": [
            {
                "id": "episode:e1",
                "type": "Episode",
                "properties": {
                    "podcast_id": "podcast:p",
                    "title": "T",
                    "publish_date": "2024-01-01T00:00:00Z",
                },
            },
            {
                "id": "person:alice",
                "type": "Entity",
                "properties": {"name": "Alice", "kind": "person", "role": "host"},
            },
            {
                "id": "org:acme",
                "type": "Entity",
                "properties": {"name": "ACME", "kind": "org"},
            },
        ],
        "edges": [
            {"type": "MENTIONS", "from": "person:alice", "to": "episode:e1", "properties": {}},
        ],
    }
    out = migrate_kg_document_v2(data)
    assert out["schema_version"] == "2.0"
    types_by_id = {n["id"]: n["type"] for n in out["nodes"]}
    assert types_by_id["person:alice"] == "Person"
    assert types_by_id["org:acme"] == "Organization"
    # Legacy `kind` property must be dropped — node type carries it now.
    alice = next(n for n in out["nodes"] if n["id"] == "person:alice")
    assert "kind" not in alice["properties"]
    # Edges untouched.
    assert any(e["type"] == "MENTIONS" for e in out["edges"])


def test_migrate_kg_document_v2_idempotent() -> None:
    """Running the v2 migration twice yields the same document."""
    data = {
        "schema_version": "1.2",
        "episode_id": "e1",
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2024-01-01T00:00:00Z",
            "transcript_ref": "t.txt",
        },
        "nodes": [
            {
                "id": "episode:e1",
                "type": "Episode",
                "properties": {
                    "podcast_id": "p",
                    "title": "T",
                    "publish_date": "2024-01-01T00:00:00Z",
                },
            },
            {
                "id": "person:x",
                "type": "Entity",
                "properties": {"name": "X", "kind": "person", "role": "host"},
            },
        ],
        "edges": [],
    }
    once = migrate_kg_document_v2(data)
    twice = migrate_kg_document_v2(once)
    assert once == twice


def test_migrate_kg_document_v2_from_legacy_v1_chain() -> None:
    """v1.0 (entity:person: + entity_kind) → v2.0 in one call."""
    data = {
        "schema_version": "1.0",
        "episode_id": "e1",
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2024-01-01T00:00:00Z",
            "transcript_ref": "t.txt",
        },
        "nodes": [
            {
                "id": "entity:person:alice",
                "type": "Entity",
                "properties": {"name": "Alice", "entity_kind": "person", "role": "host"},
            },
            {
                "id": "entity:organization:acme",
                "type": "Entity",
                "properties": {"name": "ACME", "entity_kind": "organization"},
            },
        ],
        "edges": [],
    }
    out = migrate_kg_document_v2(data)
    assert out["schema_version"] == "2.0"
    ids = {n["id"] for n in out["nodes"]}
    assert "person:alice" in ids
    assert "org:acme" in ids
    types = {n["id"]: n["type"] for n in out["nodes"]}
    assert types["person:alice"] == "Person"
    assert types["org:acme"] == "Organization"


def test_migrate_gi_document_v3_typed_mentions_and_insight_type() -> None:
    """v3.0: rewrite generic MENTIONS → MENTIONS_PERSON/MENTIONS_ORG + normalise vocab."""
    data = {
        "schema_version": "2.0",
        "episode_id": "e1",
        "model_version": "m",
        "prompt_version": "v1",
        "nodes": [
            {
                "id": "episode:e1",
                "type": "Episode",
                "properties": {
                    "podcast_id": "p",
                    "title": "T",
                    "publish_date": "2024-01-01T00:00:00Z",
                },
            },
            {
                "id": "insight:i1",
                "type": "Insight",
                "properties": {
                    "text": "Some claim.",
                    "episode_id": "e1",
                    "grounded": False,
                    "insight_type": "fact",
                },
            },
        ],
        "edges": [
            {"type": "MENTIONS", "from": "insight:i1", "to": "person:alice"},
            {"type": "MENTIONS", "from": "insight:i1", "to": "org:acme"},
        ],
    }
    out = migrate_gi_document_v3(data)
    assert out["schema_version"] == "3.0"
    types = {(e["from"], e["to"], e["type"]) for e in out["edges"]}
    assert ("insight:i1", "person:alice", "MENTIONS_PERSON") in types
    assert ("insight:i1", "org:acme", "MENTIONS_ORG") in types
    assert all(t[2] != "MENTIONS" for t in types)
    # Legacy "fact" → "claim" via synonym map.
    ins = next(n for n in out["nodes"] if n["id"] == "insight:i1")
    assert ins["properties"]["insight_type"] == "claim"


def test_migrate_gi_document_v3_out_of_vocab_insight_type_to_unknown() -> None:
    """Out-of-vocab insight types fall to "unknown" (schema valid)."""
    data = {
        "schema_version": "2.0",
        "episode_id": "e1",
        "model_version": "m",
        "prompt_version": "v1",
        "nodes": [
            {
                "id": "insight:i1",
                "type": "Insight",
                "properties": {
                    "text": "x",
                    "episode_id": "e1",
                    "grounded": False,
                    "insight_type": "weird",
                },
            },
        ],
        "edges": [],
    }
    out = migrate_gi_document_v3(data)
    ins = next(n for n in out["nodes"] if n["id"] == "insight:i1")
    assert ins["properties"]["insight_type"] == "unknown"


def test_migrate_gi_document_v3_idempotent() -> None:
    """Running the v3 migration twice yields the same document."""
    data = {
        "schema_version": "3.0",
        "episode_id": "e1",
        "model_version": "m",
        "prompt_version": "v1",
        "nodes": [
            {
                "id": "insight:i1",
                "type": "Insight",
                "properties": {
                    "text": "claim",
                    "episode_id": "e1",
                    "grounded": True,
                    "insight_type": "claim",
                },
            },
        ],
        "edges": [{"type": "MENTIONS_PERSON", "from": "insight:i1", "to": "person:p"}],
    }
    once = migrate_gi_document_v3(data)
    twice = migrate_gi_document_v3(once)
    assert once == twice


def test_compute_position_hints_for_document_uses_rss_duration() -> None:
    """Backfill uses Episode.duration_ms when present (step 1)."""
    data = {
        "schema_version": "2.0",
        "episode_id": "e1",
        "model_version": "m",
        "prompt_version": "v1",
        "nodes": [
            {
                "id": "episode:e1",
                "type": "Episode",
                "properties": {
                    "podcast_id": "p",
                    "title": "T",
                    "publish_date": "2024-01-01T00:00:00Z",
                    "duration_ms": 100000,
                },
            },
            {
                "id": "insight:i1",
                "type": "Insight",
                "properties": {"text": "x", "episode_id": "e1", "grounded": True},
            },
            {
                "id": "quote:q1",
                "type": "Quote",
                "properties": {
                    "text": "y",
                    "episode_id": "e1",
                    "char_start": 0,
                    "char_end": 1,
                    "timestamp_start_ms": 25000,
                    "timestamp_end_ms": 26000,
                    "transcript_ref": "t",
                },
            },
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "insight:i1", "to": "quote:q1"}],
    }
    out = compute_position_hints_for_document(data)
    ins = next(n for n in out["nodes"] if n["id"] == "insight:i1")
    assert ins["properties"]["position_hint"] == 0.25  # 25000 / 100000


def test_compute_position_hints_for_document_uses_segments_fallback() -> None:
    """When duration_ms is missing, segments-end (step 2) is used."""
    data = {
        "schema_version": "2.0",
        "episode_id": "e1",
        "model_version": "m",
        "prompt_version": "v1",
        "nodes": [
            {
                "id": "episode:e1",
                "type": "Episode",
                "properties": {
                    "podcast_id": "p",
                    "title": "T",
                    "publish_date": "2024-01-01T00:00:00Z",
                },
            },
            {
                "id": "insight:i1",
                "type": "Insight",
                "properties": {"text": "x", "episode_id": "e1", "grounded": True},
            },
            {
                "id": "quote:q1",
                "type": "Quote",
                "properties": {
                    "text": "y",
                    "episode_id": "e1",
                    "char_start": 0,
                    "char_end": 1,
                    "timestamp_start_ms": 50000,
                    "timestamp_end_ms": 60000,
                    "transcript_ref": "t",
                },
            },
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "insight:i1", "to": "quote:q1"}],
    }
    out = compute_position_hints_for_document(
        data,
        transcript_segments=[{"start": 0.0, "end": 200.0}],
    )
    ins = next(n for n in out["nodes"] if n["id"] == "insight:i1")
    # 50000 / (200 * 1000) = 0.25
    assert ins["properties"]["position_hint"] == 0.25


def test_compute_position_hints_for_document_skips_when_no_quote_starts() -> None:
    """Step 4: no Quote timestamp_start_ms → position_hint absent."""
    data = {
        "schema_version": "2.0",
        "episode_id": "e1",
        "model_version": "m",
        "prompt_version": "v1",
        "nodes": [
            {
                "id": "insight:i1",
                "type": "Insight",
                "properties": {"text": "x", "episode_id": "e1", "grounded": False},
            },
        ],
        "edges": [],
    }
    out = compute_position_hints_for_document(data)
    ins = next(n for n in out["nodes"] if n["id"] == "insight:i1")
    assert "position_hint" not in ins["properties"]


# ─── RFC-097 retroactive sweep: end-to-end migration → strict validation ───


def test_migrate_kg_v2_output_passes_strict_schema() -> None:
    """A v1.2 KG artifact migrated via migrate_kg_document_v2 passes strict validation."""
    from podcast_scraper.kg.schema import validate_artifact as validate_kg

    data = {
        "schema_version": "1.2",
        "episode_id": "e1",
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2024-01-01T00:00:00Z",
            "transcript_ref": "t.txt",
        },
        "nodes": [
            {
                "id": "episode:e1",
                "type": "Episode",
                "properties": {
                    "podcast_id": "podcast:p",
                    "title": "T",
                    "publish_date": "2024-01-01T00:00:00Z",
                },
            },
            {
                "id": "person:alice",
                "type": "Entity",
                "properties": {"name": "Alice", "kind": "person", "role": "host"},
            },
            {
                "id": "org:acme",
                "type": "Entity",
                "properties": {"name": "ACME", "kind": "org"},
            },
        ],
        "edges": [
            {"type": "MENTIONS", "from": "person:alice", "to": "episode:e1", "properties": {}},
        ],
    }
    out = migrate_kg_document_v2(data)
    # Must validate against the v2.0 schema in strict mode.
    validate_kg(out, strict=True)


def test_migrate_gi_v3_output_passes_strict_schema() -> None:
    """A v2.0 GI artifact migrated via migrate_gi_document_v3 passes strict validation."""
    from podcast_scraper.gi.schema import validate_artifact as validate_gi

    data = {
        "schema_version": "2.0",
        "episode_id": "e1",
        "model_version": "stub",
        "prompt_version": "v1",
        "nodes": [
            {
                "id": "episode:e1",
                "type": "Episode",
                "properties": {
                    "podcast_id": "podcast:p",
                    "title": "T",
                    "publish_date": "2024-01-01T00:00:00Z",
                },
            },
            {
                "id": "insight:i1",
                "type": "Insight",
                "properties": {
                    "text": "Some claim.",
                    "episode_id": "e1",
                    "grounded": False,
                    "insight_type": "fact",  # legacy vocab — migration must normalise
                },
            },
            # NB: target Person node MUST be present pre-migration for strict
            # validation, otherwise the rewritten MENTIONS_PERSON edge would
            # dangle (schema allows but consumers complain).
            {
                "id": "person:alice",
                "type": "Person",
                "properties": {"name": "Alice"},
            },
        ],
        "edges": [
            {"type": "MENTIONS", "from": "insight:i1", "to": "person:alice"},
        ],
    }
    out = migrate_gi_document_v3(data)
    validate_gi(out, strict=True)
    # Confirm legacy "fact" → "claim" landed.
    ins = next(n for n in out["nodes"] if n["id"] == "insight:i1")
    assert ins["properties"]["insight_type"] == "claim"
    # Confirm typed edge replaced generic MENTIONS.
    assert any(e["type"] == "MENTIONS_PERSON" for e in out["edges"])
    assert not any(e["type"] == "MENTIONS" for e in out["edges"])
