"""Unit tests for RFC-072 GI/KG JSON migrations."""

import pytest

from podcast_scraper.migrations.rfc072 import migrate_gil_document, migrate_kg_document

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
