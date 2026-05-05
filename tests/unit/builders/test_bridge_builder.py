"""Unit tests for ``bridge_builder``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from podcast_scraper.builders.bridge_builder import build_bridge
from podcast_scraper.providers.ml import embedding_loader


def test_person_gi_only_org_kg_only_topic_both() -> None:
    gi = {
        "nodes": [
            {"id": "person:alice", "type": "Person", "properties": {"name": "Alice"}},
            {"id": "topic:climate", "type": "Topic", "properties": {"label": "Climate"}},
        ]
    }
    kg = {
        "nodes": [
            {"id": "org:acme", "type": "Entity", "properties": {"name": "Acme", "kind": "org"}},
            {"id": "topic:climate", "type": "Topic", "properties": {"label": "Climate"}},
        ]
    }
    out = build_bridge("episode:test", gi, kg, fuzzy_reconcile=False)
    assert out["schema_version"] == "1.0"
    assert out["episode_id"] == "episode:test"
    assert "emitted_at" in out
    by_id = {i["id"]: i for i in out["identities"]}
    assert set(by_id) == {"person:alice", "org:acme", "topic:climate"}
    assert by_id["person:alice"]["sources"] == {"gi": True, "kg": False}
    assert by_id["org:acme"]["sources"] == {"gi": False, "kg": True}
    assert by_id["topic:climate"]["sources"] == {"gi": True, "kg": True}


def test_aliases_and_display_merged_across_layers() -> None:
    gi = {
        "nodes": [
            {
                "id": "person:bob",
                "type": "Person",
                "properties": {"name": "Bob", "aliases": ["Bobby"]},
            }
        ]
    }
    kg = {
        "nodes": [
            {
                "id": "g:person:bob",
                "type": "Entity",
                "properties": {"name": "Robert", "aliases": ["Bob Smith"], "kind": "person"},
            }
        ]
    }
    out = build_bridge("e1", gi, kg, fuzzy_reconcile=False)
    bob = next(i for i in out["identities"] if i["id"] == "person:bob")
    assert bob["sources"] == {"gi": True, "kg": True}
    assert set(bob["aliases"]) == {"Bobby", "Bob Smith"}
    assert bob["display_name"] == "Robert"


def test_strips_g_prefix_on_kg_entity_id() -> None:
    gi: dict[str, Any] = {"nodes": []}
    kg = {
        "nodes": [
            {
                "id": "g:person:zoe",
                "type": "Entity",
                "properties": {"name": "Zoe", "kind": "person"},
            }
        ]
    }
    out = build_bridge("e2", gi, kg, fuzzy_reconcile=False)
    ids = {i["id"] for i in out["identities"]}
    assert ids == {"person:zoe"}


def test_string_alias_on_person_node() -> None:
    gi = {
        "nodes": [
            {
                "id": "person:sam",
                "type": "Person",
                "properties": {"name": "Sam", "aliases": "Sammy"},
            }
        ]
    }
    out = build_bridge("e3", gi, {}, fuzzy_reconcile=False)
    sam = next(i for i in out["identities"] if i["id"] == "person:sam")
    assert sam["aliases"] == ["Sammy"]


def test_ignores_non_cil_nodes() -> None:
    gi = {"nodes": [{"id": "insight:1", "type": "Insight", "properties": {"text": "x"}}]}
    kg = {"nodes": [{"id": "kg:episode:u1", "type": "Episode", "properties": {}}]}
    out = build_bridge("e1", gi, kg, fuzzy_reconcile=False)
    assert out["identities"] == []


def test_fuzzy_reconcile_reuses_process_embedding_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """GH-742: bridge fuzzy pass must use get_embedding_model, not load_embedding_model."""
    loads: list[int] = []

    class FakeEmbedder:
        def encode(self, texts: list[str], normalize_embeddings: bool = True, **kwargs: Any):
            n = len(texts)
            row = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            row = row / float(np.linalg.norm(row))
            return np.tile(row, (n, 1))

    def fake_load(*args: Any, **kwargs: Any) -> FakeEmbedder:
        loads.append(1)
        return FakeEmbedder()

    monkeypatch.setattr(embedding_loader, "load_embedding_model", fake_load)
    monkeypatch.setattr(embedding_loader, "_embedding_models", {})

    gi = {
        "nodes": [
            {"id": "person:alice", "type": "Person", "properties": {"name": "Pat Lee"}},
        ]
    }
    kg = {
        "nodes": [
            {
                "id": "person:zed",
                "type": "Entity",
                "properties": {"name": "Pat Lee", "kind": "person"},
            }
        ],
    }
    build_bridge("ep-1", gi, kg, fuzzy_reconcile=True)
    build_bridge("ep-2", gi, kg, fuzzy_reconcile=True)

    assert len(loads) == 1
