"""Integration checks for RFC-072 bridge artifact assembly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.builders.bridge_builder import build_bridge

pytestmark = [pytest.mark.integration]


def test_build_bridge_identities_superset_of_gi_kg_cil_ids(tmp_path: Path) -> None:
    gi_path = tmp_path / "x.gi.json"
    kg_path = tmp_path / "x.kg.json"
    bridge_path = tmp_path / "x.bridge.json"

    gi = {
        "schema_version": "2.0",
        "episode_id": "ep:int",
        "nodes": [
            {"id": "person:p1", "type": "Person", "properties": {"name": "One"}},
            {"id": "topic:t1", "type": "Topic", "properties": {"label": "T"}},
        ],
        "edges": [],
    }
    kg = {
        "schema_version": "1.2",
        "episode_id": "ep:int",
        "extraction": {"model_version": "stub"},
        "nodes": [
            {"id": "org:o1", "type": "Entity", "properties": {"name": "Org", "kind": "org"}},
            {"id": "topic:t1", "type": "Topic", "properties": {"label": "T"}},
        ],
        "edges": [],
    }
    gi_path.write_text(json.dumps(gi), encoding="utf-8")
    kg_path.write_text(json.dumps(kg), encoding="utf-8")

    loaded_gi = json.loads(gi_path.read_text(encoding="utf-8"))
    loaded_kg = json.loads(kg_path.read_text(encoding="utf-8"))
    bridge_doc = build_bridge("ep:int", loaded_gi, loaded_kg, fuzzy_reconcile=False)
    bridge_path.write_text(json.dumps(bridge_doc), encoding="utf-8")

    on_disk = json.loads(bridge_path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "1.0"
    bridge_ids = {i["id"] for i in on_disk["identities"]}

    def cil_ids(artifact: dict) -> set[str]:
        out: set[str] = set()
        for n in artifact.get("nodes") or []:
            if not isinstance(n, dict):
                continue
            raw = n.get("id")
            if not isinstance(raw, str):
                continue
            s = raw.strip()
            if s.startswith("person:") or s.startswith("org:") or s.startswith("topic:"):
                out.add(s)
        return out

    gi_ids = cil_ids(loaded_gi)
    kg_ids = cil_ids(loaded_kg)
    assert gi_ids | kg_ids <= bridge_ids
