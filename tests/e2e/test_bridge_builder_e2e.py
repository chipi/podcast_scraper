"""Pytest E2E: RFC-072 ``build_bridge`` on disk fixtures (``builders/`` package)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.builders.bridge_builder import build_bridge

pytestmark = [pytest.mark.e2e, pytest.mark.critical_path]


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def test_build_bridge_from_ci_gi_and_minimal_kg(project_root: Path) -> None:
    gi_path = (
        project_root / "tests" / "fixtures" / "gil_kg_ci_enforce" / "metadata" / "ci_sample.gi.json"
    )
    kg_path = project_root / "tests" / "fixtures" / "kg" / "minimal.kg.json"
    assert gi_path.is_file()
    assert kg_path.is_file()

    gi_doc = json.loads(gi_path.read_text(encoding="utf-8"))
    kg_doc = json.loads(kg_path.read_text(encoding="utf-8"))

    bridge = build_bridge("ci-fixture", gi_doc, kg_doc)
    assert bridge.get("schema_version") == "1.0"
    assert bridge.get("episode_id") == "ci-fixture"
    assert isinstance(bridge.get("identities"), list)


def test_build_bridge_empty_artifacts() -> None:
    bridge = build_bridge("noop", None, None)
    assert bridge["schema_version"] == "1.0"
    assert bridge["episode_id"] == "noop"
    assert bridge["identities"] == []
    assert isinstance(bridge.get("emitted_at"), str)


def test_build_bridge_merges_prefixed_nodes_across_gi_and_kg() -> None:
    gi = {
        "nodes": [
            {
                "id": "g:person:pat",
                "properties": {"name": "Pat", "aliases": ["P."]},
            }
        ]
    }
    kg = {
        "nodes": [
            {
                "id": "k:person:pat",
                "properties": {"label": "Patrick", "aliases": "Patricia"},
            }
        ]
    }
    bridge = build_bridge("ep-merge", gi, kg)
    assert len(bridge["identities"]) == 1
    row = bridge["identities"][0]
    assert row["id"] == "person:pat"
    assert row["sources"] == {"gi": True, "kg": True}
    assert "P." in row["aliases"]
    assert "Patricia" in row["aliases"]
    assert row["display_name"] == "Patrick"


def test_build_bridge_strips_kg_colon_topic_prefix() -> None:
    kg = {
        "nodes": [
            {
                "id": "kg:topic:climate",
                "properties": {"label": "Climate"},
            }
        ]
    }
    bridge = build_bridge("ep-topic", {}, kg)
    assert len(bridge["identities"]) == 1
    assert bridge["identities"][0]["id"] == "topic:climate"
    assert bridge["identities"][0]["type"] == "topic"
