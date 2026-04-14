"""Unit tests for RFC-072 artifact path helpers (single source for bridge/GI/KG stems)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.builders.rfc072_artifact_paths import (
    bridge_json_path_adjacent_to_metadata,
    bridge_path_next_to_gi_json,
    gi_and_kg_json_paths_next_to_bridge,
)


def test_bridge_json_path_adjacent_to_metadata_json_yaml_yml() -> None:
    assert bridge_json_path_adjacent_to_metadata("m/metadata/ep.metadata.json") == (
        "m/metadata/ep.bridge.json"
    )
    assert bridge_json_path_adjacent_to_metadata("m/metadata/ep.metadata.yaml") == (
        "m/metadata/ep.bridge.json"
    )
    assert bridge_json_path_adjacent_to_metadata("m/metadata/ep.metadata.yml") == (
        "m/metadata/ep.bridge.json"
    )


def test_bridge_json_path_adjacent_to_metadata_fallback() -> None:
    assert bridge_json_path_adjacent_to_metadata("other.json") == "other.bridge.json"
    assert bridge_json_path_adjacent_to_metadata("nodot") == "nodot.bridge.json"


def test_bridge_path_next_to_gi_json() -> None:
    p = Path("/x/metadata/foo.gi.json")
    assert bridge_path_next_to_gi_json(p) == Path("/x/metadata/foo.bridge.json")


def test_bridge_path_next_to_gi_json_stem_endswith_dot_gi() -> None:
    """Path whose stem ends with ``.gi`` but name is not ``*.gi.json`` (Path stem rule)."""
    p = Path("/x/metadata/episode.gi")
    assert bridge_path_next_to_gi_json(p) == Path("/x/metadata/episode.bridge.json")


def test_bridge_path_next_to_gi_json_fallback_stem_only() -> None:
    p = Path("/x/metadata/odd.bin")
    assert bridge_path_next_to_gi_json(p) == Path("/x/metadata/odd.bridge.json")


def test_gi_and_kg_json_paths_next_to_bridge() -> None:
    b = Path("/c/ep.bridge.json")
    gi, kg = gi_and_kg_json_paths_next_to_bridge(b)
    assert gi == Path("/c/ep.gi.json")
    assert kg == Path("/c/ep.kg.json")


def test_gi_and_kg_json_paths_next_to_bridge_rejects_non_bridge() -> None:
    with pytest.raises(ValueError, match="not a bridge path"):
        gi_and_kg_json_paths_next_to_bridge(Path("/c/ep.gi.json"))
