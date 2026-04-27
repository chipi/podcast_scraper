"""Integration: ML cache inspection helpers and bridge artifact path rules.

Touches ``cache.manager`` read-only aggregation and ``builders.bridge_artifact_paths``
filename logic used by the GI/KG/bridge layout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import podcast_scraper.builders.bridge_artifact_paths as bridge_artifact_paths
from podcast_scraper.cache import manager as cache_manager

pytestmark = pytest.mark.unit


def test_format_size_human_readable() -> None:
    assert cache_manager.format_size(0) == "0.00 B"
    assert "KB" in cache_manager.format_size(2048)
    assert "PB" in cache_manager.format_size(int(1024**5 * 2))


def test_calculate_directory_size(tmp_path: Path) -> None:
    assert cache_manager.calculate_directory_size(tmp_path / "missing") == 0
    nested = tmp_path / "tree"
    nested.mkdir()
    (nested / "f.bin").write_bytes(b"12345")
    assert cache_manager.calculate_directory_size(nested) == 5


def test_get_all_cache_info_structure() -> None:
    info = cache_manager.get_all_cache_info()
    assert set(info.keys()) >= {"whisper", "transformers", "spacy", "total_size"}
    for key in ("whisper", "transformers", "spacy"):
        block = info[key]
        assert {"dir", "size", "models", "count"} <= set(block.keys())


def test_bridge_json_path_adjacent_to_metadata_variants() -> None:
    assert (
        bridge_artifact_paths.bridge_json_path_adjacent_to_metadata("show/ep.metadata.json")
        == "show/ep.bridge.json"
    )
    assert bridge_artifact_paths.bridge_json_path_adjacent_to_metadata("x.metadata.yaml") == (
        "x.bridge.json"
    )
    assert bridge_artifact_paths.bridge_json_path_adjacent_to_metadata("x.metadata.yml") == (
        "x.bridge.json"
    )
    assert bridge_artifact_paths.bridge_json_path_adjacent_to_metadata("plain") == (
        "plain.bridge.json"
    )
    assert bridge_artifact_paths.bridge_json_path_adjacent_to_metadata("a.b.c") == "a.b.bridge.json"


def test_bridge_path_next_to_gi_json(tmp_path: Path) -> None:
    gi = tmp_path / "stem.gi.json"
    br = bridge_artifact_paths.bridge_path_next_to_gi_json(gi)
    assert br.name == "stem.bridge.json"
    odd = tmp_path / "odd.txt"
    br_odd = bridge_artifact_paths.bridge_path_next_to_gi_json(odd)
    assert br_odd.name == "odd.bridge.json"


def test_gi_and_kg_json_paths_next_to_bridge(tmp_path: Path) -> None:
    bridge = tmp_path / "episode.bridge.json"
    gi_p, kg_p = bridge_artifact_paths.gi_and_kg_json_paths_next_to_bridge(bridge)
    assert gi_p.name == "episode.gi.json"
    assert kg_p.name == "episode.kg.json"

    with pytest.raises(ValueError, match="not a bridge"):
        bridge_artifact_paths.gi_and_kg_json_paths_next_to_bridge(tmp_path / "x.json")
