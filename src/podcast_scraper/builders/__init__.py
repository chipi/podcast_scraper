"""Artifact builders (e.g. RFC-072 bridge)."""

from .bridge_builder import build_bridge
from .rfc072_artifact_paths import (
    bridge_json_path_adjacent_to_metadata,
    bridge_path_next_to_gi_json,
    gi_and_kg_json_paths_next_to_bridge,
)

__all__ = [
    "build_bridge",
    "bridge_json_path_adjacent_to_metadata",
    "bridge_path_next_to_gi_json",
    "gi_and_kg_json_paths_next_to_bridge",
]
