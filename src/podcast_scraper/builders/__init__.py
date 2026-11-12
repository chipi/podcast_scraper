"""Artifact path helpers (the cross-layer ``bridge.json`` and ``context.json`` file names).

The bridge builder itself left with GI (E04 move); its file names stay here, read by the catalog
and the metadata writer.
"""

from .bridge_artifact_paths import (
    bridge_json_path_adjacent_to_metadata,
    bridge_path_next_to_gi_json,
    gi_and_kg_json_paths_next_to_bridge,
)

__all__ = [
    "bridge_json_path_adjacent_to_metadata",
    "bridge_path_next_to_gi_json",
    "gi_and_kg_json_paths_next_to_bridge",
]
