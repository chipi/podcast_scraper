"""Resolve KG artifact paths by episode id (scan metadata/*.kg.json)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast, Dict, Optional

from .io import read_artifact


def find_kg_artifact_by_episode_id(output_dir: Path, episode_id: str) -> Optional[Path]:
    """Scan output_dir/metadata/*.kg.json for artifact with given episode_id."""
    metadata_dir = output_dir / "metadata"
    if not metadata_dir.is_dir():
        return None
    for path in metadata_dir.glob("*.kg.json"):
        try:
            artifact = read_artifact(path, validate=False)
            if artifact.get("episode_id") == episode_id:
                return path
        except Exception:
            continue
    return None


def episode_node(artifact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the Episode node from a KG artifact, if any."""
    for n in artifact.get("nodes", []):
        if n.get("type") == "Episode":
            return cast(Dict[str, Any], n)
    return None
