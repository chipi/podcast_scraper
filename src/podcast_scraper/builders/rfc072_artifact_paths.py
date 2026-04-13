"""RFC-072 artifact filename rules: ``bridge.json`` next to metadata / GI / KG (single source)."""

from __future__ import annotations

from pathlib import Path


def bridge_json_path_adjacent_to_metadata(metadata_path: str) -> str:
    """Sibling ``*.bridge.json`` for a metadata document path (abs or rel, any separator).

    Handles ``.metadata.json``, ``.metadata.yaml``, ``.metadata.yml``; otherwise
    ``splitext`` + ``.bridge.json``.
    """
    mp = metadata_path
    if mp.endswith(".metadata.json"):
        return mp[: -len(".metadata.json")] + ".bridge.json"
    if mp.endswith(".metadata.yaml"):
        return mp[: -len(".metadata.yaml")] + ".bridge.json"
    if mp.endswith(".metadata.yml"):
        return mp[: -len(".metadata.yml")] + ".bridge.json"
    stem, _dot, _ext = mp.rpartition(".")
    if not _dot:
        return f"{mp}.bridge.json"
    return f"{stem}.bridge.json"


def bridge_path_next_to_gi_json(gi_path: Path) -> Path:
    """Sibling ``*.bridge.json`` for ``*.gi.json`` (same stem rule as catalog / CIL)."""
    name = gi_path.name
    if name.endswith(".gi.json"):
        return gi_path.with_name(name[: -len(".gi.json")] + ".bridge.json")
    stem = gi_path.stem
    if stem.endswith(".gi"):
        return gi_path.with_name(stem[: -len(".gi")] + ".bridge.json")
    return gi_path.with_name(f"{stem}.bridge.json")


def gi_and_kg_json_paths_next_to_bridge(bridge_path: Path) -> tuple[Path, Path]:
    """Sibling ``*.gi.json`` and ``*.kg.json`` for a ``*.bridge.json`` path."""
    name = bridge_path.name
    if not name.endswith(".bridge.json"):
        raise ValueError(f"not a bridge path: {bridge_path}")
    stem = name[: -len(".bridge.json")]
    parent = bridge_path.parent
    return parent / f"{stem}.gi.json", parent / f"{stem}.kg.json"
