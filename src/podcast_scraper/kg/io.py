"""KG artifact I/O: read and write per-episode kg.json files."""

import json
from pathlib import Path
from typing import Any, cast, Dict

from .schema import validate_artifact


def write_artifact(path: Path, payload: Dict[str, Any], validate: bool = True) -> None:
    """Write a KG artifact to path (e.g. episode.kg.json).

    Args:
        path: Output file path.
        payload: Dict with schema_version, episode_id, extraction, nodes, edges.
        validate: If True, run validation before writing.
    """
    if validate:
        validate_artifact(payload, strict=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def read_artifact(path: Path) -> Dict[str, Any]:
    """Read a KG artifact from path.

    Args:
        path: Path to .kg.json file.

    Returns:
        Parsed artifact dict (not validated by default).
    """
    with open(path, encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))
