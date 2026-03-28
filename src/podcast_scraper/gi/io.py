"""GIL artifact I/O: read and write gi.json files."""

import json
from pathlib import Path
from typing import Any, cast, Dict

from .schema import validate_artifact


def write_artifact(path: Path, payload: Dict[str, Any], validate: bool = True) -> None:
    """Write a GIL artifact to path (e.g. episode.gi.json).

    Args:
        path: Output file path.
        payload: Dict with schema_version, model_version, prompt_version, episode_id, nodes, edges.
        validate: If True, run minimal validation before writing.
    """
    if validate:
        validate_artifact(payload, strict=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_artifact(path: Path) -> Dict[str, Any]:
    """Read a GIL artifact from path.

    Args:
        path: Path to .gi.json file.

    Returns:
        Parsed artifact dict (not validated by default).
    """
    with open(path, encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))
