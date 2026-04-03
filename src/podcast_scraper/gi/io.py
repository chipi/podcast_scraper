"""GIL artifact I/O: read and write gi.json files."""

import json
from pathlib import Path
from typing import Any, cast, Dict, List

from .schema import validate_artifact


def collect_gi_paths_from_inputs(paths: List[Path]) -> List[Path]:
    """Expand files and directories to a sorted list of .gi.json paths.

    Args:
        paths: Files must end with ``.gi.json``; directories are scanned recursively.

    Raises:
        FileNotFoundError: If a path does not exist.
        ValueError: If a file path is not a ``.gi.json`` file.
    """
    result: List[Path] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")
        if p.is_file():
            if not p.name.endswith(".gi.json"):
                raise ValueError(f"Not a .gi.json file: {p}")
            result.append(p)
            continue
        for child in p.rglob("*.gi.json"):
            result.append(child)
    return sorted(set(result))


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
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def read_artifact(
    path: Path,
    *,
    validate: bool = True,
    strict: bool = False,
) -> Dict[str, Any]:
    """Read a GIL artifact from path.

    Args:
        path: Path to .gi.json file.
        validate: If True, run ``validate_artifact`` (non-strict by default).
        strict: Passed through when validate is True.

    Returns:
        Parsed artifact dict.
    """
    with open(path, encoding="utf-8") as f:
        data = cast(Dict[str, Any], json.load(f))
    if validate:
        validate_artifact(data, strict=strict)
    return data
