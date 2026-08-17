"""KG artifact I/O: read and write per-episode kg.json files."""

import json
from pathlib import Path
from typing import Any, cast, Dict

from podcast_scraper.utils.atomic_io import write_json_atomic

from .schema import validate_artifact


def write_artifact(path: Path, payload: Dict[str, Any], validate: bool = True) -> None:
    """Write a KG artifact to path (e.g. episode.kg.json).

    ATOMIC: a failed or interrupted write leaves the PREVIOUS artifact intact. Same reasoning as
    ``gi.io.write_artifact`` — see ``utils.atomic_io``.

    Args:
        path: Output file path.
        payload: Dict with schema_version, episode_id, extraction, nodes, edges.
        validate: If True, run validation before writing.
    """
    if validate:
        validate_artifact(payload, strict=False)
    write_json_atomic(
        Path(path),
        payload,
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
    """Read a KG artifact from path.

    Args:
        path: Path to .kg.json file.
        validate: If True, run minimal (and optional strict JSON Schema) validation.
        strict: Passed to ``validate_artifact`` when validate is True.

    Returns:
        Parsed artifact dict.
    """
    with open(path, encoding="utf-8") as f:
        data = cast(Dict[str, Any], json.load(f))
    if validate:
        validate_artifact(data, strict=strict)
    return data
