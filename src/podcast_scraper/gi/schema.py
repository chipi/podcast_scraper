"""GIL artifact schema: load gi.schema.json and validate payloads.

Validation is minimal (required keys, types) when the JSON Schema file
is not available; full validation when docs/architecture/gi/gi.schema.json exists.
"""

from pathlib import Path
from typing import Any, cast, Dict, Optional

_SCHEMA_CACHE: Optional[Dict[str, Any]] = None


def get_schema_path() -> Optional[Path]:
    """Return path to gi.schema.json when present (e.g. in repo docs/architecture/gi/)."""
    try:
        from podcast_scraper.cache import get_project_root

        root = get_project_root()
        path = root / "docs" / "architecture" / "gi" / "gi.schema.json"
        return path if path.exists() else None
    except Exception:
        return None


def load_schema() -> Optional[Dict[str, Any]]:
    """Load gi.schema.json; returns None if file not found.

    Uses a module-level cache so the file is read at most once.
    """
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    path = get_schema_path()
    if not path:
        return None
    import json

    with open(path, encoding="utf-8") as f:
        schema = cast(Dict[str, Any], json.load(f))
    _SCHEMA_CACHE = schema
    return schema


def _minimal_validate(data: Dict[str, Any]) -> None:
    """Check required top-level keys and types; raise ValueError on first error."""
    required = ("schema_version", "model_version", "prompt_version", "episode_id", "nodes", "edges")
    for key in required:
        if key not in data:
            raise ValueError(f"GIL artifact missing required key: {key!r}")
    if not isinstance(data.get("schema_version"), str):
        raise ValueError("GIL artifact 'schema_version' must be a string")
    if data.get("schema_version") != "1.0":
        raise ValueError("GIL artifact 'schema_version' must be '1.0'")
    if not isinstance(data.get("nodes"), list):
        raise ValueError("GIL artifact 'nodes' must be an array")
    if not isinstance(data.get("edges"), list):
        raise ValueError("GIL artifact 'edges' must be an array")


def validate_artifact(data: Dict[str, Any], strict: bool = False) -> None:
    """Validate a GIL artifact dict.

    Always runs minimal validation (required keys, schema_version, nodes/edges types).
    If strict=True and gi.schema.json is available, runs full JSON Schema validation.

    Raises:
        ValueError: On validation failure.
    """
    _minimal_validate(data)
    if not strict:
        return
    schema = load_schema()
    if schema is None:
        return
    try:
        import jsonschema

        jsonschema.validate(instance=data, schema=schema)
    except ImportError:
        pass
    except jsonschema.ValidationError as e:
        raise ValueError(f"GIL artifact schema validation failed: {str(e)}") from e
