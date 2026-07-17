"""KG artifact schema: load kg.schema.json and validate payloads."""

from pathlib import Path
from typing import Any, cast, Dict, Optional

_SCHEMA_CACHE: Optional[Dict[str, Any]] = None


def get_schema_path() -> Optional[Path]:
    """Return path to kg.schema.json when present (e.g. in repo docs/architecture/kg/)."""
    try:
        from podcast_scraper.cache import get_project_root

        root = get_project_root()
        path = root / "docs" / "architecture" / "kg" / "kg.schema.json"
        return path if path.exists() else None
    except Exception:
        return None


_SCHEMA_LOCK = __import__("threading").Lock()


def load_schema() -> Optional[Dict[str, Any]]:
    """Load kg.schema.json; returns None if file not found.

    Uses a module-level cache so the file is read at most once. The load-and-set
    is locked so two threads don't both read the file (review low/kg-schema).
    """
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    with _SCHEMA_LOCK:
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
    required = ("schema_version", "episode_id", "extraction", "nodes", "edges")
    for key in required:
        if key not in data:
            raise ValueError(f"KG artifact missing required key: {key!r}")
    if not isinstance(data.get("schema_version"), str):
        raise ValueError("KG artifact 'schema_version' must be a string")
    sv = data.get("schema_version")
    # RFC-097 chunk 9 (ADR-101, 2026-06-22): legacy 1.0/1.1/1.2 shape rejected.
    # Migration scripts read input as raw JSON (json.load), not via
    # validate_artifact, so this strict version gate does not block migration
    # of legacy corpora.
    if sv != "2.0":
        raise ValueError(
            "KG artifact 'schema_version' must be '2.0' (RFC-097 v2). Legacy "
            "1.0/1.1/1.2 shape is no longer accepted; migrate via "
            "scripts/migrate_kg_entity_to_person_org.py."
        )
    ext = data.get("extraction")
    if not isinstance(ext, dict):
        raise ValueError("KG artifact 'extraction' must be an object")
    for k in ("model_version", "extracted_at", "transcript_ref"):
        if k not in ext:
            raise ValueError(f"KG artifact extraction missing {k!r}")
    if not isinstance(data.get("nodes"), list):
        raise ValueError("KG artifact 'nodes' must be an array")
    if not isinstance(data.get("edges"), list):
        raise ValueError("KG artifact 'edges' must be an array")


def validate_artifact(data: Dict[str, Any], strict: bool = False) -> None:
    """Validate a KG artifact dict.

    Always runs minimal validation. If strict=True and kg.schema.json exists,
    runs full JSON Schema validation.

    Raises:
        ValueError: On validation failure.
    """
    _minimal_validate(data)
    if not strict:
        return
    schema = load_schema()
    if schema is None:
        return
    import jsonschema

    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"KG artifact schema validation failed: {str(e)}") from e
