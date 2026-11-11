"""KG artifact schema: load kg.schema.json (KG 2.x) and kg.v3.schema.json (KG 3.0) and validate
payloads."""

from pathlib import Path
from typing import Any, cast, Dict, Optional

_SCHEMA_CACHE: Optional[Dict[str, Any]] = None
_V3_SCHEMA_CACHE: Optional[Dict[str, Any]] = None

#: The KG 3.0 (ontology v3) schema file, beside the 2.x one.
_V3_SCHEMA_FILE = "kg.v3.schema.json"


def get_schema_path(filename: str = "kg.schema.json") -> Optional[Path]:
    """Return path to a KG schema file when present (e.g. in repo docs/architecture/kg/)."""
    try:
        from podcast_scraper.cache import get_project_root

        root = get_project_root()
        path = root / "docs" / "architecture" / "kg" / filename
        return path if path.exists() else None
    except Exception:
        return None


_SCHEMA_LOCK = __import__("threading").Lock()


def _load(filename: str, cached: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if cached is not None:
        return cached
    path = get_schema_path(filename)
    if not path:
        return None
    import json

    with open(path, encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))


def load_schema() -> Optional[Dict[str, Any]]:
    """Load kg.schema.json (KG 2.x); returns None if file not found.

    Uses a module-level cache so the file is read at most once. The load-and-set
    is locked so two threads don't both read the file (review low/kg-schema).
    """
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    with _SCHEMA_LOCK:
        _SCHEMA_CACHE = _load("kg.schema.json", _SCHEMA_CACHE)
    return _SCHEMA_CACHE


def load_v3_schema() -> Optional[Dict[str, Any]]:
    """Load kg.v3.schema.json (KG 3.0, ontology v3); returns None if file not found."""
    global _V3_SCHEMA_CACHE
    if _V3_SCHEMA_CACHE is not None:
        return _V3_SCHEMA_CACHE
    with _SCHEMA_LOCK:
        _V3_SCHEMA_CACHE = _load(_V3_SCHEMA_FILE, _V3_SCHEMA_CACHE)
    return _V3_SCHEMA_CACHE


#: KG artifact versions this reader accepts.
#:
#: 2.1 (#2057) adds the ``Object`` node type — a named thing that is neither a person nor a body
#: of people, and the catch-all for an entity whose kind the extractor omitted. 2.0 stays accepted
#: because it is a strict subset: a 2.0 artifact simply contains no Object nodes, and every other
#: shape is identical. Migration m0008 stamps 2.0 -> 2.1.
#:
#: 3.0 is ontology v3 (ADR-178, docs/architecture/corpus/ontology-v3.md): written by new content
#: types from the core release, by podcasts after R3b. Readers accept both until E06.
_V3 = "3.0"
_ACCEPTED_SCHEMA_VERSIONS = frozenset({"2.0", "2.1", _V3})


def _minimal_validate(data: Dict[str, Any]) -> None:
    """Check required top-level keys and types; raise ValueError on first error."""
    sv = data.get("schema_version")
    if "schema_version" not in data:
        raise ValueError("KG artifact missing required key: 'schema_version'")
    if not isinstance(sv, str):
        raise ValueError("KG artifact 'schema_version' must be a string")
    # RFC-097 chunk 9 (ADR-101, 2026-06-22): legacy 1.0/1.1/1.2 shape rejected.
    # Migration scripts read input as raw JSON (json.load), not via
    # validate_artifact, so this strict version gate does not block migration
    # of legacy corpora.
    if sv not in _ACCEPTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"KG artifact 'schema_version' must be one of "
            f"{sorted(_ACCEPTED_SCHEMA_VERSIONS)}. Legacy 1.0/1.1/1.2 shape is no longer "
            "accepted; migrate via `cli upgrade run` (m0006 → v2.0 typed Person/Organization, "
            "m0008 → v2.1 Object node)."
        )
    if sv == _V3:
        required: tuple[str, ...] = ("item_id", "content_type", "extraction", "nodes", "edges")
        extraction_keys: tuple[str, ...] = (
            "model_version",
            "extracted_at",
            "document_ref",
            "variant",
        )
    else:
        required = ("episode_id", "extraction", "nodes", "edges")
        extraction_keys = ("model_version", "extracted_at", "transcript_ref")
    for key in required:
        if key not in data:
            raise ValueError(f"KG artifact missing required key: {key!r}")
    ext = data.get("extraction")
    if not isinstance(ext, dict):
        raise ValueError("KG artifact 'extraction' must be an object")
    for k in extraction_keys:
        if k not in ext:
            raise ValueError(f"KG artifact extraction missing {k!r}")
    if not isinstance(data.get("nodes"), list):
        raise ValueError("KG artifact 'nodes' must be an array")
    if not isinstance(data.get("edges"), list):
        raise ValueError("KG artifact 'edges' must be an array")


def validate_artifact(data: Dict[str, Any], strict: bool = False) -> None:
    """Validate a KG artifact dict.

    Always runs minimal validation. If strict=True and the schema file for the artifact's version
    exists (kg.schema.json for 2.x, kg.v3.schema.json for 3.0), runs full JSON Schema validation.

    Raises:
        ValueError: On validation failure.
    """
    _minimal_validate(data)
    if not strict:
        return
    schema = load_v3_schema() if data.get("schema_version") == _V3 else load_schema()
    if schema is None:
        return
    import jsonschema

    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"KG artifact schema validation failed: {str(e)}") from e
