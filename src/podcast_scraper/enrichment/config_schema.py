"""Operator-config schema validator for the ``enrichment:`` YAML block.

The JSON Schema lives at ``config/schema/enrichment.schema.json``
(shipped in this sub-commit). This module loads it on demand and
provides ``validate_enrichment_block(block)`` for the CLI startup
path. Deferred jsonschema import so envs that never enrich (e.g.
test_unit-only contributors) don't need the extra dependency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Resolve the schema robustly across dev (editable install with source
# tree at ``<repo>/config/schema/...``) and prod (wheel install with no
# repo-side path). The package-data copy under
# ``enrichment/_schema/enrichment.schema.json`` is the canonical source;
# the legacy ``<repo>/config/schema/...`` path is kept as a fallback so
# editable installs that haven't rebuilt the package still resolve.
_PACKAGE_SCHEMA_PATH = Path(__file__).resolve().parent / "_schema" / "enrichment.schema.json"
_LEGACY_REPO_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "config"
    / "schema"
    / "enrichment.schema.json"
)


def _resolve_schema_path() -> Path:
    if _PACKAGE_SCHEMA_PATH.is_file():
        return _PACKAGE_SCHEMA_PATH
    return _LEGACY_REPO_SCHEMA_PATH


_SCHEMA_PATH = _resolve_schema_path()


class ConfigSchemaError(ValueError):
    """Raised when the enrichment YAML block fails JSON Schema validation."""


def load_schema() -> dict[str, Any]:
    """Load and return the JSON Schema dict.

    Raises ``ConfigSchemaError`` if the file is missing or malformed
    — those are deployment bugs, not operator bugs.
    """
    if not _SCHEMA_PATH.is_file():
        raise ConfigSchemaError(f"schema not found at {_SCHEMA_PATH}")
    try:
        data = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigSchemaError(f"cannot read schema: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigSchemaError("schema root must be a JSON object")
    return data


def validate_enrichment_block(block: dict[str, Any]) -> None:
    """Validate the ``enrichment:`` block against the JSON Schema.

    Raises ``ConfigSchemaError`` on validation failure. When
    ``jsonschema`` isn't installed, falls back to a minimal manual
    check (logs WARNING) so envs without the optional dep still get
    basic structural validation.
    """
    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "jsonschema not installed; using minimal manual validation for "
            "enrichment block. Install for full schema enforcement."
        )
        _minimal_validate(block)
        return

    schema = load_schema()
    try:
        jsonschema.validate(instance=block, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ConfigSchemaError(str(exc)) from exc


def _minimal_validate(block: dict[str, Any]) -> None:
    """Manual fallback validator — checks the structural skeleton.

    Catches the obvious operator typos when jsonschema isn't
    available. Full constraint validation requires jsonschema.
    """
    if not isinstance(block, dict):
        raise ConfigSchemaError("enrichment block must be a mapping")
    if "enabled" in block and not isinstance(block["enabled"], bool):
        raise ConfigSchemaError("enrichment.enabled must be a boolean")
    if "max_total_cost_usd_per_run" in block and not isinstance(
        block["max_total_cost_usd_per_run"], (int, float)
    ):
        raise ConfigSchemaError("enrichment.max_total_cost_usd_per_run must be a number")
    if "fail_on_run_cost_cap" in block and not isinstance(block["fail_on_run_cost_cap"], bool):
        raise ConfigSchemaError("enrichment.fail_on_run_cost_cap must be a boolean")
    enrichers = block.get("enrichers")
    if enrichers is not None:
        if not isinstance(enrichers, dict):
            raise ConfigSchemaError("enrichment.enrichers must be a mapping")
        for eid, cfg in enrichers.items():
            if not isinstance(cfg, dict):
                raise ConfigSchemaError(f"enrichment.enrichers.{eid} must be a mapping")
            if "enabled" in cfg and not isinstance(cfg["enabled"], bool):
                raise ConfigSchemaError(f"enrichment.enrichers.{eid}.enabled must be a boolean")
            if "opt_in" in cfg and not isinstance(cfg["opt_in"], bool):
                raise ConfigSchemaError(f"enrichment.enrichers.{eid}.opt_in must be a boolean")
            if "max_cost_usd_per_run" in cfg and not isinstance(
                cfg["max_cost_usd_per_run"], (int, float)
            ):
                raise ConfigSchemaError(
                    f"enrichment.enrichers.{eid}.max_cost_usd_per_run must be a number"
                )
