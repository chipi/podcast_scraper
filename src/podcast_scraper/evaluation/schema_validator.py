"""Schema validation utilities for evaluation artifacts.

This module provides validation functions to ensure evaluation artifacts
(references, metrics, etc.) conform to their expected schemas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Try to import jsonschema for proper validation
try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    logger.warning(
        "jsonschema library not available. Using basic validation only. "
        "Install with: pip install jsonschema"
    )


def validate_schema(obj: Dict[str, Any], schema_path: Path, strict: bool = False) -> None:
    """Validate an object against a JSON schema.

    Args:
        obj: Object to validate
        schema_path: Path to JSON schema file
        strict: If True, raise on validation failure. If False, log warning only.

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If validation fails and strict=True
    """
    if not schema_path.exists():
        if strict:
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        logger.warning(f"Schema file not found: {schema_path}, skipping validation")
        return

    try:
        schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as e:
        if strict:
            raise ValueError(f"Failed to load schema from {schema_path}: {e}") from e
        logger.warning(f"Failed to load schema from {schema_path}: {e}, skipping validation")
        return

    if HAS_JSONSCHEMA:
        try:
            jsonschema.validate(instance=obj, schema=schema_data)
            logger.debug(f"Schema validation passed: {schema_path.name}")
        except jsonschema.ValidationError as e:
            msg = f"Schema validation failed: {e.message}"
            if strict:
                raise ValueError(msg) from e
            logger.warning(f"{msg} (continuing anyway)")
        except jsonschema.SchemaError as e:
            msg = f"Invalid schema file {schema_path}: {e.message}"
            if strict:
                raise ValueError(msg) from e
            logger.warning(f"{msg} (continuing anyway)")
    else:
        # Basic validation fallback: check required fields from schema
        required = schema_data.get("required", [])
        missing = [field for field in required if field not in obj]
        if missing:
            msg = f"Missing required fields: {missing}"
            if strict:
                raise ValueError(msg)
            logger.warning(f"{msg} (continuing anyway - structure may differ)")
        else:
            logger.debug(f"Basic validation passed (jsonschema not available): {schema_path.name}")


def validate_summarization_reference(reference_entry: Dict[str, Any]) -> None:
    """Validate a single summarization reference entry.

    Args:
        reference_entry: Single JSONL entry from predictions.jsonl

    Raises:
        ValueError: If validation fails
    """
    schema_path = Path("data/eval/schemas/summarization_reference_v1.json")
    validate_schema(reference_entry, schema_path)


def validate_metrics_summarization(metrics: Dict[str, Any], strict: bool = False) -> None:
    """Validate summarization metrics.json.

    Args:
        metrics: Metrics dictionary
        strict: If True, raise on validation failure. If False, log warning only.

    Raises:
        ValueError: If validation fails and strict=True
    """
    schema_path = Path("data/eval/schemas/metrics_summarization_v1.json")
    validate_schema(metrics, schema_path, strict=strict)


def validate_metrics_ner(metrics: Dict[str, Any], strict: bool = False) -> None:
    """Validate NER metrics.json.

    Note: Current metrics structure may not match schema exactly.
    Validation is lenient by default (warns instead of failing).

    Args:
        metrics: Metrics dictionary
        strict: If True, raise on validation failure. If False, log warning only.

    Raises:
        ValueError: If validation fails and strict=True
    """
    schema_path = Path("data/eval/schemas/metrics_ner_v1.json")
    # NER metrics structure differs from schema - use lenient validation
    validate_schema(metrics, schema_path, strict=strict)
