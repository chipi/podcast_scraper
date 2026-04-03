"""Schema validation utilities for evaluation artifacts.

This module provides validation functions to ensure evaluation artifacts
(references, metrics, etc.) conform to their expected schemas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from podcast_scraper.utils.log_redaction import format_exception_for_log

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


def validate_schema(obj: Dict[str, Any], schema_path: Path, strict: bool = False) -> bool:
    """Validate an object against a JSON schema.

    Args:
        obj: Object to validate
        schema_path: Path to JSON schema file
        strict: If True, raise on validation failure. If False, log warning only.

    Returns:
        True if validation succeeded or was skipped (missing schema file, load error in
        non-strict mode). False if validation ran and the instance did not match.

    Raises:
        FileNotFoundError: If schema file doesn't exist and strict=True
        ValueError: If validation fails and strict=True
    """
    if not schema_path.exists():
        if strict:
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        logger.warning(f"Schema file not found: {schema_path}, skipping validation")
        return True

    try:
        schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as e:
        if strict:
            raise ValueError(f"Failed to load schema from {schema_path}: {e}") from e
        logger.warning(
            "Failed to load schema from %s: %s, skipping validation",
            schema_path,
            format_exception_for_log(e),
        )
        return True

    if HAS_JSONSCHEMA:
        try:
            jsonschema.validate(instance=obj, schema=schema_data)
            logger.debug(f"Schema validation passed: {schema_path.name}")
            return True
        except jsonschema.ValidationError as e:
            msg = f"Schema validation failed: {e.message}"
            if strict:
                raise ValueError(msg) from e
            logger.warning(f"{msg} (continuing anyway)")
            return False
        except jsonschema.SchemaError as e:
            msg = f"Invalid schema file {schema_path}: {e.message}"
            if strict:
                raise ValueError(msg) from e
            logger.warning(f"{msg} (continuing anyway)")
            return False
    else:
        # Basic validation fallback: check required fields from schema
        required = schema_data.get("required", [])
        missing = [field for field in required if field not in obj]
        if missing:
            msg = f"Missing required fields: {missing}"
            if strict:
                raise ValueError(msg)
            logger.warning(f"{msg} (continuing anyway - structure may differ)")
            return False
        logger.debug(f"Basic validation passed (jsonschema not available): {schema_path.name}")
        return True


def _normalize_summarization_reference_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy with top-level ``summary`` for schema validation when needed.

    Run/baseline JSONL uses ``output.summary_final`` (or legacy ``summary_long``); the v1
    JSON Schema still requires top-level ``summary``. We mirror that field so validation
    matches real artifacts without duplicating the summary in stored files.
    """
    out = dict(entry)
    summary = out.get("summary")
    if isinstance(summary, str) and summary.strip():
        return out
    payload = out.get("output")
    if isinstance(payload, dict):
        text = payload.get("summary_final") or payload.get("summary_long")
        if isinstance(text, str) and text.strip():
            out["summary"] = text
    return out


def validate_summarization_reference(reference_entry: Dict[str, Any], strict: bool = False) -> bool:
    """Validate a single summarization reference entry (JSONL line).

    Accepts legacy rows with top-level ``summary`` or run/baseline rows with
    ``output.summary_final`` / ``output.summary_long`` (RFC baseline format).

    Args:
        reference_entry: Single JSONL object from ``predictions.jsonl``
        strict: Passed through to :func:`validate_schema`.

    Returns:
        True if validation passed or was skipped; False if the instance did not match.

    Raises:
        ValueError: If ``strict=True`` and validation fails
    """
    schema_path = Path("data/eval/schemas/summarization_reference_v1.json")
    normalized = _normalize_summarization_reference_entry(reference_entry)
    return validate_schema(normalized, schema_path, strict=strict)


def validate_metrics_summarization(metrics: Dict[str, Any], strict: bool = False) -> bool:
    """Validate summarization metrics.json (scorer layout: ``metrics_summarization_v2``).

    Legacy top-level ``metrics_summarization_v1`` (flat gates/length) is no longer emitted;
    see ``data/eval/schemas/metrics_summarization_v1.json`` for the historical shape.

    Args:
        metrics: Metrics dictionary
        strict: If True, raise on validation failure. If False, log warning only.

    Returns:
        True if validation passed or was skipped; False if instance did not match schema.

    Raises:
        ValueError: If validation fails and strict=True
    """
    schema_path = Path("data/eval/schemas/metrics_summarization_v2.json")
    return validate_schema(metrics, schema_path, strict=strict)


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


def validate_metrics_gil_reference(metrics: Dict[str, Any], strict: bool = False) -> None:
    """Validate per-reference GIL metrics dict (under ``vs_reference``)."""
    schema_path = Path("data/eval/schemas/metrics_gil_v1.json")
    validate_schema(metrics, schema_path, strict=strict)


def validate_metrics_kg_reference(metrics: Dict[str, Any], strict: bool = False) -> None:
    """Validate per-reference KG metrics dict (under ``vs_reference``)."""
    schema_path = Path("data/eval/schemas/metrics_kg_v1.json")
    validate_schema(metrics, schema_path, strict=strict)


def validate_metrics_gil_eval_run(metrics: Dict[str, Any], strict: bool = False) -> None:
    """Validate top-level ``metrics.json`` for grounded_insights experiment runs."""
    schema_path = Path("data/eval/schemas/metrics_gil_eval_run_v1.json")
    validate_schema(metrics, schema_path, strict=strict)


def validate_metrics_kg_eval_run(metrics: Dict[str, Any], strict: bool = False) -> None:
    """Validate top-level ``metrics.json`` for knowledge_graph experiment runs."""
    schema_path = Path("data/eval/schemas/metrics_kg_eval_run_v1.json")
    validate_schema(metrics, schema_path, strict=strict)
