"""Output envelope validation for enricher artifacts.

Every enricher output file carries this envelope shape — ``derived: true``,
``computed_at``, ``enricher_id``, ``enricher_version``, ``schema_version``,
``status``, ``error?``, ``data``, plus the resilience-aware fields
(``retry_count``, ``circuit_state``, ``duration_ms``, ``records_written``).

The framework writes the envelope; the enricher returns an ``EnricherResult``
which carries the inner ``data`` dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from podcast_scraper.enrichment.protocol import (
    ALL_STATUSES,
    EnricherResult,
    STATUS_OK,
)


class EnvelopeShapeError(ValueError):
    """Raised when an on-disk enrichment output file fails envelope validation.

    Classified as ``non-retryable`` in the resilience failure taxonomy —
    indicates a bug in the enricher, not a transient backend issue.
    """


@dataclass(frozen=True)
class EnvelopePayload:
    """The canonical on-disk shape for an enricher output file."""

    derived: bool
    computed_at: str  # ISO 8601 UTC, seconds precision
    enricher_id: str
    enricher_version: str
    schema_version: str
    status: str
    data: dict[str, Any] | None
    error: str | None
    error_class: str | None
    retry_count: int
    circuit_state: str | None
    duration_ms: int
    records_written: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for JSON serialization."""
        return {
            "derived": self.derived,
            "computed_at": self.computed_at,
            "enricher_id": self.enricher_id,
            "enricher_version": self.enricher_version,
            "schema_version": self.schema_version,
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "error_class": self.error_class,
            "retry_count": self.retry_count,
            "circuit_state": self.circuit_state,
            "duration_ms": self.duration_ms,
            "records_written": self.records_written,
        }


def utc_iso_now() -> str:
    """ISO 8601 UTC timestamp with seconds precision (e.g. ``2026-06-26T15:01:42Z``)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_envelope(
    *,
    result: EnricherResult,
    enricher_id: str,
    enricher_version: str,
    schema_version: str,
    computed_at: str | None = None,
) -> EnvelopePayload:
    """Wrap an ``EnricherResult`` in the on-disk envelope shape.

    ``data`` is only populated when ``status == "ok"``; failed / timeout /
    quarantined / cancelled / skipped results carry ``None`` data + the
    ``error`` / ``error_class`` diagnostic fields.
    """
    return EnvelopePayload(
        derived=True,
        computed_at=computed_at or utc_iso_now(),
        enricher_id=enricher_id,
        enricher_version=enricher_version,
        schema_version=schema_version,
        status=result.status,
        data=result.data if result.status == STATUS_OK else None,
        error=result.error,
        error_class=result.error_class,
        retry_count=result.retry_count,
        circuit_state=result.circuit_state,
        duration_ms=result.duration_ms,
        records_written=result.records_written,
    )


_REQUIRED_KEYS = (
    "derived",
    "computed_at",
    "enricher_id",
    "enricher_version",
    "schema_version",
    "status",
)


def validate_envelope(payload: dict[str, Any]) -> None:
    """Validate a loaded JSON dict against the envelope contract.

    Raises ``EnvelopeShapeError`` on any contract violation. Used by
    the envelope writer (round-trip self-check) and by consumers
    loading enrichment files from disk.
    """
    for key in _REQUIRED_KEYS:
        if key not in payload:
            raise EnvelopeShapeError(f"envelope missing required key: {key!r}")
    if payload["derived"] is not True:
        raise EnvelopeShapeError(f"envelope.derived must be True (got {payload['derived']!r})")
    status = payload["status"]
    if status not in ALL_STATUSES:
        raise EnvelopeShapeError(
            f"envelope.status must be one of {sorted(ALL_STATUSES)}, " f"got {status!r}"
        )
    if status == STATUS_OK and payload.get("data") is None:
        raise EnvelopeShapeError("envelope.status=='ok' requires non-None data")
    for field_name in ("enricher_id", "enricher_version", "schema_version"):
        value = payload[field_name]
        if not isinstance(value, str) or not value:
            raise EnvelopeShapeError(f"envelope.{field_name} must be a non-empty string")
