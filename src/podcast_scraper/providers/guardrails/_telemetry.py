"""Telemetry helpers shared by the per-service guardrail check modules.

Prometheus counter + structured WARN log + exception raise — centralised so
the three-part contract stays in lockstep across every check.

Counter name: ``inference_guardrail_violations_total{service, reason}``.
Renamed from ``dgx_guardrail_violations_total`` on 2026-06-15 — DGX is just
a deploy detail; the counter is for any inference-service guardrail
violation (self-hosted or cloud). Metric history reset is acceptable per
the rename.

Cardinality discipline: only ``service`` and ``reason`` are labels, both
drawn from a fixed per-service enum (see ``REASON_*`` constants in each
service's module). **Never** add high-cardinality fields (audio filename,
request ID, response content) as labels — those belong in the structured
log body / Sentry context. If the active series count for this counter
ever climbs above ~50, that's a bug.
"""

from __future__ import annotations

import logging
from typing import Any

from .exceptions import GuardrailViolation

logger = logging.getLogger(__name__)


# Lazy-imported on first use, cached at module scope so the Counter is
# registered exactly once even when this module is reloaded under pytest.
_COUNTER: Any = None


def _record(service: str, reason: str) -> None:
    """Increment the Prometheus counter for a guardrail violation."""
    global _COUNTER
    if _COUNTER is None:
        try:
            from prometheus_client import Counter

            _COUNTER = Counter(
                "inference_guardrail_violations_total",
                "Count of response-shape guardrail violations from inference "
                "services (self-hosted or cloud). ADR-099. Labels: service, reason.",
                labelnames=("service", "reason"),
            )
        except Exception:  # pragma: no cover - prometheus_client optional
            _COUNTER = False  # sentinel — don't retry the import
            return
    if _COUNTER:
        try:
            _COUNTER.labels(service=service, reason=reason).inc()
        except Exception:  # pragma: no cover - telemetry never breaks the caller
            pass


def raise_violation(service: str, reason: str, summary: str = "") -> None:
    """Emit telemetry then raise ``GuardrailViolation``.

    Centralised so the log + counter pair stays in lockstep across every
    per-service callback. The exception itself propagates the caller-side
    fallback path (see :class:`GuardrailViolation` docstring).
    """
    logger.warning(
        "inference guardrail violation: service=%s reason=%s summary=%s",
        service,
        reason,
        summary[:200] if summary else "",
    )
    _record(service, reason)
    raise GuardrailViolation(service, reason, summary)


__all__ = ["raise_violation"]
