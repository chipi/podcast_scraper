"""Exception class for response-shape guardrail violations (ADR-099).

A self-deployed (or cloud) inference service returned a successful HTTP
response whose content fails a structural sanity check. Consumers catch
this as a sibling of ``resilience.TimeoutLike`` — same fallback path,
no retry on the failing endpoint.
"""

from __future__ import annotations


class GuardrailViolation(Exception):
    """A self-hosted (or cloud) inference service returned a successful HTTP
    response whose content fails a structural sanity check (ADR-099).

    Behaviour: callers that already handle ``resilience.TimeoutLike`` should
    add a sibling ``except GuardrailViolation`` block. The fallback path is
    identical — the offending call counted as a failure, breaker records it,
    cloud / local fallback fires. No retry against the same endpoint: a
    guardrail-violating response under contention is unlikely to repair on
    immediate retry, and the fallback path is already paid for.

    Attributes carry enough context for log + telemetry without needing
    high-cardinality label values on the Prometheus side (those live in the
    log body and Sentry event scope).
    """

    def __init__(self, service: str, reason: str, response_summary: str = "") -> None:
        self.service = service
        self.reason = reason
        # Truncated to ~200 chars to keep log lines bounded; full response
        # body goes to the consumer's debug log if it needs investigation.
        self.response_summary = (response_summary or "")[:200]
        super().__init__(
            f"guardrail violation: service={service} reason={reason} "
            f"summary={self.response_summary!r}"
        )


__all__ = ["GuardrailViolation"]
