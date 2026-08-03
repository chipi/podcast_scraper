"""ADR-145: one bounded in-place re-roll for an invalid structured LLM response before fallover.

Response-shape violations are non-retryable *in place* under ADR-100 — they fall over to another
provider. But a transient truncated/invalid structured response (observed on vLLM: valid on a
re-roll because the server is not bit-deterministic even at temperature 0) is cheaper to recover by
re-issuing the SAME request on the SAME endpoint once, before paying for a provider swap. This seam
does exactly that: call → validate → on failure, re-roll up to ``max_reroll`` times → then raise a
GuardrailViolation so the existing ADR-100 FallbackAware chain engages. The re-roll is PREPENDED to
fallover, not a replacement.

Each attempt is a real LLM call and ticks the per-episode call-budget fuse — keep ``max_reroll``
small (default 1). Invalid-response carries no HTTP status, so it never trips the #697 circuit
breaker (a healthy endpoint returning bad content must not be parked).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from ._telemetry import raise_violation
from .chat import REASON_CHAT_BAD_JSON

logger = logging.getLogger(__name__)


def structured_call_with_reroll(
    make_call: Callable[[], str],
    validate: Callable[[str], None],
    *,
    service: str,
    max_reroll: int = 1,
) -> str:
    """Call ``make_call`` and ``validate`` its structured response; re-roll on the same endpoint up
    to ``max_reroll`` times on a validation failure, then raise ``GuardrailViolation``.

    ``validate(content)`` must return ``None`` on success and raise on an invalid response (a
    ``GuardrailViolation`` from ``check_chat_response``, a ``json``/schema error, etc. — any
    exception is treated as "invalid response, re-roll"). Returns the first valid ``content``.
    """
    last: Optional[Exception] = None
    for attempt in range(max_reroll + 1):
        content = make_call()
        try:
            validate(content)
            return content
        except Exception as exc:  # noqa: BLE001 - any validator failure == invalid response
            last = exc
            if attempt < max_reroll:
                logger.warning(
                    "structured response invalid (service=%s attempt=%d/%d) — re-rolling: %s",
                    service,
                    attempt + 1,
                    max_reroll + 1,
                    exc,
                )
    # Exhausted the in-place re-rolls: surface as a GuardrailViolation so the ADR-100 fallover chain
    # (FallbackAware) engages, then the episode fails if every provider yields bad content.
    raise_violation(service, REASON_CHAT_BAD_JSON, f"invalid after {max_reroll} re-roll(s): {last}")
    raise AssertionError("unreachable — raise_violation always raises")  # for the type-checker
