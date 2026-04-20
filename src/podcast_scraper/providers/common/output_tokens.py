"""Shared helpers for cloud-LLM output-token budget enforcement.

Discovered via Flightcast summary failure on 2026-04-20: all 6 cloud provider
summarize() paths flow ``params.max_length or cfg.summary_reduce_params.max_new_tokens``
(default 650, tuned for LED-base local ML) into the API's ``max_tokens`` /
``max_output_tokens`` cap. For long transcripts + structured JSON output, 650
is below the budget Gemini allocates, so Gemini silently truncates mid-JSON
and downstream parsers reject the response.

The fix: enforce a cloud-LLM-specific floor (``cloud_llm_structured_min_output_tokens``,
default 4096) at each provider's summarize() call site. Local ML paths (LED-base
bart-led) are unchanged — they continue to use the 650 default that's tuned
for their decoder.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


_DEFAULT_CLOUD_STRUCTURED_MIN_OUTPUT_TOKENS = 4096


def cloud_structured_max_output_tokens(cfg: Any, requested: Optional[int]) -> int:
    """Clamp requested max_output_tokens up to the cloud-LLM structured floor.

    Cloud LLMs emit proportionally larger structured JSON for longer transcripts
    (title + summary + bullets + schema overhead). Applying the LED-base 650-token
    default to cloud LLM calls causes mid-JSON truncation on long episodes
    (observed 2026-04-20 on Flightcast with Gemini 2.5-flash-lite).

    Args:
        cfg: Config object (reads ``cloud_llm_structured_min_output_tokens`` if set).
        requested: Caller-provided max_output_tokens (e.g. from
            params.max_length or cfg.summary_reduce_params.max_new_tokens).

    Returns:
        ``max(requested, cloud_llm_structured_min_output_tokens)``.
    """
    floor = getattr(cfg, "cloud_llm_structured_min_output_tokens", None) or (
        _DEFAULT_CLOUD_STRUCTURED_MIN_OUTPUT_TOKENS
    )
    current = int(requested or 0)
    if current >= floor:
        return current
    if current > 0 and current < floor:
        logger.debug(
            "cloud_structured_max_output_tokens: clamping %d -> %d (cloud-LLM floor)",
            current,
            floor,
        )
    return int(floor)


def warn_if_output_truncated(
    provider_name: str,
    finish_reason: Optional[str],
    output_tokens: Optional[int],
    max_output_tokens: int,
    episode_id: Optional[str] = None,
) -> None:
    """Log a warning if the API response looks truncated.

    Two independent signals:
      1. ``finish_reason`` contains MAX_TOKENS / LENGTH / STOP_BEYOND_MAX (provider-dependent).
      2. ``output_tokens`` is within 5% of ``max_output_tokens`` — the model
         likely filled the budget and would have kept going.

    Args:
        provider_name: e.g. "gemini", "openai", for log attribution.
        finish_reason: Raw finish_reason string from provider response.
        output_tokens: Actual completion tokens used (if reported).
        max_output_tokens: The cap passed in the request.
        episode_id: Episode id for log attribution, optional.
    """
    hit_reason = False
    if finish_reason:
        fr = str(finish_reason).upper()
        hit_reason = any(x in fr for x in ("MAX_TOKENS", "LENGTH", "SAFETY", "RECITATION"))

    hit_budget = False
    if output_tokens is not None and max_output_tokens > 0:
        hit_budget = output_tokens >= int(max_output_tokens * 0.95)

    if hit_reason or hit_budget:
        ep = f" ep={episode_id}" if episode_id else ""
        logger.warning(
            "[%s]%s possible output truncation: finish_reason=%s tokens=%s/%s (>=95%%=%s). "
            "Consider raising cloud_llm_structured_min_output_tokens or "
            "summary_reduce_params.max_new_tokens.",
            provider_name,
            ep,
            finish_reason,
            output_tokens,
            max_output_tokens,
            hit_budget,
        )
