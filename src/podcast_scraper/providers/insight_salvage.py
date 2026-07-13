"""Salvage a truncated insight list instead of losing the episode.

``generate_insights`` asks for a newline-delimited list. When the reply overruns
``max_output_tokens`` the provider reports ``finish_reason=length``, the chat guardrail raises
(correctly — a truncated JSON body IS unusable), the exception is swallowed upstream, and the
episode falls back to the stub with a single placeholder insight. The run then reports success.

For a *line list* that reaction is far too destructive. Truncation cuts the final line mid-word
and leaves every earlier line intact. Discarding forty good insights because the forty-first was
clipped is worse than the truncation it is guarding against.

So: for this one shape — a line list, a length truncation, non-empty content — drop the partial
last line and keep the rest. Every other guardrail reason still raises. The guardrail is not
weakened; the recoverable case is handled.

Measured: the intermittent length overrun hit 1 of 3 eval episodes and 8 of 15 probe runs, each
time costing the episode its entire insight set.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .guardrails.chat import REASON_CHAT_FINISH_LENGTH
from .guardrails.exceptions import GuardrailViolation

logger = logging.getLogger(__name__)


def resolve_insight_temperature(cfg: Any, provider: str) -> float:
    """Temperature for insight generation, from config.

    Every provider hardcoded 0.3 and ignored the configured value, so the pipeline was not
    reproducible: the same config on the same 3 episodes gave 28.0 vs 18.3 insights/episode and
    1.51 vs 6.00 quotes/insight, with grounding straddling the ADR-053 line (79.8% vs 94.5%).
    Evals need to pin this to 0.
    """
    from .. import config_constants

    value = getattr(cfg, f"{provider}_temperature", None)
    if value is None:
        return float(config_constants.GI_INSIGHT_TEMPERATURE_DEFAULT)
    return float(value)


def strip_json_fence(content: Optional[str]) -> str:
    """Unwrap a ```json ... ``` fence.

    Anthropic wraps JSON replies in a markdown fence even when asked not to, so a strict
    ``expect_json`` guardrail rejects an otherwise perfect response. Cheap to strip, and harmless
    for providers that never fence.
    """
    body = (content or "").strip()
    if not body.startswith("```"):
        return body
    lines = body.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def salvage_truncated_lines(exc: GuardrailViolation, content: Optional[str]) -> Optional[str]:
    """Return the usable prefix of a truncated line list, or ``None`` if unsalvageable.

    Args:
        exc: the guardrail violation just raised.
        content: the (partial) response body.

    Returns:
        The content minus its truncated final line, when the violation is a length truncation and
        at least one complete line survives. ``None`` otherwise — callers must re-raise.
    """
    if getattr(exc, "reason", None) != REASON_CHAT_FINISH_LENGTH:
        return None

    body = (content or "").strip()
    if not body:
        return None

    lines = body.splitlines()
    if len(lines) < 2:
        # A single line that was itself cut off tells us nothing reliable.
        return None

    # The last line is the one truncation landed in; drop it.
    kept = [ln for ln in lines[:-1] if ln.strip()]
    if not kept:
        return None

    logger.warning(
        "insight list truncated at max_output_tokens; salvaged %d complete lines and dropped the "
        "partial last one, rather than losing the episode to the stub fallback",
        len(kept),
    )
    return "\n".join(kept)
