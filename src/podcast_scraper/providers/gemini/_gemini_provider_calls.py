"""Module-level helpers gemini_provider and its GI/naming methods share (moved verbatim from
``gemini_provider``, which imports them back)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...models import Episode
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]

logger = logging.getLogger("podcast_scraper.providers.gemini.gemini_provider")


def _gemini_finish_reason(response: Any) -> Optional[str]:
    """Best-effort extraction of Gemini's finish_reason for guardrail purposes.

    Gemini's response shape varies by SDK version; ``candidates[0].finish_reason``
    is the canonical path but the value is sometimes an enum and sometimes a
    string. Normalises to a lowercase string ("length", "stop", etc.) so the
    chat-guardrail's ``finish_reason == "length"`` comparison works.
    Returns None when the field isn't present (the guardrail then skips the
    finish-reason check, which is the safe default).
    """
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        if reason is None:
            return None
        name = getattr(reason, "name", None)
        raw = str(name).lower() if name else str(reason).lower()
        # ADR-100 normalization: Gemini emits ``MAX_TOKENS`` for the
        # over-budget condition; the guardrail trips on the canonical
        # ``"length"`` value (OpenAI/DeepSeek convention). Map at the
        # per-SDK boundary so the helper stays service-neutral.
        if raw == "max_tokens":
            return "length"
        return raw
    except Exception:  # noqa: BLE001 - defensive; never break the caller
        return None


def _record_gemini_llm_call(
    response: Any,
    pipeline_metrics: Optional[Any],
    *,
    recorder_name: str,
    cfg: Any,
    model: str,
) -> None:
    """Record tokens + cost_usd for a Gemini LLM call into pipeline_metrics.

    Matches OpenAI/Anthropic equivalents (#650/#651). Gemini uses
    ``response.usage_metadata.prompt_token_count`` / ``.candidates_token_count``
    for usage. Uses the ``summarization`` pricing capability for all text calls.
    """
    if pipeline_metrics is None or not hasattr(pipeline_metrics, recorder_name):
        return
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    in_raw = getattr(usage, "prompt_token_count", None)
    out_raw = getattr(usage, "candidates_token_count", None)
    if not isinstance(in_raw, (int, float)) or not isinstance(out_raw, (int, float)):
        return
    in_tok = int(in_raw)
    out_tok = int(out_raw)
    from ...workflow.helpers import calculate_provider_cost

    cost = calculate_provider_cost(
        cfg=cfg,
        provider_type="gemini",
        capability="summarization",
        model=model,
        prompt_tokens=in_tok,
        completion_tokens=out_tok,
    )
    getattr(pipeline_metrics, recorder_name)(in_tok, out_tok, cost_usd=cost)


def _should_disable_thinking_for_model(model: str) -> bool:
    """True for Gemini 2.5 Flash (non-lite): default thinking consumes max_output_tokens.

    Note: 2.5-PRO cannot disable thinking (the API rejects budget 0: "This model only works in
    thinking mode"). It is handled separately in ``_merge_generate_content_config`` with a small
    bounded budget + extra output headroom, so its thinking does not truncate the content."""
    m = (model or "").lower()
    if "flash-lite" in m:
        return False
    return "2.5-flash" in m


# 2.5-pro requires thinking (budget 0 is rejected), and thinking shares the max_output_tokens
# budget with the content — a small-budget call (chunked-GI insights) truncates the CONTENT
# (finish_reason=MAX_TOKENS -> guardrail rejects the chunk -> no insights). Give pro a modest,
# bounded thinking budget and add that headroom on top of the requested output tokens so content
# always fits, keeping pro comparable to the other (reasoning-off) bake-off models (#1356).
_GEMINI_PRO_THINKING_BUDGET = 1024


def _should_bound_thinking_for_pro(model: str) -> bool:
    m = (model or "").lower()
    return "2.5-pro" in m and "flash-lite" not in m


def _merge_generate_content_config(
    model: str, base: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Merge generation config; set thinking_budget=0 for 2.5-flash unless thinking_config set."""
    merged: Dict[str, Any] = dict(base or ())
    # Explicit None is treated as absent so callers do not block injection by mistake.
    if merged.get("thinking_config") is None and "thinking_config" in merged:
        del merged["thinking_config"]
    if "thinking_config" not in merged and _should_disable_thinking_for_model(model):
        merged["thinking_config"] = {"thinking_budget": 0}
    elif "thinking_config" not in merged and _should_bound_thinking_for_pro(model):
        merged["thinking_config"] = {"thinking_budget": _GEMINI_PRO_THINKING_BUDGET}
        mot = merged.get("max_output_tokens")
        if isinstance(mot, int):
            merged["max_output_tokens"] = mot + _GEMINI_PRO_THINKING_BUDGET
    # Disable Automatic Function Calling. We never declare tools, so AFC has
    # nothing to do — disabling suppresses the "AFC is enabled with max remote
    # calls: 10" INFO line the SDK emits on every request. This is the
    # SDK-creator-recommended way to opt out (vs. silencing the logger).
    if "automatic_function_calling" not in merged:
        merged["automatic_function_calling"] = {"disable": True}
    return merged
