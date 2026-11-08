"""Module-level helpers anthropic_provider and its GI/naming methods share (moved verbatim from
``anthropic_provider``, which imports them back)."""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...models import Episode
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]

logger = logging.getLogger("podcast_scraper.providers.anthropic.anthropic_provider")


def _anthropic_finish_reason(response: Any) -> Optional[str]:
    """Normalize Anthropic's ``stop_reason`` to the guardrail's vocabulary.

    Anthropic emits ``"max_tokens"`` for the over-budget condition; the
    chat guardrail trips on the canonical ``"length"`` value (OpenAI /
    DeepSeek convention). Map at the per-SDK boundary so the helper stays
    service-neutral. Returns None when the field isn't present (the
    guardrail then skips the finish-reason check, which is the safe
    default).
    """
    raw = getattr(response, "stop_reason", None)
    if raw == "max_tokens":
        return "length"
    return raw


def _record_anthropic_llm_call(
    response: Any,
    pipeline_metrics: Optional[Any],
    *,
    recorder_name: str,
    cfg: Any,
    model: str,
) -> None:
    """Record tokens + cost_usd for an Anthropic LLM call into pipeline_metrics.

    Matches the behaviour of OpenAI's inline wiring (#650/#651) for cleaning,
    GI, and KG capabilities. Uses the ``summarization`` pricing capability for
    all text calls — the YAML rate row is per (provider, text-vs-transcription)
    regardless of which pipeline stage invoked it.
    """
    if pipeline_metrics is None or not hasattr(pipeline_metrics, recorder_name):
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    in_raw = getattr(usage, "input_tokens", None)
    out_raw = getattr(usage, "output_tokens", None)
    if not isinstance(in_raw, (int, float)) or not isinstance(out_raw, (int, float)):
        return
    in_tok = int(in_raw)
    out_tok = int(out_raw)
    from ...workflow.helpers import calculate_provider_cost

    cost = calculate_provider_cost(
        cfg=cfg,
        provider_type="anthropic",
        capability="summarization",
        model=model,
        prompt_tokens=in_tok,
        completion_tokens=out_tok,
    )
    getattr(pipeline_metrics, recorder_name)(in_tok, out_tok, cost_usd=cost)
