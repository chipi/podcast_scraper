"""Module-level helpers mistral_provider and its GI/naming methods share (moved verbatim from
``mistral_provider``, which imports them back)."""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...models import Episode
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]

logger = logging.getLogger("podcast_scraper.providers.mistral.mistral_provider")


def _record_mistral_llm_call(
    response: Any,
    pipeline_metrics: Optional[Any],
    *,
    recorder_name: str,
    cfg: Any,
    model: str,
) -> None:
    """Record tokens + cost_usd for a Mistral LLM call into pipeline_metrics.

    Mirrors OpenAI/Anthropic/Gemini wiring (#650/#651). Mistral uses
    ``response.usage.prompt_tokens`` / ``.completion_tokens`` (OpenAI-compat).
    """
    if pipeline_metrics is None or not hasattr(pipeline_metrics, recorder_name):
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    in_raw = getattr(usage, "prompt_tokens", None)
    out_raw = getattr(usage, "completion_tokens", None)
    if not isinstance(in_raw, (int, float)) or not isinstance(out_raw, (int, float)):
        return
    in_tok = int(in_raw)
    out_tok = int(out_raw)
    from ...workflow.helpers import calculate_provider_cost

    cost = calculate_provider_cost(
        cfg=cfg,
        provider_type="mistral",
        capability="summarization",
        model=model,
        prompt_tokens=in_tok,
        completion_tokens=out_tok,
    )
    getattr(pipeline_metrics, recorder_name)(in_tok, out_tok, cost_usd=cost)
