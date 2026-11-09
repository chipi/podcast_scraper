"""Module-level helpers openai_provider and its GI/naming methods share (moved verbatim from
``openai_provider``, which imports them back)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from ... import models

if TYPE_CHECKING:
    from ...models import Episode
else:
    Episode = models.Episode  # type: ignore[assignment]

logger = logging.getLogger("podcast_scraper.providers.openai.openai_provider")


def _repetition_signal(text: str, n: int = 12) -> Tuple[int, str]:
    """``(count, gram)`` for the most-repeated ``n``-word run — the loop tell.

    Computed AT CAPTURE TIME so the signal survives even if the reply itself is truncated by
    the size cap. A healthy JSON reply repeats a 12-gram once or twice (shared phrasing); a
    decoding loop repeats one run dozens of times, which is what separates "the model got stuck"
    from "the model genuinely had more to say" (#2053).

    NOT sufficient alone — it is whitespace-tokenised and therefore blind to two degenerate shapes.
    Always read it alongside :func:`_tail_degeneracy`.
    """
    words = (text or "").split()
    if len(words) < n * 3:
        return 0, ""
    counts: Dict[str, int] = {}
    for i in range(len(words) - n):
        gram = " ".join(words[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    if not counts:
        return 0, ""
    gram, count = max(counts.items(), key=lambda kv: kv[1])
    return count, gram[:200]


def _openai_chat_usage_tokens(response: Any) -> Tuple[Optional[int], Optional[int]]:
    """Best-effort prompt/completion token counts from a chat completion response."""
    if not hasattr(response, "usage") or not response.usage:
        return None, None
    prompt_tokens = getattr(response.usage, "prompt_tokens", None)
    completion_tokens = getattr(response.usage, "completion_tokens", None)
    input_tokens = int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None
    output_tokens = int(completion_tokens) if isinstance(completion_tokens, (int, float)) else None
    return input_tokens, output_tokens


def _openai_response_cost_usd(response: Any) -> Optional[float]:
    """PA3: the ACTUAL cost of a call, straight from the upstream response, when it exposes one.

    OpenRouter returns ``usage.cost`` (real spend for the chosen backend); LiteLLM surfaces
    ``_hidden_params["response_cost"]``. Preferring this over the local pricing-table estimate fixes
    the gateway/OpenRouter ``estimated_cost_usd: 0.0`` gap (a table lookup misses aliased /
    OpenRouter model ids). Returns None when no upstream cost is present so the caller falls back.
    """
    usage = getattr(response, "usage", None)
    cost = getattr(usage, "cost", None) if usage is not None else None
    if not isinstance(cost, (int, float)):
        hidden = getattr(response, "_hidden_params", None)
        if isinstance(hidden, dict):
            cost = hidden.get("response_cost")
    return float(cost) if isinstance(cost, (int, float)) and cost > 0 else None
