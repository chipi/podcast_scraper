"""Tokens are the ground truth; cost is a projection of tokens through a price list.

THE INCIDENT this fixes (Problem 1): our cost telemetry was computed AT EMIT TIME and dropped when
it came out zero — so a provider with no pricing row (OpenAI gpt-5.x) logged *nothing*, and every
GI / evidence / cleaning call (whose ``capability`` the pricing lookup never mapped) was silently
unpriced and dropped. Mistral's "12× undercount" was exactly that: its log carried only
``summarization`` events. You cannot slice or reconcile spend you never recorded.

The fix inverts the dependency. We instrument the ONE thing every provider reliably returns — the
token counts on the request/response — as a normalised ``TokenUsage`` (input / output / cached-read
/ cache-write), and record it ALWAYS, regardless of whether a price is known. Cost then becomes a
pure function ``cost_from_tokens(usage, rates)`` applied AFTER THE FACT, so the same raw telemetry
can be re-priced, sliced per request / operation / episode / run, and reconciled vs a dashboard.

Providers disagree on how they report cache tokens, and this module hides that:

* OpenAI / DeepSeek / Gemini — ``cached`` is a SUBSET of the prompt/input count (prompt_tokens
  already includes the cache-read portion). ``prompt_tokens_details.cached_tokens`` /
  ``prompt_cache_hit_tokens`` / ``cached_content_token_count``.
* Anthropic — ``input_tokens`` is the NON-cached input; ``cache_read_input_tokens`` and
  ``cache_creation_input_tokens`` are SEPARATE, additive fields.

We normalise both into one convention: ``input_tokens`` is the TOTAL input processed (cached +
uncached + cache-write), ``cached_input_tokens`` is the cache-read portion (billed cheap), and
``cache_write_tokens`` is the cache-creation portion (billed at a premium). Then ``uncached_input``
is always ``input - cached - cache_write`` and cost is uniform across providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


@dataclass(frozen=True)
class TokenUsage:
    """Normalised token counts for one LLM request/response — the ground-truth unit of accounting.

    ``input_tokens`` is the TOTAL input processed (includes cached-read and cache-write portions).
    ``cached_input_tokens`` is the cache-read portion (billed at the cached rate).
    ``cache_write_tokens`` is the cache-creation portion (billed at a write premium, Anthropic).
    ``output_tokens`` is the completion. Any field may be ``None`` when a provider omits it.
    """

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None

    @property
    def uncached_input_tokens(self) -> Optional[int]:
        """Input billed at the full rate: total input minus cache-read minus cache-write."""
        if self.input_tokens is None:
            return None
        return max(
            0, self.input_tokens - (self.cached_input_tokens or 0) - (self.cache_write_tokens or 0)
        )

    def is_empty(self) -> bool:
        """True when the call reported no input or output tokens (nothing to record)."""
        return not (self.input_tokens or self.output_tokens)

    def as_dict(self) -> Dict[str, Optional[int]]:
        """The four token counts as a plain dict (for the cost event / rollup)."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


def _openai_style_usage(usage: Any) -> TokenUsage:
    """OpenAI-compatible (openai / deepseek / grok / mistral): prompt_tokens INCLUDES cached."""
    input_tokens = _as_int(getattr(usage, "prompt_tokens", None))
    output_tokens = _as_int(getattr(usage, "completion_tokens", None))
    cached = None
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = _as_int(getattr(details, "cached_tokens", None))
        if cached is None and isinstance(details, dict):
            cached = _as_int(details.get("cached_tokens"))
    # DeepSeek exposes its own cache-hit field alongside the OpenAI shape.
    if cached is None:
        cached = _as_int(getattr(usage, "prompt_cache_hit_tokens", None))
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached,
        cache_write_tokens=None,
    )


def _anthropic_usage(usage: Any) -> TokenUsage:
    """Anthropic: input_tokens is NON-cached; cache_read/creation are separate → fold into total."""
    base_input = _as_int(getattr(usage, "input_tokens", None)) or 0
    cache_read = _as_int(getattr(usage, "cache_read_input_tokens", None)) or 0
    cache_write = _as_int(getattr(usage, "cache_creation_input_tokens", None)) or 0
    output_tokens = _as_int(getattr(usage, "output_tokens", None))
    total_input = base_input + cache_read + cache_write
    return TokenUsage(
        input_tokens=total_input or None,
        output_tokens=output_tokens,
        cached_input_tokens=cache_read or None,
        cache_write_tokens=cache_write or None,
    )


def _gemini_usage(usage: Any) -> TokenUsage:
    """Gemini: prompt_token_count INCLUDES the cached_content portion."""
    input_tokens = _as_int(getattr(usage, "prompt_token_count", None))
    output_tokens = _as_int(getattr(usage, "candidates_token_count", None))
    cached = _as_int(getattr(usage, "cached_content_token_count", None))
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached,
        cache_write_tokens=None,
    )


# OpenAI-compatible SDK is the transport for these providers.
_OPENAI_COMPATIBLE = frozenset({"openai", "deepseek", "grok", "mistral"})


def extract_token_usage(provider: str, response: Any) -> TokenUsage:
    """Normalise a provider's response into a canonical :class:`TokenUsage` (never raises).

    Picks the reader by provider family; unknown providers get a best-effort OpenAI-style read. A
    response with no usable usage block yields an all-None ``TokenUsage`` (``is_empty()`` True).
    """
    if response is None:
        return TokenUsage()
    p = (provider or "").strip().lower()
    try:
        if p == "anthropic":
            usage = getattr(response, "usage", None)
            return _anthropic_usage(usage) if usage else TokenUsage()
        if p == "gemini":
            usage = getattr(response, "usage_metadata", None)
            return _gemini_usage(usage) if usage else TokenUsage()
        # openai/deepseek/grok/mistral and any unknown → OpenAI-compatible shape.
        usage = getattr(response, "usage", None)
        return _openai_style_usage(usage) if usage else TokenUsage()
    except Exception:  # noqa: BLE001 - telemetry must never break a provider call
        return TokenUsage()


def cost_from_tokens(usage: TokenUsage, rates: Dict[str, float]) -> Optional[float]:
    """Project a :class:`TokenUsage` through a per-1M-token price list into USD.

    ``rates`` keys (all optional, USD per 1M tokens): ``input_cost_per_1m_tokens``,
    ``output_cost_per_1m_tokens``, ``cached_input_cost_per_1m_tokens`` (defaults to the input rate
    if absent), ``cache_write_cost_per_1m_tokens`` (defaults to input rate). Returns ``None`` if no
    input/output rate is available (cost genuinely unknown — keep the tokens, drop the projection),
    else the USD cost. Cached and cache-write tokens are billed at their own rates; the rest of the
    input at the full input rate.
    """
    if not rates:
        return None
    in_rate = rates.get("input_cost_per_1m_tokens")
    out_rate = rates.get("output_cost_per_1m_tokens")
    if in_rate is None and out_rate is None:
        return None
    in_rate = float(in_rate or 0.0)
    out_rate = float(out_rate or 0.0)
    # Cached-input rate is keyed ``cache_hit_input_cost_per_1m_tokens`` in pricing_assumptions.yaml;
    # accept the shorter alias too. Absent → falls back to the full input rate.
    cached_rate_raw = rates.get("cache_hit_input_cost_per_1m_tokens")
    if cached_rate_raw is None:
        cached_rate_raw = rates.get("cached_input_cost_per_1m_tokens", in_rate)
    cached_rate = float(cached_rate_raw if cached_rate_raw is not None else in_rate)
    write_rate = float(rates.get("cache_write_cost_per_1m_tokens", in_rate) or in_rate)

    cost = 0.0
    uncached = usage.uncached_input_tokens
    if uncached:
        cost += (uncached / 1_000_000) * in_rate
    if usage.cached_input_tokens:
        cost += (usage.cached_input_tokens / 1_000_000) * cached_rate
    if usage.cache_write_tokens:
        cost += (usage.cache_write_tokens / 1_000_000) * write_rate
    if usage.output_tokens:
        cost += (usage.output_tokens / 1_000_000) * out_rate
    return round(cost, 6)
