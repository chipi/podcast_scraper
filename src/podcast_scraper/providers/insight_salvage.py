"""Salvage a truncated insight list instead of losing the episode.

``generate_insights`` asks for a newline-delimited list. When the reply overruns
``max_output_tokens`` the provider reports ``finish_reason=length``, the chat guardrail raises
(correctly — a truncated JSON body IS unusable), the exception is swallowed upstream, and the
episode ends up with no insights at all. The run then reports success.

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
    1.51 vs 6.00 quotes/insight, with grounding landing either side of the 80% floor (79.8% vs
    94.5%). Evals need to pin this to 0.

    Un-hardcoding it was not enough, because there was still no knob an eval could TURN. This read
    only ``<provider>_temperature``, so an arm YAML saying ``temperature: 0.0`` pinned nothing —
    the key mapped to no Config field and extraction sampled at 0.3 regardless. A head-to-head run
    that way partly measures the sampler: re-run the SAME model and it disagrees with itself, and
    that disagreement is indistinguishable from "the other model found different knowledge".

    ``gi_insight_temperature`` is that knob, and it wins when set. It is deliberately NOT
    ``<provider>_temperature``, which also drives summarisation and speaker detection — pinning
    insight extraction must not silently re-tune two unrelated stages.
    """
    from .. import config_constants

    pinned = getattr(cfg, "gi_insight_temperature", None)
    if pinned is not None:
        return float(pinned)
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


def _bump(metrics: Any, name: str, amount: int = 1) -> None:
    """Increment a counter on the pipeline metrics object; never raise on telemetry."""
    if metrics is None or not amount:
        return
    try:
        setattr(metrics, name, getattr(metrics, name, 0) + amount)
    except Exception:  # noqa: BLE001 - telemetry must never break a run
        pass


def record_overgeneration(pipeline_metrics: Any, produced: int, ceiling: int) -> None:
    """Log + count a response that exceeded the requested insight ceiling.

    Shared by every provider because the block is identical in all six and the counters were
    not: only ``openai_provider`` bumped them, under a comment claiming "these counters make
    the ratio aggregatable". Production (vLLM / LiteLLM) inherits ``OpenAICompatibleProvider``
    so prod was covered by luck, but an ollama or cloud-native run measured nothing.

    Over-production is a signal, not a detail to swallow. Truncating silently is what hid the
    model returning 300+ lines while we kept 50 — which read as "it obediently returned exactly
    the cap". On a self-hosted box cost is $0 by definition, so waste has no economic alarm; it
    has to be counted or it is invisible.
    """
    if produced <= ceiling:
        return
    logger.warning(
        "generate_insights: model returned %d insights for a ceiling of %d; keeping %d spread "
        "across the episode. The prompt is not constraining the count.",
        produced,
        ceiling,
        ceiling,
    )
    _bump(pipeline_metrics, "gi_insight_overgeneration_events")
    _bump(pipeline_metrics, "gi_insight_overgenerated_total", produced - ceiling)
    if produced >= 5 * max(1, ceiling):
        _bump(pipeline_metrics, "gi_insight_overgeneration_severe_events")


def take_within_ceiling(insights: list[str], ceiling: int) -> list[str]:
    """Reduce *insights* to at most *ceiling* items, preserving whole-episode coverage (#1919).

    Providers used to end with ``cleaned[:max_insights]``. That reads as "keep the best N" and is
    in fact "keep the earliest N" — and the model emits insights in transcript order, so it means
    "keep the beginning of the episode and discard the rest".

    Measured on a real episode (73 insights, GI's own ``rank`` / ``position_hint`` properties):

        Pearson(rank, position_hint) = 0.904
        kept by head-slice(25): episode position 0.04 - 0.29   (mean 0.17)
        discarded (48):         episode position 0.03 - 0.92   (mean 0.55)

    Everything said between the 30% and 92% marks was dropped. On interview shows — most of the
    Batch A corpus — the substance is usually *after* the setup, so the head-slice removed the
    part worth keeping.

    Even-stride sampling instead. Strictly better under uncertainty: when arrival order tracks the
    transcript (it does) stride preserves coverage end to end; if the order were arbitrary, stride
    is no worse than a head-slice. Order is preserved, so downstream ``rank`` stays monotonic in
    transcript position.

    Ranking by ``salience`` would be better still, but the pipeline computes salience *downstream*
    of this cut, on the survivors only. Moving that upstream is a larger change (#1919 discussion).
    """
    if ceiling <= 0:
        return []
    if len(insights) <= ceiling:
        return list(insights)
    # Evenly spaced indices across the full list, endpoints included, no duplicates by
    # construction (stride >= 1 because len > ceiling).
    step = (len(insights) - 1) / (ceiling - 1) if ceiling > 1 else 0.0
    picks = [round(i * step) for i in range(ceiling)]
    return [insights[i] for i in picks]


def salvage_truncated_lines(
    exc: GuardrailViolation,
    content: Optional[str],
    pipeline_metrics: Optional[Any] = None,
) -> Optional[str]:
    """Return the usable prefix of a truncated line list, or ``None`` if unsalvageable.

    Counters, not just a log line. The 2026-08-31 DGX batch had this path firing ~1.2x per
    episode — a recovery mechanism carrying that much traffic is a load-bearing part of the
    pipeline, and a WARNING is not something you can aggregate, alert on, or trend. Both
    outcomes are counted, because they mean opposite things:

    * ``gi_insight_salvage_events`` / ``..._lines_recovered`` — truncation happened and we
      kept the good prefix. Waste, not damage.
    * ``gi_insight_salvage_failed_events`` — truncation happened and NOTHING was recoverable,
      so the caller re-raises and the episode loses its entire insight set. This is the
      outcome that actually costs data, and until now it had no signal at all: it looked
      identical to any other guardrail violation.

    A rising ratio of failed-to-successful salvage is the early warning that the token budget
    is now too tight, which is exactly the knob that was lowered 150 -> 50 on 2026-08-31.

    Args:
        exc: the guardrail violation just raised.
        content: the (partial) response body.
        pipeline_metrics: optional metrics sink; telemetry only, never affects the result.

    Returns:
        The content minus its truncated final line, when the violation is a length truncation and
        at least one complete line survives. ``None`` otherwise — callers must re-raise.
    """
    if getattr(exc, "reason", None) != REASON_CHAT_FINISH_LENGTH:
        # Not a truncation at all — a different guardrail. Not this function's business, and
        # deliberately NOT counted as a salvage failure: that counter means "we lost insights
        # to truncation", and diluting it with unrelated violations would hide the trend.
        return None

    body = (content or "").strip()
    lines = body.splitlines() if body else []
    # A single line that was itself cut off tells us nothing reliable.
    kept = [ln for ln in lines[:-1] if ln.strip()] if len(lines) >= 2 else []

    if not kept:
        _bump(pipeline_metrics, "gi_insight_salvage_failed_events")
        logger.warning(
            "insight list truncated at max_output_tokens with nothing recoverable "
            "(%d line(s) in the partial body); the episode loses its entire insight set",
            len(lines),
        )
        return None

    _bump(pipeline_metrics, "gi_insight_salvage_events")
    _bump(pipeline_metrics, "gi_insight_salvage_lines_recovered", len(kept))
    logger.warning(
        "insight list truncated at max_output_tokens; salvaged %d complete lines and dropped the "
        "partial last one, rather than losing every insight in the batch",
        len(kept),
    )
    return "\n".join(kept)
