"""Value gate: drop insights that carry no real knowledge.

The extractor cannot be made selective by prompting. Measured on 3 episodes with a blind
vendor-disjoint judge, the CORE count barely moves however the prompt is written:

    prompt              emitted   CORE   USEFUL+   FILLER
    quota (v1)             50.0   13.3      37.7     12.3
    down-biased bar        30.0   10.3      24.0      6.0
    neutral bar (v2)       41.3   12.0      29.7     11.7

An episode contains roughly a dozen genuinely important insights and no prompt conjures more.
All the prompt controls is how much filler rides along — and every attempt to suppress the
filler also suppresses real content, because the model cannot reliably tell them apart while
generating.

So this is a gate, not a prompt problem. Generate broadly, then trim — the same shape as the QA
and NLI gates on the evidence path. Filler is removed *after* the fact, where the decision is a
classification rather than a generation.

FAIL-OPEN. If the gate errors, or the provider cannot classify, every insight is kept. A broken
gate must never empty an episode — that is the failure mode this codebase keeps producing, and
it is worse than the filler the gate exists to remove.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Tiers the gate assigns. Kept here so the prompt, the config bound and the metrics agree.
TIER_FILLER = 0
TIER_MINOR = 1
TIER_USEFUL = 2
TIER_CORE = 3

DEFAULT_MIN_TIER = TIER_USEFUL


_judge_cache: Dict[str, Any] = {}
_judge_lock = threading.Lock()


def _resolve_judge(provider: Optional[Any], cfg: Optional[Any]) -> Optional[Any]:
    """Return the provider that grades the insights.

    By default the extractor grades its own output. That is the #939 same-vendor bias: a model
    asked to judge its own work is lenient. Measured across 7 providers, self-grading drops ~10%
    of insights where an independent judge drops ~25% of the same output — roughly half as strict.

    It also makes comparisons unfair: if gemini grades gemini and qwen grades qwen, each arm is
    filtered by a different strictness and the surviving counts are not comparable. Set
    ``gi_value_gate_provider`` to pin ONE judge across every arm.
    """
    if cfg is None:
        return provider
    name = getattr(cfg, "gi_value_gate_provider", None)
    if not name:
        return provider

    cached = _judge_cache.get(name)
    if cached is not None:
        return cached

    with _judge_lock:
        # Re-check under the lock: concurrent episodes must not each build a judge (and torch's
        # lazy init races when they do — see gi/about_edges.py).
        cached = _judge_cache.get(name)
        if cached is not None:
            return cached
        try:
            from ..summarization.factory import create_summarization_provider

            judge_cfg = cfg.model_copy(update={"summary_provider": name})
            judge = create_summarization_provider(judge_cfg)
            judge.initialize()
            _judge_cache[name] = judge
            logger.info("value gate: grading with a pinned judge (%s), not the extractor", name)
            return judge
        except Exception as exc:  # noqa: BLE001 — fail-open, as everywhere in this module
            logger.warning(
                "value gate: could not build the pinned judge %r (%s); falling back to the "
                "extractor grading its own output, which is lenient: %s",
                name,
                type(exc).__name__,
                exc,
            )
            return provider


def apply_value_gate(
    insight_specs: List[Tuple[str, str]],
    *,
    provider: Optional[Any],
    cfg: Optional[Any] = None,
    pipeline_metrics: Optional[Any] = None,
) -> List[Tuple[str, str]]:
    """Drop insights scoring below ``gi_value_gate_min_tier``.

    Args:
        insight_specs: ``[(text, insight_type), ...]`` as resolved upstream.
        provider: the summarization provider. Must expose ``classify_insights(texts)`` returning
            one integer tier per insight, in order. Providers without it skip the gate.
        cfg: config. Reads ``gi_value_gate_enabled`` and ``gi_value_gate_min_tier``.
        pipeline_metrics: optional counters.

    Returns:
        The surviving specs, order preserved. Every spec on any failure.
    """
    if not insight_specs:
        return insight_specs

    enabled = bool(getattr(cfg, "gi_value_gate_enabled", False)) if cfg else False
    if not enabled:
        return insight_specs

    min_tier = int(getattr(cfg, "gi_value_gate_min_tier", DEFAULT_MIN_TIER) or DEFAULT_MIN_TIER)

    judge = _resolve_judge(provider, cfg)
    classify = getattr(judge, "classify_insights", None)
    if not callable(classify):
        logger.debug(
            "value gate enabled but provider %s cannot classify insights; keeping all %d",
            type(provider).__name__,
            len(insight_specs),
        )
        _bump(pipeline_metrics, "gi_value_gate_unsupported")
        return insight_specs

    texts = [t for t, _ in insight_specs]
    try:
        tiers = classify(texts)
    except Exception as exc:  # noqa: BLE001 — fail-open is the whole point
        logger.warning(
            "value gate failed (%s); keeping all %d insights ungated: %s",
            type(exc).__name__,
            len(insight_specs),
            exc,
        )
        _bump(pipeline_metrics, "gi_value_gate_failures")
        return insight_specs

    if not isinstance(tiers, list) or len(tiers) != len(insight_specs):
        logger.warning(
            "value gate returned %s tiers for %d insights; keeping all (ungated)",
            len(tiers) if isinstance(tiers, list) else type(tiers).__name__,
            len(insight_specs),
        )
        _bump(pipeline_metrics, "gi_value_gate_failures")
        return insight_specs

    kept: List[Tuple[str, str]] = []
    dropped = 0
    for spec, tier in zip(insight_specs, tiers):
        try:
            t = int(tier)
        except (TypeError, ValueError):
            t = TIER_CORE  # unparsable tier: keep, do not silently discard real content
        if t >= min_tier:
            kept.append(spec)
        else:
            dropped += 1

    # Never let the gate empty an episode. If nothing clears the bar, the gate is more likely
    # broken (or the rubric mismatched) than the episode genuinely worthless.
    if not kept:
        logger.warning(
            "value gate rejected ALL %d insights (min_tier=%d); keeping them ungated rather "
            "than emitting an empty episode",
            len(insight_specs),
            min_tier,
        )
        _bump(pipeline_metrics, "gi_value_gate_rejected_all")
        return insight_specs

    _bump(pipeline_metrics, "gi_value_gate_calls")
    _bump(pipeline_metrics, "gi_insights_dropped_by_value_gate", dropped)
    if dropped:
        logger.info(
            "value gate: dropped %d/%d insights below tier %d",
            dropped,
            len(insight_specs),
            min_tier,
        )
    return kept


def _bump(metrics: Optional[Any], name: str, amount: int = 1) -> None:
    if metrics is None or not amount:
        return
    try:
        setattr(metrics, name, getattr(metrics, name, 0) + amount)
    except Exception:  # noqa: BLE001
        pass
