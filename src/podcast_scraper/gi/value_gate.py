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
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

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

            update: Dict[str, Any] = {"summary_provider": name}
            # The judge model must be explicit. Inheriting the provider's default model is how a
            # full 10-episode run silently completed with the gate failing open on a 404.
            model = getattr(cfg, "gi_value_gate_model", None)
            if model:
                update[f"{name}_summary_model"] = model
            judge_cfg = cfg.model_copy(update=update)
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


def format_insight_for_judging(text: str, evidence: Optional["InsightEvidence"]) -> str:
    """The insight, WITH the evidence that grounds it — the thing the judge was never shown.

    The rubric asks for "a substantive position a NAMED PERSON took", "a real disagreement BETWEEN
    SPEAKERS", and "an AD or sponsor read". The judge could see none of those: it was handed a bare
    sentence. It is asked to grade evidence-backed-ness while blind to the evidence.

    Worse, it ran BEFORE grounding, so the quotes did not exist yet — the same defect as ADR-110,
    one layer up: a decision made at a point in the pipeline where its evidence has not been
    computed. So an insight with no verbatim support at all looked identical to one quoted from the
    host, and "no quote exists for this" — the strongest FILLER signal there is — was invisible.
    """
    if evidence is None or not evidence.quote:
        return f"{text}\n    EVIDENCE: NONE — no verbatim quote in the transcript supports this."
    who = evidence.speaker or "an unnamed voice"
    kind = f" [{evidence.voice_type}]" if evidence.voice_type else ""
    return f'{text}\n    EVIDENCE: "{evidence.quote}" — said by {who}{kind}'


class InsightEvidence(NamedTuple):
    """What grounds an insight: the verbatim span, who said it, and what kind of voice that is."""

    quote: Optional[str]
    speaker: Optional[str]
    voice_type: Optional[str]


def value_gate_keep_mask(
    insight_specs: List[Tuple[str, str]],
    *,
    provider: Optional[Any],
    cfg: Optional[Any] = None,
    pipeline_metrics: Optional[Any] = None,
    evidence: Optional[List[Optional[InsightEvidence]]] = None,
) -> List[bool]:
    """Which insights clear ``gi_value_gate_min_tier`` — one boolean per insight, in order.

    Args:
        insight_specs: ``[(text, insight_type), ...]`` as resolved upstream.
        provider: the summarization provider. Must expose ``classify_insights(texts)`` returning
            one integer tier per insight, in order. Providers without it skip the gate.
        cfg: config. Reads ``gi_value_gate_enabled`` and ``gi_value_gate_min_tier``.
        pipeline_metrics: optional counters.
        evidence: per-insight grounding — the quote, its speaker, and the voice type. When supplied
            the judge grades the CLAIM AND ITS SUPPORT; without it, the bare sentence as before.

    Returns:
        A keep-mask, order preserved. All-True on any failure — the gate fails OPEN.
    """
    if not insight_specs:
        return []

    enabled = bool(getattr(cfg, "gi_value_gate_enabled", False)) if cfg else False
    if not enabled:
        return [True] * len(insight_specs)

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
        return [True] * len(insight_specs)

    if evidence is not None and len(evidence) == len(insight_specs):
        texts = [format_insight_for_judging(t, ev) for (t, _), ev in zip(insight_specs, evidence)]
        grounded = sum(1 for ev in evidence if ev and ev.quote)
        logger.info(
            "value gate: grading %d insights WITH their evidence (%d grounded, %d unsupported)",
            len(texts),
            grounded,
            len(texts) - grounded,
        )
    else:
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
        return [True] * len(insight_specs)

    if not isinstance(tiers, list) or len(tiers) != len(insight_specs):
        logger.warning(
            "value gate returned %s tiers for %d insights; keeping all (ungated)",
            len(tiers) if isinstance(tiers, list) else type(tiers).__name__,
            len(insight_specs),
        )
        _bump(pipeline_metrics, "gi_value_gate_failures")
        return [True] * len(insight_specs)

    # A KEEP MASK, not a filtered list. The caller must drop each insight's QUOTES along with it,
    # and they are index-aligned — identity cannot be used to re-pair them, because CPython shares
    # one object for two equal constant tuples and an episode that says the same thing twice would
    # then keep the wrong evidence. A quote attached to the wrong insight is a fabricated
    # attribution, which is worse than the filler the gate exists to remove.
    keep: List[bool] = []
    dropped = 0
    for tier in tiers:
        try:
            t = int(tier)
        except (TypeError, ValueError):
            t = TIER_CORE  # unparsable tier: keep, do not silently discard real content
        keep.append(t >= min_tier)
        if t < min_tier:
            dropped += 1

    # Never let the gate empty an episode. If nothing clears the bar, the gate is more likely
    # broken (or the rubric mismatched) than the episode genuinely worthless.
    if not any(keep):
        logger.warning(
            "value gate rejected ALL %d insights (min_tier=%d); keeping them ungated rather "
            "than emitting an empty episode",
            len(insight_specs),
            min_tier,
        )
        _bump(pipeline_metrics, "gi_value_gate_rejected_all")
        return [True] * len(insight_specs)

    _bump(pipeline_metrics, "gi_value_gate_calls")
    _bump(pipeline_metrics, "gi_insights_dropped_by_value_gate", dropped)
    if dropped:
        logger.info(
            "value gate: dropped %d/%d insights below tier %d",
            dropped,
            len(insight_specs),
            min_tier,
        )
    return keep


def _bump(metrics: Optional[Any], name: str, amount: int = 1) -> None:
    if metrics is None or not amount:
        return
    try:
        setattr(metrics, name, getattr(metrics, name, 0) + amount)
    except Exception:  # noqa: BLE001
        pass


def apply_value_gate(
    insight_specs: List[Tuple[str, str]],
    *,
    provider: Optional[Any],
    cfg: Optional[Any] = None,
    pipeline_metrics: Optional[Any] = None,
    evidence: Optional[List[Optional[InsightEvidence]]] = None,
) -> List[Tuple[str, str]]:
    """The surviving specs, for callers that carry nothing alongside them."""
    mask = value_gate_keep_mask(
        insight_specs,
        provider=provider,
        cfg=cfg,
        pipeline_metrics=pipeline_metrics,
        evidence=evidence,
    )
    return [spec for spec, keep in zip(insight_specs, mask) if keep]
