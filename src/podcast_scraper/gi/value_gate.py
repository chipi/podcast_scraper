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


_gate_model_cache: Dict[str, Any] = {}
_gate_model_lock = threading.Lock()


def _provider_can_gate(cfg: Optional[Any]) -> bool:
    """Is there an LLM on this path at all?

    The gate is an LLM being ASKED whether an insight is worth surfacing. On the pure-ML path —
    sentence-transformers, summllama, the local extractive stack — there is no LLM, so there is no
    rater and there cannot be one. The gate is INAPPLICABLE there, not merely switched off.

    The registry's resolver already refuses to hand a gate model to a profile with no LLM. This is
    the same rule at the point of USE, for a caller that builds a bare Config with no profile: the
    flag now defaults to True, so without this the gate would fall through to "the extractor rates
    itself" and press a summarisation model into service as a rater — loading a transformers model
    to answer a question it has no way to answer.

    Deliberately a POSITIVE identification of a non-LLM, not an inference about one: anything we
    cannot positively name as LLM-less is left alone, so this can only ever turn the gate OFF where
    it provably cannot run.
    """
    # Imported lazily and from the REGISTRY: one definition of "is an LLM", not a second copy here
    # that can drift from the resolver's. Lazy because providers.ml pulls in the whole torch stack,
    # which this module must not drag in just to answer a question about a string.
    from ..providers.ml.model_registry import _LLM_PROVIDERS

    name = getattr(cfg, "summary_provider", None)
    return not (isinstance(name, str) and name not in _LLM_PROVIDERS)


def _resolve_gate_model(
    provider: Optional[Any], cfg: Optional[Any], pipeline_metrics: Optional[Any] = None
) -> Optional[Any]:
    """Return the provider instance that rates the insights, or None when nothing here can rate.

    This is a per-insight tier RATER inside the production pipeline — not a bake-off judge. The
    registry (`resolve_value_gate`) picks it to ride the summariser's own route: same provider, same
    proxy, same credential, stronger sibling model. Do not re-import the evaluation rule
    (vendor-disjointness, #939) here; that governs autoresearch cohorts, and applying it to this
    stage is what sent litellm-routed production runs to the Anthropic API mid-pipeline.

    Where a route has no curated sibling the rater IS the summariser's own model. That is lenient —
    ~10% of insights dropped against ~25% for a distinct rater, measured across 7 providers — so it
    is logged at WARNING rather than left silently true.
    """
    if not _provider_can_gate(cfg):
        return None
    if cfg is None:
        return provider
    name = getattr(cfg, "gi_value_gate_provider", None)
    if not name:
        # No rater configured -> the extractor rates itself. Legitimate, but ~half as strict, and
        # this used to be the SILENT path: no log, no metric, indistinguishable from a full-strength
        # run in o11y. cloud_split_dgx_down and experiment_dgx_moss both sat here unnoticed.
        logger.warning(
            "value gate: no rater pinned for summariser %r — self-grading, which is lenient "
            "(~10%% of insights dropped vs ~25%%). Insight counts are not comparable with a "
            "curated-rater run. Pin gi_value_gate_provider/_model on this profile.",
            getattr(cfg, "summary_provider", None),
        )
        _bump(pipeline_metrics, "gi_value_gate_self_grade")
        return provider

    model = getattr(cfg, "gi_value_gate_model", None)

    # Keyed on route AND model. Provider alone used to be enough only because the rater was chosen
    # to
    # be a DIFFERENT vendor from the summariser; now that it always rides the same route, two
    # profiles sharing a provider but pinning different gate models would otherwise silently share
    # one instance and the second would rate with the first's model.
    cache_key = f"{name}::{model or ''}"

    cached = _gate_model_cache.get(cache_key)
    if cached is not None:
        return cached

    with _gate_model_lock:
        # Re-check under the lock: concurrent episodes must not each build a rater (and torch's
        # lazy init races when they do — see gi/about_edges.py).
        cached = _gate_model_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            from ..summarization.factory import create_summarization_provider

            update: Dict[str, Any] = {"summary_provider": name}
            # The gate model must be explicit. Inheriting the provider's default model is how a
            # full 10-episode run silently completed with the gate failing open on a 404.
            if model:
                update[f"{name}_summary_model"] = model
            gate_cfg = cfg.model_copy(update=update)
            gate_provider = create_summarization_provider(gate_cfg)
            gate_provider.initialize()
            _gate_model_cache[cache_key] = gate_provider

            # Self-grade is a legitimate outcome (an uncurated route rates with its own model), but
            # it is HALF AS STRICT — ~10% dropped vs ~25%. It must never be a silent property of a
            # run: someone reading insight counts has to be able to see which strictness produced
            # them.
            summariser_model = getattr(cfg, f"{name}_summary_model", None) or getattr(
                cfg, "summary_model", None
            )
            if model and summariser_model and model == summariser_model:
                _bump(pipeline_metrics, "gi_value_gate_self_grade")
                logger.warning(
                    "value gate: SELF-GRADING — %s is rating its own output with %r. This is "
                    "lenient (~10%% of insights dropped vs ~25%% for a distinct rater); counts are "
                    "not comparable with a curated-rater run. Add a stronger sibling for %r to "
                    "_PREFERRED_GATE_MODEL to fix.",
                    name,
                    model,
                    name,
                )
            else:
                logger.info(
                    "value gate: rating with %s/%s on the summariser's own route", name, model
                )
            return gate_provider
        except Exception as exc:  # noqa: BLE001 — fail-open, as everywhere in this module
            # A rater that fails to BUILD (bad alias, 404, gateway not serving the model) degrades
            # every episode to self-grade. Without this metric that degradation is invisible: the
            # run succeeds, the counts look plausible, and nothing says the gate ran at half
            # strength.
            _bump(pipeline_metrics, "gi_value_gate_rater_build_failures")
            _bump(pipeline_metrics, "gi_value_gate_self_grade")
            logger.warning(
                "value gate: could not build the pinned rater %r/%r (%s); falling back to the "
                "extractor rating its own output, which is lenient: %s",
                name,
                model,
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


def _coerce_tier(tier: Any) -> int:
    """A tier as int; an unparsable one is treated as CORE (keep, never silently discard)."""
    try:
        return int(tier)
    except (TypeError, ValueError):
        return TIER_CORE


def _classify_tiers(
    insight_specs: List[Tuple[str, str]],
    *,
    provider: Optional[Any],
    cfg: Optional[Any],
    pipeline_metrics: Optional[Any],
    evidence: Optional[List[Optional[InsightEvidence]]],
) -> Optional[List[int]]:
    """One judge classification → the raw per-insight tier list, or ``None`` when the gate can't or
    shouldn't classify (disabled, unsupported provider, or a failed/malformed judge call). Never
    raises. Callers treat ``None`` as "ungated: keep everything at CORE"."""
    if not insight_specs:
        return []

    enabled = bool(getattr(cfg, "gi_value_gate_enabled", False)) if cfg else False
    if not enabled:
        return None

    rater = _resolve_gate_model(provider, cfg, pipeline_metrics)
    classify = getattr(rater, "classify_insights", None)
    if not callable(classify):
        logger.debug(
            "value gate enabled but provider %s cannot classify insights; keeping all %d",
            type(provider).__name__,
            len(insight_specs),
        )
        _bump(pipeline_metrics, "gi_value_gate_unsupported")
        return None

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
        return None

    if not isinstance(tiers, list) or len(tiers) != len(insight_specs):
        logger.warning(
            "value gate returned %s tiers for %d insights; keeping all (ungated)",
            len(tiers) if isinstance(tiers, list) else type(tiers).__name__,
            len(insight_specs),
        )
        _bump(pipeline_metrics, "gi_value_gate_failures")
        return None

    return [_coerce_tier(t) for t in tiers]


def value_gate_evaluate(
    insight_specs: List[Tuple[str, str]],
    *,
    provider: Optional[Any],
    cfg: Optional[Any] = None,
    pipeline_metrics: Optional[Any] = None,
    evidence: Optional[List[Optional[InsightEvidence]]] = None,
) -> Tuple[List[bool], List[int]]:
    """(keep-mask, per-insight tier) from ONE judge call — #1191 preserves the tier the gate already
    computes (FILLER/MINOR/USEFUL/CORE) instead of collapsing it to keep/drop. The tier is stored on
    the Insight so the corpus ranks and tags rather than truncating. Fails OPEN.

    Both lists are order-preserved and index-aligned with ``insight_specs``. When the gate is
    ungated/unsupported/failed the tier is CORE (we kept it, we did not judge it low); when the gate
    rejects everything (fail-open-empty) the REAL tiers are still returned (honest ranking data).
    """
    n = len(insight_specs)
    tiers_raw = _classify_tiers(
        insight_specs,
        provider=provider,
        cfg=cfg,
        pipeline_metrics=pipeline_metrics,
        evidence=evidence,
    )
    if tiers_raw is None:
        return [True] * n, [TIER_CORE] * n
    if not tiers_raw:
        return [], []

    min_tier = int(getattr(cfg, "gi_value_gate_min_tier", DEFAULT_MIN_TIER) or DEFAULT_MIN_TIER)
    keep = [t >= min_tier for t in tiers_raw]
    dropped = sum(1 for k in keep if not k)

    # Never let the gate empty an episode. If nothing clears the bar, the gate is more likely broken
    # (or the rubric mismatched) than the episode genuinely worthless — keep all, real tiers intact.
    if not any(keep):
        logger.warning(
            "value gate rejected ALL %d insights (min_tier=%d); keeping them ungated rather "
            "than emitting an empty episode",
            n,
            min_tier,
        )
        _bump(pipeline_metrics, "gi_value_gate_rejected_all")
        return [True] * n, tiers_raw

    _bump(pipeline_metrics, "gi_value_gate_calls")
    _bump(pipeline_metrics, "gi_insights_dropped_by_value_gate", dropped)
    if dropped:
        logger.info("value gate: dropped %d/%d insights below tier %d", dropped, n, min_tier)
    return keep, tiers_raw


def value_gate_keep_mask(
    insight_specs: List[Tuple[str, str]],
    *,
    provider: Optional[Any],
    cfg: Optional[Any] = None,
    pipeline_metrics: Optional[Any] = None,
    evidence: Optional[List[Optional[InsightEvidence]]] = None,
) -> List[bool]:
    """Which insights clear ``gi_value_gate_min_tier`` — one boolean per insight, in order. Thin
    wrapper over :func:`value_gate_evaluate` (which also returns the tiers). Fails OPEN.

    The caller MUST drop each insight's QUOTES along with it — they are index-aligned and identity
    cannot re-pair them (CPython shares one object for two equal constant tuples, so an episode that
    says the same thing twice would keep the wrong evidence; a quote on the wrong insight is a
    fabricated attribution, worse than the filler the gate removes).
    """
    keep, _ = value_gate_evaluate(
        insight_specs,
        provider=provider,
        cfg=cfg,
        pipeline_metrics=pipeline_metrics,
        evidence=evidence,
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
