"""Provider-swap fallback for LLM provider calls (RFC-089 #5).

When an operator runs a DGX-hosted (or otherwise local) LLM and the local
backend becomes unreachable, the pipeline must fall back to a cloud provider
rather than degrade silently. This module implements that contract for every
LLM-touching provider method on a single wrapped instance.

Wrapped methods: EVERY callable on the provider except an explicit exemption list
(``_NEVER_WRAPPED`` — lifecycle/introspection, plus transcription, which carries its own
RFC-106 ladder). That includes summarization, GI insight generation, GI evidence (quotes and
entailment), KG extraction, transcript cleaning, insight classification, speaker and host
detection, and anything added later.

This was previously an allowlist of eight method names, which meant a provider could be
"wrapped with failover" and still have none on ``generate_insights``, ``detect_speakers``,
``extract_kg_graph``, ``clean_transcript``, ``classify_insights``, ``detect_hosts``,
``analyze_patterns`` or ``complete_text`` — ``__getattr__`` forwarded them straight to the
broken primary. Half the LLM surface was unprotected and nothing said so. The rule is inverted
now: failover is the default and opting out is explicit.

Wrapping is opt-in via the failover ladder in the profile/config. RFC-106 (#1198): the source of
truth is the registry-emitted ``summary_fallback_providers`` (an ordered chain); the legacy
``degradation_policy.fallback_provider_on_failure`` is honoured as a one-element chain for profiles
that predate it. If neither is set, no wrapping happens and behavior is unchanged.

Call-site coverage. The wrapper kicks in at every place a provider instance is
constructed for a fallback-eligible role:

- ``workflow/orchestration.py::_create_summarization_provider`` — primary
  summary provider; reused for KG when ``kg_extraction_provider`` matches
  ``summary_provider`` (the default case).
- ``gi/deps.py::create_gil_evidence_providers`` — quote / entailment providers
  built fresh when their config differs from ``summary_provider``.
- ``workflow/metadata_generation.py`` — KG provider built fresh when
  ``kg_extraction_provider`` differs from ``summary_provider``.

The wrapper is transparent to callers: pass-through of non-protocol attributes
(e.g. ``cleaning_processor``) goes to the primary via ``__getattr__``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from .. import config
from .base import SummarizationProvider

logger = logging.getLogger(__name__)

#: The wrapper is provider-shaped, not summarization-shaped: it wraps whatever provider it is
#: given and returns something with the same interface. Typed as a TypeVar so a wrapped speaker
#: detector still type-checks as a speaker detector — the class name is historical (it began as
#: a summary-only feature) but the capability is generic, and the types now say so.
_ProviderT = TypeVar("_ProviderT")


#: Methods that must NOT be failed over. Everything else on the provider is wrapped.
#:
#: THIS USED TO BE THE OTHER WAY AROUND — an allowlist of eight method names — and that is a
#: defect that cannot be seen by reading the list, only by reading what is missing from it.
#: ``__getattr__`` forwarded every unlisted method straight to the primary, so a provider that
#: was "wrapped with failover" still had NO failover on:
#:
#:     generate_insights   classify_insights   clean_transcript   extract_kg_graph
#:     detect_speakers     detect_hosts        analyze_patterns   complete_text
#:
#: Eight LLM calls protected, eight unprotected, and nothing anywhere said so. When the
#: OpenRouter account went over its weekly limit every one of those failed outright while
#: summarisation quietly failed over and survived — which is how ``generate_insights`` came to
#: write placeholder artifacts across production episodes, and how a prod speaker-detection job died
#: with "no budget/credit left on this key" instead of failing over to native DeepSeek.
#:
#: An allowlist puts the burden on whoever adds the NEXT LLM method to remember this file. A
#: denylist puts the burden on whoever adds a method that must not fail over — a much rarer and
#: much more obvious thing to notice. Failover is now the default; opting out is explicit.
_NEVER_WRAPPED = frozenset(
    {
        # Lifecycle and introspection: no LLM call is made, and wrapping them would construct a
        # whole fallback provider just to answer a question about the primary.
        "initialize",
        "cleanup",
        "warmup",
        "clear_cache",
        "is_initialized",
        "get_capabilities",
        "get_pricing",
        # Transcription carries its OWN RFC-106 ladder (``transcription_fallback_providers``,
        # applied in transcription/factory.py). Wrapping it here would stack two independent
        # ladders on one call and fail over twice by different rules.
        "transcribe",
        "transcribe_with_segments",
    }
)


class FallbackAwareSummarizationProvider:
    """Wraps a primary summarization provider with cloud-fallback behavior.

    On any exception from a wrapped ``summarize*`` call, lazily builds a
    secondary provider of type ``fallback_provider_name`` (using the same
    config) and retries. If the fallback also fails, the original primary
    exception is re-raised so the existing degradation policy applies.

    The fallback provider is built lazily — DGX-hosted runs that don't fail
    pay no construction cost. Once built, it's reused for the rest of the run.

    Notes:

    - ``warmup`` is invoked on the primary only — the fallback is cloud and
      doesn't need warmup. If the primary's warmup itself throws, that surfaces
      to the caller (current orchestration treats warmup failures as warnings,
      not fatal).
    - Non-protocol attributes (e.g. ``cleaning_processor``, ``call_metrics``)
      are forwarded to the primary via ``__getattr__`` so the rest of the
      pipeline can't tell the wrapper is there.
    - Calls ``pipeline_metrics.record_llm_summary_fallback_active(fallback_name)``
      once per fallback activation (the first time fallback succeeds for a run).
      Subsequent calls in the same run still go through fallback but don't
      re-record the counter — the per-run "did fallback fire" signal is what
      operators need, not per-call.
    """

    def __init__(
        self,
        primary: Any,
        fallback_provider_names: Union[str, Sequence[str]],
        cfg: config.Config,
    ) -> None:
        # RFC-106 (#1198): the fallback is an ORDERED chain, tried in sequence. A bare string is
        # accepted for back-compat (the RFC-089 single-fallback shape) and normalised to one tier.
        if isinstance(fallback_provider_names, str):
            fallback_provider_names = [fallback_provider_names]
        self._primary = primary
        self._fallback_names: List[str] = [
            str(n).strip().lower() for n in fallback_provider_names if str(n).strip()
        ]
        self._cfg = cfg
        self._fallbacks: Dict[str, SummarizationProvider] = {}
        self._fallback_recorded = False

    def initialize(self) -> None:
        """Initialize the primary provider. Fallback tiers are built lazily on first failure."""
        self._primary.initialize()

    def cleanup(self) -> None:
        """Release primary, then every built fallback tier. Each release is independent — one tier
        raising on cleanup does not leak the others."""
        try:
            self._primary.cleanup()
        finally:
            for name, fb in self._fallbacks.items():
                if hasattr(fb, "cleanup"):
                    try:
                        fb.cleanup()
                    except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                        logger.warning("fallback tier '%s' cleanup failed: %s", name, exc)

    def warmup(self, timeout_s: int = 600) -> None:
        """Warm up the primary if it supports it. Fallback is cloud, no warmup needed."""
        warmup_fn = getattr(self._primary, "warmup", None)
        if callable(warmup_fn):
            warmup_fn(timeout_s=timeout_s)

    def __getattr__(self, name: str) -> Any:
        """Forward to the primary, wrapping every LLM call in the failover chain.

        Default-wrap, not default-forward. Attributes the wrapper does not itself define land
        here, and anything callable that is not explicitly exempt gets the chain — so a provider
        method added tomorrow is protected without anyone editing this file.
        """
        attr = getattr(self._primary, name)
        if name.startswith("_") or name in _NEVER_WRAPPED or not callable(attr):
            # Data attributes (``cleaning_processor``, ``call_metrics``) and the exempt
            # lifecycle/transcription methods pass through untouched, as before.
            return attr
        return self._wrap_call(name, attr)

    def _walk_fallback_chain(
        self, method_name: str, primary_exc: Exception, args: Any, kwargs: Any
    ) -> Any:
        """Walk the ordered fallback chain for one call; first tier that succeeds wins.

        Shared by the exception path (``_wrap_call``) and the explicit escalation path
        (``call_via_fallback``) so there is exactly ONE chain-walking implementation.
        Raises ``primary_exc`` when the chain is empty or exhausted.
        """
        for fb_name in self._fallback_names:
            logger.warning(
                "Primary provider failed on %s(); trying fallback tier '%s'. Primary error: %s",
                method_name,
                fb_name,
                primary_exc,
            )
            try:
                fallback = self._get_or_build_fallback(fb_name)
                fallback_fn = getattr(fallback, method_name, None)
                if fallback_fn is None:
                    logger.error(
                        "Fallback tier '%s' does not implement %s; trying next tier",
                        fb_name,
                        method_name,
                    )
                    continue
                result = fallback_fn(*args, **kwargs)
                self._record_fallback_once(kwargs.get("pipeline_metrics"), fb_name)
                return result
            except Exception as fallback_exc:  # noqa: BLE001
                logger.error(
                    "Fallback tier '%s' also failed on %s: %s; trying next tier",
                    fb_name,
                    method_name,
                    fallback_exc,
                )
                continue
        # Chain exhausted (or empty): surface the primary error so the existing degradation
        # policy applies.
        raise primary_exc

    def call_via_fallback(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke ``method_name`` on the fallback chain ONLY — the primary is not tried.

        The escalation path for failures the exception contract cannot see (#1878): a schema
        failure happens at PARSE time, after the primary's call returned "successfully", so
        ``_wrap_call`` never fires — which is how an episode shipped with no summary while a
        healthy fallback tier sat unused. The caller (metadata_generation, after the ADR-148
        re-roll also failed) uses this to hand the same request to the next vendor instead of
        giving up. Raises when the chain is empty or every tier fails, so the caller's existing
        recoverable-failure path still applies.
        """
        return self._walk_fallback_chain(
            method_name,
            RuntimeError(
                f"call_via_fallback({method_name}): no fallback tier available or all tiers failed"
            ),
            args,
            kwargs,
        )

    def _wrap_call(self, method_name: str, primary_fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return primary_fn(*args, **kwargs)
            except Exception as primary_exc:  # noqa: BLE001 — fallback contract (RFC-089/RFC-106)
                # One exception class does not belong to the chain: a failure the CALLER can fix
                # by resending a smaller request. Failing over answers it from a weaker model
                # when the primary was willing and able — strictly worse than the caller's own
                # retry, and it hides the overflow because the run then looks successful.
                #
                # This is deliberately narrow. It keys off a marker the exception sets, not the
                # method name, because the SAME method must still fail over when the endpoint is
                # actually down. See ``BundleOutputBudgetExceeded``.
                if getattr(primary_exc, "caller_can_retry_smaller", False):
                    logger.warning(
                        "%s() hit a caller-recoverable limit; NOT failing over so the caller can "
                        "retry smaller: %s",
                        method_name,
                        primary_exc,
                    )
                    raise
                # Walk the ordered chain; the first tier that succeeds wins. This preserves the
                # RFC-089 contract of falling back on ANY primary failure (the LLM stage does not
                # apply is_infra_failure — a DGX-down summary retries on cloud regardless).
                return self._walk_fallback_chain(method_name, primary_exc, args, kwargs)

        wrapper.__name__ = method_name
        return wrapper

    def _get_or_build_fallback(self, name: str) -> SummarizationProvider:
        fallback = self._fallbacks.get(name)
        if fallback is None:
            from .factory import create_summarization_provider

            logger.info("Building fallback summarization provider '%s' on first failure", name)
            fallback = create_summarization_provider(self._cfg, provider_type_override=name)
            if hasattr(fallback, "initialize"):
                fallback.initialize()
            self._fallbacks[name] = fallback
        return fallback

    def _record_fallback_once(self, pipeline_metrics: Any, fallback_name: str) -> None:
        if self._fallback_recorded:
            return
        if pipeline_metrics is None:
            return
        record_fn = getattr(pipeline_metrics, "record_llm_summary_fallback_active", None)
        if callable(record_fn):
            try:
                record_fn(fallback_name)
                self._fallback_recorded = True
            except Exception as exc:  # noqa: BLE001 — metrics are best-effort
                logger.debug("Failed to record fallback metric: %s", exc)


def _summary_fallback_chain(cfg: config.Config, primary_name: Optional[str] = None) -> List[str]:
    """The ordered LLM/summary failover ladder for ``cfg`` (RFC-106 / #1198).

    Prefers the registry-emitted ``summary_fallback_providers`` (the source of truth). Falls back to
    the legacy ``degradation_policy.fallback_provider_on_failure`` (RFC-089) as a one-element chain
    for profiles that predate the registry chain. Any tier equal to the primary is dropped — there
    is no point failing over to the provider that just failed.

    ``primary_name`` names the provider being wrapped. It defaults to ``cfg.summary_provider``
    because this ladder began as a summary-only feature, but the wrapper now protects the speaker
    detector too, and that stage's primary is ``speaker_detector_provider``. Passing the real
    primary is what makes "drop the tier that just failed" mean the right thing for each stage.
    Read with ``getattr``: a config (or a test double) need not carry every provider field.
    """
    if primary_name is None:
        primary_name = str(getattr(cfg, "summary_provider", None) or "")
    primary_name = str(primary_name or "").strip().lower()
    chain = [
        str(p).strip().lower()
        for p in (getattr(cfg, "summary_fallback_providers", None) or [])
        if str(p).strip()
    ]
    if not chain:
        policy: Dict[str, Any] = getattr(cfg, "degradation_policy", None) or {}
        legacy = policy.get("fallback_provider_on_failure") if isinstance(policy, dict) else None
        if legacy:
            chain = [str(legacy).strip().lower()]
    return [name for name in chain if name and name != primary_name]


def preflight_fallback_chain(
    cfg: config.Config,
    primary_name: Optional[str] = None,
) -> List[Tuple[str, Optional[str]]]:
    """Check every declared failover tier can actually be BUILT. Returns the broken ones.

    Each entry is ``(tier_name, error_message)`` for a tier that could not be constructed; an
    empty list means the ladder is sound.

    WHY THIS EXISTS: a ladder is only worth what it can construct, and the tiers are built
    LAZILY — ``_get_or_build_fallback`` runs on the first failure, which is the single worst
    moment to discover a missing credential. Proven live on 2026-08-16, acceptance run feed 1::

        WARNING  Primary provider failed on extract_quotes_bundled(); trying fallback tier
                 'deepseek'
        ERROR    Fallback tier 'deepseek' also failed: DeepSeek API key required

    Twice in one episode. 612fd451 had configured a cross-vendor ladder across 11 profiles, all
    pointing at the ``deepseek`` tier, and none of them could be built. The ladder detected the
    failure, logged it, and recovered nothing. Nothing before this ever asked whether it could.

    It also answered a question left open by #20: the ladder fires on ORDINARY provider faults
    (malformed JSON here), not only on budget exhaustion — so a ladder that cannot build is a
    routine liability, not a doomsday one.

    Construction only. This deliberately does NOT make a network call: it must be cheap enough
    to run unconditionally at startup and must not spend money or fail on a transient blip. A
    missing key raises at construction, which is the failure mode actually observed.
    """
    chain = _summary_fallback_chain(cfg, primary_name)
    broken: List[Tuple[str, Optional[str]]] = []
    for name in chain:
        try:
            from .factory import create_summarization_provider

            create_summarization_provider(cfg, provider_type_override=name)
        except Exception as exc:  # noqa: BLE001 — any construction failure disqualifies the tier
            broken.append((name, str(exc) or exc.__class__.__name__))
    return broken


def log_fallback_chain_preflight(
    cfg: config.Config,
    primary_name: Optional[str] = None,
    *,
    stage: str = "summary",
) -> List[Tuple[str, Optional[str]]]:
    """Run the pre-flight and say so LOUDLY, at startup, once. Returns the broken tiers.

    Warns rather than exits. A ladder is a safety net: an unbuildable one means the run has no
    protection, which is bad — but a run that would otherwise succeed on its primary should not
    be blocked by it. That is the opposite trade-off from ffmpeg (#26), where the missing
    component breaks every episode outright rather than removing a contingency.
    """
    broken = preflight_fallback_chain(cfg, primary_name)
    if not broken:
        return broken
    for name, error in broken:
        logger.warning(
            "FAILOVER LADDER BROKEN [%s]: tier '%s' is declared but CANNOT BE BUILT (%s). "
            "If the primary fails, this tier will recover nothing — the run is unprotected. "
            "Fix the credential/config now, not during an outage.",
            stage,
            name,
            error,
        )
    return broken


def wrap_with_fallback_if_configured(
    primary: _ProviderT,
    cfg: config.Config,
    primary_name: Optional[str] = None,
) -> _ProviderT:
    """Wrap ``primary`` in ``FallbackAwareSummarizationProvider`` if the config declares an LLM
    failover ladder (registry-emitted ``summary_fallback_providers`` or the legacy
    ``degradation_policy.fallback_provider_on_failure``).

    Returns the primary unchanged when no fallback is configured. The returned object satisfies the
    same protocol as the primary. Provider-agnostic: it wraps whatever the summary primary is —
    DGX-served vLLM (an ``openai``-protocol provider) or ``ollama`` — and fails over to the cloud
    tier(s) in the chain.

    ADR-122: under the **hold** failure strategy this fallover is deliberately suppressed, mirroring
    the ASR/self-hosted factory guard. HOLD optimises *consistency* — the chosen LLM is the only
    LLM, so a DGX/Ollama-served summary must never silently degrade to a cloud provider and produce
    a mixed-backend corpus. The primary is returned unwrapped; the per-provider LLM circuit breaker
    (backoff/hold) still protects the chosen model, and a sustained outage surfaces to the operator
    rather than falling over. The **failover** strategy (serve default) keeps today's
    availability-first fallover. The strategy is a standalone knob defaulted by run context
    (reprocess -> hold) and overridable per profile.
    """
    chain = _summary_fallback_chain(cfg, primary_name)
    if not chain:
        return primary
    from ..providers.resilience import FailureStrategy, resolve_failure_strategy

    if resolve_failure_strategy(cfg) is FailureStrategy.HOLD:
        logger.info(
            "ADR-122 HOLD strategy: NOT wrapping summary provider '%s' in cross-LLM fallover "
            "(chain %s suppressed) — the chosen model is the only model; consistency over "
            "availability",
            # getattr, not attribute access: this wrapper is now applied to the speaker detector
            # too, and is reachable with config objects (and test doubles) that carry no
            # ``summary_provider``. A log line must not be the thing that fails a run.
            str(getattr(cfg, "summary_provider", None) or "").strip().lower(),
            chain,
        )
        return primary
    # The wrapper is a transparent proxy: every attribute it does not define forwards to the
    # primary, so it satisfies the primary's interface structurally without inheriting it. mypy
    # cannot express "same duck as the input", hence the cast — the TypeVar is what keeps the
    # CALLER correctly typed (a wrapped speaker detector stays a speaker detector).
    return cast("_ProviderT", FallbackAwareSummarizationProvider(primary, chain, cfg))
