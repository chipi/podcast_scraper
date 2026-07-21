"""Provider-swap fallback for LLM provider calls (RFC-089 #5).

When an operator runs a DGX-hosted (or otherwise local) LLM and the local
backend becomes unreachable, the pipeline must fall back to a cloud provider
rather than degrade silently. This module implements that contract for every
LLM-touching provider method on a single wrapped instance.

Wrapped methods:

- Summarization: ``summarize``, ``summarize_bundled``,
  ``summarize_mega_bundled``, ``summarize_extraction_bundled``.
- GI evidence: ``extract_quotes``, ``extract_quotes_bundled``,
  ``score_entailment``, ``score_entailment_bundled``. The same wrapped
  instance backs all three roles (summary / quote / entailment) when their
  provider config is the same; without this coverage, ``__getattr__`` would
  forward the GI methods to the broken primary and silently lose the
  evidence stack (insights with ``gi_require_grounding: true`` would all
  drop out).

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
from typing import Any, Callable, Dict, List, Sequence, Union

from .. import config
from .base import SummarizationProvider

logger = logging.getLogger(__name__)


_WRAPPED_METHODS = (
    # Summarization
    "summarize",
    "summarize_bundled",
    "summarize_mega_bundled",
    "summarize_extraction_bundled",
    # GI evidence (quotes + entailment, staged + bundled)
    "extract_quotes",
    "extract_quotes_bundled",
    "score_entailment",
    "score_entailment_bundled",
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
        primary: SummarizationProvider,
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
        if name in _WRAPPED_METHODS:
            primary_fn = getattr(self._primary, name, None)
            if primary_fn is None:
                raise AttributeError(name)
            return self._wrap_call(name, primary_fn)
        return getattr(self._primary, name)

    def _wrap_call(self, method_name: str, primary_fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return primary_fn(*args, **kwargs)
            except Exception as primary_exc:  # noqa: BLE001 — fallback contract (RFC-089/RFC-106)
                # Walk the ordered chain; the first tier that succeeds wins. This preserves the
                # RFC-089 contract of falling back on ANY primary failure (the LLM stage does not
                # apply is_infra_failure — a DGX-down summary retries on cloud regardless).
                for fb_name in self._fallback_names:
                    logger.warning(
                        "Primary summarization provider failed on %s; trying fallback tier '%s'. "
                        "Primary error: %s",
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


def _summary_fallback_chain(cfg: config.Config) -> List[str]:
    """The ordered LLM/summary failover ladder for ``cfg`` (RFC-106 / #1198).

    Prefers the registry-emitted ``summary_fallback_providers`` (the source of truth). Falls back to
    the legacy ``degradation_policy.fallback_provider_on_failure`` (RFC-089) as a one-element chain
    for profiles that predate the registry chain. Any tier equal to the primary is dropped — there
    is no point failing over to the provider that just failed.
    """
    primary_name = str(cfg.summary_provider or "").strip().lower()
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


def wrap_with_fallback_if_configured(
    primary: SummarizationProvider,
    cfg: config.Config,
) -> SummarizationProvider:
    """Wrap ``primary`` in ``FallbackAwareSummarizationProvider`` if the config declares an LLM
    failover ladder (registry-emitted ``summary_fallback_providers`` or the legacy
    ``degradation_policy.fallback_provider_on_failure``).

    Returns the primary unchanged when no fallback is configured. The returned object satisfies the
    same protocol as the primary. Provider-agnostic: it wraps whatever the summary primary is —
    DGX-served vLLM (an ``openai``-protocol provider) or ``ollama`` — and fails over to the cloud
    tier(s) in the chain.

    ADR-119: under the **hold** failure strategy this fallover is deliberately suppressed, mirroring
    the ASR/self-hosted factory guard. HOLD optimises *consistency* — the chosen LLM is the only
    LLM, so a DGX/Ollama-served summary must never silently degrade to a cloud provider and produce
    a mixed-backend corpus. The primary is returned unwrapped; the per-provider LLM circuit breaker
    (backoff/hold) still protects the chosen model, and a sustained outage surfaces to the operator
    rather than falling over. The **failover** strategy (serve default) keeps today's
    availability-first fallover. The strategy is a standalone knob defaulted by run context
    (reprocess -> hold) and overridable per profile.
    """
    chain = _summary_fallback_chain(cfg)
    if not chain:
        return primary
    from ..providers.resilience import FailureStrategy, resolve_failure_strategy

    if resolve_failure_strategy(cfg) is FailureStrategy.HOLD:
        logger.info(
            "ADR-119 HOLD strategy: NOT wrapping summary provider '%s' in cross-LLM fallover "
            "(chain %s suppressed) — the chosen model is the only model; consistency over "
            "availability",
            str(cfg.summary_provider or "").strip().lower(),
            chain,
        )
        return primary
    return FallbackAwareSummarizationProvider(primary, chain, cfg)
