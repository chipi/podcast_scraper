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

Wrapping is opt-in via ``degradation_policy.fallback_provider_on_failure`` in
the profile/config. If unset, no wrapping happens and behavior is unchanged.

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
from typing import Any, Callable, Dict, Optional

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
        fallback_provider_name: str,
        cfg: config.Config,
    ) -> None:
        self._primary = primary
        self._fallback_name = str(fallback_provider_name).strip().lower()
        self._cfg = cfg
        self._fallback: Optional[SummarizationProvider] = None
        self._fallback_recorded = False

    def initialize(self) -> None:
        self._primary.initialize()

    def cleanup(self) -> None:
        try:
            self._primary.cleanup()
        finally:
            if self._fallback is not None and hasattr(self._fallback, "cleanup"):
                self._fallback.cleanup()

    def warmup(self, timeout_s: int = 600) -> None:
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
            except Exception as primary_exc:  # noqa: BLE001 — fallback contract
                logger.warning(
                    "Primary summarization provider failed on %s; attempting fallback "
                    "to '%s'. Primary error: %s",
                    method_name,
                    self._fallback_name,
                    primary_exc,
                )
                try:
                    fallback = self._get_or_build_fallback()
                    fallback_fn = getattr(fallback, method_name, None)
                    if fallback_fn is None:
                        logger.error(
                            "Fallback provider '%s' does not implement %s; re-raising primary",
                            self._fallback_name,
                            method_name,
                        )
                        raise primary_exc
                    result = fallback_fn(*args, **kwargs)
                    self._record_fallback_once(kwargs.get("pipeline_metrics"))
                    return result
                except Exception as fallback_exc:  # noqa: BLE001
                    logger.error(
                        "Fallback provider '%s' also failed on %s: %s; re-raising primary",
                        self._fallback_name,
                        method_name,
                        fallback_exc,
                    )
                    raise primary_exc

        wrapper.__name__ = method_name
        return wrapper

    def _get_or_build_fallback(self) -> SummarizationProvider:
        if self._fallback is None:
            from .factory import create_summarization_provider

            logger.info(
                "Building fallback summarization provider '%s' on first failure",
                self._fallback_name,
            )
            fallback = create_summarization_provider(
                self._cfg, provider_type_override=self._fallback_name
            )
            if hasattr(fallback, "initialize"):
                fallback.initialize()
            self._fallback = fallback
        return self._fallback

    def _record_fallback_once(self, pipeline_metrics: Any) -> None:
        if self._fallback_recorded:
            return
        if pipeline_metrics is None:
            return
        record_fn = getattr(pipeline_metrics, "record_llm_summary_fallback_active", None)
        if callable(record_fn):
            try:
                record_fn(self._fallback_name)
                self._fallback_recorded = True
            except Exception as exc:  # noqa: BLE001 — metrics are best-effort
                logger.debug("Failed to record fallback metric: %s", exc)


def wrap_with_fallback_if_configured(
    primary: SummarizationProvider,
    cfg: config.Config,
) -> SummarizationProvider:
    """Wrap ``primary`` in ``FallbackAwareSummarizationProvider`` if the config
    declares ``degradation_policy.fallback_provider_on_failure``.

    Returns the primary unchanged when no fallback is configured. The returned
    object satisfies the same protocol as the primary.
    """
    policy: Dict[str, Any] = getattr(cfg, "degradation_policy", None) or {}
    fallback_name = policy.get("fallback_provider_on_failure") if isinstance(policy, dict) else None
    if not fallback_name:
        return primary
    if str(fallback_name).strip().lower() == str(cfg.summary_provider or "").strip().lower():
        logger.warning(
            "degradation_policy.fallback_provider_on_failure (%s) matches summary_provider "
            "(%s); skipping fallback wrapper (no point falling back to the same provider).",
            fallback_name,
            cfg.summary_provider,
        )
        return primary
    return FallbackAwareSummarizationProvider(primary, fallback_name, cfg)
