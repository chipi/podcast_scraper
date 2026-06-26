"""Optional Langfuse LLM tracing (#1052).

The AI-quality lens over the pipeline's LLM calls — per-call generation spans
(model / token usage / cost / stage), grouped per run. (No latency yet: the span
is emitted post-call at the cost choke point, which carries no per-call timing —
a phase-2 item once duration is threaded onto ProviderCallMetrics.) Coexists with the
own cost/ops solution (Loki ``llm_cost`` events + ``corpus_manifest.cost_rollup``
+ Sentry); both emit at the same provider cost choke point
(:func:`podcast_scraper.utils.provider_metrics.record_provider_call_cost`), so
adding tracing touched one place rather than all 8 providers.

Enable-when-secret-present (mirrors :mod:`podcast_scraper.utils.sentry_init`):
a **true no-op** unless both ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY``
are set, so dev / CI / offline boots stay silent and the pipeline never pays an
import cost or a network hop it didn't ask for. These are the Langfuse SDK's
own env-var names, so the same keys work for any langfuse tooling. Host-agnostic:
``LANGFUSE_BASE_URL`` (or the SDK's ``LANGFUSE_HOST``) points at Langfuse Cloud or
a self-hosted instance; unset uses the SDK default (cloud).

Every public entrypoint is wrapped so a tracing failure can **never** break a
pipeline run — the worst case is a missing span plus a debug log.
"""

from __future__ import annotations

import atexit
import logging
import os
from typing import Any, Optional

_LOGGER = logging.getLogger(__name__)

# Langfuse SDK-native env-var names (so the same keys drive any langfuse tooling).
_PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
_SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
_BASE_URL_ENVS = ("LANGFUSE_BASE_URL", "LANGFUSE_HOST")

# Lazily-initialised singleton client. ``_init_attempted`` distinguishes
# "not tried yet" from "tried and got None" so a missing SDK / bad keys don't
# re-attempt (and re-log) on every LLM call.
_client: Any = None
_init_attempted = False


def langfuse_enabled() -> bool:
    """True when both Langfuse keys are present (the enable signal)."""
    return bool(
        os.environ.get(_PUBLIC_KEY_ENV, "").strip() and os.environ.get(_SECRET_KEY_ENV, "").strip()
    )


def get_langfuse_client() -> Optional[Any]:
    """Return a cached Langfuse client, or ``None`` on the no-op path.

    ``None`` when: keys unset, the ``langfuse`` SDK isn't installed, or init
    failed. Callers treat ``None`` as "tracing disabled" and do nothing.
    """
    global _client, _init_attempted
    if _init_attempted:
        return _client
    _init_attempted = True

    if not langfuse_enabled():
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        _LOGGER.debug(
            "langfuse tracing requested (keys set) but the `langfuse` SDK is not "
            "installed; install the [langfuse] extra to enable spans"
        )
        return None

    try:
        kwargs: dict[str, Any] = {
            "public_key": os.environ[_PUBLIC_KEY_ENV].strip(),
            "secret_key": os.environ[_SECRET_KEY_ENV].strip(),
        }
        base_url = next(
            (os.environ[k].strip() for k in _BASE_URL_ENVS if os.environ.get(k, "").strip()),
            "",
        )
        if base_url:
            kwargs["base_url"] = base_url
        _client = Langfuse(**kwargs)
        # Pipeline runs are short-lived subprocesses; flush queued spans on exit
        # so a fast run doesn't drop its trailing generations.
        atexit.register(_safe_flush)
    except Exception as exc:  # pragma: no cover - defensive; never break the run
        _LOGGER.debug("langfuse init failed: %s", exc)
        _client = None
    return _client


def _safe_flush() -> None:
    if _client is not None:
        try:
            _client.flush()
        except Exception as exc:  # pragma: no cover - best-effort
            _LOGGER.debug("langfuse flush skipped: %s", exc)


def flush_langfuse() -> None:
    """Flush queued spans (call at run end; also registered via ``atexit``)."""
    _safe_flush()


def emit_langfuse_span(
    *,
    provider: str,
    capability: str,
    model: str,
    cost: Optional[float],
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    run_seed: Optional[str] = None,
    episode_id: Optional[str] = None,
    feed_id: Optional[str] = None,
    triggered_guardrail: bool = False,
    env: Optional[str] = None,
    # Enrichment-layer correlation (RFC-088 / Epic #1101). Future LLM
    # query enrichers pass these so the Langfuse trace is filterable
    # by enricher via ``prod_recent_traces`` + the enrichment MCP tools.
    enricher_id: Optional[str] = None,
    enricher_tier: Optional[str] = None,
) -> None:
    """Emit one ``generation`` observation for a single LLM call.

    Silent no-op when tracing is disabled. ``run_seed`` (the run_id correlation key,
    #1053) deterministically groups all of a run's calls under one Langfuse trace —
    so the trace is addressable as ``create_trace_id(seed=run_seed)`` — and is stamped
    into the span metadata alongside ``episode_id`` so the signals join.

    When ``enricher_id`` is set (LLM-tier query enricher provider calls),
    it lands in the span metadata so Langfuse traces can be filtered
    by enricher in addition to ``run_id``. Never raises.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        trace_context = None
        if run_seed:
            trace_context = {"trace_id": client.create_trace_id(seed=str(run_seed))}

        usage: dict[str, int] = {}
        if prompt_tokens is not None:
            usage["input"] = int(prompt_tokens)
        if completion_tokens is not None:
            usage["output"] = int(completion_tokens)

        metadata: dict[str, Any] = {
            "provider": provider,
            "stage": capability,
            "run_id": run_seed,
            "episode_id": episode_id,
            "feed_id": feed_id,
            "triggered_guardrail": bool(triggered_guardrail),
            "env": env,
        }
        if enricher_id is not None:
            metadata["enricher_id"] = enricher_id
        if enricher_tier is not None:
            metadata["enricher_tier"] = enricher_tier

        observation = client.start_observation(
            trace_context=trace_context,
            name=f"{capability}:{model}",
            as_type="generation",
            model=str(model),
            metadata=metadata,
            usage_details=usage or None,
            cost_details=({"total": float(cost)} if cost is not None else None),
        )
        observation.end()
    except Exception as exc:  # pragma: no cover - defensive; never break the run
        _LOGGER.debug("langfuse span emission skipped: %s", exc)


def _reset_for_tests() -> None:
    """Test hook: drop the cached client so env changes re-init."""
    global _client, _init_attempted
    _client = None
    _init_attempted = False
