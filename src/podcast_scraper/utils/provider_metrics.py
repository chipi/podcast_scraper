"""Provider call metrics tracking utilities.

This module provides utilities for tracking per-call metrics from providers,
including retries, rate limit sleep time, and token usage.
"""

from __future__ import annotations

import importlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from podcast_scraper.utils.log_redaction import format_exception_for_log
from podcast_scraper.utils.retryable_errors import get_retry_reason, is_retryable_error

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ProviderCallMetrics:
    """Metrics from a single provider call (transcription or summarization)."""

    prompt_tokens: Optional[int] = None  # Input tokens used
    completion_tokens: Optional[int] = None  # Output tokens used
    retries: int = 0  # Number of retries attempted
    rate_limit_sleep_sec: float = 0.0  # Time spent sleeping due to rate limits
    estimated_cost: Optional[float] = None  # Estimated cost in USD
    _retry_count: int = field(default=0, init=False, repr=False)  # Internal retry counter
    _rate_limit_sleep_total: float = field(
        default=0.0, init=False, repr=False
    )  # Internal sleep tracker
    _provider_name: str = field(
        default="unknown", init=False, repr=False
    )  # Provider name for logging
    _breaker_config: Optional[Any] = field(
        default=None, init=False, repr=False
    )  # LLMCircuitBreakerConfig derived from cfg, populated lazily.
    _resilience_profile: Optional[Any] = field(
        default=None, init=False, repr=False
    )  # Per-model ResilienceProfile; retry_with_metrics reads retries/backoff from it.

    def set_breaker_config_from_cfg(self, cfg: Any) -> None:
        """Attach the per-model resilience profile AND (if enabled) the circuit-breaker config.

        ADR-100 follow-up: lets cloud providers wire per-provider resilience without threading it
        through every :func:`retry_with_metrics` call site — they already call this. The per-model
        profile (retries/backoff, and the breaker's own threshold/cooldown) is resolved here from
        ``_provider_name`` + the summary model, so gemini-2.5-flash-lite backs off harder than a
        heavier model WITHOUT any call-site changes.
        """
        if cfg is None:
            return
        # Resolve the per-model resilience profile UNCONDITIONALLY (independent of the breaker
        # toggle): its retries/backoff apply on every retry_with_metrics call.
        from .llm_resilience import resolve_resilience

        model = getattr(cfg, f"{self._provider_name}_summary_model", None)
        profile = resolve_resilience(self._provider_name, model)
        self._resilience_profile = profile

        if not getattr(cfg, "llm_circuit_breaker_enabled", False):
            return
        from .llm_circuit_breaker import LLMCircuitBreakerConfig

        # The model's profile wins on breaker threshold/cooldown so flash-lite trips sooner and
        # cools down longer than the global default.
        self._breaker_config = LLMCircuitBreakerConfig(
            enabled=True,
            failure_threshold=profile.breaker_failure_threshold,
            window_seconds=float(getattr(cfg, "llm_circuit_breaker_window_seconds", 30.0)),
            cooldown_seconds=profile.breaker_cooldown_seconds,
        )

    def set_provider_name(self, name: str) -> None:
        """Set provider name for logging.

        Args:
            name: Provider name (e.g., "openai", "gemini")
        """
        self._provider_name = name

    def record_retry(self, sleep_seconds: float = 0.0, reason: str = "") -> None:
        """Record a retry attempt.

        Args:
            sleep_seconds: Time spent sleeping before retry (for rate limits)
            reason: Reason for retry (e.g., "429", "500", "connection_reset")
        """
        self._retry_count += 1
        if sleep_seconds > 0:
            self._rate_limit_sleep_total += sleep_seconds

    def finalize(self) -> None:
        """Finalize metrics (call after operation completes)."""
        self.retries = self._retry_count
        self.rate_limit_sleep_sec = self._rate_limit_sleep_total

    def set_tokens(self, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> None:
        """Set token counts.

        Args:
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
        """
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def set_cost(self, cost: Optional[float]) -> None:
        """Set estimated cost.

        Args:
            cost: Estimated cost in USD
        """
        self.estimated_cost = cost


def apply_estimated_cost_if_missing(
    call_metrics: ProviderCallMetrics,
    *,
    cfg: Any,
    provider_type: str,
    capability: str,
    model: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    audio_minutes: Optional[float] = None,
) -> None:
    """Populate ``estimated_cost`` from pricing YAML when providers omit it (#823)."""
    if call_metrics.estimated_cost is not None:
        return
    if not provider_type or not model:
        return
    try:
        from podcast_scraper.workflow.helpers import calculate_provider_cost

        cost = calculate_provider_cost(
            cfg=cfg,
            provider_type=provider_type,
            capability=capability,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            audio_minutes=audio_minutes,
        )
        if cost is not None:
            record_provider_call_cost(
                call_metrics,
                float(cost),
                cfg=cfg,
                provider_type=provider_type,
                capability=capability,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                audio_minutes=audio_minutes,
            )
    except Exception:
        pass


def record_provider_call_cost(
    call_metrics: ProviderCallMetrics,
    cost: Optional[float],
    *,
    cfg: Any,
    provider_type: str,
    capability: str,
    model: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    audio_minutes: Optional[float] = None,
    triggered_guardrail: bool = False,
) -> None:
    """Set per-call USD, backfill when null, and emit ``llm_cost_event`` (#823 / #804).

    ``triggered_guardrail`` (added ADR-100): forwarded into the
    ``llm_cost_event`` so cost-rollup can pivot on paid-but-rejected
    spend (the cloud provider charged us for a response that tripped a
    response-shape guardrail and got routed to a fallback).
    """
    if cost is not None:
        call_metrics.set_cost(cost)
    else:
        apply_estimated_cost_if_missing(
            call_metrics,
            cfg=cfg,
            provider_type=provider_type,
            capability=capability,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            audio_minutes=audio_minutes,
        )
    final = call_metrics.estimated_cost
    # NOT gated on cost>0 any more: a call with tokens but no known price (an unpriced model, or a
    # capability the pricing lookup can't resolve) must still record its token usage — tokens are
    # ground truth, cost is a projection. emit_llm_cost_event drops only a truly-empty call, stamps
    # run/episode from correlation itself, and emits the Langfuse span (the SINGLE tracing choke
    # point) — so we no longer emit a span here (that would double-count).
    # #1053: the correlation join key for every signal emitted for this LLM call.
    from podcast_scraper.utils import correlation

    run_id = correlation.get_run_id()

    try:
        from podcast_scraper.workflow.cost_monitoring import emit_llm_cost_event

        emit_llm_cost_event(
            cfg,
            provider=provider_type,
            stage=capability,
            model=model,
            estimated_cost_usd=float(final or 0.0),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            run_id=run_id,
            triggered_guardrail=triggered_guardrail,
        )
    except Exception as exc:
        logger.debug("llm_cost_event emission skipped: %s", exc)


def transcription_model_for_cfg(cfg: Any) -> str:
    """Resolve transcription model name for pricing / cost backfill."""
    provider = str(getattr(cfg, "transcription_provider", None) or "whisper")
    if provider == "whisper":
        return str(getattr(cfg, "whisper_model", None) or "base")
    model_field_by_provider = {
        "openai": "openai_transcription_model",
        "gemini": "gemini_transcription_model",
        "mistral": "mistral_transcription_model",
        "deepgram": "deepgram_model",
    }
    field = model_field_by_provider.get(provider, f"{provider}_transcription_model")
    model = getattr(cfg, field, None)
    if model:
        return str(model)
    return ""


def _safe_openai_retryable() -> tuple[type[Exception], ...]:
    """Return retryable OpenAI exception classes with fallback.

    When ``openai`` is mocked in tests the imported names are Mock
    objects, not real exception classes.  Using them in an ``except``
    clause raises ``TypeError``.  This helper catches that and falls
    back to ``(Exception,)`` so ``retry_with_metrics`` still works.
    """
    try:
        from openai import (
            APIError as _APIError,
            RateLimitError as _RLError,
        )

        if (
            isinstance(_RLError, type)
            and issubclass(_RLError, Exception)
            and isinstance(_APIError, type)
            and issubclass(_APIError, Exception)
        ):
            return (_RLError, _APIError, ConnectionError)
    except (ImportError, AttributeError):
        pass
    return (Exception,)


def _safe_gemini_retryable() -> tuple[type[Exception], ...]:
    """Return retryable Google API exception classes with fallback.

    Same rationale as :func:`_safe_openai_retryable` but covers both
    SDKs used by the Gemini provider:

    - ``google.api_core.exceptions`` — legacy ``google-generativeai`` SDK.
    - ``google.genai.errors`` — new ``google-genai`` SDK (the one our
      GeminiProvider uses since #415). Its ``ServerError`` wraps 5xx
      responses including 503 UNAVAILABLE (model-overload throttling).

    Missing the new-SDK ``ServerError`` caused mega_bundled to fall back
    to staged on the very first 503 instead of retrying with backoff
    (discovered during the 10-feed cloud_balanced production run).
    """
    classes: list[type[Exception]] = [ConnectionError]
    try:
        from google.api_core import exceptions as _gexc

        _re = _gexc.ResourceExhausted
        _su = _gexc.ServiceUnavailable
        if (
            isinstance(_re, type)
            and issubclass(_re, Exception)
            and isinstance(_su, type)
            and issubclass(_su, Exception)
        ):
            classes.extend([_re, _su])
    except (ImportError, AttributeError):
        pass
    try:
        from google.genai import errors as _genai_errors

        _se = _genai_errors.ServerError
        if isinstance(_se, type) and issubclass(_se, Exception):
            classes.append(_se)
    except (ImportError, AttributeError):
        pass
    if len(classes) == 1:  # only ConnectionError survived — broaden to be safe
        return (Exception,)
    return tuple(classes)


def _safe_anthropic_retryable() -> tuple[type[Exception], ...]:
    """Return retryable Anthropic exception classes with fallback.

    Same rationale as :func:`_safe_openai_retryable` but for the
    ``anthropic`` SDK used by the Anthropic provider.
    """
    try:
        import anthropic as _anth

        _ae = _anth.APIError
        if isinstance(_ae, type) and issubclass(_ae, Exception):
            return (_ae, ConnectionError, TimeoutError)
    except (ImportError, AttributeError):
        pass
    return (Exception,)


def _safe_mistral_retryable() -> tuple[type[Exception], ...]:
    """Return retryable Mistral exception classes with fallback.

    Same rationale as :func:`_safe_openai_retryable` but for the
    ``mistralai`` SDK used by the Mistral provider.
    """
    for mod_name in ("mistralai.client.errors", "mistralai"):
        try:
            mod = importlib.import_module(mod_name)
            sdk_err = getattr(mod, "SDKError", None)
            if isinstance(sdk_err, type) and issubclass(sdk_err, Exception):
                return (sdk_err, ConnectionError, TimeoutError)
        except (ImportError, AttributeError, TypeError):
            continue
    return (Exception,)


def _safe_deepgram_retryable() -> tuple[type[Exception], ...]:
    """Return retryable Deepgram exception classes with fallback.

    Same rationale as :func:`_safe_openai_retryable` but for the ``deepgram-sdk``.
    """
    for mod_name in ("deepgram.errors", "deepgram"):
        try:
            mod = importlib.import_module(mod_name)
            err = getattr(mod, "DeepgramApiError", None) or getattr(mod, "DeepgramError", None)
            if isinstance(err, type) and issubclass(err, Exception):
                return (err, ConnectionError, TimeoutError)
        except (ImportError, AttributeError, TypeError):
            continue
    return (Exception,)


_CANONICAL_STAGES = (
    "summarization",
    "gi",
    "kg",
    "cleaning",
    "speaker_detection",
    "transcription",
    "diarization",
)
_CANONICAL_STAGE_KEYWORDS = (
    # Order matters. Specific compound tags first, then summary/cleaning, then
    # GIL/KG/speaker/transcribe/diarize. The compound clean+summary bundled
    # call carries BOTH ``clean`` and ``summary`` keywords; we want it grouped
    # under summarization because it IS the summary call — the cleaning step
    # is incidental to that one round-trip.
    ("bundled_clean_summary", "summarization"),
    ("clean_summary", "summarization"),
    ("extraction_bundle", "summarization"),
    ("gil", "gi"),
    ("extract_quotes", "gi"),
    ("score_entailment", "gi"),
    ("gi_", "gi"),
    ("kg_graph", "kg"),
    ("extract_kg", "kg"),
    ("kg", "kg"),
    ("megabundle", "summarization"),
    ("mega_bundle", "summarization"),
    ("summar", "summarization"),
    ("summary", "summarization"),
    ("clean", "cleaning"),
    ("speaker", "speaker_detection"),
    ("transcribe", "transcription"),
    ("transcript", "transcription"),
    ("diarize", "diarization"),
    ("diariz", "diarization"),
)


def canonical_stage(stage: Optional[str]) -> str:
    """Map a provider-specific stage tag to a canonical pipeline stage (#988).

    Provider call sites tag their ``retry_context`` with descriptive stage
    strings like ``gemini_transcribe`` / ``gemini_gil_extract_quotes`` /
    ``deepseek_summarize``. The reliability eval needs canonical
    pipeline-stage buckets (``transcription`` / ``summarization`` / ``gi`` /
    etc.) so 503 rates can be compared across providers.

    Unknown / empty inputs fall through to ``"other"`` so attribution still
    happens but lands in a clearly-labeled catch-all bucket rather than a
    mis-attributed pipeline-stage.
    """
    if not stage:
        return "other"
    s = str(stage).strip().lower()
    if s in _CANONICAL_STAGES:
        return s
    for needle, canonical in _CANONICAL_STAGE_KEYWORDS:
        if needle in s:
            return canonical
    return "other"


def retry_with_metrics(  # noqa: C901
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    metrics: Optional[ProviderCallMetrics] = None,
    jitter: bool = True,
    error_context: str = "default",
    circuit_breaker_config: Optional[Any] = None,
    pipeline_metrics: Optional[Any] = None,
    retry_context: Optional[Dict[str, Any]] = None,
) -> T:
    """Retry a function with exponential backoff, jitter, and metrics tracking.

    Args:
        func: Function to retry (must be callable with no arguments)
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 30.0)
        retryable_exceptions: Tuple of exception types that should trigger retry
                             (default: all exceptions)
        metrics: Optional metrics object to track retries and sleep time
        jitter: Whether to add random jitter to delays (default: True).
                Jitter prevents thundering herd by randomizing retry timing.
                Adds ±10% random variation to delay.
        error_context: Passed to :func:`is_retryable_error` (e.g. ``\"ollama_local\"``).
        retry_context: Optional small dict (stage, episode id, etc.) included in the
            terminal ``provider_retries_exhausted`` log when all attempts fail.

    Returns:
        Result of calling func()

    Raises:
        Exception: The last exception raised by func() if all retries are exhausted

    Note:
        Jitter is applied to prevent multiple clients from retrying simultaneously,
        which can cause a "thundering herd" problem. The jitter adds ±10% random
        variation to the calculated delay.
    """
    # Per-model resilience profile (attached to metrics by set_breaker_config_from_cfg). When
    # present it OVERRIDES the call site's hardcoded retries/backoff, so flash-lite retries
    # more patiently with longer backoff than a heavier model — without touching the ~75 call sites.
    _profile = getattr(metrics, "_resilience_profile", None) if metrics is not None else None
    if _profile is not None:
        max_retries = _profile.max_retries
        initial_delay = _profile.initial_delay
        max_delay = _profile.max_delay

    last_exception: Optional[Exception] = None
    delay = initial_delay
    total_retry_sleep_seconds = 0.0

    # #697: optional per-provider circuit breaker for cloud-API 503 storms.
    # When provided, ``circuit_breaker_config`` is an LLMCircuitBreakerConfig
    # and the provider name is read from ``metrics._provider_name``. Each
    # attempt waits if the breaker is in cooldown; failures with overload
    # status (5xx / 429) are recorded; successes clear the breaker. Module-
    # level lazy import so adding the breaker doesn't widen unit-test imports.
    _breaker: Optional[Any] = None
    _provider_name = (
        getattr(metrics, "_provider_name", "unknown") if metrics is not None else "unknown"
    )
    # Fall back to a breaker config attached to ``metrics`` (ADR-100
    # follow-up) so cloud providers can wire the breaker once on
    # ProviderCallMetrics creation rather than threading it through every
    # call site. Explicit ``circuit_breaker_config`` wins if provided.
    if circuit_breaker_config is None and metrics is not None:
        circuit_breaker_config = getattr(metrics, "_breaker_config", None)
    if circuit_breaker_config is not None and getattr(circuit_breaker_config, "enabled", False):
        from . import llm_circuit_breaker as _llm_breaker_module

        _breaker = _llm_breaker_module

    from . import llm_call_fuse as _llm_fuse

    for attempt in range(max_retries + 1):
        # THE LLM CALL FUSE. Ticked once per attempt (retries cost money too), OUTSIDE the try so a
        # blown budget raises straight through as a hard abort instead of being swallowed by the
        # retry handler below. This is the count ceiling neither the failure breaker nor the HTTP
        # resilience layer provides — it is what would have stopped gpt-5.5's ~3,500-call runaway.
        # A no-op when no budget is installed (unit tests, ad-hoc scripts).
        _llm_fuse.tick()
        try:
            if _breaker is not None:
                _breaker.wait_if_overloaded(
                    _provider_name, circuit_breaker_config, metrics=pipeline_metrics
                )
            result = func()
            if _breaker is not None:
                _breaker.record_success(_provider_name, circuit_breaker_config)
            return result
        except retryable_exceptions as e:
            last_exception = e
            # TERMINAL first — out of money / credit / access. No backoff fixes this, so do NOT
            # retry: raise a clear hard stop (the money/access fuse, mirroring the call-count fuse).
            # This is the class the old binary retryable-or-not check missed — it lumped every
            # "quota" string into retryable and looped on the exact 400 that stopped our Anthropic
            # account mid-run.
            from .llm_error_taxonomy import (
                classify_llm_error,
                LLMErrorClass,
                LLMTerminalError,
                terminal_message,
            )

            if classify_llm_error(e) is LLMErrorClass.TERMINAL:
                msg = terminal_message(_provider_name, e)
                logger.error(msg)
                raise LLMTerminalError(msg) from e
            # #697: record overload-class failures into the circuit breaker so
            # repeated 503s within the window trip the breaker for the next call.
            if _breaker is not None:
                _err_str = str(e)
                _status = 0
                for code in (429, 500, 502, 503, 504):
                    if str(code) in _err_str:
                        _status = code
                        break
                if _status:
                    _breaker.record_failure(
                        _provider_name,
                        circuit_breaker_config,
                        _status,
                        metrics=pipeline_metrics,
                    )
            # Check if error is actually retryable using improved classification
            if not is_retryable_error(e, error_context=error_context):
                # Non-retryable error - re-raise immediately
                logger.debug("Non-retryable error detected: %s", format_exception_for_log(e))
                raise

            if attempt < max_retries:
                # Determine if this is a rate limit error (for special handling)
                error_msg = str(e).lower()
                is_rate_limit = (
                    "429" in str(e)
                    or "rate limit" in error_msg
                    or "quota" in error_msg
                    or "resource exhausted" in error_msg
                )

                # Extract retry_after if available
                sleep_time = delay
                if is_rate_limit and hasattr(e, "retry_after"):
                    try:
                        sleep_time = float(e.retry_after)
                    except (ValueError, TypeError):
                        pass

                # Apply jitter to prevent thundering herd
                if jitter:
                    # Add ±10% random variation to delay
                    jitter_factor = random.uniform(0.9, 1.1)
                    sleep_time = sleep_time * jitter_factor
                    # Ensure sleep_time doesn't go negative or exceed max_delay
                    sleep_time = max(0.0, min(sleep_time, max_delay))

                # Record retry in metrics (use original delay before jitter for metrics)
                if metrics is not None:
                    reason = get_retry_reason(e)
                    # Record the actual sleep time (with jitter) for accurate metrics
                    metrics.record_retry(sleep_seconds=sleep_time, reason=reason)
                    # Log compact retry line as requested
                    provider_name = getattr(metrics, "_provider_name", "unknown")
                    logger.info(
                        f"provider_retry: provider={provider_name} attempt={attempt + 2} "
                        f"sleep={sleep_time:.1f} reason={reason}"
                    )
                # #988: per-stage attribution. Only fires when the caller wired
                # both ``pipeline_metrics`` and a stage hint in ``retry_context``.
                if pipeline_metrics is not None and retry_context:
                    stage_hint = (
                        retry_context.get("stage") if isinstance(retry_context, dict) else None
                    )
                    if stage_hint and hasattr(pipeline_metrics, "record_provider_retry"):
                        reason_for_stage = reason if metrics is not None else get_retry_reason(e)
                        try:
                            pipeline_metrics.record_provider_retry(
                                canonical_stage(stage_hint), reason_for_stage
                            )
                        except Exception as attribution_exc:  # noqa: BLE001
                            logger.debug(
                                "per-stage retry attribution skipped: %s",
                                attribution_exc,
                            )

                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries + 1,
                    format_exception_for_log(e),
                    sleep_time,
                )
                total_retry_sleep_seconds += sleep_time
                time.sleep(sleep_time)
                # Exponential backoff: double the delay, but cap at max_delay
                delay = min(delay * 2, max_delay)
            else:
                ctx_blob = "{}"
                if retry_context:
                    try:
                        ctx_blob = json.dumps(retry_context, default=str, sort_keys=True)
                    except (TypeError, ValueError):
                        ctx_blob = repr(retry_context)
                logger.error(
                    "provider_retries_exhausted: provider=%s attempts=%d "
                    "total_retry_sleep_s=%.3f last_error=%s context=%s",
                    _provider_name,
                    max_retries + 1,
                    total_retry_sleep_seconds,
                    format_exception_for_log(e),
                    ctx_blob,
                )
        except Exception as e:
            # Non-retryable exception - re-raise immediately
            logger.debug("Non-retryable exception: %s", format_exception_for_log(e))
            raise

    # All retries exhausted
    if last_exception:
        raise last_exception

    # This should never be reached, but type checker needs it
    raise RuntimeError("Retry logic error: no exception but function failed")


def openai_compatible_chat_usage_tokens(response: Any) -> Tuple[Optional[int], Optional[int]]:
    """Best-effort prompt/completion token counts from OpenAI-compatible chat responses."""
    if not hasattr(response, "usage") or not response.usage:
        return None, None
    prompt_tokens = getattr(response.usage, "prompt_tokens", None)
    completion_tokens = getattr(response.usage, "completion_tokens", None)
    in_tok = int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None
    out_tok = int(completion_tokens) if isinstance(completion_tokens, (int, float)) else None
    return in_tok, out_tok


def anthropic_message_usage_tokens(response: Any) -> Tuple[Optional[int], Optional[int]]:
    """Best-effort token counts from Anthropic messages API responses."""
    if not hasattr(response, "usage") or not response.usage:
        return None, None
    usage = response.usage
    input_raw = getattr(usage, "input_tokens", None)
    output_raw = getattr(usage, "output_tokens", None)
    in_tok = int(input_raw) if isinstance(input_raw, (int, float)) else None
    out_tok = int(output_raw) if isinstance(output_raw, (int, float)) else None
    return in_tok, out_tok


def gemini_generate_usage_tokens(response: Any) -> Tuple[Optional[int], Optional[int]]:
    """Best-effort token counts from Gemini generate_content responses."""
    if not hasattr(response, "usage_metadata") or not response.usage_metadata:
        return None, None
    usage = response.usage_metadata
    pt = getattr(usage, "prompt_token_count", None)
    ct = getattr(usage, "candidates_token_count", None)
    try:
        in_tok = int(pt) if pt is not None else None
    except (TypeError, ValueError):
        in_tok = None
    try:
        out_tok = int(ct) if ct is not None else None
    except (TypeError, ValueError):
        out_tok = None
    return in_tok, out_tok


def apply_gil_evidence_llm_call_metrics(
    call_metrics: ProviderCallMetrics,
    pipeline_metrics: Any,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    cfg: Optional[Any] = None,
    provider_type: Optional[str] = None,
    model: Optional[str] = None,
    stage: Optional[str] = None,
) -> None:
    """Finalize a GIL evidence LLM call and merge into pipeline metrics.

    Records aggregate GI tokens when both token counts are known; always records
    retry and rate-limit sleep from ``call_metrics`` after :meth:`finalize`.

    When ``cfg`` + ``provider_type`` + ``model`` are provided and
    ``call_metrics.estimated_cost`` has not been set, computes cost from
    ``calculate_provider_cost`` and populates both ``call_metrics`` (for
    per-episode attribution) and the pipeline's ``llm_gi_cost_usd`` aggregate
    (#650 Finding 17). Without those args the legacy behaviour stands —
    callers that already `set_cost` on ``call_metrics`` will still flow that
    value through.

    When ``stage`` is one of ``"extract_quotes"`` or ``"score_entailment"`` and
    the pipeline metrics object exposes ``record_llm_gi_evidence_stage_call``,
    the call is attributed to that substage bucket (#698 Phase 1). The parent
    ``llm_gi_*`` aggregate is updated in either path so legacy dashboards keep
    working.
    """
    if prompt_tokens is not None and completion_tokens is not None:
        call_metrics.set_tokens(prompt_tokens, completion_tokens)
    # Compute and attach cost if the provider didn't already.
    cost_event_emitted = False
    if (
        call_metrics.estimated_cost is None
        and cfg is not None
        and provider_type
        and model
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        try:
            from podcast_scraper.workflow.helpers import calculate_provider_cost

            cost = calculate_provider_cost(
                cfg=cfg,
                provider_type=provider_type,
                capability="summarization",
                model=model,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
            )
            if cost is not None:
                record_provider_call_cost(
                    call_metrics,
                    cost,
                    cfg=cfg,
                    provider_type=provider_type,
                    capability="summarization",
                    model=model,
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                )
                cost_event_emitted = True
        except Exception:
            # Pricing is best-effort at this layer — a missing rate row
            # shouldn't fail a GIL evidence call.
            pass
    if (
        not cost_event_emitted
        and call_metrics.estimated_cost is not None
        and call_metrics.estimated_cost > 0
        and cfg is not None
        and provider_type
        and model
    ):
        try:
            from podcast_scraper.workflow.cost_monitoring import emit_llm_cost_event

            emit_llm_cost_event(
                cfg,
                provider=provider_type,
                stage="summarization",
                model=model,
                estimated_cost_usd=float(call_metrics.estimated_cost),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            logger.debug("llm_cost_event emission skipped: %s", exc)
    call_metrics.finalize()
    if pipeline_metrics is None:
        return
    if hasattr(pipeline_metrics, "record_llm_gi_evidence_call_metrics"):
        pipeline_metrics.record_llm_gi_evidence_call_metrics(call_metrics)
    if prompt_tokens is None or completion_tokens is None:
        return
    gi_cost = getattr(call_metrics, "estimated_cost", None)
    # Substage attribution updates the substage bucket AND the parent aggregate
    # via ``record_llm_gi_call`` internally. Without ``stage``, fall back to the
    # legacy parent-only path so other providers' behaviour is unchanged.
    if stage in ("extract_quotes", "score_entailment") and hasattr(
        pipeline_metrics, "record_llm_gi_evidence_stage_call"
    ):
        pipeline_metrics.record_llm_gi_evidence_stage_call(
            stage, prompt_tokens, completion_tokens, cost_usd=gi_cost
        )
    elif hasattr(pipeline_metrics, "record_llm_gi_call"):
        pipeline_metrics.record_llm_gi_call(prompt_tokens, completion_tokens, cost_usd=gi_cost)


def merge_gil_evidence_call_metrics_on_failure(
    call_metrics: ProviderCallMetrics,
    pipeline_metrics: Any,
) -> None:
    """After a failed GIL evidence LLM call, record retries and rate-limit sleep only."""
    call_metrics.finalize()
    if pipeline_metrics is None:
        return
    if hasattr(pipeline_metrics, "record_llm_gi_evidence_call_metrics"):
        pipeline_metrics.record_llm_gi_evidence_call_metrics(call_metrics)
