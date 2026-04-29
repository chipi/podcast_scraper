"""Provider call metrics tracking utilities.

This module provides utilities for tracking per-call metrics from providers,
including retries, rate limit sleep time, and token usage.
"""

from __future__ import annotations

import importlib
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, TypeVar

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


def retry_with_metrics(
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

    Returns:
        Result of calling func()

    Raises:
        Exception: The last exception raised by func() if all retries are exhausted

    Note:
        Jitter is applied to prevent multiple clients from retrying simultaneously,
        which can cause a "thundering herd" problem. The jitter adds ±10% random
        variation to the calculated delay.
    """
    last_exception: Optional[Exception] = None
    delay = initial_delay

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
    if circuit_breaker_config is not None and getattr(circuit_breaker_config, "enabled", False):
        from . import llm_circuit_breaker as _llm_breaker_module

        _breaker = _llm_breaker_module

    for attempt in range(max_retries + 1):
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

                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries + 1,
                    format_exception_for_log(e),
                    sleep_time,
                )
                time.sleep(sleep_time)
                # Exponential backoff: double the delay, but cap at max_delay
                delay = min(delay * 2, max_delay)
            else:
                logger.error(
                    "All %d attempts failed. Last error: %s",
                    max_retries + 1,
                    format_exception_for_log(e),
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
    """
    if prompt_tokens is not None and completion_tokens is not None:
        call_metrics.set_tokens(prompt_tokens, completion_tokens)
    # Compute and attach cost if the provider didn't already.
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
                call_metrics.set_cost(cost)
        except Exception:
            # Pricing is best-effort at this layer — a missing rate row
            # shouldn't fail a GIL evidence call.
            pass
    call_metrics.finalize()
    if pipeline_metrics is None:
        return
    if hasattr(pipeline_metrics, "record_llm_gi_evidence_call_metrics"):
        pipeline_metrics.record_llm_gi_evidence_call_metrics(call_metrics)
    if (
        prompt_tokens is not None
        and completion_tokens is not None
        and hasattr(pipeline_metrics, "record_llm_gi_call")
    ):
        gi_cost = getattr(call_metrics, "estimated_cost", None)
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
