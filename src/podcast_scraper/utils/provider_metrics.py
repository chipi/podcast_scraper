"""Provider call metrics tracking utilities.

This module provides utilities for tracking per-call metrics from providers,
including retries, rate limit sleep time, and token usage.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar

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


def retry_with_metrics(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    metrics: Optional[ProviderCallMetrics] = None,
    jitter: bool = True,
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

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            # Check if error is actually retryable using improved classification
            if not is_retryable_error(e):
                # Non-retryable error - re-raise immediately
                logger.debug(f"Non-retryable error detected: {e}")
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
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {sleep_time:.1f}s..."
                )
                time.sleep(sleep_time)
                # Exponential backoff: double the delay, but cap at max_delay
                delay = min(delay * 2, max_delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
        except Exception as e:
            # Non-retryable exception - re-raise immediately
            logger.debug(f"Non-retryable exception: {e}")
            raise

    # All retries exhausted
    if last_exception:
        raise last_exception

    # This should never be reached, but type checker needs it
    raise RuntimeError("Retry logic error: no exception but function failed")
