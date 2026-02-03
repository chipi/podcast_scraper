"""Retry utilities with exponential backoff for transient errors.

This module provides retry functionality with exponential backoff for handling
transient errors such as network failures, timeouts, and temporary I/O issues.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    func: Callable[[], Any],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Any:
    """Retry a function with exponential backoff on transient errors.

    Args:
        func: Function to retry (must be callable with no arguments)
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 30.0)
        retryable_exceptions: Tuple of exception types that should trigger retry
                             (default: all exceptions)

    Returns:
        Result of calling func()

    Raises:
        Exception: The last exception raised by func() if all retries are exhausted
    """
    last_exception: Optional[Exception] = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
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
