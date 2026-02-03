"""Timeout utilities for long-running operations.

This module provides timeout enforcement for transcription and summarization
operations to prevent hangs and ensure graceful degradation (Issue #379).
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when an operation exceeds the timeout."""

    pass


@contextmanager
def timeout_context(seconds: Optional[int], operation_name: str = "operation"):
    """Context manager for enforcing timeouts on operations.

    Args:
        seconds: Timeout in seconds (None disables timeout)
        operation_name: Name of operation for logging

    Yields:
        None

    Raises:
        TimeoutError: If operation exceeds timeout

    Example:
        >>> with timeout_context(30, "transcription"):
        ...     result = transcribe_audio(audio_file)
    """
    if seconds is None or seconds <= 0:
        # No timeout
        yield
        return

    # Use threading.Timer for cross-platform timeout (signal.alarm is Unix-only)
    timeout_occurred = threading.Event()

    def timeout_handler():
        timeout_occurred.set()
        logger.warning(f"Timeout occurred for {operation_name} after {seconds} seconds")

    timer = threading.Timer(seconds, timeout_handler)
    timer.start()

    try:
        yield
        if timeout_occurred.is_set():
            raise TimeoutError(f"{operation_name} exceeded timeout of {seconds} seconds")
    finally:
        timer.cancel()


def with_timeout(
    func: Callable[..., T],
    timeout_seconds: Optional[int],
    operation_name: str = "operation",
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute a function with a timeout.

    Args:
        func: Function to execute
        timeout_seconds: Timeout in seconds (None disables timeout)
        operation_name: Name of operation for logging
        *args: Positional arguments to pass to function
        **kwargs: Keyword arguments to pass to function

    Returns:
        Function result

    Raises:
        TimeoutError: If operation exceeds timeout

    Example:
        >>> result = with_timeout(transcribe_audio, 30, "transcription", audio_file)
    """
    if timeout_seconds is None or timeout_seconds <= 0:
        # No timeout
        return func(*args, **kwargs)

    result: Optional[T] = None
    exception: Optional[Exception] = None
    timeout_occurred = threading.Event()

    def target():
        nonlocal result, exception
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exception = e

    def timeout_handler():
        timeout_occurred.set()
        logger.warning(f"Timeout occurred for {operation_name} after {timeout_seconds} seconds")

    thread = threading.Thread(target=target, daemon=True)
    timer = threading.Timer(timeout_seconds, timeout_handler)

    thread.start()
    timer.start()

    thread.join(timeout=timeout_seconds + 1)  # Add small buffer
    timer.cancel()

    if timeout_occurred.is_set():
        raise TimeoutError(f"{operation_name} exceeded timeout of {timeout_seconds} seconds")

    if exception:
        raise exception

    if result is None:
        raise TimeoutError(f"{operation_name} did not complete within {timeout_seconds} seconds")

    return result
