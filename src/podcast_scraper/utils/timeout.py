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
    """Observe — but do NOT enforce — a deadline on a block of code.

    .. warning::
       **This cannot interrupt anything.** The wrapped block runs to completion; the
       ``TimeoutError`` is raised only *after* control returns from the ``yield``. A call
       that blocks forever holds this context manager open forever and no exception is
       ever raised.

       Do not use it as protection against hangs. Issue #379 introduced it "to prevent
       hangs" and it prevents none. On 2026-08-12 a production run hung for 4h15m while
       wrapped in a 1200s ``timeout_context``.

    What it actually provides:

    * an ERROR log line once the deadline passes, while the operation is still running —
      i.e. a *detection* signal, which is the useful part; and
    * a ``TimeoutError`` afterwards, useful only for recording that something overran.

    To genuinely bound an operation, in order of preference:

    1. Pass a transport-level timeout to the underlying call (``requests``/``httpx``
       ``timeout=``, SDK deadline parameters). This is the only approach that interrupts a
       blocked socket read, which is where real hangs live.
    2. Run the work in a worker and bound it with ``concurrent.futures``
       ``future.result(timeout=...)``, accepting that the abandoned worker keeps running.
    3. Only as a last resort, a signal-based alarm — Unix-only and main-thread-only.

    Args:
        seconds: Deadline in seconds (None or <= 0 disables observation entirely)
        operation_name: Name used in the deadline log line

    Yields:
        None

    Raises:
        TimeoutError: after the block completes, if the deadline had already passed

    Example:
        >>> with timeout_context(30, "transcription"):  # observes only
        ...     result = transcribe_audio(audio_file, timeout=30)  # this enforces
    """
    if seconds is None or seconds <= 0:
        # No timeout
        yield
        return

    # Use threading.Timer for cross-platform timeout (signal.alarm is Unix-only)
    timeout_occurred = threading.Event()

    def timeout_handler():
        timeout_occurred.set()
        # ERROR, not warning: this is the ONLY signal a caller gets while an operation is
        # overrunning, and it is emitted from a timer thread while the blocked operation is
        # still stuck. During the 2026-08-12 wedge the pipeline produced zero log output for
        # four hours; a line like this one is the difference between a detectable stall and
        # silence. Downstream alerting keys on it.
        logger.error(
            "DEADLINE EXCEEDED: %s has been running longer than %ss and is STILL RUNNING. "
            "This context manager cannot interrupt it — see the docstring. If this repeats, "
            "the fix is a transport-level timeout on the underlying call, not a larger value "
            "here.",
            operation_name,
            seconds,
        )

    timer = threading.Timer(seconds, timeout_handler)
    timer.daemon = True  # never keep the interpreter alive waiting to log a deadline
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
