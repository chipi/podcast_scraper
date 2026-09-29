"""Timeout utilities for long-running operations.

This module provides timeout enforcement for transcription and summarization
operations to prevent hangs and ensure graceful degradation (Issue #379).
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
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
        # No timeout. Still publish a no-op deadline so a nested `deadline_credit` call does not
        # have to ask whether observation is enabled — it never should have to.
        token = _ACTIVE_DEADLINE.set(None)
        try:
            yield
        finally:
            _ACTIVE_DEADLINE.reset(token)
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

    state = _DeadlineState(
        operation_name=operation_name,
        budget_s=float(seconds),
        occurred=timeout_occurred,
        handler=timeout_handler,
    )
    state.start()
    token = _ACTIVE_DEADLINE.set(state)

    try:
        yield
        if timeout_occurred.is_set():
            raise TimeoutError(
                f"{operation_name} exceeded timeout of {seconds} seconds"
                + (f" (excluding {state.credited_s:.1f}s credited)" if state.credited_s else "")
            )
    finally:
        _ACTIVE_DEADLINE.reset(token)
        state.cancel()


class _DeadlineState:
    """The live timer behind one ``timeout_context``, extendable by work it should not measure.

    Kept internal: callers reach it through :func:`deadline_credit`, which finds it on a
    contextvar. Threading a handle through the thirty-odd parameters between the deadline and the
    stage that needs to credit it was the alternative, and a parameter that must be passed
    correctly at five call sites is a parameter that will not be.
    """

    def __init__(
        self,
        *,
        operation_name: str,
        budget_s: float,
        occurred: threading.Event,
        handler: Callable[[], None],
    ) -> None:
        self._operation_name = operation_name
        self._occurred = occurred
        self._handler = handler
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._expires_at = time.monotonic() + budget_s
        self.credited_s = 0.0

    def start(self) -> None:
        self._arm(self._expires_at - time.monotonic())

    def _arm(self, delay_s: float) -> None:
        timer = threading.Timer(max(delay_s, 0.0), self._handler)
        timer.daemon = True  # never keep the interpreter alive waiting to log a deadline
        self._timer = timer
        timer.start()

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def credit(self, seconds: float, *, reason: str) -> None:
        """Push the deadline out by *seconds* of work it was never meant to measure.

        A CREDIT, NOT A LARGER BUDGET. The budget stays what it was configured to be; this says
        "that much of the elapsed time belongs to something else". The distinction matters because
        the alternative — adding a translation allowance to the configured deadline — requires
        guessing a number nobody has measured yet (S2.10 is the slice that measures it), and a
        wrong guess either raises false alarms or hides real ones.

        WHY THIS EXISTS. ``processing.py``'s deadline wraps summary + GI + KG under the config key
        named ``summarization_timeout``, and every one of the 22 overruns measured on
        2026-08-31 was GI reported under the summariser's name — sending whoever read the alert
        to debug the innocent stage. Translation runs inside the same block and would do exactly
        that again, at a cost nobody has bounded yet.

        If the deadline has ALREADY fired, the credit clears the flag, so no ``TimeoutError`` is
        raised and no overrun is counted. The ERROR line that was already logged stands and
        cannot be unlogged — which is the honest outcome: an operation slow enough to burn the
        whole metadata budget inside translation has earned a log line.
        """
        if seconds <= 0:
            return
        with self._lock:
            self.credited_s += float(seconds)
            self._expires_at += float(seconds)
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            was_set = self._occurred.is_set()
            if was_set:
                self._occurred.clear()
            self._arm(self._expires_at - time.monotonic())
        logger.debug(
            "deadline credit: %s +%.1fs (%s); total credited %.1fs%s",
            self._operation_name,
            seconds,
            reason,
            self.credited_s,
            " — a fired deadline was cleared" if was_set else "",
        )


#: The deadline currently being observed, if any. Set by ``timeout_context``.
_ACTIVE_DEADLINE: contextvars.ContextVar[Optional[_DeadlineState]] = contextvars.ContextVar(
    "podcast_scraper_active_deadline", default=None
)


def deadline_credit(seconds: float, *, reason: str) -> bool:
    """Exclude *seconds* from the enclosing ``timeout_context``. Returns whether it applied.

    A NO-OP WHEN THERE IS NO DEADLINE, which is the common case: the relabel and rediarize
    paths, every unit test, and any run with the deadline disabled. So a caller never has to
    ask whether it is inside an observed block, and there is no branch to get wrong.
    """
    state = _ACTIVE_DEADLINE.get()
    if state is None or seconds <= 0:
        return False
    state.credit(seconds, reason=reason)
    return True


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
