"""Hard wall-clock deadline primitive for any long-blocking call.

httpx's own read/write timeout has been observed to never fire when a
co-tenant workload (e.g. a vLLM crash-loop, #954) intermittently stalls
the underlying compute: the multipart upload trickles, each accepted
chunk resets httpx's per-write timeout, and the request hangs
indefinitely. The watchdog guarantees the caller regains control even
when the inner call ignores its own timeout.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, cast, TypeVar

T = TypeVar("T")

# Default wall-clock slack added on top of a request's duration-scaled timeout
# before the watchdog gives up (covers upload + response serialisation).
WATCHDOG_GRACE_SEC = 30.0


def run_with_watchdog(fn: Callable[[], T], deadline_sec: float, *, label: str) -> T:
    """Run ``fn()`` under a hard wall-clock deadline; raise ``TimeoutError`` if it
    overruns.

    Runs ``fn`` in a daemon worker thread and waits at most ``deadline_sec``. If the
    call hasn't returned by then we stop waiting and raise — guaranteeing the caller
    regains control even when ``fn`` (e.g. a trickling httpx upload) ignores its own
    timeout. The orphaned worker is a daemon: it holds at most one connection, never
    blocks process exit, and unwinds whenever the underlying call finally errors.
    Exceptions raised inside ``fn`` propagate to the caller unchanged.
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["res"] = fn()
        except BaseException as exc:  # noqa: BLE001 - propagate to caller thread
            box["err"] = exc

    worker = threading.Thread(target=_run, name=label, daemon=True)
    worker.start()
    worker.join(deadline_sec)
    if worker.is_alive():
        raise TimeoutError(
            f"{label} exceeded hard wall-clock deadline {deadline_sec:.0f}s; "
            "abandoning request and failing over"
        )
    if "err" in box:
        raise box["err"]
    return cast(T, box["res"])


__all__ = ["WATCHDOG_GRACE_SEC", "run_with_watchdog"]
