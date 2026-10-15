"""One computation per key at a time; concurrent callers share its result.

A heavy, read-only listing (``/api/artifacts``, ``/api/index/stats``) costs seconds of CPU per
call. Each caller computed it independently, so a client that retries before the previous call
finishes adds a whole computation to an already saturated process. The 2026-10-15 post-deploy smoke
did exactly that: 10 s timeouts and 15 s retries piled /api/artifacts up to ~600 s per call with the
API above 100% CPU, and the deploy failed.

Here the first caller computes and everyone who arrives while it runs waits for the same result (or
the same exception). Nothing is kept once it finishes: the next call computes afresh, so this
cannot serve anything staler than a call already in flight.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Hashable, Optional, TypeVar

T = TypeVar("T")


class _Flight:
    __slots__ = ("done", "result", "error")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: object = None
        self.error: Optional[BaseException] = None


_lock = threading.Lock()
_flights: Dict[Hashable, _Flight] = {}


def single_flight(key: Hashable, compute: Callable[[], T]) -> T:
    """Run ``compute`` once for every caller that asks for ``key`` while it is running."""
    with _lock:
        flight = _flights.get(key)
        leader = flight is None
        if flight is None:
            flight = _Flight()
            _flights[key] = flight
    if not leader:
        flight.done.wait()
        if flight.error is not None:
            raise flight.error
        return flight.result  # type: ignore[return-value]
    try:
        result = compute()
        flight.result = result
        return result
    except BaseException as exc:
        flight.error = exc
        raise
    finally:
        with _lock:
            _flights.pop(key, None)
        flight.done.set()


__all__ = ["single_flight"]
