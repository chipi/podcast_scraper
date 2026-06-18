"""Exception classes shared by the resilience primitives.

Currently just the ``TimeoutLike`` tuple — extracted here so consumers can
import a single symbol without pulling in the entire ``sockets`` module.
"""

from __future__ import annotations

# Timeout-class errors: the server is slow/contended and (likely) still working
# the request, so retrying would double the load. Resolved lazily so a missing
# httpx doesn't break import.
try:  # pragma: no cover - import guard
    import httpx as _httpx

    TimeoutLike: tuple[type[Exception], ...] = (_httpx.TimeoutException, TimeoutError)
except ImportError:  # pragma: no cover
    TimeoutLike = (TimeoutError,)


__all__ = ["TimeoutLike"]
