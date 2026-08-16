"""In-process per-principal rate limiter for the OAuth endpoints (RFC-112 T-13 residue).

The shared Caddy/nginx edge already rate-limits per **IP** (fail2ban + `limit_req`); this adds the
per-**principal** dimension the edge can't see — per OAuth *client* on the token endpoint and per
*IP* on Dynamic Client Registration — to blunt token brute-force + DCR spam.

Deliberately simple: a fixed in-memory sliding window guarded by a lock. The player-api runs a
**single** uvicorn worker, so one process holds the whole counter; the only caveat is that counters
**reset on restart** (fine for rate-limiting — a restart is not an attack vector). A crude size cap
bounds memory against unbounded distinct keys. Not a distributed limiter; if the app ever scales to
multiple workers/hosts this must move to a shared store (documented in THREAT_MODEL T-13).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

_LOCK = threading.Lock()
# OrderedDict for O(1) LRU eviction: touched keys move to the end, so the oldest are at the front.
_HITS: "OrderedDict[str, list[float]]" = OrderedDict()
_MAX_KEYS = 20_000  # bound on distinct keys; over this we evict the OLDEST, never clear all


def allow(key: str, *, limit: int, window_s: float) -> bool:
    """Record a hit for ``key``; return whether it is within ``limit`` per ``window_s`` seconds."""
    now = time.monotonic()
    cutoff = now - window_s
    with _LOCK:
        hits = [t for t in _HITS.get(key, ()) if t >= cutoff]
        allowed = len(hits) < limit
        if allowed:
            hits.append(now)
        _HITS[key] = hits
        _HITS.move_to_end(key)  # most-recently-touched → end
        # Evict the OLDEST keys, not the whole table: a flood of distinct (spoofable) keys must not
        # be able to wipe live counters for legitimate principals (a reset attack). H3.
        while len(_HITS) > _MAX_KEYS:
            _HITS.popitem(last=False)
        return allowed


def reset() -> None:
    """Clear all counters (test helper)."""
    with _LOCK:
        _HITS.clear()
