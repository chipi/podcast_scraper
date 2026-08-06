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

_LOCK = threading.Lock()
_HITS: dict[str, list[float]] = {}
_MAX_KEYS = (
    20_000  # crude bound: if distinct keys blow past this, drop everything (safe for a limiter)
)


def allow(key: str, *, limit: int, window_s: float) -> bool:
    """Record a hit for ``key``; return whether it is within ``limit`` per ``window_s`` seconds."""
    now = time.monotonic()
    cutoff = now - window_s
    with _LOCK:
        if len(_HITS) > _MAX_KEYS:
            _HITS.clear()
        hits = [t for t in _HITS.get(key, ()) if t >= cutoff]
        if len(hits) >= limit:
            _HITS[key] = hits
            return False
        hits.append(now)
        _HITS[key] = hits
        return True


def reset() -> None:
    """Clear all counters (test helper)."""
    with _LOCK:
        _HITS.clear()
