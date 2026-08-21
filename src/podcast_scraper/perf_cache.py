"""Central in-process cache for perf-sensitive, index/corpus-derived reads.

One place for the mtime-token-keyed caches that back the api's hot read paths
(search index stats, digest topic bands, catalog rows / feeds). Previously each
call site hand-rolled its own ``dict + lock + mtime token + clear()``; this
consolidates them so the caching **strategy lives in one place** and every cache
gets consistent invalidation + stats.

Freshness model — an entry self-invalidates when its **token** changes:

* :func:`lance_mtime` — the LanceDB index dir mtime; bumps on **reindex**.
* :func:`corpus_mtime` — the corpus run-summary mtime; bumps on **ingest**.

Pick the token that matches what the cached value derives from (index vs corpus).

In-memory only today. This module is the single seam to add file / TTL / LRU
strategies later without touching any call site. Per-namespace hit/miss/size
stats are exposed via :func:`stats` (surfaced at ``GET /api/ops/cache-stats``).

Not for: warm-handle pools that do more than store a value (``search.index_pool``
pre-opens LanceDB tables; ``corpus_graph`` builds graphs) — those stay bespoke.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Tuple

_LOCK = threading.Lock()
# namespace -> { key -> (token, value) }
_STORE: Dict[str, Dict[Hashable, Tuple[float, Any]]] = {}
# namespace -> [hits, misses, build_seconds_total] — build time lets stats() report whether the
# cache is actually EARNING its keep (time saved on hits) vs just being overhead.
_STATS: Dict[str, list] = {}


def _ns(namespace: str) -> Dict[Hashable, Tuple[float, Any]]:
    store = _STORE.get(namespace)
    if store is None:
        store = {}
        _STORE[namespace] = store
        _STATS[namespace] = [0, 0, 0.0]
    return store


def get_or_compute(namespace: str, key: Hashable, token: float, compute: Callable[[], Any]) -> Any:
    """Return the cached value for ``(namespace, key)`` when its stored token
    matches *token*; otherwise call *compute*, store, and return it.

    ``compute()`` runs OUTSIDE the lock (it may be slow / do IO), so a concurrent
    duplicate compute is possible and acceptable — last writer wins. The reads
    this backs are idempotent, so that is safe.
    """
    with _LOCK:
        store = _ns(namespace)
        hit = store.get(key)
        if hit is not None and hit[0] == token:
            _STATS[namespace][0] += 1
            return hit[1]
        _STATS[namespace][1] += 1
    started = time.perf_counter()
    value = compute()
    elapsed = time.perf_counter() - started
    with _LOCK:
        _ns(namespace)[key] = (token, value)
        _STATS[namespace][2] += elapsed
    return value


def clear(namespace: str | None = None) -> None:
    """Drop one namespace, or the whole cache when *namespace* is None.

    For tests and explicit ingest/reindex hooks; caches also self-invalidate via
    their token, so this is belt-and-suspenders.
    """
    with _LOCK:
        if namespace is None:
            _STORE.clear()
            _STATS.clear()
        else:
            _STORE.pop(namespace, None)
            _STATS.pop(namespace, None)


def stats() -> Dict[str, Dict[str, Any]]:
    """Per-namespace cache health snapshot.

    Beyond hit/miss/size, reports whether the cache is EARNING its keep: ``avg_build_ms`` is a
    miss's compute cost, and ``est_saved_seconds`` (= hits × avg build) is the wall-clock saved by
    not recomputing on hits. High hit rate + high ``avg_build_ms`` = paying off; low hit rate + low
    ``avg_build_ms`` = ~free overhead to drop — "helping or hurting?" without a profiler.
    """
    with _LOCK:
        out: Dict[str, Dict[str, Any]] = {}
        for ns, counters in _STATS.items():
            hits, misses, build_s = counters[0], counters[1], counters[2]
            total = hits + misses
            avg_build_s = (build_s / misses) if misses else 0.0
            out[ns] = {
                "hits": hits,
                "misses": misses,
                "entries": len(_STORE.get(ns, {})),
                "hit_rate_pct": round(100.0 * hits / total, 1) if total else 0.0,
                "build_seconds_total": round(build_s, 3),
                "avg_build_ms": round(avg_build_s * 1000.0, 1),
                "est_saved_seconds": round(hits * avg_build_s, 2),
            }
        return out


def lance_mtime(lance_dir: Path | str) -> float:
    """Reindex signal: the LanceDB index dir mtime (mirrors ``search.index_pool``)."""
    try:
        return os.path.getmtime(lance_dir)
    except OSError:
        return -1.0


def corpus_mtime(root: Path | str) -> float:
    """Ingest signal: the corpus run-summary mtime (rewritten each run), falling
    back to the manifest, then the corpus dir mtime."""
    root = Path(root)
    for name in ("corpus_run_summary.json", "corpus_manifest.json"):
        try:
            # callers pass a validated corpus root (platform anchor or _resolve_corpus output);
            # name is a constant; getmtime only stats it for the cache token.
            # codeql[py/path-injection] -- validated corpus root + constant filename (Type 1).
            return os.path.getmtime(root / name)
        except OSError:
            continue
    try:
        # codeql[py/path-injection] -- same validated corpus root; getmtime stats only (Type 1).
        return os.path.getmtime(root)
    except OSError:
        return -1.0
