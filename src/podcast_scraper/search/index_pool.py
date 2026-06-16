"""Process-scoped pool of search-index handles (ADR-099, #995).

Opening a LanceDB backend — ``lancedb.connect`` + opening the segment/insight/aux
tables + loading their IVF-PQ (vector) and FTS (BM25) index readers — costs ~0.8 s on a
99-episode corpus; the actual query on a *warm* table is ~7 ms. The serving path used to
construct a fresh ``LanceDBBackend`` per query (``hybrid_search``) and reload the vector
store per query (``corpus_search``), paying the open cost every time. That is the
"open a database per request" anti-pattern.

This module caches one **warm** handle per index directory and reuses it for the life of
the process, rebuilding only when the on-disk index changes (directory mtime). Handles
are built under a lock and the LanceDB read tables are pre-opened before publication, so
concurrent borrowers (FastAPI's sync-handler threadpool) never trigger a cold open — which
also removes the concurrent-cold-init SIGSEGV. The LanceDB-only / no_index contract
(ADR-098) is unchanged; pooling applies to whichever backend the caller selects.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

logger = logging.getLogger(__name__)

# LanceDB read tables to pre-open when warming a backend (see LanceDBBackend.TABLES).
_LANCE_TIERS: Tuple[str, ...] = ("segment", "insight", "aux")

_lock = threading.Lock()
# resolved index path -> (freshness_token, handle)
_lance_pool: Dict[str, Tuple[float, Any]] = {}


def _freshness_token(index_dir: Path) -> float:
    """Cheap change signal: the index directory's mtime.

    A (re)index via ``cli index-two-tier`` / ``cli index`` rewrites the directory, bumping
    its mtime, which invalidates the cached handle. A stat per call is sub-microsecond.
    Schema-version bumps are handled separately by the caller's ``lance_index_is_stale``
    guard (we never get here for a stale-schema index).
    """
    try:
        return os.path.getmtime(index_dir)
    except OSError:
        return -1.0


def _prewarm_lance(backend: Any) -> None:
    """Open the read tables so the search path is read-only on the backend's table cache."""
    opener = getattr(backend, "_open_if_exists", None)
    if not callable(opener):
        return
    for tier in _LANCE_TIERS:
        try:
            opener(tier)
        except Exception as exc:  # noqa: BLE001 - a missing tier table is fine (opened lazily)
            logger.debug("index_pool: prewarm tier %s skipped: %s", tier, exc)


def get_lance_backend(index_dir: str | os.PathLike[str], build: Callable[[], Any]) -> Any:
    """Return a warm, cached ``LanceDBBackend`` for *index_dir*, rebuilt on index change.

    *build* constructs a fresh backend (injected so this module stays free of the lancedb
    import). The returned backend has its read tables pre-opened.
    """
    path = Path(index_dir)
    key = str(path.resolve())
    token = _freshness_token(path)
    cached = _lance_pool.get(key)
    if cached is not None and cached[0] == token:
        return cached[1]
    with _lock:
        cached = _lance_pool.get(key)  # re-check: another thread may have built it
        if cached is not None and cached[0] == token:
            return cached[1]
        backend = build()
        _prewarm_lance(backend)
        _lance_pool[key] = (token, backend)
        logger.info("index_pool: warmed LanceDB backend for %s", key)
        return backend


def clear() -> None:
    """Drop all pooled handles. For tests and explicit reindex hooks."""
    with _lock:
        _lance_pool.clear()
