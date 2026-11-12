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

import logging
import os
import queue
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Iterator, Tuple

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
# namespace -> { key -> (token, value) }
_STORE: Dict[str, Dict[Hashable, Tuple[float, Any]]] = {}
# namespace -> [hits, misses, build_seconds_total, stale_served] — build time lets stats() report
# whether the cache is actually EARNING its keep (time saved on hits) vs just being overhead.
_STATS: Dict[str, list] = {}
# (namespace, key, token) -> set when the build in progress for it finishes (single-flight).
_IN_FLIGHT: Dict[Tuple[str, Hashable, float], threading.Event] = {}
# depth > 0 on a thread inside fresh(): no stale values for it.
_FRESH = threading.local()
# Stale rebuilds queued for the one background builder, not yet started: flight key -> compute.
# A caller that needs the value now (inside fresh(), or without stale_ok) claims it and builds it
# itself, so a build that needs a key queued behind it cannot deadlock the builder.
_QUEUED: Dict[Tuple[str, Hashable, float], Callable[[], Any]] = {}
_ORDER: "queue.SimpleQueue[Tuple[str, Hashable, float]]" = queue.SimpleQueue()
_BUILDER: list[threading.Thread] = []


def _ns(namespace: str) -> Dict[Hashable, Tuple[float, Any]]:
    store = _STORE.get(namespace)
    if store is None:
        store = {}
        _STORE[namespace] = store
        _STATS[namespace] = [0, 0, 0.0, 0]
    return store


def cache_peek(namespace: str, key: Hashable, token: float) -> Any:
    """The cached value for ``(namespace, key)`` if its token matches, else ``None``.

    A lock-free read for callers that guard an EXPENSIVE ``compute()`` behind their own
    single-flight lock: peek first, and only contend for that lock on a genuine miss. Without it a
    warm caller would queue behind whoever is currently building.

    Counts as neither a hit nor a miss — the caller's own :func:`get_or_compute` records the
    outcome, and double-counting would make the hit rate meaningless. ``None`` is indistinguishable
    from a cached ``None``; no caller stores one, and the cost of getting that wrong is one extra
    compute, not a wrong answer.
    """
    with _LOCK:
        hit = _ns(namespace).get(key)
    return hit[1] if hit is not None and hit[0] == token else None


@contextmanager
def fresh() -> Iterator[None]:
    """Inside this block every :func:`get_or_compute` returns a value for the CURRENT token.

    For the cache warmer, whose job is building current values, and for every ``compute()`` (it
    runs inside one), so a value cached under a new token is never built from a stale input.
    """
    depth = getattr(_FRESH, "depth", 0)
    _FRESH.depth = depth + 1
    try:
        yield
    finally:
        _FRESH.depth = depth


def get_or_compute(
    namespace: str,
    key: Hashable,
    token: float,
    compute: Callable[[], Any],
    *,
    stale_ok: bool = False,
) -> Any:
    """Return the cached value for ``(namespace, key)`` when its stored token
    matches *token*; otherwise call *compute*, store, and return it.

    ``compute()`` runs OUTSIDE the lock (it may be slow / do IO). SINGLE-FLIGHT: a caller that
    misses while another is already building the same ``(namespace, key, token)`` waits for that
    build instead of starting its own. Duplicates looked harmless for a 100 ms build, but the
    corpus digest's cold build is seconds of embedding-model and LanceDB warm-up, and on the
    2026-10-03 deploys every retry of the post-deploy smoke started another one: the duplicates
    starved each other (api at 160-180% CPU) and the first answer took 131 s against ~17 s alone,
    so the smoke's digest probe never passed on a cold api. If the build raises, waiters fall
    back to computing themselves.

    ``stale_ok``: when the entry holds an OLDER token's value, return that value at once and queue
    the current one for the background builder, unless the caller is inside :func:`fresh`. One
    builder, not a thread per key: several rebuilds at once competed with the requests they were
    meant to spare. For corpus-wide caches the warmer rebuilds after each ingest: an ingest
    invalidates all of them at the same moment, and requests that arrived before the warmer
    rebuilt them each built several in turn, 8-10 s for Home's burst on a prod-shaped corpus
    against 1.6 s warm (2026-11-13), and 14.4 s for one /your-week on prod (2026-11-07).
    """
    flight_key = (namespace, key, token)
    while True:
        with _LOCK:
            store = _ns(namespace)
            hit = store.get(key)
            if hit is not None and hit[0] == token:
                _STATS[namespace][0] += 1
                return hit[1]
            if stale_ok and hit is not None and not getattr(_FRESH, "depth", 0):
                _STATS[namespace][3] += 1
                if flight_key not in _IN_FLIGHT:
                    _STATS[namespace][1] += 1
                    _IN_FLIGHT[flight_key] = threading.Event()
                    _QUEUED[flight_key] = compute
                    _ORDER.put(flight_key)
                    _ensure_builder()
                return hit[1]
            in_flight = _IN_FLIGHT.get(flight_key)
            if in_flight is not None and _QUEUED.pop(flight_key, None) is not None:
                done = in_flight  # queued, not started: build it here rather than wait on it
                break
            if in_flight is None:
                _STATS[namespace][1] += 1
                done = threading.Event()
                _IN_FLIGHT[flight_key] = done
                break
        in_flight.wait()
        with _LOCK:
            hit = _ns(namespace).get(key)
            if hit is not None and hit[0] == token:
                _STATS[namespace][0] += 1
                return hit[1]
            if flight_key in _IN_FLIGHT:
                continue  # another waiter already took over the build, or it is queued
            _STATS[namespace][1] += 1
            done = threading.Event()
            _IN_FLIGHT[flight_key] = done
            break  # the build we waited on failed: compute ourselves
    return _build(namespace, key, token, compute, flight_key, done)


def _build(
    namespace: str,
    key: Hashable,
    token: float,
    compute: Callable[[], Any],
    flight_key: Tuple[str, Hashable, float],
    done: threading.Event,
) -> Any:
    try:
        started = time.perf_counter()
        with fresh():
            value = compute()
        elapsed = time.perf_counter() - started
        with _LOCK:
            _ns(namespace)[key] = (token, value)
            _STATS[namespace][2] += elapsed
        return value
    finally:
        with _LOCK:
            _IN_FLIGHT.pop(flight_key, None)
        done.set()


def _ensure_builder() -> None:
    """Start the background builder if it is not running. Call with ``_LOCK`` held."""
    if _BUILDER and _BUILDER[0].is_alive():
        return
    thread = threading.Thread(target=_builder_loop, name="perf-cache-rebuild", daemon=True)
    _BUILDER[:] = [thread]
    thread.start()


def _builder_loop() -> None:
    while True:
        flight_key = _ORDER.get()
        with _LOCK:
            compute = _QUEUED.pop(flight_key, None)
            done = _IN_FLIGHT.get(flight_key)
        if compute is None or done is None:
            continue  # a caller that needed it now claimed it
        namespace, key, token = flight_key
        try:
            _build(namespace, key, token, compute, flight_key, done)
        except Exception:  # noqa: BLE001 - the stale value keeps serving; the next caller retries
            logger.warning("perf_cache: background rebuild of %s failed", namespace, exc_info=True)


def wait_for_rebuilds(timeout: float = 30.0) -> bool:
    """Block until no build is queued or in progress; ``False`` if *timeout* passed first.

    For tests and tools that change the corpus and then need the rebuilt values, which stale
    readers only see once the background builder has finished.
    """
    deadline = time.monotonic() + timeout
    while True:
        with _LOCK:
            pending = list(_IN_FLIGHT.values())
        if not pending:
            return True
        for event in pending:
            if not event.wait(max(0.0, deadline - time.monotonic())):
                return False


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
                "stale_served": counters[3],
            }
        return out


def lance_mtime(lance_dir: Path | str) -> float:
    """Reindex signal: the LanceDB index dir mtime (mirrors ``search.index_pool``)."""
    try:
        return os.path.getmtime(lance_dir)
    except OSError:
        return -1.0


#: Files every out-of-band writer of corpus ARTIFACTS touches (see :func:`corpus_mtime`).
_CONTENT_STAMP_NAMES = (
    "corpus_run_summary.json",
    "corpus_manifest.json",
    "upgrade_ledger.json",
    "corpus_edges_stamp.json",
)

#: The identity ledger. Its own write must move the token the projections key on, so
#: person pages and the KG index pick up a rebuilt ledger without a restart.
_IDENTITY_LEDGER_NAME = "identity_ledger.json"


def corpus_mtime(root: Path | str) -> float:
    """Ingest signal: the corpus run-summary mtime (rewritten each run), falling
    back to the manifest, then the corpus dir mtime. Includes the identity ledger."""
    return _stamp_mtime(Path(root), _CONTENT_STAMP_NAMES + (_IDENTITY_LEDGER_NAME,), True)


def corpus_content_mtime(root: Path | str) -> float:
    """:func:`corpus_mtime` without the identity ledger: the state of the corpus CONTENT.

    The ledger records this as the corpus it was built from. A token that included the ledger
    could never match after the ledger is written. For the same reason it never falls
    back to the corpus root's own mtime, which the ledger write moves: a corpus with none of the
    stamp files has no content mtime (-1), and staleness then rests on the kg count.
    """
    return _stamp_mtime(Path(root), _CONTENT_STAMP_NAMES, False)


def _stamp_mtime(root: Path, names: tuple, fall_back_to_root: bool) -> float:
    """MAX mtime over ``names`` under ``root``; else the root's mtime (if allowed), else -1."""
    stamps = []
    for name in names:
        try:
            # callers pass a validated corpus root (platform anchor or _resolve_corpus output);
            # name is a constant; getmtime only stats it for the cache token.
            # codeql[py/path-injection] -- validated corpus root + constant filename (Type 1).
            stamps.append(os.path.getmtime(root / name))
        except OSError:
            continue
    if stamps:
        # MAX, not first-found. A corpus MIGRATION rewrites `*.kg.json` and the upgrade ledger and
        # touches neither run-summary nor manifest (#2065 advisor S6), so tokening on the first
        # file present left every projection serving pre-migration roles until the next ingest or
        # a process restart — the KG entity index, the catalog, momentum person-roles (the exact
        # field m0009 changes), top-persons and the per-artifact loader. An operator who migrated
        # and then looked at the app would see nothing change and conclude it had done nothing.
        #
        # `corpus_edges_stamp.json` is the same hole one path over: `search enrich-edges` rewrites
        # gi.json — SPOKEN_BY, the edges insight attribution reads — and is neither an ingest nor
        # a migration, so it moved none of the three names above. Every out-of-band writer of
        # corpus artifacts needs a name in THIS tuple; adding one is the price of not silently
        # serving stale projections.
        return max(stamps)
    if not fall_back_to_root:
        return -1.0
    try:
        # codeql[py/path-injection] -- same validated corpus root; getmtime stats only (Type 1).
        return os.path.getmtime(root)
    except OSError:
        return -1.0
