"""Best-effort warming of the consumer read caches (perf follow-up).

Warming is pure optimization: it calls the **same** build functions a request would, so the first
request after a start or an ingest doesn't pay the O(corpus) build by itself. It is best-effort BY
DESIGN — every step is guarded, so a failed warm silently degrades to the lazy path and never breaks
serving (the caches are never load-bearing for correctness). That is the anti-fragility guarantee.

Simplicity: one daemon thread + a sleep loop, standard library only — no scheduler, no new deps —
mirroring the existing search warmup in ``server.app``. Two triggers, one thread:

* **A — startup**: warm once when the process boots (the first request after a deploy/restart).
* **B — post-ingest**: poll the corpus mtime (the exact token the caches key on) and re-warm when it
  moves, so a live corpus swap doesn't leave the next user paying the rebuild.

Disable with ``APP_CACHE_WARMING=0`` (wired in the lifespan); then the caches simply fill lazily.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from podcast_scraper import perf_cache

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_S = 30.0


def warm_caches(root: Path) -> None:
    """Populate the consumer read caches for ``root`` (catalog, slug index, KG index).

    Best-effort: each step is independently guarded so one failure (e.g. a missing artifact) does
    not skip the rest. Imports are local so a warm never drags card/relational modules into a
    caller that would not otherwise need them.
    """
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.app_kg_index import get_kg_index
    from podcast_scraper.server.app_slugs import resolve_slug

    steps = (
        ("catalog", lambda: cached_catalog(root)),
        # resolve_slug builds + caches the slug index on any non-empty query (then misses harmless).
        ("slug_index", lambda: resolve_slug(root, "\x00warm")),
        ("kg_index", lambda: get_kg_index(root)),
    )
    for label, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - warming is advisory; the request path re-reports
            logger.debug("cache warm %s skipped: %s", label, exc)


def _warm_loop(root: Path, interval_s: float, stop: threading.Event) -> None:
    """Warm once, then re-warm whenever the corpus mtime changes, until ``stop`` is set."""
    last_token: float | None = None
    while not stop.is_set():
        try:
            token = perf_cache.corpus_mtime(root)
            if token != last_token:
                warm_caches(root)
                last_token = token
        except Exception as exc:  # noqa: BLE001 - the loop must never die on a transient error
            logger.debug("cache warm loop error: %s", exc)
        stop.wait(interval_s)


def start_cache_warmer(root: Path, *, interval_s: float = _DEFAULT_INTERVAL_S) -> threading.Event:
    """Start the daemon warmer (warm-once + re-warm-on-ingest); return a stop ``Event``.

    Never blocks and never raises — a warmer that cannot start just means lazy cache fills, the
    pre-warming behaviour before this existed.
    """
    stop = threading.Event()
    threading.Thread(
        target=_warm_loop, args=(root, interval_s, stop), name="cache-warmer", daemon=True
    ).start()
    logger.info("cache warmer started (interval %.0fs)", interval_s)
    return stop
