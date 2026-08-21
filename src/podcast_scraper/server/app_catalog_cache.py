"""Shared, corpus-mtime-cached catalog + slug index for the consumer ``/api/app/*`` plane.

``build_catalog_rows_cumulative()`` is a full-corpus scan (~0.5 ms/episode: open + JSON-parse each
``*.metadata.json`` and stat three sidecars). Its result only changes on **ingest**, yet nearly
every consumer route re-ran it *per request* — the dominant scaling cost. Uncached it is ~50 ms at
100 episodes, ~350 ms at 700, and projects to **~5 s/request at 10k**; because the scan is
synchronous CPU + GIL-bound JSON parsing, concurrent requests serialize (measured: 20 concurrent
``/discover`` went from 54 ms to 2.7 s each).

This caches the catalog through the shared :mod:`podcast_scraper.perf_cache`, keyed by the corpus
run-summary mtime (``corpus_mtime`` — bumps on ingest) — the *same* token + pattern the operator
``catalog_feeds`` route already uses (``routes/corpus_library.py``). One warm catalog is then shared
across every consumer route; the scan runs once per ingest, not once per request.

Rows are immutable (``@dataclass(frozen=True)``), so sharing row objects is safe; we hand back a
shallow **copy** of the list so a caller that sorts/filters in place (e.g. ``/discover``) cannot
corrupt the shared entry. The slug index is read-only (callers only ``.get``) so it is shared as-is.

Scope: both planes. The consumer ``/api/app/*`` routes and the operator ``/api/corpus/*`` viewer
routes share this cache — the scan and its cost are identical, so there is no reason to cache it
twice. Two variants, because the corpus catalog has two legitimate shapes (see
:mod:`corpus_catalog`): :func:`cached_catalog` wraps the cumulative-unique scan (all runs), and
:func:`cached_catalog_last_run` wraps the last-run-only scan. Callers keep whichever shape they
already used — this is a perf cache, not a semantics change.
"""

from __future__ import annotations

from pathlib import Path

from podcast_scraper import perf_cache
from podcast_scraper.server.corpus_catalog import (
    build_catalog_rows,
    build_catalog_rows_cumulative,
    CatalogEpisodeRow,
)

_CATALOG_NS = "app_catalog_rows"
_CATALOG_LAST_RUN_NS = "catalog_rows_last_run"


def _key(root: Path) -> str:
    return str(Path(root).resolve())


def cached_catalog(root: Path) -> list[CatalogEpisodeRow]:
    """The cumulative-unique catalog, cached by corpus mtime; a fresh list copy per call.

    Drop-in for ``build_catalog_rows_cumulative(root)``. The copy is O(rows) pointer work
    (microseconds) and lets callers sort/filter in place without touching the shared cache entry.
    """
    rows = perf_cache.get_or_compute(
        _CATALOG_NS,
        _key(root),
        perf_cache.corpus_mtime(root),
        lambda: build_catalog_rows_cumulative(root),
    )
    return list(rows)


def cached_catalog_last_run(root: Path) -> list[CatalogEpisodeRow]:
    """The last-run-only catalog, cached by corpus mtime; a fresh list copy per call.

    Drop-in for ``build_catalog_rows(root)`` — the last-run-only shape the operator episode-detail /
    similar / resolve routes use. Kept distinct from :func:`cached_catalog` (different namespace)
    because the two scans return different row sets; swapping one for the other would change
    behaviour, not just latency.
    """
    rows = perf_cache.get_or_compute(
        _CATALOG_LAST_RUN_NS,
        _key(root),
        perf_cache.corpus_mtime(root),
        lambda: build_catalog_rows(root),
    )
    return list(rows)
