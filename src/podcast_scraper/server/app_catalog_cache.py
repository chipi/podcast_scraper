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

Scope: the **consumer** plane only. Operator library/metrics routes keep their existing behaviour
(``routes/corpus_library.py`` deliberately notes "shared ``build_catalog_rows_cumulative`` stays
uncached" and caches per-route itself).
"""

from __future__ import annotations

from pathlib import Path

from podcast_scraper import perf_cache
from podcast_scraper.server.corpus_catalog import (
    build_catalog_rows_cumulative,
    CatalogEpisodeRow,
)

_CATALOG_NS = "app_catalog_rows"


def _key(root: Path) -> str:
    return str(Path(root).resolve())


def cached_catalog(root: Path) -> list[CatalogEpisodeRow]:
    """The cumulative-unique catalog, cached by corpus mtime; a fresh list copy per call.

    Drop-in for ``build_catalog_rows_cumulative(root)`` on the consumer plane. The copy is O(rows)
    pointer work (microseconds) and lets callers sort/filter in place without touching the shared
    cache entry.
    """
    rows = perf_cache.get_or_compute(
        _CATALOG_NS,
        _key(root),
        perf_cache.corpus_mtime(root),
        lambda: build_catalog_rows_cumulative(root),
    )
    return list(rows)
