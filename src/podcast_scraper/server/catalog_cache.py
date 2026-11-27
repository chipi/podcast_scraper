"""The episode catalog for both planes, read from the corpus rows and cached by a row token.

``catalog_rows.catalog_from_rows`` builds the catalog's :class:`CatalogEpisodeRow` list from the
served items in one statement (``intents.catalog_items``), over the ``corpus_ro`` engine. There is
no file scan and no fallback: with no corpus URL a call raises ``DatabaseNotConfigured`` (r-intents;
agreed with the Player 2026-11-26, the way back is the cut-over's rollback).

The cache is the shared :mod:`podcast_scraper.perf_cache`, keyed by :func:`catalog_token`: the
count and newest ``updated_at`` of ``content_items`` and the corpus revision, in one small query.
Every writer of an item row sets ``updated_at``, and a row migration bumps the revision, so the
token moves with every change the catalog shows. The corpus revision alone would not: only corpus
migrations bump it, not an ingest.

A read token is reused for :data:`TOKEN_HOLD_S` outside :func:`perf_cache.fresh`, so a request
that reads the catalog several times pays one token query, not one a read (the Player's
round-trip budgets, 2026-11-27). A change then reaches the catalog that much later; inside
``fresh()``, as :func:`current_catalog` reads, the token is read every time.

The file era had two shapes, the cumulative scan over every run and the last run's. Rows hold one
item per episode, the import's and the writer's newest run, so both names return the same rows; the
``_last_run`` names stay so their callers need no change (the two scans returned the same 3,522
rows on snapshot-prod-20261124).

Rows are immutable (``@dataclass(frozen=True)``); callers get a shallow copy of the list, so one
that sorts or filters in place cannot corrupt the shared entry.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import func, select

from podcast_scraper import perf_cache
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow

_CATALOG_NS = "app_catalog_rows"
CATALOG_LAST_RUN_NS = _CATALOG_NS
_TOKEN_BITS = (1 << 52) - 1
#: Seconds a read row token is reused outside ``perf_cache.fresh()``.
TOKEN_HOLD_S = 2.0


def _key(root: Path) -> str:
    return str(Path(root).resolve())


def _read_token() -> float:
    """The catalog's cache token: it changes whenever an item row is written or a row migration
    runs."""
    from podcast_scraper.db.engines import corpus_ro_engine, read_connection
    from podcast_scraper.db.schema import content_items, corpus_revision

    revision = select(corpus_revision.c.revision).where(corpus_revision.c.id == 1)
    with read_connection(corpus_ro_engine()) as conn:
        count, newest, rev = conn.execute(
            select(
                func.count(), func.max(content_items.c.updated_at), revision.scalar_subquery()
            ).select_from(content_items)
        ).one()
    return float(hash((count, newest, rev)) & _TOKEN_BITS)


class _TokenHold:
    """A token reader that reuses its last answer for ``seconds`` outside ``perf_cache.fresh()``."""

    def __init__(self, read: Callable[[], float], seconds: float) -> None:
        self._read = read
        self._seconds = seconds
        self._held: Optional[tuple[float, float]] = None
        self._lock = threading.Lock()

    def __call__(self) -> float:
        if not perf_cache.in_fresh():
            with self._lock:
                if self._held is not None and time.monotonic() - self._held[0] < self._seconds:
                    return self._held[1]
        token = self._read()
        with self._lock:
            self._held = (time.monotonic(), token)
        return token


_held_token = _TokenHold(lambda: _read_token(), TOKEN_HOLD_S)


def catalog_token() -> float:
    """The catalog's cache token: it changes whenever an item row is written or a row migration
    runs; reused for :data:`TOKEN_HOLD_S` outside ``perf_cache.fresh()``."""
    return _held_token()


def _catalog_rows(root: Path) -> list[CatalogEpisodeRow]:
    from podcast_scraper.db.engines import corpus_ro_engine, read_connection
    from podcast_scraper.server.catalog_rows import catalog_from_rows

    with read_connection(corpus_ro_engine()) as conn:
        return catalog_from_rows(conn, Path(root))


def cached_catalog(root: Path) -> list[CatalogEpisodeRow]:
    """The catalog of the served podcast items, cached by :func:`catalog_token`; a fresh list copy
    per call."""
    rows = perf_cache.get_or_compute(
        _CATALOG_NS, _key(root), catalog_token(), lambda: _catalog_rows(root), stale_ok=True
    )
    return list(rows)


def cached_catalog_last_run(root: Path) -> list[CatalogEpisodeRow]:
    """:func:`cached_catalog`: on rows the last run's catalog and the cumulative one are one."""
    return cached_catalog(root)


def current_catalog(root: Path) -> list[CatalogEpisodeRow]:
    """:func:`cached_catalog` for the CURRENT rows, never the previous ones while it rebuilds.

    The stale answer spares the Player's cold reads after an ingest; the operator's routes need what
    the job just wrote: the Workbench refetches once after a job and showed the pre-job lists until
    reload (#2486). Waits for the rebuild instead (single-flight, so one build serves every waiter).
    """
    with perf_cache.fresh():
        return cached_catalog(root)


def current_catalog_last_run(root: Path) -> list[CatalogEpisodeRow]:
    """:func:`current_catalog` (see :func:`cached_catalog_last_run`)."""
    return current_catalog(root)
