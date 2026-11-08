"""The search index's coverage on ``/metrics``: rows per LanceDB table, and per index kind whether
it exists and how many rows it does not cover.

A dense table with no vector index, or with many rows outside it, is searched by a full scan of
its embeddings: on a prod-sized segment table one such query took +1.8 GB for half a second and
four at once +5.9 GB (2026-11-08), past the API's memory cap. The OOM kills of 2026-10-26/27 were
sub-second jumps of that size. These gauges make that state visible before it kills a process;
the O11Y ``database`` surface reads them.

Measured at scrape time, at most once a minute (opening the tables and reading their index
statistics is cheap, but a scrape every few seconds need not repeat it). Never raises (ADR-120).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

_REFRESH_SECONDS = 60.0


def measure(index_dir: Path) -> List[Dict[str, Any]]:
    """``LanceDBBackend.index_coverage()`` for ``index_dir``, or ``[]`` when there is no index."""
    if not index_dir.is_dir():
        return []
    from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend

    return LanceDBBackend(str(index_dir)).index_coverage()


class _Memo:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.lock = threading.Lock()
        self.at = 0.0
        self.rows: List[Dict[str, Any]] = []

    def get(self) -> List[Dict[str, Any]]:
        with self.lock:
            if time.monotonic() - self.at >= _REFRESH_SECONDS:
                try:
                    self.rows = measure(self.index_dir)
                except Exception as exc:  # noqa: BLE001 - a scrape must never fail on this
                    logger.debug("search index coverage not measured: %s", exc)
                    self.rows = []
                self.at = time.monotonic()
            return self.rows


def install_metrics(corpus_root: Path, registry: Any = None) -> bool:
    """Publish the coverage of ``<corpus_root>/search/lance_index`` on ``/metrics``. No index
    exports nothing. Never raises. ``registry`` defaults to the process registry (tests pass
    their own)."""
    try:
        from prometheus_client import REGISTRY
        from prometheus_client.core import GaugeMetricFamily
    except Exception:  # noqa: BLE001 - metrics extras absent; run without these gauges
        return False

    from podcast_scraper.search.hybrid_search import lance_index_dir

    memo = _Memo(lance_index_dir(Path(corpus_root)))

    class _SearchIndexCollector:
        def collect(self):  # noqa: ANN202 - prometheus_client's duck-typed collector protocol
            rows = memo.get()
            if not rows:
                return
            total = GaugeMetricFamily(
                "podcast_search_index_rows", "Rows in a search-index table.", labels=["table"]
            )
            present = GaugeMetricFamily(
                "podcast_search_index_present",
                "1 when the table has this kind of index (vector / fts), 0 when it has none.",
                labels=["table", "index"],
            )
            unindexed = GaugeMetricFamily(
                "podcast_search_index_unindexed_rows",
                "Rows this index does not cover (all rows when the index is missing): "
                "searched by a full scan.",
                labels=["table", "index"],
            )
            seen: set[str] = set()
            for r in rows:
                if r["table"] not in seen:
                    seen.add(r["table"])
                    total.add_metric([r["table"]], float(r["rows"]))
                key: Tuple[str, str] = (r["table"], r["index"])
                present.add_metric(list(key), 1.0 if r["present"] else 0.0)
                unindexed.add_metric(list(key), float(r["unindexed_rows"]))
            yield total
            yield present
            yield unindexed

    try:
        (registry or REGISTRY).register(_SearchIndexCollector())  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - duplicate registration on app re-create is harmless
        return False
    return True
