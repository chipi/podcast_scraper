"""Equal scores rank the same way every time (found 2026-11-12).

Two search indexes built over identical rows answered the same query in different orders: rows
with equal scores kept the order the table or the input lists returned them in, and that order
differs between builds. Both places that rank now break a tie by the document id (then the tier),
so the order depends on the rows alone.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from podcast_scraper.search.backend import ScoredResult, SearchQuery
from podcast_scraper.search.fusion import rrf_fuse

pytestmark = pytest.mark.unit


def _r(doc_id: str, rank: int, signal: str) -> ScoredResult:
    return ScoredResult(
        doc_id=doc_id, score=1.0, rank=rank, payload={}, signal=signal, source_tier="segment"
    )


def test_fused_ties_do_not_depend_on_the_order_of_the_lists() -> None:
    bm25 = [_r("b", 1, "bm25"), _r("a", 2, "bm25")]
    vector = [_r("a", 1, "vector"), _r("b", 2, "vector")]
    one = [r.doc_id for r in rrf_fuse([bm25, vector])]
    other = [r.doc_id for r in rrf_fuse([vector, bm25])]
    assert one == other == ["a", "b"]


class _Backend:
    """``LanceDBBackend._run`` over a table that returns the given rows, in the given order."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend

        self._impl = LanceDBBackend.__new__(LanceDBBackend)
        self._impl._tables_for_tier = lambda tier, keyword=False: ["segment"]  # type: ignore
        self._impl._where_for = lambda query, tier: None  # type: ignore
        self._impl._fresh_read = lambda tier, run: [dict(r) for r in rows]  # type: ignore

    def bm25_ids(self) -> List[str]:
        query = SearchQuery(text="q", embedding=[0.0], k=10)
        return [r.doc_id for r in self._impl.search_bm25(query)]


def test_a_legs_ties_do_not_depend_on_the_tables_order() -> None:
    rows = [{"id": "b", "_score": 2.0}, {"id": "a", "_score": 2.0}, {"id": "c", "_score": 3.0}]
    assert _Backend(rows).bm25_ids() == _Backend(list(reversed(rows))).bm25_ids() == ["c", "a", "b"]
