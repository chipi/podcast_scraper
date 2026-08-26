"""One duplicate row must never abort a flush (the 2026-08-26 whole-reindex zero).

LanceDB merge_insert refuses ambiguous batches outright, so the buffer must reach it
duplicate-free — last row wins, loudly logged.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from podcast_scraper.search.two_tier_indexer import _dedupe_rows_by_id

pytestmark = [pytest.mark.unit]


@dataclass
class _Row:
    id: str
    payload: str = ""


def test_duplicate_ids_collapse_last_wins() -> None:
    buf = [_Row("a", "first"), _Row("b"), _Row("a", "second")]
    _dedupe_rows_by_id("aux", buf)
    assert [r.id for r in buf] == ["b", "a"] or [r.id for r in buf] == ["a", "b"]
    kept = next(r for r in buf if r.id == "a")
    assert kept.payload == "second", "last row must win (freshest artifact state)"


def test_clean_buffer_untouched() -> None:
    buf = [_Row("a"), _Row("b"), _Row("c")]
    _dedupe_rows_by_id("segment", buf)
    assert [r.id for r in buf] == ["a", "b", "c"]
