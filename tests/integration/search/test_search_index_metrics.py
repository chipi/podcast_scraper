"""The search index's coverage, as the backend reads it and as ``/metrics`` exports it.

A dense table with no vector index, or rows outside it, is scanned whole by every vector query.
These pin that the gauges say so.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from podcast_scraper.search import index_metrics
from podcast_scraper.search.backend import SegmentDocument
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend

pytestmark = pytest.mark.integration


def _seg(i: int) -> SegmentDocument:
    return SegmentDocument(
        id=f"s{i}",
        text=f"row {i} about the topic",
        show_id="A",
        episode_id=f"e{i // 10}",
        start_time=0.0,
        end_time=1.0,
        embedding=[((i * k) % 97) / 97.0 for k in (1, 3, 7, 11)],
    )


def _backend(root: Path, n: int) -> LanceDBBackend:
    b = LanceDBBackend(str(root / "search" / "lance_index"), embed_dim=4)
    b.upsert_segments([_seg(i) for i in range(n)])
    return b


def _segment(coverage: list[dict], index: str) -> dict:
    return next(r for r in coverage if r["table"] == "segments" and r["index"] == index)


def test_indexed_then_rows_added_then_counted_as_unindexed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PODCAST_SEARCH_EXACT_VECTORS", raising=False)
    b = _backend(tmp_path, LanceDBBackend._MIN_VECTOR_INDEX_ROWS + 44)
    b.create_indices()
    vec = _segment(b.index_coverage(), "vector")
    assert vec["present"] and vec["unindexed_rows"] == 0 and vec["rows"] == 300
    assert _segment(b.index_coverage(), "fts")["present"]

    b.upsert_segments([_seg(i) for i in range(1000, 1025)])
    vec = _segment(b.index_coverage(), "vector")
    assert vec["present"] and vec["unindexed_rows"] == 25 and vec["rows"] == 325


def test_a_missing_vector_index_counts_every_row(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_SEARCH_EXACT_VECTORS", "1")  # create_indices builds FTS only
    b = _backend(tmp_path, 300)
    b.create_indices()
    vec = _segment(b.index_coverage(), "vector")
    assert not vec["present"] and vec["unindexed_rows"] == 300


def test_the_gauges_on_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_SEARCH_EXACT_VECTORS", "1")
    _backend(tmp_path, 300).create_indices()
    registry = CollectorRegistry()
    assert index_metrics.install_metrics(tmp_path, registry)
    text = generate_latest(registry).decode()
    assert 'podcast_search_index_rows{table="segments"} 300.0' in text
    assert 'podcast_search_index_present{index="vector",table="segments"} 0.0' in text
    assert 'podcast_search_index_present{index="fts",table="segments"} 1.0' in text
    assert 'podcast_search_index_unindexed_rows{index="vector",table="segments"} 300.0' in text


def test_no_index_exports_nothing(tmp_path: Path) -> None:
    registry = CollectorRegistry()
    assert index_metrics.install_metrics(tmp_path, registry)
    assert "podcast_search_index" not in generate_latest(registry).decode()
