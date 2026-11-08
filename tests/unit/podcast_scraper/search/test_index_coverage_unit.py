"""Search-index coverage without LanceDB: the backend's reading of tables, the failed-ANN warning,
and the /metrics gauges with their once-a-minute memo.

The integration tests (tests/integration/search/test_search_index_metrics.py) run the same on a
real LanceDB table; these pin the branches a real table rarely takes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from podcast_scraper.search import index_metrics
from podcast_scraper.search.backends import lancedb_backend
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend


class _Table:
    def __init__(self, rows: int, indices: dict[str, int | None], *, broken: bool = False) -> None:
        self.rows = rows
        self.indices = indices  # index name -> unindexed rows
        self.broken = broken

    def count_rows(self) -> int:
        if self.broken:
            raise RuntimeError("unreadable table")
        return self.rows

    def list_indices(self) -> list[Any]:
        kinds = {"text_idx": "FTS", "embedding_idx": "IvfPq"}
        return [SimpleNamespace(name=n, index_type=kinds[n]) for n in self.indices]

    def index_stats(self, name: str) -> Any:
        return SimpleNamespace(num_unindexed_rows=self.indices[name])

    def create_fts_index(self, *_a: Any, **_k: Any) -> None:
        pass

    def create_index(self, *_a: Any, **_k: Any) -> None:
        raise RuntimeError("ivf training failed")


def _backend(monkeypatch: pytest.MonkeyPatch, tables: dict[str, _Table]) -> LanceDBBackend:
    b = object.__new__(LanceDBBackend)
    monkeypatch.setattr(b, "_open_if_exists", lambda tier: tables.get(tier))
    return b


def test_coverage_reads_presence_and_unindexed_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    b = _backend(
        monkeypatch,
        {
            "segment": _Table(300, {"text_idx": 0, "embedding_idx": 25}),
            "insight": _Table(400, {"text_idx": 0}),
            "aux": _Table(9, {}, broken=True),
            "segment_nonen": _Table(50, {"text_idx": 4}),
        },
    )
    got = {(r["table"], r["index"]): r for r in b.index_coverage()}
    seg = got[("segments", "vector")]
    assert seg["present"] and seg["unindexed_rows"] == 25 and seg["rows"] == 300
    insight_vec = got[(LanceDBBackend.TABLES["insight"], "vector")]
    assert not insight_vec["present"] and insight_vec["unindexed_rows"] == 400
    nonen = LanceDBBackend.TABLES["segment_nonen"]
    assert (nonen, "vector") not in got  # no embedding column: no vector index expected
    assert got[(nonen, "fts")]["unindexed_rows"] == 4
    assert not any(t == LanceDBBackend.TABLES["aux"] for t, _ in got)  # unreadable: left out


def test_a_failed_ann_build_warns_with_the_table(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv(lancedb_backend.EXACT_VECTORS_ENV, raising=False)
    b = _backend(monkeypatch, {"segment": _Table(1000, {})})
    with caplog.at_level(logging.WARNING, logger=lancedb_backend.__name__):
        b.create_indices()
    msgs = [r.getMessage() for r in caplog.records]
    assert any("vector index NOT built" in m and "1000 rows" in m for m in msgs), msgs


def test_the_gauges_from_measured_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = [
        {
            "table": "segments",
            "index": "vector",
            "rows": 300,
            "present": False,
            "unindexed_rows": 300,
        },
        {"table": "segments", "index": "fts", "rows": 300, "present": True, "unindexed_rows": 0},
    ]
    monkeypatch.setattr(index_metrics, "measure", lambda _d: rows)
    registry = CollectorRegistry()
    assert index_metrics.install_metrics(tmp_path, registry)
    text = generate_latest(registry).decode()
    assert text.count("podcast_search_index_rows{") == 1
    assert 'podcast_search_index_present{index="vector",table="segments"} 0.0' in text
    assert 'podcast_search_index_unindexed_rows{index="vector",table="segments"} 300.0' in text


def test_the_memo_measures_once_a_minute_and_survives_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[int] = []
    clock = [1000.0]

    def measure(_d: Path) -> list[dict]:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("index being rewritten")
        return [{"table": "t", "index": "fts", "rows": 1, "present": True, "unindexed_rows": 0}]

    monkeypatch.setattr(index_metrics, "measure", measure)
    monkeypatch.setattr(index_metrics.time, "monotonic", lambda: clock[0])
    memo = index_metrics._Memo(tmp_path)
    assert memo.get() and memo.get() and len(calls) == 1  # within the minute: no new measurement
    clock[0] += index_metrics._REFRESH_SECONDS
    assert memo.get() == [] and len(calls) == 2  # a failed measurement exports nothing, not stale
    clock[0] += index_metrics._REFRESH_SECONDS
    assert memo.get() and len(calls) == 3


def test_no_index_directory_measures_nothing(tmp_path: Path) -> None:
    assert index_metrics.measure(tmp_path / "absent") == []


def test_a_refused_registration_returns_false_not_raises(tmp_path: Path) -> None:
    def register(_collector: object) -> None:
        raise ValueError("Duplicated timeseries in CollectorRegistry")

    assert index_metrics.install_metrics(tmp_path, SimpleNamespace(register=register)) is False


def test_the_wait_histogram_never_breaks_a_search(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Broken:
        def observe(self, _s: float) -> None:
            raise RuntimeError("registry gone")

    monkeypatch.setitem(lancedb_backend._VECTOR_WAIT, "h", _Broken())
    lancedb_backend._observe_vector_wait(0.1)  # must not raise


def test_a_malformed_setting_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(lancedb_backend.VECTOR_CONCURRENCY_ENV, "many")
    assert lancedb_backend._vector_concurrency() == lancedb_backend._DEFAULT_VECTOR_CONCURRENCY
