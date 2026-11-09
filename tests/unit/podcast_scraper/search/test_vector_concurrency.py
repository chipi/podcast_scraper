"""Vector searches per process, capped by ``PODCAST_SEARCH_VECTOR_CONCURRENCY`` (off by default).

A vector query over rows no ANN index covers scans their embeddings whole, with memory growing with
the rows; the cap bounds how many such scans run at once. These pin how many run at once for each
setting.
"""

from __future__ import annotations

import threading
import time

import pytest

from podcast_scraper.search.backend import SearchQuery
from podcast_scraper.search.backends import lancedb_backend
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend


def _most_at_once(monkeypatch: pytest.MonkeyPatch, setting: str | None, callers: int = 6) -> int:
    if setting is None:
        monkeypatch.delenv(lancedb_backend.VECTOR_CONCURRENCY_ENV, raising=False)
    else:
        monkeypatch.setenv(lancedb_backend.VECTOR_CONCURRENCY_ENV, setting)
    lock = threading.Lock()
    now = [0]
    most = [0]

    def fake_run(self, query, *, query_type, score_key):
        with lock:
            now[0] += 1
            most[0] = max(most[0], now[0])
        time.sleep(0.05)
        with lock:
            now[0] -= 1
        return []

    monkeypatch.setattr(LanceDBBackend, "_run", fake_run)
    backend = object.__new__(LanceDBBackend)  # no connection: _run is replaced
    query = SearchQuery(text="", embedding=[0.0, 0.0, 0.0, 0.0], tier="segment")
    threads = [
        threading.Thread(target=backend.search_vector, args=(query,)) for _ in range(callers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return most[0]


def test_no_cap_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _most_at_once(monkeypatch, None) == 6


def test_one_is_one_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _most_at_once(monkeypatch, "1") == 1


def test_the_setting_is_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _most_at_once(monkeypatch, "3") == 3


def test_zero_turns_the_cap_off(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _most_at_once(monkeypatch, "0") == 6


def test_keyword_search_is_not_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        lancedb_backend.VECTOR_CONCURRENCY_ENV, "1"
    )  # cap on: bm25 still bypasses it
    calls: list[str] = []
    monkeypatch.setattr(
        LanceDBBackend, "_run", lambda self, q, *, query_type, score_key: calls.append(query_type)
    )
    monkeypatch.setattr(lancedb_backend, "_vector_slot", None)  # would raise if bm25 used it
    backend = object.__new__(LanceDBBackend)
    backend.search_bm25(SearchQuery(text="x", embedding=[0.0] * 4, tier="segment"))
    assert calls == ["fts"]
