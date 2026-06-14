"""Unit tests for the process-scoped search-index pool (ADR-099, #995).

The pool must (a) reuse one handle per index dir, (b) rebuild when the index changes,
and (c) build exactly once under concurrent access (the property that removes the
concurrent-cold-init segfault).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from podcast_scraper.search import index_pool


class _FakeBackend:
    """Stands in for LanceDBBackend: records prewarm calls, exposes _open_if_exists."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.opened: list[str] = []

    def _open_if_exists(self, tier: str):  # noqa: D401 - matches backend surface
        self.opened.append(tier)
        return object()


@pytest.fixture(autouse=True)
def _clear_pool():
    index_pool.clear()
    yield
    index_pool.clear()


def test_lance_backend_is_reused_for_same_dir(tmp_path: Path) -> None:
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return _FakeBackend(f"b{calls['n']}")

    b1 = index_pool.get_lance_backend(tmp_path, build)
    b2 = index_pool.get_lance_backend(tmp_path, build)
    assert b1 is b2
    assert calls["n"] == 1  # built once, reused


def test_prewarm_opens_read_tables(tmp_path: Path) -> None:
    b = index_pool.get_lance_backend(tmp_path, lambda: _FakeBackend("b"))
    # The three read tiers are opened at warm time so the search path never cold-opens.
    assert set(b.opened) == {"segment", "insight", "aux"}


def test_rebuilds_when_index_dir_changes(tmp_path: Path) -> None:
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return _FakeBackend(f"b{calls['n']}")

    b1 = index_pool.get_lance_backend(tmp_path, build)
    # Bump the dir mtime to simulate a reindex; the pool must rebuild.
    import os

    st = tmp_path.stat()
    os.utime(tmp_path, (st.st_atime, st.st_mtime + 10))
    b2 = index_pool.get_lance_backend(tmp_path, build)
    assert b1 is not b2
    assert calls["n"] == 2


def test_clear_forces_rebuild(tmp_path: Path) -> None:
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return _FakeBackend("b")

    index_pool.get_lance_backend(tmp_path, build)
    index_pool.clear()
    index_pool.get_lance_backend(tmp_path, build)
    assert calls["n"] == 2


def test_concurrent_get_builds_exactly_once(tmp_path: Path) -> None:
    """The segfault-prevention property: N threads racing → one cold build, one handle."""
    calls = {"n": 0}
    lock = threading.Lock()

    def build():
        with lock:
            calls["n"] += 1
        # simulate a slow cold open so threads genuinely overlap
        threading.Event().wait(0.02)
        return _FakeBackend("b")

    results: list[object] = []
    rlock = threading.Lock()

    def worker():
        b = index_pool.get_lance_backend(tmp_path, build)
        with rlock:
            results.append(b)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1  # built exactly once despite 16 concurrent callers
    assert len({id(r) for r in results}) == 1  # everyone got the same handle


def test_faiss_store_is_reused_for_same_dir(tmp_path: Path) -> None:
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return object()

    s1 = index_pool.get_faiss_store(tmp_path, build)
    s2 = index_pool.get_faiss_store(tmp_path, build)
    assert s1 is s2
    assert calls["n"] == 1
