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


def test_freshness_token_returns_minus_one_on_oserror(tmp_path: Path, monkeypatch) -> None:
    """_freshness_token degrades to -1.0 when stat fails (line 46-47).

    A -1.0 token is still a stable comparison key, so a cached handle whose token is also
    -1.0 (same unreadable dir) is reused rather than rebuilt on every call.
    """
    import os

    def _boom(_p):
        raise OSError("stat failed")

    monkeypatch.setattr(os.path, "getmtime", _boom)
    assert index_pool._freshness_token(tmp_path) == -1.0

    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return _FakeBackend(f"b{calls['n']}")

    b1 = index_pool.get_lance_backend(tmp_path, build)
    b2 = index_pool.get_lance_backend(tmp_path, build)
    assert b1 is b2 and calls["n"] == 1  # -1.0 token is stable -> reused, not rebuilt


def test_prewarm_swallows_opener_exception(tmp_path: Path) -> None:
    """A backend whose _open_if_exists raises must still be pooled (line 58-59).

    Prewarm is best-effort: a missing/erroring tier is logged + skipped (tables open
    lazily on the read path), so build() must still publish the handle.
    """

    class _RaisingBackend:
        def __init__(self) -> None:
            self.attempts: list[str] = []

        def _open_if_exists(self, tier: str):
            self.attempts.append(tier)
            raise RuntimeError(f"cold open failed for {tier}")

    backend = _RaisingBackend()
    got = index_pool.get_lance_backend(tmp_path, lambda: backend)
    assert got is backend  # handle published despite every prewarm raising
    # All three tiers were attempted (each raise caught independently).
    assert set(backend.attempts) == {"segment", "insight", "aux"}


def test_prewarm_skips_backend_without_opener(tmp_path: Path) -> None:
    """A backend lacking a callable _open_if_exists is pooled without prewarm (line 52-53)."""

    class _NoOpener:
        pass

    backend = _NoOpener()
    assert index_pool.get_lance_backend(tmp_path, lambda: backend) is backend


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
