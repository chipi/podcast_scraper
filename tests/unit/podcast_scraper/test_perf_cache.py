"""Unit tests for the central perf cache (podcast_scraper.perf_cache)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from podcast_scraper import perf_cache

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean():
    perf_cache.clear()
    yield
    perf_cache.clear()


def test_hit_miss_and_token_invalidation():
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return calls["n"]

    # First call at token 1.0 → miss + compute.
    assert perf_cache.get_or_compute("ns", "k", 1.0, compute) == 1
    assert calls["n"] == 1
    # Same token → hit, no recompute.
    assert perf_cache.get_or_compute("ns", "k", 1.0, compute) == 1
    assert calls["n"] == 1
    # New token → miss, recompute.
    assert perf_cache.get_or_compute("ns", "k", 2.0, compute) == 2
    assert calls["n"] == 2

    s = perf_cache.stats()["ns"]
    assert s["hits"] == 1 and s["misses"] == 2 and s["entries"] == 1
    assert s["hit_rate_pct"] == pytest.approx(33.3, abs=0.1)


def test_stats_report_build_cost_and_time_saved():
    # A ~10ms build so the time-saved signal is measurable, not noise.
    def slow_compute():
        import time

        time.sleep(0.01)
        return "v"

    perf_cache.get_or_compute("slow", "k", 1.0, slow_compute)  # miss → one build
    for _ in range(4):
        perf_cache.get_or_compute("slow", "k", 1.0, slow_compute)  # 4 hits, no rebuild

    s = perf_cache.stats()["slow"]
    assert s["misses"] == 1 and s["hits"] == 4
    # One build was timed; avg_build_ms reflects it, and est_saved ≈ hits × avg build.
    assert s["avg_build_ms"] >= 5.0
    assert s["build_seconds_total"] >= 0.005
    assert s["est_saved_seconds"] == pytest.approx(4 * s["avg_build_ms"] / 1000.0, abs=0.01)


def test_stats_zero_build_cost_when_never_computed():
    # A namespace touched only for stats has no builds → zeroed cost fields, no divide-by-zero.
    perf_cache.get_or_compute("cheap", "k", 1.0, lambda: 1)
    s = perf_cache.stats()["cheap"]
    assert s["misses"] == 1
    perf_cache.get_or_compute("cheap", "k", 1.0, lambda: 1)  # a hit
    s = perf_cache.stats()["cheap"]
    assert s["avg_build_ms"] >= 0.0 and s["est_saved_seconds"] >= 0.0


def test_distinct_keys_and_namespaces_isolated():
    perf_cache.get_or_compute("a", "k1", 1.0, lambda: "v1")
    perf_cache.get_or_compute("a", "k2", 1.0, lambda: "v2")
    perf_cache.get_or_compute("b", "k1", 1.0, lambda: "v3")
    stats = perf_cache.stats()
    assert stats["a"]["entries"] == 2
    assert stats["b"]["entries"] == 1


def test_clear_namespace_and_all():
    perf_cache.get_or_compute("a", "k", 1.0, lambda: 1)
    perf_cache.get_or_compute("b", "k", 1.0, lambda: 1)
    perf_cache.clear("a")
    assert "a" not in perf_cache.stats()
    assert "b" in perf_cache.stats()
    perf_cache.clear()
    assert perf_cache.stats() == {}


def test_caches_none_result():
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return None

    assert perf_cache.get_or_compute("ns", "k", 1.0, compute) is None
    assert perf_cache.get_or_compute("ns", "k", 1.0, compute) is None
    assert calls["n"] == 1  # None is a valid cached value


def test_lance_mtime_and_corpus_mtime(tmp_path):
    # Absent paths → -1.0 sentinel, never raises.
    assert perf_cache.lance_mtime(tmp_path / "nope") == -1.0
    assert perf_cache.corpus_mtime(tmp_path / "nope") == -1.0
    # corpus_mtime prefers corpus_run_summary.json over the dir mtime.
    (tmp_path / "corpus_run_summary.json").write_text("{}", encoding="utf-8")
    import os

    assert perf_cache.corpus_mtime(tmp_path) == pytest.approx(
        os.path.getmtime(tmp_path / "corpus_run_summary.json")
    )


class TestTheTokenSeesAMigration:
    """A corpus migration must invalidate the caches (advisor S6).

    ``corpus_mtime`` tokened on ``corpus_run_summary.json`` / ``corpus_manifest.json`` — both
    written by an INGEST. The m0009 migration rewrites ``*.kg.json`` and the upgrade ledger and
    touches neither, so every mtime-tokened projection kept serving pre-migration roles until a
    later ingest or a process restart: the KG entity index, the catalog, momentum person-roles
    (the exact field the migration changes), top-persons, and the per-artifact loader.

    An operator who runs the migration and then looks at the app sees no change and concludes it
    did nothing.
    """

    def test_the_ledger_moves_the_token(self, tmp_path: Path) -> None:
        import os

        summary = tmp_path / "corpus_run_summary.json"
        summary.write_text("{}", encoding="utf-8")
        os.utime(summary, (1000.0, 1000.0))
        before = perf_cache.corpus_mtime(tmp_path)

        ledger = tmp_path / "upgrade_ledger.json"
        ledger.write_text("{}", encoding="utf-8")
        os.utime(ledger, (2000.0, 2000.0))

        assert (
            perf_cache.corpus_mtime(tmp_path) != before
        ), "a migration that writes only the ledger must still invalidate the caches"

    def test_an_ingest_still_moves_the_token(self, tmp_path: Path) -> None:
        import os

        summary = tmp_path / "corpus_run_summary.json"
        summary.write_text("{}", encoding="utf-8")
        os.utime(summary, (1000.0, 1000.0))
        first = perf_cache.corpus_mtime(tmp_path)
        os.utime(summary, (3000.0, 3000.0))
        assert perf_cache.corpus_mtime(tmp_path) != first

    def test_no_corpus_files_is_still_safe(self, tmp_path: Path) -> None:
        assert isinstance(perf_cache.corpus_mtime(tmp_path), float)


class TestEveryOutOfBandWriterMovesTheToken:
    """A writer that changes corpus artifacts but not the token serves stale projections forever.

    Two of these have already shipped as defects: a MIGRATION rewrites `*.kg.json` (#2065 advisor
    S6) and `search enrich-edges` rewrites gi.json — including SPOKEN_BY, which insight
    attribution reads. Neither is an ingest, so neither moved the run-summary or the manifest, and
    the API kept answering from the pre-change graph until the next ingest or a restart.

    The token is `max()` over a fixed tuple of filenames, so coverage is exactly that tuple. These
    tests assert each name is load-bearing rather than decorative.
    """

    @pytest.mark.parametrize(
        "name,writer",
        [
            ("corpus_run_summary.json", "an ingest run"),
            ("corpus_manifest.json", "a manifest rebuild"),
            ("upgrade_ledger.json", "a corpus migration"),
            ("corpus_edges_stamp.json", "search enrich-edges"),
        ],
    )
    def test_each_watched_file_moves_the_token(self, tmp_path: Path, name: str, writer: str):
        import os

        before = perf_cache.corpus_mtime(tmp_path)
        stamp = tmp_path / name
        stamp.write_text("{}", encoding="utf-8")
        # Force a clearly-later mtime: the filesystem's resolution is coarser than this test.
        os.utime(stamp, (before + 1000, before + 1000))
        after = perf_cache.corpus_mtime(tmp_path)
        assert after > before, f"{name} is written by {writer}; without it the caches go stale"

    def test_the_token_is_the_max_so_one_stale_file_cannot_mask_a_fresh_one(self, tmp_path: Path):
        import os

        old, new = tmp_path / "corpus_run_summary.json", tmp_path / "corpus_edges_stamp.json"
        old.write_text("{}", encoding="utf-8")
        new.write_text("{}", encoding="utf-8")
        base = perf_cache.corpus_mtime(tmp_path)
        os.utime(old, (base + 10, base + 10))
        os.utime(new, (base + 5000, base + 5000))
        # First-found ordering would return the run-summary's older stamp and lose the edge write.
        assert perf_cache.corpus_mtime(tmp_path) == pytest.approx(base + 5000)


def _wait_until(predicate) -> None:
    deadline = time.time() + 5
    while not predicate():
        assert time.time() < deadline, "timed out"
        time.sleep(0.01)


class TestStaleWhileRebuilding:
    def test_the_previous_value_serves_while_one_background_build_runs(self):
        perf_cache.get_or_compute("swr", "k", 1.0, lambda: "old", stale_ok=True)
        started, release = threading.Event(), threading.Event()

        def compute():
            started.set()
            assert release.wait(5)
            return "new"

        assert perf_cache.get_or_compute("swr", "k", 2.0, compute, stale_ok=True) == "old"
        assert started.wait(5)
        # A second reader during the build gets the old value too, and starts no second build.
        assert perf_cache.get_or_compute("swr", "k", 2.0, lambda: "dup", stale_ok=True) == "old"
        release.set()
        _wait_until(lambda: perf_cache.cache_peek("swr", "k", 2.0) is not None)
        assert perf_cache.get_or_compute("swr", "k", 2.0, lambda: "dup", stale_ok=True) == "new"
        assert perf_cache.stats()["swr"]["stale_served"] == 2

    def test_nothing_cached_still_computes_inline(self):
        assert perf_cache.get_or_compute("swr", "k", 1.0, lambda: "v", stale_ok=True) == "v"

    def test_without_stale_ok_a_new_token_gets_the_current_value(self):
        perf_cache.get_or_compute("plain", "k", 1.0, lambda: "old")
        assert perf_cache.get_or_compute("plain", "k", 2.0, lambda: "new") == "new"

    def test_fresh_gets_the_current_value_where_stale_is_allowed(self):
        perf_cache.get_or_compute("swr", "k", 1.0, lambda: "old", stale_ok=True)
        with perf_cache.fresh():
            assert perf_cache.get_or_compute("swr", "k", 2.0, lambda: "new", stale_ok=True) == "new"

    def test_a_compute_reads_its_inputs_fresh(self):
        # An outer value cached under a new token must not be built from a stale inner one.
        perf_cache.get_or_compute("inner", "k", 1.0, lambda: "inner-old", stale_ok=True)

        def outer():
            inner = perf_cache.get_or_compute("inner", "k", 2.0, lambda: "inner-new", stale_ok=True)
            return f"outer({inner})"

        assert perf_cache.get_or_compute("outer", "k", 2.0, outer) == "outer(inner-new)"

    def test_a_failed_background_build_keeps_the_old_value_and_the_next_reader_retries(self):
        perf_cache.get_or_compute("swr", "k", 1.0, lambda: "old", stale_ok=True)

        def boom():
            raise RuntimeError("build failed")

        assert perf_cache.get_or_compute("swr", "k", 2.0, boom, stale_ok=True) == "old"
        _wait_until(lambda: ("swr", "k", 2.0) not in perf_cache._IN_FLIGHT)
        assert perf_cache.get_or_compute("swr", "k", 2.0, lambda: "new", stale_ok=True) == "old"
        _wait_until(lambda: perf_cache.cache_peek("swr", "k", 2.0) is not None)
        assert perf_cache.cache_peek("swr", "k", 2.0) == "new"

    def test_a_build_that_needs_a_key_queued_behind_it_builds_that_key_itself(self):
        # One builder: if the outer build waited for the queued inner one, nothing would build it.
        perf_cache.get_or_compute("inner", "k", 1.0, lambda: "inner-old", stale_ok=True)
        perf_cache.get_or_compute("outer", "k", 1.0, lambda: "outer-old", stale_ok=True)
        release = threading.Event()

        def outer():
            assert release.wait(5)
            inner = perf_cache.get_or_compute("inner", "k", 2.0, lambda: "inner-new", stale_ok=True)
            return f"outer({inner})"

        assert perf_cache.get_or_compute("outer", "k", 2.0, outer, stale_ok=True) == "outer-old"
        assert (
            perf_cache.get_or_compute("inner", "k", 2.0, lambda: "inner-new", stale_ok=True)
            == "inner-old"
        )
        release.set()
        _wait_until(lambda: perf_cache.cache_peek("outer", "k", 2.0) is not None)
        assert perf_cache.cache_peek("outer", "k", 2.0) == "outer(inner-new)"
        assert perf_cache.cache_peek("inner", "k", 2.0) == "inner-new"

    def test_one_background_build_runs_at_a_time(self):
        running, peak = [0], [0]
        lock = threading.Lock()

        def compute():
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.02)
            with lock:
                running[0] -= 1
            return "new"

        for i in range(5):
            perf_cache.get_or_compute("swr", i, 1.0, lambda: "old", stale_ok=True)
        for i in range(5):
            assert perf_cache.get_or_compute("swr", i, 2.0, compute, stale_ok=True) == "old"
        _wait_until(lambda: all(perf_cache.cache_peek("swr", i, 2.0) for i in range(5)))
        assert peak[0] == 1
