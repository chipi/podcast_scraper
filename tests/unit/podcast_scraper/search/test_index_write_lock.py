"""One index writer at a time, and a time limit that fits a real backlog.

Prod 2026-10-07: the post-run index update timed out at 1800 s twice (367 changed episodes to
re-embed), and the index had no write lock, so a manual catch-up rebuild could overlap a pipeline
run's own update.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from filelock import FileLock

from podcast_scraper.search import indexer, reindex

pytestmark = [pytest.mark.unit]


def _calls(monkeypatch) -> list:
    seen: list = []

    def fake_locked(out, cfg, stats, rebuild, backbone):
        seen.append(out)
        return stats

    monkeypatch.setattr(indexer, "_index_corpus_locked", fake_locked)
    return seen


def test_an_update_waits_for_the_one_holding_the_lock_and_gives_up_non_fatally(
    tmp_path: Path, monkeypatch
) -> None:
    seen = _calls(monkeypatch)
    monkeypatch.setenv("PODCAST_INDEX_LOCK_WAIT_SECONDS", "0.3")
    held = FileLock(str(tmp_path / "search" / ".index-write.lock"))
    (tmp_path / "search").mkdir()
    with held:
        stats = indexer.index_corpus(str(tmp_path), cfg=None)  # type: ignore[arg-type]
    assert seen == []
    assert stats.errors and "gave up" in stats.errors[0]


def test_another_process_holding_the_lock_blocks_the_update(tmp_path: Path, monkeypatch) -> None:
    # The real overlap is two PROCESSES: the pipeline's index subprocess and the API's thread.
    seen = _calls(monkeypatch)
    monkeypatch.setenv("PODCAST_INDEX_LOCK_WAIT_SECONDS", "0.3")
    (tmp_path / "search").mkdir()
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import sys, time
                from filelock import FileLock
                with FileLock({str(tmp_path / "search" / ".index-write.lock")!r}):
                    print("held", flush=True)
                    time.sleep(5)
                """),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        stats = indexer.index_corpus(str(tmp_path), cfg=None)  # type: ignore[arg-type]
    finally:
        holder.kill()
        holder.wait()
    assert seen == [] and stats.errors


def test_the_lock_is_released_after_an_update_even_one_that_raises(
    tmp_path: Path, monkeypatch
) -> None:
    seen = _calls(monkeypatch)
    indexer.index_corpus(str(tmp_path), cfg=None)  # type: ignore[arg-type]
    assert seen == [tmp_path]

    def boom(*_a, **_k):
        raise RuntimeError("index failed")

    monkeypatch.setattr(indexer, "_index_corpus_locked", boom)
    with pytest.raises(RuntimeError):
        indexer.index_corpus(str(tmp_path), cfg=None)  # type: ignore[arg-type]
    other = FileLock(str(tmp_path / "search" / ".index-write.lock"))
    other.acquire(timeout=0)  # free again
    other.release()


@pytest.mark.parametrize(
    ("env", "expected"),
    [(None, 7200.0), ("3600", 3600.0), ("nonsense", 7200.0), ("0", 7200.0), ("-5", 7200.0)],
)
def test_the_index_time_limit_is_two_hours_unless_overridden(env, expected, monkeypatch) -> None:
    if env is None:
        monkeypatch.delenv("PODCAST_INDEX_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("PODCAST_INDEX_TIMEOUT_SECONDS", env)
    assert reindex.index_timeout_seconds() == expected


def test_the_subprocess_uses_the_resolved_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_INDEX_TIMEOUT_SECONDS", "123")
    seen: dict = {}

    class Done:
        returncode = 0

    def fake_run(argv, env, timeout, check):
        seen["timeout"] = timeout
        return Done()

    monkeypatch.setattr(reindex.subprocess, "run", fake_run)

    class Cfg:
        def model_dump(self, mode):
            return {}

    assert reindex.run_index_in_subprocess(str(tmp_path), Cfg())  # type: ignore[arg-type]
    assert seen["timeout"] == 123.0
