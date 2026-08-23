"""Unit tests for corpus_lock — #1810 concurrent-run guard.

Covers:
- Single-feed run acquires the lock (previously only multi-feed did).
- Second acquisition while the lock is held raises RuntimeError with the
  holder PID/hostname/start-time in the message.
- A lock whose holder PID is dead is reclaimed and a fresh acquire succeeds.
- Holder file is written on acquire and removed on release.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _acquire_once(corpus_dir: Path) -> None:
    """Context-manager helper: acquire + immediately release (checks no exception)."""
    from podcast_scraper.utils.corpus_lock import corpus_parent_lock

    with corpus_parent_lock(corpus_dir):
        pass


# ---------------------------------------------------------------------------
# 1. Single-feed acquire — lock file and holder file are created and cleaned up
# ---------------------------------------------------------------------------


def test_holder_file_written_on_acquire_and_removed_on_release(tmp_path: Path) -> None:
    from podcast_scraper.utils.corpus_lock import (
        _HOLDER_BASENAME,
        corpus_parent_lock,
    )

    with corpus_parent_lock(tmp_path):
        assert (tmp_path / _HOLDER_BASENAME).is_file(), "holder file must exist while lock is held"
        holder = json.loads((tmp_path / _HOLDER_BASENAME).read_text())
        assert holder["pid"] == os.getpid()
        assert "hostname" in holder
        assert "started_at" in holder

    assert not (tmp_path / _HOLDER_BASENAME).exists(), "holder file must be gone after release"
    # Lock file itself may or may not linger (filelock behaviour); that is ok.


# ---------------------------------------------------------------------------
# 2. Contention: second acquire while lock held → RuntimeError naming holder
# ---------------------------------------------------------------------------


def test_contention_raises_runtime_error_with_holder_info(tmp_path: Path) -> None:
    from filelock import FileLock

    from podcast_scraper.utils.corpus_lock import (
        _HOLDER_BASENAME,
        corpus_parent_lock,
        LOCK_BASENAME,
    )

    lock_path = tmp_path / LOCK_BASENAME
    holder_path = tmp_path / _HOLDER_BASENAME

    # Simulate a live concurrent holder by pre-acquiring via FileLock and
    # writing a holder file that names this very PID (it IS alive).
    outer = FileLock(str(lock_path), timeout=0)
    outer.acquire()
    holder_data = {
        "pid": os.getpid(),
        "hostname": "test-host",
        "started_at": "2026-01-01T00:00:00+00:00",
    }
    holder_path.write_text(json.dumps(holder_data))

    try:
        with pytest.raises(RuntimeError) as exc_info:
            with corpus_parent_lock(tmp_path):
                pass  # must never reach here
        msg = str(exc_info.value)
        assert str(os.getpid()) in msg, f"PID not in error message: {msg!r}"
        assert "test-host" in msg, f"hostname not in error message: {msg!r}"
        assert "2026-01-01" in msg, f"start time not in error message: {msg!r}"
    finally:
        outer.release()
        holder_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. Stale-lock (dead PID) is reclaimed; fresh acquire succeeds
# ---------------------------------------------------------------------------


def test_stale_lock_with_dead_pid_is_reclaimed(tmp_path: Path) -> None:
    from filelock import FileLock

    from podcast_scraper.utils.corpus_lock import (
        _HOLDER_BASENAME,
        corpus_parent_lock,
        LOCK_BASENAME,
    )

    lock_path = tmp_path / LOCK_BASENAME
    holder_path = tmp_path / _HOLDER_BASENAME

    # Create a stale lock (fd closed → flock released, but the file stays).
    stale = FileLock(str(lock_path), timeout=0)
    stale.acquire()
    stale.release()
    # Write a holder that names a PID that is certainly not alive.
    dead_pid = 2**30  # implausibly large PID
    holder_path.write_text(
        json.dumps(
            {"pid": dead_pid, "hostname": "ghost-host", "started_at": "2025-01-01T00:00:00+00:00"}
        )
    )

    # Patch _is_pid_alive to report dead_pid as dead (safe: it's enormous).
    import podcast_scraper.utils.corpus_lock as _mod

    original = _mod._is_pid_alive

    def fake_alive(pid: int) -> bool:
        if pid == dead_pid:
            return False
        return original(pid)

    with patch.object(_mod, "_is_pid_alive", side_effect=fake_alive):
        # Should NOT raise — stale lock must be reclaimed.
        reclaimed = False
        with corpus_parent_lock(tmp_path):
            reclaimed = True
            # Holder file must be refreshed with OUR pid.
            holder = json.loads(holder_path.read_text())
            assert holder["pid"] == os.getpid()

    assert reclaimed, "lock was not reclaimed"
    assert not holder_path.exists(), "holder file must be cleaned up after reclaim"


# ---------------------------------------------------------------------------
# 4. Lock disabled (PODCAST_SCRAPER_CORPUS_LOCK=0) is a no-op
# ---------------------------------------------------------------------------


def test_lock_disabled_env_var_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from podcast_scraper.utils.corpus_lock import _HOLDER_BASENAME, corpus_parent_lock

    monkeypatch.setenv("PODCAST_SCRAPER_CORPUS_LOCK", "0")

    with corpus_parent_lock(tmp_path):
        # Holder file must NOT be written when locking is disabled.
        assert not (tmp_path / _HOLDER_BASENAME).exists()
