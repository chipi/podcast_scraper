"""Advisory exclusive lock for corpus directories (single-writer discipline).

Every pipeline run that touches a corpus directory must hold this lock for
the duration of the run — single-feed and multi-feed alike.  The sweep-prod-audio
workflow deletes media files; an undetected concurrent run is a data-loss shape.

Holder file
-----------
On acquire, a JSON sidecar ``<corpus>/.podcast_scraper.lock.holder`` is written
with ``{"pid": <int>, "started_at": "<iso>", "hostname": "<str>"}``.  Ops
workflows can read this without shelling into a container to answer "is anything
touching the corpus right now?"

Stale-lock detection
--------------------
A lock whose recorded PID is not alive is reclaimable.  On acquire failure the
code checks the holder file; if the PID is dead the lock file is removed and
the acquire retried (logged at WARNING level).  Two concurrent starters that
both detect a stale lock cannot both acquire: the second ``FileLock.acquire``
after the unlink will still race through the OS-level ``flock``/``fcntl``,
so only one caller gets the fd.

Loud contention
---------------
When the lock is genuinely held (live PID), the ``RuntimeError`` message names
the holder PID, hostname, and start time so ops can find the running container
without further investigation.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

LOCK_BASENAME = ".podcast_scraper.lock"
_HOLDER_BASENAME = ".podcast_scraper.lock.holder"


def corpus_lock_enabled() -> bool:
    """Return False when ``PODCAST_SCRAPER_CORPUS_LOCK`` is ``0``/``false``/``off``."""
    raw = os.environ.get("PODCAST_SCRAPER_CORPUS_LOCK", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _is_pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* is running on this host."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM: process exists but we can't signal it — still alive.
        return True
    except OSError:
        return False


def _read_holder(holder_path: Path) -> Optional[Dict[str, object]]:
    """Parse the holder JSON; return None on any error."""
    try:
        raw = json.loads(holder_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw  # type: ignore[return-value]
        return None
    except Exception:  # noqa: BLE001
        return None


def _write_holder(holder_path: Path) -> None:
    """Record this process as the current lock holder."""
    data = {
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }
    holder_path.write_text(json.dumps(data), encoding="utf-8")


def _remove_holder(holder_path: Path) -> None:
    """Remove the holder file; ignore errors (e.g. already deleted)."""
    try:
        holder_path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001  # pragma: no cover - best-effort cleanup
        pass


def _contention_message(lock_path: Path, holder_path: Path) -> str:
    """Build the loud contention error message naming the current holder."""
    holder = _read_holder(holder_path)
    if holder:
        pid = holder.get("pid", "?")
        hostname = holder.get("hostname", "?")
        started = holder.get("started_at", "?")
        return (
            f"Corpus directory is locked by PID {pid} on {hostname} "
            f"(started {started}). Wait for the other process to finish or "
            "set PODCAST_SCRAPER_CORPUS_LOCK=0 to disable locking. "
            f"Lock file: {lock_path}"
        )
    return (
        f"Corpus directory is locked ({lock_path}). Wait for the other process "
        "or set PODCAST_SCRAPER_CORPUS_LOCK=0 to disable locking."
    )


@contextmanager
def corpus_parent_lock(
    corpus_parent: Union[str, Path],
    *,
    logger: Optional[logging.Logger] = None,
) -> Iterator[None]:
    """Hold an exclusive lock on ``corpus_parent`` for the duration of the block.

    ``corpus_parent`` may be a path string or a ``pathlib.Path``.

    Creates ``corpus_parent`` if missing.  Uses ``filelock.FileLock`` (non-blocking
    acquire).  Set ``PODCAST_SCRAPER_CORPUS_LOCK=0`` to disable for tests or
    advanced workflows.

    On acquire:
    - Writes a holder file recording this PID/hostname/start-time.
    - If the lock is held by a dead PID, reclaims it (logged at WARNING).
    - If held by a live process, raises ``RuntimeError`` naming the holder.

    On release:
    - Removes the holder file.

    Raises:
        RuntimeError: If the lock is already held by another live process.
    """
    if not corpus_lock_enabled():
        yield
        return

    root = Path(corpus_parent).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / LOCK_BASENAME
    holder_path = root / _HOLDER_BASENAME

    from filelock import FileLock, Timeout

    lock = FileLock(str(lock_path), timeout=0)
    try:
        lock.acquire()
    except Timeout:  # pragma: no cover - contention/crash-recovery path, not hit by single-run e2e
        # Check whether the holder is still alive; reclaim if dead.
        holder = _read_holder(holder_path)
        if holder:
            pid = holder.get("pid")
            if isinstance(pid, int) and not _is_pid_alive(pid):
                started = holder.get("started_at", "?")
                hostname = holder.get("hostname", "?")
                if logger is not None:
                    logger.warning(
                        "Corpus lock held by dead PID %s (hostname=%s, started=%s) — "
                        "reclaiming lock at %s",
                        pid,
                        hostname,
                        started,
                        lock_path,
                    )
                # Remove the stale lock file so the re-acquire can succeed.
                try:
                    lock_path.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
                _remove_holder(holder_path)
                # Re-acquire after reclaim.
                lock2 = FileLock(str(lock_path), timeout=0)
                try:
                    lock2.acquire()
                except Timeout as exc2:  # pragma: no cover - rare reclaim double-race
                    # Another process beat us to the reclaim; give the loud message.
                    msg = _contention_message(lock_path, holder_path)
                    if logger is not None:
                        logger.error("%s", msg)
                    raise RuntimeError(msg) from exc2
                # Reclaim succeeded — continue with lock2.
                _write_holder(holder_path)
                try:
                    yield
                finally:
                    lock2.release()
                    _remove_holder(holder_path)
                return

        msg = _contention_message(lock_path, holder_path)
        if logger is not None:
            logger.error("%s", msg)
        raise RuntimeError(msg)

    _write_holder(holder_path)
    try:
        yield
    finally:
        lock.release()
        _remove_holder(holder_path)
