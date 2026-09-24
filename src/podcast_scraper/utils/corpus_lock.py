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
    """Return True if a process with *pid* is running on THIS host.

    Only meaningful for a holder recorded on this same hostname — callers should use
    :func:`_holder_is_reclaimable`.
    """
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


def _holder_is_reclaimable(holder: Dict[str, object]) -> bool:
    """True only when we can PROVE the recorded holder is gone.

    A PID is meaningful only on the host that recorded it. Every container's init is
    PID 1, so a holder written as ``{"pid": 1, "hostname": "f0c5969c4647"}`` — exactly
    what a ``docker compose run`` reprocess records — reads as ALIVE from any other
    container, because that container has its own PID 1. The hostname was already being
    recorded and simply never compared, so a hard-dead container's holder could never be
    reclaimed automatically.

    Unknown is not dead: a holder from a different hostname returns False rather than
    guessing. That is the safe direction — a false "reclaimable" would admit two writers
    to one corpus, the data-loss shape this lock exists to prevent. The OS ``flock``
    stays the real mutual exclusion; this only decides whether a retry is worth trying.
    """
    pid = holder.get("pid")
    if not isinstance(pid, int):
        return False
    recorded_host = str(holder.get("hostname") or "")
    if recorded_host and recorded_host != socket.gethostname():
        return False
    return not _is_pid_alive(pid)


def corpus_lock_state(corpus_parent: Union[str, Path]) -> Dict[str, object]:
    """Answer "is anything holding this corpus right now?" — the ops-facing question.

    THE LOCK FILE'S EXISTENCE MEANS NOTHING. ``filelock`` never unlinks it on release:
    only the ``flock`` is dropped and the holder sidecar removed, so
    ``.podcast_scraper.lock`` sits there, 0 bytes, after every run that has ever
    succeeded. On 2026-09-22..24 its presence was read as "locked" repeatedly, concluded
    to be a stale lock, and hand-removed — and one of those readings reached a runbook as
    "the lock does NOT release on cancel — verified twice". It does release. Nothing was
    stale. The removals were unnecessary.

    ``held`` is decided by probing the ``flock``, which is the only authority. Returns
    ``{held, holder, lock_path, lock_file_exists, reason}`` and never raises.
    """
    root = Path(corpus_parent).expanduser().resolve()
    lock_path = root / LOCK_BASENAME
    holder_path = root / _HOLDER_BASENAME
    holder = _read_holder(holder_path)
    out: Dict[str, object] = {
        "lock_path": str(lock_path),
        "lock_file_exists": lock_path.exists(),
        "holder": holder,
    }
    if not corpus_lock_enabled():
        out.update(held=False, reason="locking disabled via PODCAST_SCRAPER_CORPUS_LOCK")
        return out
    if not lock_path.exists():
        out.update(held=False, reason="no lock file — this corpus has never been locked")
        return out
    try:
        from filelock import FileLock, Timeout

        probe = FileLock(str(lock_path), timeout=0)
        try:
            probe.acquire()
        except Timeout:
            out.update(
                held=True,
                reason="flock is held by a live process"
                + (f" ({holder})" if holder else " (no holder file)"),
            )
            return out
        probe.release()
        out.update(
            held=False,
            reason=(
                "lock file exists but the flock is FREE — the normal state after any "
                "completed run; the file is never unlinked on release"
            ),
        )
        return out
    except Exception as exc:  # noqa: BLE001 — a status probe must never raise
        # UNKNOWN IS NOT "NOT HELD". Reporting held=False here would be the same unsafe
        # direction this function exists to end: a PermissionError (non-root probing a
        # root-owned corpus — a shape prod has had) would read as "nothing is running" while
        # a run holds the lock. Callers must treat None as "go look", not as a green light.
        out.update(held=None, reason=f"COULD NOT PROBE the flock: {type(exc).__name__}: {exc}")
        return out


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
            # Hostname-aware: a PID from ANOTHER host tells us nothing, and every
            # container's PID 1 exists, so the old bare `_is_pid_alive` never reclaimed a
            # dead container's holder. See `_holder_is_reclaimable`.
            if _holder_is_reclaimable(holder):
                pid = holder.get("pid")
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
