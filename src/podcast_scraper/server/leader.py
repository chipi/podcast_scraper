"""One leader among a host's API workers, for the work that must run once per host.

With ``serve --workers N`` each worker is its own process, and anything started from the app's
lifespan starts N times: the cron scheduler, the job-queue sweeper. A file lock picks one leader:
``fcntl.flock`` on a file every worker of the host opens. The kernel releases the lock when the
leader's process dies, however it dies, and a follower takes it over at its next try (every
``RETRY_SECONDS``).

A Postgres advisory lock would do the same across hosts, but a session advisory lock breaks behind
PgBouncer in transaction mode (ADR-173). Workers of one ``serve`` share a host, which is all a file
lock needs. With one worker (the default) the process is always the leader and nothing waits.
"""

from __future__ import annotations

import fcntl
import logging
import os
import threading
from pathlib import Path
from typing import Callable, IO

logger = logging.getLogger(__name__)

RETRY_SECONDS = 30.0

#: Set by ``serve --workers N`` for its workers; absent or 1 means a single process.
ENV_WORKERS = "PODCAST_SERVE_WORKERS"
#: The lock file; every worker of one ``serve`` must name the same one.
ENV_LOCK_FILE = "PODCAST_SERVE_LEADER_LOCK"


def worker_count() -> int:
    """The number of workers this process is one of (1 when unset or malformed)."""
    try:
        return max(1, int(os.environ.get(ENV_WORKERS, "1")))
    except ValueError:
        return 1


def default_lock_file(port: int) -> Path:
    """The lock file for a ``serve`` on ``port`` (one leader per listening port and host)."""
    base = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
    return base / f"podcast-serve-leader-{port}.lock"


class Leadership:
    """Try for the lock now, then every ``retry_seconds`` until held; ``on_lead`` runs once, on
    the thread that took the lock. ``stop()`` ends the retries and releases a held lock."""

    def __init__(
        self,
        lock_file: Path,
        on_lead: Callable[[], None],
        *,
        retry_seconds: float = RETRY_SECONDS,
    ) -> None:
        self._path = lock_file
        self._on_lead = on_lead
        self._retry = retry_seconds
        self._fd: IO[bytes] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_leader(self) -> bool:
        return self._fd is not None

    def _try(self) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = open(self._path, "ab")  # noqa: SIM115 - held open for the process's leadership
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fd.close()
            return False
        self._fd = fd
        logger.info("leader: pid %s leads (%s)", os.getpid(), self._path)
        try:
            self._on_lead()
        except BaseException:
            # A leader whose work did not start must not keep the lock: release it, so this
            # process or another takes over at the next try.
            self._release()
            raise
        return True

    def _release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None

    def start(self) -> None:
        """Lead now if the lock is free, else retry on a daemon thread. A failed start does not
        fail the worker's startup: it is logged and retried like a held lock."""
        try:
            if self._try():
                return
        except Exception:  # noqa: BLE001 - the app serves requests without leading
            logger.exception("leader: starting the leader's work failed; retrying")
        logger.info("leader: pid %s follows; retrying every %.0fs", os.getpid(), self._retry)
        self._thread = threading.Thread(target=self._loop, name="leader-retry", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._retry):
            try:
                if self._try():
                    return
            except Exception:  # noqa: BLE001 - keep trying; a failed start logs and retries
                logger.exception("leader: taking over failed; retrying")

    def stop(self) -> None:
        self._stop.set()
        self._release()
