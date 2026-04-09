"""Advisory exclusive lock for multi-feed corpus parent (single-writer discipline)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

LOCK_BASENAME = ".podcast_scraper.lock"


def corpus_lock_enabled() -> bool:
    """Return False when ``PODCAST_SCRAPER_CORPUS_LOCK`` is ``0``/``false``/``off``."""
    raw = os.environ.get("PODCAST_SCRAPER_CORPUS_LOCK", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


@contextmanager
def corpus_parent_lock(
    corpus_parent: Union[str, Path],
    *,
    logger: Optional[logging.Logger] = None,
) -> Iterator[None]:
    """Hold an exclusive lock on ``corpus_parent`` for the duration of the block.

    ``corpus_parent`` may be a path string (e.g. normalized CLI output dir) or a
    ``pathlib.Path``.

    Creates ``corpus_parent`` if missing. Uses ``filelock.FileLock`` (non-blocking
    acquire). Set ``PODCAST_SCRAPER_CORPUS_LOCK=0`` to disable for tests or
    advanced workflows.

    Raises:
        RuntimeError: If the lock is already held by another process.
    """
    if not corpus_lock_enabled():
        yield
        return

    root = Path(corpus_parent).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / LOCK_BASENAME

    from filelock import FileLock, Timeout

    lock = FileLock(str(lock_path), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        msg = (
            f"Corpus directory is locked ({lock_path}). Wait for the other process "
            "or set PODCAST_SCRAPER_CORPUS_LOCK=0 to disable locking."
        )
        if logger is not None:
            logger.error("%s", msg)
        raise RuntimeError(msg) from exc
    try:
        yield
    finally:
        lock.release()
