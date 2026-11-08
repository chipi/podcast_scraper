"""Hand memory a corpus-wide build freed back to the operating system."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging

logger = logging.getLogger(__name__)

_libc = None


def release_freed_memory() -> None:
    """Ask glibc to return free heap pages to the OS (``malloc_trim``); a no-op elsewhere.

    A corpus-wide build (the entity registry, the corpus graph) allocates and frees gigabytes, and
    glibc keeps the freed pages in the process: the MCP server held about 2 GB at idle on prod
    (2026-11-08). On a prod snapshot, trimming after the registry build took the process from
    4,089 to 2,733 MiB resident with the same live objects.
    """
    global _libc
    try:
        if _libc is None:
            name = ctypes.util.find_library("c")
            if not name:
                return
            _libc = ctypes.CDLL(name)
        trim = getattr(_libc, "malloc_trim", None)
        if trim is not None:
            trim(0)
    except OSError as exc:  # not glibc, or no libc to load: nothing to release
        logger.debug("malloc_trim unavailable: %s", exc)
