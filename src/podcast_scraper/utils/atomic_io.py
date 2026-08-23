"""Write a file without ever leaving a half-written one at the destination.

WHY THIS EXISTS AS A SHARED MODULE
``open(path, "w")`` truncates the destination the moment it is called, so the previous contents
are gone before the first byte of the new contents is written. Anything that interrupts the write
— a kill, a full disk, a serialization error — leaves a truncated file at the real path. For a
cache or a status file that is an annoyance; for a corpus artifact it is data loss, and for a
``gi.json`` it is worse than that: ``gi.repair`` refuses to rewrite an artifact it cannot parse,
so a kill mid-repair makes that episode permanently unrepairable by the only tool that repairs it.

This repo had already worked that out SIX times independently — ``monitor/status.py``,
``utils/audio_cache.py``, ``utils/storage_backend.py``, and three ``upgrade/migrations/*``, each
with its own copy of tmp-then-``os.replace``. The GI and KG artifact writers, which need it most,
had none. This module exists so the next writer that needs it is the seventh USE rather than the
seventh COPY.

THE TWO NON-OBVIOUS REQUIREMENTS
1. The temp file must live in the SAME DIRECTORY as the destination. ``os.replace`` is atomic
   only within a single filesystem; a temp file in ``/tmp`` crossing a mount point degrades to
   copy-then-delete and reintroduces exactly the torn-write window this is meant to close.
2. ``fsync`` before the replace. Without it the rename can reach the disk before the data does,
   so a power loss leaves an intact directory entry pointing at unwritten blocks — the same
   truncated file, arrived at by a different route.

WHAT THIS DOES NOT PROMISE
Durability of the *directory entry* itself (that needs an fsync on the parent directory) and
protection against two processes writing the same path concurrently — the last replace wins,
whole and valid, but which one wins is unspecified. Neither matters for artifact writes: the
corpus lock serializes writers, and a whole-but-older artifact is a re-run, not corruption.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, payload: Any, **json_kwargs: Any) -> None:
    """Serialize *payload* to *path* as JSON, atomically.

    On success *path* holds the complete new document. On ANY failure *path* is left exactly as
    it was — including not existing — and no temp file survives.

    ``json_kwargs`` are passed straight through to ``json.dump`` with no defaults applied, so
    each caller keeps its own formatting contract (``ensure_ascii``, ``sort_keys``, ``indent``,
    ``allow_nan``) rather than silently inheriting this module's opinion. Changing a corpus
    artifact's byte layout is not something an atomicity fix should do as a side effect.

    Args:
        path: Destination file. Parent directories are created if missing.
        payload: Any JSON-serializable object.
        **json_kwargs: Forwarded verbatim to ``json.dump``.

    Raises:
        Whatever ``json.dump`` or the filesystem raises. The destination is untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Named after the destination so an orphan left by SIGKILL (which no cleanup can catch) is
    # traceable to its origin. The leading dot hides it, and the ``.tmp`` suffix keeps it out of
    # the ``rglob("*.gi.json")`` / ``rglob("*.metadata.json")`` scans the corpus gates run — an
    # orphan that matched those would be read as a real artifact.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, **json_kwargs)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt during a long corpus repair is the
        # realistic interruption, and it must not leave the temp file behind either.
        try:
            if tmp_path.is_file():
                tmp_path.unlink()
        except OSError:
            pass
        raise
