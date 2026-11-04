"""Keep app-data files owned by the app user when an operator runs a command as root.

Prod admin commands run through ``docker exec`` (root by default) while the API runs as uid 1000.
A file root creates under the app data dir (``profile.json``, ``.profile.lock``, JSONL logs) then
can no longer be written by the API. Two guards, both no-ops unless the process is root:

* :func:`drop_root_to_owner` at a CLI entry point: become the data dir's owner before any write.
* :func:`match_data_owner` / :class:`OwnedFileLock` on the write paths: whatever root still
  creates (an ``exec python -c`` that never went through a CLI entry point) is chowned to the
  owner of the data dir it lives under.
"""

from __future__ import annotations

import os
import pwd
import sys
from pathlib import Path

from filelock import FileLock

KEEP_ROOT_ENV = "PODCAST_KEEP_ROOT"


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _lstat(path: Path) -> os.stat_result:
    return os.lstat(path)


def drop_root_to_owner(path: Path) -> None:
    """When root and *path* is owned by another user, become that user (one stderr line).

    ``PODCAST_KEEP_ROOT=1`` (the convention of ``docker/api/entrypoint.sh``) skips the drop.
    Not root, a missing path, or a root-owned path: nothing happens.
    """
    if not _is_root() or os.environ.get(KEEP_ROOT_ENV) == "1":
        return
    try:
        st = _lstat(Path(path))
    except OSError:
        return
    if st.st_uid == 0:
        return
    try:
        pw = pwd.getpwuid(st.st_uid)
        os.initgroups(pw.pw_name, st.st_gid)
        home = pw.pw_dir
    except (KeyError, OSError):
        os.setgroups([])
        home = "/tmp"  # nosec B108 - a writable HOME for a uid that has no passwd entry
    os.setgid(st.st_gid)
    os.setuid(st.st_uid)
    os.environ["HOME"] = home
    print(
        f"dropped root to uid {st.st_uid} (owner of {path}) so files stay writable by the app; "
        f"set {KEEP_ROOT_ENV}=1 to stay root",
        file=sys.stderr,
    )


def match_data_owner(path: Path) -> None:
    """Chown *path* (and root-owned parents the write created) to the nearest non-root ancestor.

    A no-op unless running as root. Only root-owned entries are touched, so a file someone else
    owns is never taken over, and a data dir that is itself root-owned (dev) changes nothing.
    """
    if not _is_root():
        return
    chain: list[Path] = []
    cur = Path(path)
    owner: tuple[int, int] | None = None
    while True:
        try:
            st = _lstat(cur)
        except OSError:
            st = None
        if st is not None:
            if st.st_uid != 0:
                owner = (st.st_uid, st.st_gid)
                break
            chain.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    if owner is None:
        return
    for entry in chain:
        try:
            os.chown(entry, owner[0], owner[1], follow_symlinks=False)
        except OSError:
            continue


class OwnedFileLock(FileLock):
    """A ``FileLock`` whose lock file (and any dirs made for it) end up owned by the data owner."""

    def acquire(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Acquire, then hand the lock file to the data owner (root only)."""
        proxy = super().acquire(*args, **kwargs)
        match_data_owner(Path(self.lock_file))
        return proxy
