"""Pluggable storage backend for the raw-audio archive (#1199).

The #947 audio cache persists raw episode audio so reprocessing reuses it
instead of re-fetching from the (mutating / expiring) live feed. This module
lets that archive live on the **local filesystem** (default) or on **remote
object storage** (rclone-backed: a Hetzner Storage Box over SFTP, or any
S3-compatible / rclone remote), selected by ``audio_storage_backend``.

A backend is keyed by a POSIX ``rel_key`` — the sharded GUID path the audio
cache computes, e.g. ``sha256/aa/bb/<digest>.mp3`` — and does three things:

* ``exists(rel_key)`` — is the object present (non-empty)?
* ``upload(src_path, rel_key)`` — put a local file at the key (dedupe by
  existence); returns success.
* ``download(rel_key, dest_path)`` — fetch the object to a local file.

Everything above the backend (GUID hashing, extension probing, call-site
wiring) is transport-agnostic and lives in :mod:`audio_cache`.

Failure contract (the Deepgram lesson, #1195): a **misconfigured** remote fails
loud at construction — a missing rclone binary or an empty remote name raises
``StorageBackendError`` rather than silently falling through to "no archive".
An **individual** upload/download failure is logged at ERROR (visible, never
silent) but is best-effort per episode, so one transient rclone hiccup cannot
break ingestion.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Default per-operation timeout for a single rclone invocation (seconds). A
# single-object copy of a ~50 MB episode is well under this; the ceiling guards
# against a hung network mount stalling the pipeline.
DEFAULT_RCLONE_TIMEOUT_S = 300


class StorageBackendError(RuntimeError):
    """A storage backend is misconfigured or unusable — raised loud, never swallowed."""


class StorageBackend(ABC):
    """Key-addressed blob store for the raw-audio archive."""

    @abstractmethod
    def exists(self, rel_key: str) -> bool:
        """True if a non-empty object is stored at ``rel_key``."""

    @abstractmethod
    def upload(self, src_path: str, rel_key: str) -> bool:
        """Store ``src_path`` at ``rel_key`` (dedupe by existence). Return success."""

    @abstractmethod
    def download(self, rel_key: str, dest_path: str) -> bool:
        """Fetch the object at ``rel_key`` into ``dest_path``. Return success."""

    def describe(self) -> str:
        """Short human label for logs."""
        return type(self).__name__


class LocalStorageBackend(StorageBackend):
    """Filesystem backend — the historical #947 behaviour, ``rel_key`` under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def describe(self) -> str:
        return f"local:{self.root}"

    def _path(self, rel_key: str) -> Path:
        return self.root / rel_key

    def exists(self, rel_key: str) -> bool:
        p = self._path(rel_key)
        try:
            return p.is_file() and p.stat().st_size > 0
        except OSError:
            return False

    def upload(self, src_path: str, rel_key: str) -> bool:
        if not src_path or not os.path.isfile(src_path):
            return False
        try:
            if os.path.getsize(src_path) <= 0:
                return False
        except OSError:
            return False
        dest = self._path(rel_key)
        try:
            if dest.is_file() and dest.stat().st_size > 0:
                return True  # already archived (dedupe)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Stage to a sibling ``.tmp`` then atomic-rename so a partial or
            # racing write never leaves a torn archive file.
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            shutil.copy2(src_path, tmp)
            os.replace(tmp, dest)
            return True
        except OSError as exc:
            logger.error("audio archive (local): failed to store %s -> %s: %s", src_path, dest, exc)
            return False

    def download(self, rel_key: str, dest_path: str) -> bool:
        src = self._path(rel_key)
        try:
            if not (src.is_file() and src.stat().st_size > 0):
                return False
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            shutil.copy2(src, dest_path)
            return True
        except OSError as exc:
            logger.error("audio archive (local): failed to fetch %s -> %s: %s", src, dest_path, exc)
            return False


# A command runner is injectable so tests can drive rclone deterministically
# without the binary (and CI never touches a real remote). Signature mirrors the
# subset of ``subprocess.run`` we use.
RcloneRunner = Callable[[Sequence[str], int], "subprocess.CompletedProcess"]


def _default_runner(args: Sequence[str], timeout: int) -> "subprocess.CompletedProcess":
    return subprocess.run(  # noqa: S603 - args are built from config, never shell-interpolated
        list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


class RcloneStorageBackend(StorageBackend):
    """Remote backend that shells out to ``rclone``.

    Works against any configured rclone remote — a Hetzner Storage Box (SFTP) or
    an S3 / object-storage bucket — so the transport is chosen in ``rclone
    config`` on the host, not in application code, and no cloud SDK is a Python
    dependency. Credentials live in the rclone config, never in our config.
    """

    def __init__(
        self,
        remote: str,
        base_path: str = "",
        rclone_bin: str = "rclone",
        *,
        timeout_s: int = DEFAULT_RCLONE_TIMEOUT_S,
        extra_args: Optional[Sequence[str]] = None,
        runner: Optional[RcloneRunner] = None,
    ) -> None:
        if not remote or not str(remote).strip():
            raise StorageBackendError(
                "audio_storage_backend='remote' requires audio_remote_rclone_remote "
                "(the rclone remote name); none was configured."
            )
        self.remote = str(remote).strip().rstrip(":")
        self.base_path = str(base_path or "").strip().strip("/")
        self.rclone_bin = rclone_bin or "rclone"
        self.timeout_s = int(timeout_s)
        self.extra_args: List[str] = list(extra_args or [])
        self._run = runner or _default_runner
        # Fail loud on a missing binary — never silently degrade to "no archive".
        # Skipped when a runner is injected (tests supply their own rclone).
        if runner is None and shutil.which(self.rclone_bin) is None:
            raise StorageBackendError(
                f"audio_storage_backend='remote' needs the '{self.rclone_bin}' binary on PATH; "
                "it was not found. Install rclone in the pipeline image / host."
            )

    def describe(self) -> str:
        return f"rclone:{self.remote}:{self.base_path}"

    def _target(self, rel_key: str) -> str:
        key = str(rel_key).strip().lstrip("/")
        path = f"{self.base_path}/{key}" if self.base_path else key
        return f"{self.remote}:{path}"

    def _rclone(self, *args: str) -> "subprocess.CompletedProcess":
        cmd = [self.rclone_bin, *args, *self.extra_args]
        return self._run(cmd, self.timeout_s)

    def exists(self, rel_key: str) -> bool:
        target = self._target(rel_key)
        try:
            proc = self._rclone("lsjson", "--no-modtime", "--files-only", target)
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.error(
                "audio archive (rclone): exists check timed out/failed for %s: %s", target, exc
            )
            return False
        if proc.returncode != 0:
            # rclone exits non-zero when the object's parent doesn't exist — that
            # is a normal cache miss, not an error worth shouting about.
            return False
        try:
            entries = json.loads(proc.stdout or "[]")
        except (ValueError, TypeError):
            return False
        return any(e.get("Size", 0) and not e.get("IsDir", False) for e in entries)

    def upload(self, src_path: str, rel_key: str) -> bool:
        if not src_path or not os.path.isfile(src_path):
            return False
        try:
            if os.path.getsize(src_path) <= 0:
                return False
        except OSError:
            return False
        if self.exists(rel_key):
            return True  # already archived (dedupe)
        target = self._target(rel_key)
        try:
            proc = self._rclone("copyto", src_path, target)
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.error(
                "audio archive (rclone): upload timed out/failed %s -> %s: %s",
                src_path,
                target,
                exc,
            )
            return False
        if proc.returncode != 0:
            logger.error(
                "audio archive (rclone): upload failed %s -> %s (rc=%s): %s",
                src_path,
                target,
                proc.returncode,
                (proc.stderr or "").strip()[:400],
            )
            return False
        return True

    def download(self, rel_key: str, dest_path: str) -> bool:
        target = self._target(rel_key)
        try:
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            proc = self._rclone("copyto", target, dest_path)
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.error(
                "audio archive (rclone): download timed out/failed %s -> %s: %s",
                target,
                dest_path,
                exc,
            )
            return False
        if proc.returncode != 0:
            logger.error(
                "audio archive (rclone): download failed %s -> %s (rc=%s): %s",
                target,
                dest_path,
                proc.returncode,
                (proc.stderr or "").strip()[:400],
            )
            return False
        try:
            return os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0
        except OSError:
            return False
