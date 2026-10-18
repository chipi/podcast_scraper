"""Which files a migration's decision reads for one episode, so its plan row can hash exactly them.

The migration outcome hook diffs prod's per-episode plan against the clone's (method D15). A row
that hashes fewer files than its decision reads lets a changed input pass unseen; one that hashes
more reports differences that change nothing. So the inputs are not listed by hand: the decision
runs inside ``recording(root)``, which notes every file under ``root`` opened for reading
(``builtins.open``, ``io.open`` — what ``pathlib`` uses — and ``os.open``) and every existence check
(``Path.exists`` / ``Path.is_file``, ``os.path.exists`` / ``isfile``). ``inputs()`` then gives
``{relative path: sha256}``, with ``"absent"`` for a file that was looked for and is not there.

Only used around a read-only plan (``dry_run_report``); single-threaded, restored on exit.
"""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import os
import os.path
from pathlib import Path
from typing import Any, Dict, Iterator, Set

ABSENT = "absent"


class Recorder:
    """The paths read or looked for under one root."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.paths: Set[Path] = set()

    def note(self, path: Any) -> None:
        try:
            p = Path(os.fspath(path)).resolve()
        except (TypeError, ValueError, OSError):
            return
        if p == self.root or self.root in p.parents:
            self.paths.add(p)

    def inputs(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for p in sorted(self.paths):
            if p.is_dir():
                continue
            rel = p.relative_to(self.root).as_posix()
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ABSENT
        return out


@contextlib.contextmanager
def recording(root: Path) -> Iterator[Recorder]:
    """Record reads and existence checks under ``root`` while the block runs."""
    rec = Recorder(root)
    real_open, real_io_open, real_os_open = builtins.open, io.open, os.open
    real_exists, real_is_file = Path.exists, Path.is_file
    real_path_exists, real_path_isfile = os.path.exists, os.path.isfile

    def _open(file: Any, mode: str = "r", *a: Any, **k: Any) -> Any:
        if isinstance(file, (str, bytes, os.PathLike)) and not any(c in mode for c in "wax+"):
            rec.note(file)
        return real_open(file, mode, *a, **k)

    def _io_open(file: Any, mode: str = "r", *a: Any, **k: Any) -> Any:
        if isinstance(file, (str, bytes, os.PathLike)) and not any(c in mode for c in "wax+"):
            rec.note(file)
        return real_io_open(file, mode, *a, **k)

    def _os_open(path: Any, flags: int, *a: Any, **k: Any) -> Any:
        if not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            rec.note(path)
        return real_os_open(path, flags, *a, **k)

    def _exists(self: Path, *a: Any, **k: Any) -> bool:
        rec.note(self)
        return real_exists(self, *a, **k)

    def _is_file(self: Path, *a: Any, **k: Any) -> bool:
        rec.note(self)
        return real_is_file(self, *a, **k)

    def _p_exists(path: Any) -> bool:
        rec.note(path)
        return real_path_exists(path)

    def _p_isfile(path: Any) -> bool:
        rec.note(path)
        return real_path_isfile(path)

    builtins.open = _open  # type: ignore[assignment]
    io.open = _io_open  # type: ignore[assignment]
    os.open = _os_open  # type: ignore[assignment]
    Path.exists = _exists  # type: ignore[method-assign]
    Path.is_file = _is_file  # type: ignore[method-assign]
    os.path.exists = _p_exists
    os.path.isfile = _p_isfile
    try:
        yield rec
    finally:
        builtins.open, io.open, os.open = real_open, real_io_open, real_os_open  # type: ignore
        Path.exists, Path.is_file = real_exists, real_is_file  # type: ignore[method-assign]
        os.path.exists, os.path.isfile = real_path_exists, real_path_isfile
