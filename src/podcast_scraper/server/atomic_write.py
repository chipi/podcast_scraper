"""Atomic text writes for viewer-managed corpus files."""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write ``content`` to ``path`` via a same-directory replace (best-effort atomic).

    *path* is only built by server routes after ``resolve_corpus_path_param`` /
    ``normpath_if_under_root`` (see ``docs/ci/CODEQL_DISMISSALS.md`` Type 1).
    """
    # codeql[py/path-injection] -- path from corpus-anchored callers only (Type 1).
    path.parent.mkdir(parents=True, exist_ok=True)
    # codeql[py/path-injection] -- tmp is a same-directory sibling of path (see above).
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # codeql[py/path-injection] -- tmp is a same-dir sibling of path (see above).
    tmp.write_text(content, encoding=encoding)
    # codeql[py/path-injection] -- replace target is path (see above).
    tmp.replace(path)
