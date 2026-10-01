"""Files a migration writes must belong to whoever owns the corpus, not to the migration's user.

Prod runs ``upgrade run`` inside the api container as root while the pipeline writes as the
corpus owner (uid 1000). m0011 created ``.podcast_scraper/corpus-art/`` as ``0:0`` mode 755 on
2026-10-01, and every later artwork download would have failed to create its folder until it was
chowned by hand. A file a migration REPLACES is the same hazard one level down.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def match_corpus_owner(corpus_root: Path, paths: Iterable[Path]) -> None:
    """Give *paths* the corpus root's uid/gid. A no-op unless running as root (only root may)."""
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    st = Path(corpus_root).stat()
    for path in paths:
        try:
            os.chown(path, st.st_uid, st.st_gid)
        except OSError:
            continue


def created_dirs(start: Path, stop: Path) -> Iterable[Path]:
    """*start* and its parents up to (not including) *stop* — the directories a write may create."""
    path = Path(start)
    stop = Path(stop)
    while path != stop and stop in path.parents:
        yield path
        path = path.parent
