"""Walking a corpus for its CONTENT — never into its bookkeeping dot-directories.

A corpus root also holds ``.podcast_scraper/upgrade-backups/<tag>/feeds/…`` (every migration and
repair backs up what it rewrites), ``.trash/<ts>/feeds/…`` (cleanup), ``.viewer`` job logs and
caches. Those copies are real ``*.gi.json`` / ``*.kg.json`` / ``*.metadata.json`` files with real
episode ids, from BEFORE a repair. A plain ``root.rglob("*.gi.json")`` collects them as live.

On prod 2026-10-07 that was 593 GI, 401 KG, 450 bridge and 2,826 metadata backup copies: 1,444 of
the 2,587 artifacts ``/api/artifacts`` listed to the operator viewer's graph, and 125 of the 2,589
KG files the entity-id map was built from. ``upgrade verify`` hit the same thing on 2026-10-03 and
was fixed alone; every corpus scanner now goes through here instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List


def corpus_relpath_is_excluded(rel_posix: str) -> bool:
    """True when a corpus-relative path lies under a RETIRED / non-content directory (#2161).

    Corpus cleanup moves superseded artifacts to ``.trash/<ts>/feeds/<feed>/run_*/metadata/``.
    Those files are real metadata for a real episode id, so every stage that walks the corpus for
    membership used to collect them as if they were live — and because the trash path parses as
    NEITHER corpus layout it was kept unconditionally and never compared against the live copy. The
    episode was then collected twice, its id-keyed rows collided, and the index prune deleted the
    LIVE copy's rows as "superseded".

    The rule is by leading dot rather than a ``.trash`` literal: every dot-directory here
    (``.trash``, ``.podcast_scraper`` upgrade backups, ``.viewer`` job logs, caches) is bookkeeping,
    not corpus content, and a future retirement directory should be excluded on the day it is
    introduced rather than after it deletes data. Judged on the path RELATIVE to the corpus root, so
    a root that itself sits under a dotted directory is unaffected.
    """
    return any(
        seg.startswith(".") and seg not in (".", "..")
        for seg in rel_posix.replace("\\", "/").split("/")
        if seg
    )


def corpus_rglob(root: Path, pattern: str) -> Iterator[Path]:
    """``root.rglob(pattern)`` without anything under a dot-directory of *root*.

    ``root.glob("**/<p>")`` is the same walk: pass ``"<p>"``. Order is the filesystem's, as with
    ``rglob`` — sort where order matters.
    """
    root = Path(root)
    for path in root.rglob(pattern):
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.as_posix()
        if not corpus_relpath_is_excluded(rel):
            yield path


def prune_excluded_dirs(dirnames: List[str]) -> None:
    """For ``os.walk``: drop dot-directories from *dirnames* in place so the walk never enters."""
    dirnames[:] = [d for d in dirnames if not corpus_relpath_is_excluded(d)]


__all__ = ["corpus_relpath_is_excluded", "corpus_rglob", "prune_excluded_dirs"]
