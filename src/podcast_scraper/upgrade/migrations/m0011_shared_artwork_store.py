"""0011 — one artwork store per corpus, at the corpus root (#2204).

The pipeline downloaded artwork under each RUN dir (``feeds/<feed>/run_<id>/.podcast_scraper/
corpus-art/``) while every reader resolves ``image_local_relpath`` against the CORPUS root. Two
defects from one wrong root:

* **None of it was served.** On prod (2026-10-01) 0 of 2,002 served episodes resolved local art;
  the app fell back to hot-linking the publisher's image for every one.
* **Every re-run stored another copy.** 2,214 files held 1,090 distinct images — 1.49 GB on disk
  for 0.78 GB of content — and it grew with every reprocess.

The writer now stores at the corpus root (``metadata_generation.artwork_store_root``). This moves
what is already on disk: one copy of each image into ``<corpus>/.podcast_scraper/corpus-art/``,
then the per-run copies are removed.

NO METADATA IS REWRITTEN. ``image_local_relpath`` is ``.podcast_scraper/corpus-art/sha256/..``,
which is already corpus-relative in shape — it was only the FILE that sat in the wrong place.

A per-run copy is removed only once the shared copy is proven to hold the same bytes. Files are
content-addressed (name = sha256 of the bytes), so a shared file whose hash matches its name is
the same image; a name that is not a hash is compared byte-for-byte instead. Anything that fails
either check is left where it is and reported as a conflict. Nothing is lost if an image is
missing afterwards anyway: every record also keeps ``image_url``, and the readers fall back to it.

Idempotent: once moved, no per-run store remains, so the second run finds nothing to do.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

from ...utils.corpus_artwork import CORPUS_ART_REL_PREFIX
from ..migration import Migration, MigrationContext, MigrationResult

_SHA256_NAME = re.compile(r"^[0-9a-f]{64}$")


def _per_run_stores(root: Path) -> List[Path]:
    """Every run-level art store: corpus layout (``feeds/*/run_*``) and legacy (``run_*``)."""
    pattern = f"run_*/{CORPUS_ART_REL_PREFIX}"
    stores = list(root.glob(f"feeds/*/{pattern}")) + list(root.glob(pattern))
    return sorted(p for p in stores if p.is_dir())


def _files(store: Path) -> Iterable[Path]:
    return sorted(p for p in store.rglob("*") if p.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _holds(shared: Path, source: Path) -> bool:
    """Does ``shared`` hold exactly the image ``source`` holds?"""
    if not shared.is_file():
        return False
    stem = shared.name.split(".", 1)[0]
    if _SHA256_NAME.match(stem):
        return _sha256(shared) == stem
    return shared.read_bytes() == source.read_bytes()


def _copy_atomic(source: Path, dest: Path) -> None:
    """tmp + os.replace — a kill mid-copy must not leave a truncated image at the final name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, dest)


def _prune_empty_dirs(store: Path) -> None:
    """Remove the emptied store, and the ``.podcast_scraper`` dir above it if that is empty too."""
    for directory in sorted(
        (p for p in store.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    for directory in (store, store.parent):
        try:
            directory.rmdir()
        except OSError:
            break


class SharedArtworkStoreMigration(Migration):
    """Move per-run artwork into the one corpus-root store the readers use."""

    id = "0011_shared_artwork_store"
    to_version = "2.7.5"
    description = (
        "#2204: artwork was stored per RUN while every reader resolves it at the corpus root — "
        "none of it was served and every re-run added a copy. Move one copy of each image to "
        "<corpus>/.podcast_scraper/corpus-art/ and remove the verified per-run duplicates"
    )

    def _walk(self, ctx: MigrationContext) -> Tuple[int, int, int, int, int, List[str]]:
        """Plan or apply. ``(stores, files, copied, removed, bytes_freed, conflicts)``."""
        root = ctx.corpus_root
        shared_root = root / CORPUS_ART_REL_PREFIX
        stores = _per_run_stores(root)
        files = copied = removed = freed = 0
        conflicts: List[str] = []
        pending_copies: set = set()
        for store in stores:
            for source in _files(store):
                files += 1
                rel = source.relative_to(store)
                shared = shared_root / rel
                if source.name.endswith(".tmp"):
                    conflicts.append(f"{source.relative_to(root)}: partial copy, left in place")
                    continue
                if ctx.dry_run:
                    if not shared.exists() and str(rel) not in pending_copies:
                        pending_copies.add(str(rel))
                        copied += 1
                    removed += 1
                    freed += source.stat().st_size
                    continue
                if not shared.exists():
                    _copy_atomic(source, shared)
                    copied += 1
                if not _holds(shared, source):
                    conflicts.append(
                        f"{source.relative_to(root)}: differs from the shared copy, left in place"
                    )
                    continue
                size = source.stat().st_size
                source.unlink()
                removed += 1
                freed += size
            if not ctx.dry_run:
                _prune_empty_dirs(store)
        return len(stores), files, copied, removed, freed, conflicts

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would move — pure read, no writes."""
        dry = MigrationContext(corpus_root=ctx.corpus_root, dry_run=True, logger=ctx.logger)
        stores, files, copied, removed, freed, conflicts = self._walk(dry)
        if not stores:
            return "no per-run artwork stores — nothing to move"
        return (
            f"shared artwork store plan: {stores} per-run store(s), {files} file(s); "
            f"{copied} distinct image(s) to copy to the corpus root, {removed} per-run file(s) "
            f"to remove ({freed / 2**20:.0f} MB)"
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No per-run store is left holding an image. ``(ok, message)``.

        Reads the corpus directly rather than re-running ``plan``: the ledger excludes an applied
        migration, so a plan that finds nothing would be vacuous.
        """
        left = [
            str(f.relative_to(ctx.corpus_root))
            for store in _per_run_stores(ctx.corpus_root)
            for f in _files(store)
        ]
        if left:
            return False, f"{len(left)} artwork file(s) still in per-run stores: {left[:5]}"
        return True, "all artwork is in the corpus-root store"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Copy each distinct image to the corpus root, then remove the proven duplicates."""
        stores, files, copied, removed, freed, conflicts = self._walk(ctx)
        for line in conflicts:
            ctx.log(f"artwork conflict: {line}")
        verb = "would move" if ctx.dry_run else "moved"
        message = (
            f"{verb} {files} file(s) from {stores} per-run store(s): {copied} distinct image(s) "
            f"copied to the corpus root, {removed} per-run file(s) removed "
            f"({freed / 2**20:.0f} MB), {len(conflicts)} conflict(s) left in place"
        )
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "per_run_stores": stores,
                "files": files,
                "copied": copied,
                "removed": removed,
                "bytes_freed": freed,
                "conflicts": conflicts[:20],
            },
        )
