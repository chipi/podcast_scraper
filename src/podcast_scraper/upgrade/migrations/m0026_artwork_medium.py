"""0026 — every stored cover has its player-sized (≤1024px) copy on disk.

The player, the lock screen and offline downloads were served the ORIGINAL artwork: on prod
(2026-10-08) 677 of 1,318 originals are 2001-3000px, ~19 MB each once a phone decodes them, and the
same originals filled 116px cards via the episode detail. The API now asks for ``size=medium``, but
it mounts the corpus read-only, so it can only serve a medium copy that already exists — until then
it falls back to the original. The writer makes the medium copy on download; this does the same
for the images already stored.

Derived and content-addressed (``corpus-art/derived/medium/<sha256>.jpg``), so it is regenerable
and deleting one costs only bytes. An image Pillow cannot decode is recorded and left — the
original is served for it, as today. Never upscaled: a smaller original is copied at its own size.

Idempotent: a medium copy that exists is never rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from ...utils.corpus_artwork import CORPUS_ART_REL_PREFIX, medium_path, write_medium
from ..migration import Migration, MigrationContext, MigrationResult
from ..ownership import created_dirs, match_corpus_owner

FAILED_FILE = "artwork_medium_failed.jsonl"


def _originals(root: Path) -> List[Path]:
    store = root / CORPUS_ART_REL_PREFIX / "sha256"
    if not store.is_dir():
        return []
    return sorted(p for p in store.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))


def _failed(root: Path) -> set:
    try:
        lines = (root / FAILED_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    return {json.loads(line).get("original") for line in lines if line.strip()}


class ArtworkMediumMigration(Migration):
    """Write the missing player-sized copies of stored artwork."""

    id = "0026_artwork_medium"
    to_version = "2.7.20"
    description = (
        "Write corpus-art/derived/medium/<sha>.jpg (≤1024px) for every stored cover that lacks "
        "one: the player was served originals of up to 3000px"
    )

    def _missing(self, root: Path) -> List[Path]:
        return [p for p in _originals(root) if not medium_path(root, str(p)).is_file()]

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would write — pure read, no writes."""
        originals = _originals(ctx.corpus_root)
        missing = self._missing(ctx.corpus_root)
        return f"artwork medium plan: {len(originals)} image(s), {len(missing)} missing a medium"

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Every stored image has a medium copy, or is recorded as undecodable."""
        root = ctx.corpus_root
        failed = _failed(root)
        left = [
            str(p.relative_to(root))
            for p in self._missing(root)
            if str(p.relative_to(root)) not in failed
        ]
        if left:
            return False, f"{len(left)} image(s) have no medium copy: {left[:5]}"
        return True, f"every stored image has a medium copy ({len(failed)} undecodable, recorded)"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Write a medium copy for every stored cover that lacks one.

        An undecodable image is RECORDED rather than raised: one unreadable cover must not
        stop the backfill for every other episode.
        """
        root = ctx.corpus_root
        missing = self._missing(root)
        written: List[str] = []
        failed: List[str] = []
        if not ctx.dry_run:
            for original in missing:
                rel = str(original.relative_to(root))
                if write_medium(root, str(original)):
                    dst = medium_path(root, str(original))
                    match_corpus_owner(root, [dst, *created_dirs(dst.parent, root)])
                    written.append(rel)
                else:
                    failed.append(rel)
            if failed:
                with (root / FAILED_FILE).open("a", encoding="utf-8") as fh:
                    for rel in failed:
                        fh.write(json.dumps({"original": rel}) + "\n")
                match_corpus_owner(root, [root / FAILED_FILE])
        verb = "would write" if ctx.dry_run else "wrote"
        count = len(missing) if ctx.dry_run else len(written)
        message = (
            f"{verb} {count} medium copy(ies) of {len(_originals(root))} stored image(s); "
            f"{len(failed)} undecodable"
        )
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details={"missing": len(missing), "written": len(written), "failed": failed[:20]},
        )
