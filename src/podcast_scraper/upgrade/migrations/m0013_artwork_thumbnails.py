"""0013 — every stored cover has its list thumbnail on disk.

The serving API mounts the corpus read-only, so ``GET /api/app/artwork?size=thumb`` could never
write the thumbnail it looked for and served the full original instead: measured on prod
(2026-10-01) the same 200,194 bytes for ``thumb`` and ``large``, in every list and card. The
artwork writer now makes the thumbnail when it downloads an image (it has write access); this does
the same for the 1,090 images already stored.

Derived and content-addressed (``corpus-art/derived/thumb/<sha256>.jpg``), so it is regenerable and
deleting one costs only bytes. An image Pillow cannot decode is recorded and left — the original is
served for it, as today.

Idempotent: a thumbnail that exists is never rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from ...utils.corpus_artwork import CORPUS_ART_REL_PREFIX, thumbnail_path, write_thumbnail
from ..migration import Migration, MigrationContext, MigrationResult
from ..ownership import created_dirs, match_corpus_owner

FAILED_FILE = "artwork_thumbnails_failed.jsonl"


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


class ArtworkThumbnailsMigration(Migration):
    """Write the missing list thumbnails for stored artwork."""

    id = "0013_artwork_thumbnails"
    to_version = "2.7.7"
    description = (
        "Write corpus-art/derived/thumb/<sha>.jpg for every stored cover that lacks one: the "
        "serving API mounts the corpus read-only and served full-size images as thumbnails"
    )

    def _missing(self, root: Path) -> List[Path]:
        return [p for p in _originals(root) if not thumbnail_path(root, str(p)).is_file()]

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would write — pure read, no writes."""
        originals = _originals(ctx.corpus_root)
        missing = self._missing(ctx.corpus_root)
        return f"artwork thumbnails plan: {len(originals)} image(s), {len(missing)} missing a thumb"

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Every stored image has a thumbnail, or is recorded as undecodable."""
        root = ctx.corpus_root
        failed = _failed(root)
        left = [
            str(p.relative_to(root))
            for p in self._missing(root)
            if str(p.relative_to(root)) not in failed
        ]
        if left:
            return False, f"{len(left)} image(s) have no thumbnail: {left[:5]}"
        return True, f"every stored image has a thumbnail ({len(failed)} undecodable, recorded)"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        root = ctx.corpus_root
        missing = self._missing(root)
        written: List[str] = []
        failed: List[str] = []
        if not ctx.dry_run:
            for original in missing:
                rel = str(original.relative_to(root))
                if write_thumbnail(root, str(original)):
                    dst = thumbnail_path(root, str(original))
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
            f"{verb} {count} thumbnail(s) of {len(_originals(root))} stored image(s); "
            f"{len(failed)} undecodable"
        )
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details={"missing": len(missing), "written": len(written), "failed": failed[:20]},
        )
