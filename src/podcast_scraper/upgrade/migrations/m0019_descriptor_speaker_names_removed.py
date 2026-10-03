"""0019 — m0015 once more, with the publish gate as of 11e1e425e (hyphenated descriptors).

m0015 removed every published speaker name the gate refused on 2026-10-03 and is recorded applied,
so it never runs again. The gate then learned one more junk shape: a hyphenated descriptor is a
phrase about a person, not their name ("Pulitzer Prize-winning", published as a Freakonomics guest
on the voice that is Jennifer Egan's). This re-runs m0015's own decision (``refused_names``, today's
gate plus m0015's hand-kept and hand-removed lists) and m0015's own five-surface rewrite, with its
own frozen set, receipts and backups.

Nothing new is decided here beyond what the gate decides at ingest. THE SET IS FROZEN at apply time
and written to the receipts; ``verify`` judges against it. UNDO restores each file from
``.podcast_scraper/upgrade-backups/0019/`` while it is still exactly what this migration wrote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, read_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from .m0015_unpublishable_speaker_names_removed import _Episode, KEEP, refused_names, REMOVE

MIGRATION_ID = "0019_descriptor_speaker_names_removed"
RECEIPTS_FILE = "descriptor_speaker_names_removed.jsonl"
BACKUP_TAG = "0019"


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class DescriptorSpeakerNamesRemovedMigration(Migration):
    """Remove published speaker names the gate refuses since m0015 ran, from every surface."""

    id = MIGRATION_ID
    to_version = "2.7.13"
    description = (
        "m0015 with today's publish gate: a published speaker name the gate now refuses "
        "(e.g. the hyphenated descriptor 'Pulitzer Prize-winning') is removed from roster, "
        "segments, KG, GI and bridge"
    )

    def _episodes(self, root: Path, names: Set[str]) -> Iterable[_Episode]:
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta, names)
            if ep.plan():
                yield ep

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        names = refused_names(ctx.corpus_root)
        if not names:
            return "no published speaker name is refused by today's gate — nothing to remove"
        eps = list(self._episodes(ctx.corpus_root, set(names)))
        return f"refused speaker names plan: {sorted(names)}; {len(eps)} episode(s)"

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No served surface still carries a FROZEN refused name. ``(ok, message)``."""
        header, _rows = read_receipts(ctx.corpus_root, RECEIPTS_FILE)
        names = set(header.get("names") or {})
        if not names:
            return True, "no receipts — nothing was removed, nothing to verify"
        left = [e.meta.name for e in self._episodes(ctx.corpus_root, names)]
        if left:
            return False, f"{len(left)} episode(s) still carry a refused name: {left[:5]}"
        return True, f"no served surface carries any of {len(names)} refused name(s)"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Remove the frozen refused names from every speaker surface; back up, receipt."""
        root = ctx.corpus_root
        names = refused_names(root)
        totals: Dict[str, int] = {}
        touched: List[str] = []
        receipts: List[dict] = []
        for ep in self._episodes(root, set(names)):
            touched.append(str(ep.meta.relative_to(root)))
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
            if ctx.dry_run:
                continue
            for path in sorted(ep.changed):
                receipts.append(write_with_backup(root, BACKUP_TAG, path, ep.files[path]))
        if not ctx.dry_run and touched:
            append_receipts(
                root,
                RECEIPTS_FILE,
                {"names": names, "kept": KEEP, "removed_by_hand": REMOVE},
                receipts,
            )
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {len(touched)} episode(s) for {len(names)} refused name(s): "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())),
            details={
                "names": dict(sorted(names.items())),
                "episodes": touched,
                "totals": totals,
                "files_written": len(receipts),
            },
        )
