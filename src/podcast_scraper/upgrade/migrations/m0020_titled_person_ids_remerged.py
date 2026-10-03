"""0020 — m0017 once more, for titled person ids minted after it ran.

m0017 merged ``person:professor-hannah-fry`` into ``person:hannah-fry`` across the corpus and is
recorded applied, so it never runs again. The GI speaker path kept minting titled ids
(``graph_id_utils.person_node_id`` slugified the display name raw; fixed in 6ab9a27f0), so episodes
ingested after it carried them again — ``upgrade verify`` 0010 and 0017 failed on three episodes
from the 2026-10-03 deepen runs (Google DeepMind, StarTalk).

Nothing new is decided here: m0017's own maps (``plan_maps``, recomputed over the served corpus)
and its own five-surface rewrite, with this migration's own frozen maps, receipts and backups.
UNDO restores each file from ``.podcast_scraper/upgrade-backups/0020/`` while it is still exactly
what this migration wrote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, read_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from .m0017_speaker_names_canonicalised import _Episode, plan_maps

MIGRATION_ID = "0020_titled_person_ids_remerged"
RECEIPTS_FILE = "titled_person_ids_remerged.jsonl"
BACKUP_TAG = "0020"


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class TitledPersonIdsRemergedMigration(Migration):
    """Re-run m0017's rename on whatever was ingested with a titled id after it ran."""

    id = MIGRATION_ID
    to_version = "2.7.14"
    description = (
        "m0017 again: a titled person id minted after it ran (person:professor-hannah-fry, "
        "person:dr-moriba-jah) merged into or re-minted as the untitled one, across metadata, "
        "segments, KG, GI and bridge"
    )

    def _episodes(
        self, root: Path, names: Dict[str, str], ids: Dict[str, str]
    ) -> Iterable[_Episode]:
        if not names and not ids:
            return
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta)
            if not ep.load():
                continue
            ep.rewrite(names, ids)
            if ep.changed:
                yield ep

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        names, ids, conflicts = plan_maps(ctx.corpus_root)
        if not names and not ids:
            return "no published speaker name or person id needs renaming"
        episodes = sum(1 for _ in self._episodes(ctx.corpus_root, names, ids))
        return (
            f"titled ids plan: {len(names)} name(s), {len(ids)} id(s) {sorted(ids)}, "
            f"{episodes} episode(s), {len(conflicts)} conflicting id(s) skipped"
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No served surface still carries a frozen old name or id. ``(ok, message)``."""
        header, _rows = read_receipts(ctx.corpus_root, RECEIPTS_FILE)
        if not header:
            return True, "no receipts — nothing was renamed, nothing to verify"
        names, ids = dict(header.get("names") or {}), dict(header.get("ids") or {})
        left = [ep.meta.name for ep in self._episodes(ctx.corpus_root, names, ids)]
        if left:
            return False, f"{len(left)} episode(s) still carry an old name or id: {left[:5]}"
        return True, f"no served surface carries any of {len(names)} name(s) / {len(ids)} id(s)"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Freeze the maps, then rewrite each episode's surfaces together; back up and receipt."""
        root = ctx.corpus_root
        names, ids, conflicts = plan_maps(root)
        totals: Dict[str, int] = {}
        touched: List[str] = []
        receipts: List[dict] = []
        for ep in self._episodes(root, names, ids):
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
                {"names": names, "ids": ids, "conflicts": conflicts},
                receipts,
            )
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {len(touched)} episode(s): {len(names)} name(s), {len(ids)} id(s); "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())),
            details={
                "names": dict(sorted(names.items())),
                "person_ids_changed": dict(sorted(ids.items())),
                "conflicts_skipped": conflicts,
                "episodes": touched,
                "totals": totals,
                "files_written": len(receipts),
            },
        )
