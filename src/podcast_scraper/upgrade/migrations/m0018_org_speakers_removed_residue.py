"""0018 — finish m0012: an organisation it froze is nobody's voice in ANY served episode.

``upgrade verify`` after the 2026-10-03 deploy failed 0012 on two served episodes (EP41 / EP42 of
one feed, run 2026-09-11): a GI Person node "Africa Tech Summit" — in m0012's frozen organisation
set — with no roster entry, so neither m0012's apply nor m0015 (roster names only) removed it. The
check is m0012's own verify; this applies m0012's own five-surface rewrite once more with m0012's
FROZEN set (read from its receipts, never recomputed from live votes), so the two can only agree.

Nothing new is decided here. No receipts from m0012 means nothing to finish: a clean no-op. Undo
restores each file from ``.podcast_scraper/upgrade-backups/0018/`` while it is still exactly what
this migration wrote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from ..corpus_selection import select_served_artifacts
from ..file_rewrite import append_receipts, read_receipts, undo_from_receipts, write_with_backup
from ..migration import Migration, MigrationContext, MigrationResult
from .m0012_org_speakers_removed import _Episode, RECEIPTS_FILE as M0012_RECEIPTS

MIGRATION_ID = "0018_org_speakers_removed_residue"
RECEIPTS_FILE = "org_speakers_removed_residue.jsonl"
BACKUP_TAG = "0018"


def frozen_orgs(root: Path) -> Set[str]:
    """m0012's frozen organisation set (``kind_key`` form), or empty when m0012 never wrote one."""
    header, _rows = read_receipts(root, M0012_RECEIPTS)
    return set(header.get("orgs") or {})


def undo(root: Path) -> Tuple[int, List[str]]:
    """Restore each file this migration wrote, if still as left. ``(restored, refused)``."""
    return undo_from_receipts(Path(root), RECEIPTS_FILE, BACKUP_TAG, MIGRATION_ID)


class OrgSpeakersRemovedResidueMigration(Migration):
    """Re-apply m0012's rewrite, with m0012's frozen set, to whatever it left behind."""

    id = MIGRATION_ID
    to_version = "2.7.12"
    description = (
        "finish 0012: an organisation in m0012's frozen set that a served episode still carries "
        "as a person (a GI Person node with no roster entry, e.g. 'Africa Tech Summit') is "
        "removed with m0012's own five-surface rewrite"
    )

    def _episodes(self, root: Path, orgs: Set[str]) -> Iterable[_Episode]:
        for meta in select_served_artifacts(root, ".metadata.json")[0]:
            ep = _Episode(meta, orgs)
            if ep.plan():
                yield ep

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would rewrite — pure read, no writes."""
        orgs = frozen_orgs(ctx.corpus_root)
        if not orgs:
            return "m0012 froze no organisation set — nothing to finish"
        eps = list(self._episodes(ctx.corpus_root, orgs))
        return f"org residue plan: {len(eps)} episode(s): " + ", ".join(
            f"{e.meta.name}: {sorted(e.counts)}" for e in eps[:10]
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """No served surface names any of m0012's frozen orgs as a person. ``(ok, message)``."""
        orgs = frozen_orgs(ctx.corpus_root)
        if not orgs:
            return True, "m0012 froze no organisation set — nothing to verify"
        left = [e.meta.name for e in self._episodes(ctx.corpus_root, orgs)]
        if left:
            return False, f"{len(left)} episode(s) still name an m0012 org: {left[:5]}"
        return True, f"no served surface names any of m0012's {len(orgs)} org(s)"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Rewrite every episode still carrying an m0012 org; back up, receipt."""
        root = ctx.corpus_root
        orgs = frozen_orgs(root)
        totals: Dict[str, int] = {}
        touched: List[str] = []
        receipts: List[dict] = []
        for ep in self._episodes(root, orgs):
            touched.append(str(ep.meta.relative_to(root)))
            for k, v in ep.counts.items():
                totals[k] = totals.get(k, 0) + v
            if ctx.dry_run:
                continue
            for path in sorted(ep.changed):
                receipts.append(write_with_backup(root, BACKUP_TAG, path, ep.files[path]))
        if not ctx.dry_run and touched:
            append_receipts(root, RECEIPTS_FILE, {"orgs": sorted(orgs)}, receipts)
        verb = "would rewrite" if ctx.dry_run else "rewrote"
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=f"{verb} {len(touched)} episode(s) still naming an m0012 org: "
            + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())),
            details={"episodes": touched, "totals": totals, "files_written": len(receipts)},
        )
