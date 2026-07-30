"""0005 — stamp ``.gi.json`` envelopes to ``schema_version`` 3.1 (ADR-135/#1191 route-and-tag).

#1191 added the additive, OPTIONAL Insight fields ``rank``/``tier``/``routing_tag``/``salience``
and bumped the GI envelope 3.0 → 3.1. Those fields are POPULATED by the route-and-tag pipeline (a
reprocess), so this step does **not** synthesise them: a 3.0 artifact is valid-as-3.1
(``gi/schema.py`` accepts both), and this only stamps the version so consumers know the schema.
Reprocess the corpus to actually fill the new fields.

Runs AFTER 0003 (which lands the v3 typed-MENTIONS shape), so on an already-v3 corpus this is a
pure 3.0 → 3.1 version stamp. ``migrate_gi_document_v3_1`` re-applies the v3 transform first, so it
is also safe on a corpus that somehow skipped 0003.

Idempotent: a file already at 3.1 (or otherwise unchanged) is ``before == after`` and skipped
without a write. Files that fail to parse are recorded but do NOT fail the migration (mirrors
0003) — they are gitignored/junk or a separate upstream bug a corpus-upgrade run shouldn't block on.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Iterable

from ...migrations.gil_kg_identity_migrations import migrate_gi_document_v3_1
from ..migration import Migration, MigrationContext, MigrationResult


def _iter_gi_files(root: Path) -> Iterable[Path]:
    """All ``*.gi.json`` files under ``root`` (recursive). Stable order."""
    return sorted(root.rglob("*.gi.json"))


class GiV31RouteAndTagMigration(Migration):
    """Stamp all ``.gi.json`` envelopes to ``schema_version`` 3.1 (ADR-135/#1191)."""

    id = "0005_gi_v3_1_route_and_tag"
    # Shares the 2.7.1 release marker with 0003/0004 (all unreleased, landing in the same next
    # train over the deployed 2.7.0.dev0). to_version is a ledger LABEL + optional --to-version
    # ceiling, not a run gate — the runner applies any migration whose id is absent from the ledger.
    to_version = "2.7.1"
    description = (
        "ADR-135/#1191 route-and-tag: stamp GI envelope schema_version 3.1 "
        "(additive Insight rank/tier/routing_tag/salience; reprocess populates them)"
    )

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would touch — pure read, no writes."""
        files = list(_iter_gi_files(ctx.corpus_root))
        if not files:
            return "no .gi.json files under corpus — nothing to migrate"
        would_change = unparsable = 0
        for f in files:
            try:
                before = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                unparsable += 1
                continue
            after = migrate_gi_document_v3_1(copy.deepcopy(before))
            if before != after:
                would_change += 1
        return (
            f"GI 3.1 stamp plan: {len(files)} files scanned, {would_change} would change, "
            f"{unparsable} unparsable (will be skipped)"
        )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Walk all ``.gi.json`` files; write only those whose content actually changes."""
        files = list(_iter_gi_files(ctx.corpus_root))
        if not files:
            ctx.log(f"no .gi.json files under {ctx.corpus_root}")
            return MigrationResult(
                self.id,
                applied=True,
                dry_run=ctx.dry_run,
                message="no .gi.json files found",
                details={"files_scanned": 0},
            )

        changed_files: list[str] = []
        unchanged = 0
        unparsable: list[str] = []

        for f in files:
            try:
                before = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                unparsable.append(f"{f}: {exc.__class__.__name__}")
                continue
            after = migrate_gi_document_v3_1(copy.deepcopy(before))
            if before == after:
                unchanged += 1
                continue
            changed_files.append(str(f.relative_to(ctx.corpus_root)))
            if ctx.dry_run:
                continue
            # Atomic write (tmp + os.replace): a kill mid-write otherwise leaves a truncated,
            # unparsable .gi.json that is silently abandoned on replay (mirrors 0003).
            tmp = f.with_suffix(f.suffix + ".tmp")
            tmp.write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, f)

        message = (
            f"{'would write' if ctx.dry_run else 'wrote'} {len(changed_files)} files to 3.1; "
            f"{unchanged} already-current, {len(unparsable)} unparsable"
        )
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details={
                "files_scanned": len(files),
                "changed": len(changed_files),
                "unchanged": unchanged,
                "unparsable": unparsable,
            },
        )
