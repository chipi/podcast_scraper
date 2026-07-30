"""0006 — bring ``.kg.json`` envelopes to RFC-097 v2.0 (typed Person/Organization).

The KG counterpart of 0003 (GI v3). Rewrites legacy ``Entity(kind=...)`` nodes to typed
``Person`` / ``Organization`` and normalises legacy ids (``entity:person:`` → ``person:``,
``entity_kind`` → ``kind``), stamping ``schema_version`` 2.0. ``migrate_kg_document_v2`` chains the
v1.x → v1.2 id/kind normalisation first, so this single step supersedes BOTH standalone scripts it
replaces (``migrate_kg_entity_ids.py`` + ``migrate_kg_entity_to_person_org.py``).

Idempotent: a v2.0 artifact is ``before == after`` and skipped without a write. Unparsable files are
recorded but do NOT fail the migration (mirrors 0003/0005) — junk/gitignored or a separate upstream
bug a corpus-upgrade run shouldn't block on.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Iterable

from ...migrations.gil_kg_identity_migrations import migrate_kg_document_v2
from ..migration import Migration, MigrationContext, MigrationResult


def _iter_kg_files(root: Path) -> Iterable[Path]:
    """All ``*.kg.json`` files under ``root`` (recursive). Stable order."""
    return sorted(root.rglob("*.kg.json"))


class KgV2TypedEntitiesMigration(Migration):
    """Bring all ``.kg.json`` envelopes to RFC-097 v2.0 (typed Person/Organization)."""

    id = "0006_kg_v2_typed_entities"
    # Shares the 2.7.1 release marker with 0003–0005 (all unreleased, same next train). to_version
    # is a ledger LABEL, not a run gate — the runner applies any migration absent from the ledger.
    to_version = "2.7.1"
    description = (
        "RFC-097 v2.0 KG: typed Person/Organization + entity id/kind normalization "
        "(schema 2.0 bump)"
    )

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would touch — pure read, no writes."""
        files = list(_iter_kg_files(ctx.corpus_root))
        if not files:
            return "no .kg.json files under corpus — nothing to migrate"
        would_change = unparsable = 0
        for f in files:
            try:
                before = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                unparsable += 1
                continue
            after = migrate_kg_document_v2(copy.deepcopy(before))
            if before != after:
                would_change += 1
        return (
            f"KG v2.0 migration plan: {len(files)} files scanned, {would_change} would change, "
            f"{unparsable} unparsable (will be skipped)"
        )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Walk all ``.kg.json`` files; write only those whose content actually changes."""
        files = list(_iter_kg_files(ctx.corpus_root))
        if not files:
            ctx.log(f"no .kg.json files under {ctx.corpus_root}")
            return MigrationResult(
                self.id,
                applied=True,
                dry_run=ctx.dry_run,
                message="no .kg.json files found",
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
            after = migrate_kg_document_v2(copy.deepcopy(before))
            if before == after:
                unchanged += 1
                continue
            changed_files.append(str(f.relative_to(ctx.corpus_root)))
            if ctx.dry_run:
                continue
            # Atomic write (tmp + os.replace): a kill mid-write otherwise leaves a truncated,
            # unparsable .kg.json that is silently abandoned on replay (mirrors 0003/0005).
            tmp = f.with_suffix(f.suffix + ".tmp")
            tmp.write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, f)

        message = (
            f"{'would write' if ctx.dry_run else 'wrote'} {len(changed_files)} files to v2.0; "
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
