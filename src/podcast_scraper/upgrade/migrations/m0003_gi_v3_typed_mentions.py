"""0003 — GI v3 typed-mentions schema migration (RFC-097, retrofitted into #862).

Wraps the standalone ``scripts/migrate_gi_to_v3.py`` script (and its underlying
``migrate_gi_document_v3`` function) into the corpus-upgrade framework so a
fresh agent or operator can run ``make upgrade-corpus`` once and have ALL pending
work apply in registered order — no separate "did I remember to run the v3
script" step.

What this migration does, per ``.gi.json`` file under the corpus root:

- Bumps ``schema_version`` ``2.0`` → ``3.0``.
- Rewrites legacy ``MENTIONS`` edges (Insight → Person/Org) to typed
  ``MENTIONS_PERSON`` / ``MENTIONS_ORG`` based on the edge target's id prefix.
- Normalises legacy ``Insight.insight_type`` vocab (``fact``/``opinion``) to
  the v3 schema vocab (``claim``/``observation``); out-of-vocab → ``unknown``.

KG-side legacy ``MENTIONS`` edges (Topic → Episode discovery edges) stay
untouched by design — see ``gil_kg_identity_migrations.migrate_gi_document_v3``.

Idempotent: a file already at schema 3.0 with already-typed edges passes through
unchanged (we count it as ``unchanged``, write nothing). Safe to re-run.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable, Tuple

from ...migrations.gil_kg_identity_migrations import migrate_gi_document_v3
from ..migration import Migration, MigrationContext, MigrationResult


def _iter_gi_files(root: Path) -> Iterable[Path]:
    """All ``*.gi.json`` files under ``root`` (recursive). Stable order."""
    return sorted(root.rglob("*.gi.json"))


def _classify(before: dict, after: dict) -> Tuple[bool, int, int]:
    """Compute (changed, mentions_person_added, mentions_org_added)."""
    if before == after:
        return False, 0, 0
    mp = mo = 0
    for be, ae in zip(before.get("edges") or [], after.get("edges") or []):
        if be.get("type") == "MENTIONS" and ae.get("type") == "MENTIONS_PERSON":
            mp += 1
        if be.get("type") == "MENTIONS" and ae.get("type") == "MENTIONS_ORG":
            mo += 1
    return True, mp, mo


class GiV3TypedMentionsMigration(Migration):
    """Bring all .gi.json envelopes to RFC-097 v3 (typed MENTIONS_PERSON/MENTIONS_ORG)."""

    id = "0003_gi_v3_typed_mentions"
    to_version = "2.7.1"
    description = (
        "RFC-097 v3 GI schema: typed MENTIONS_PERSON/MENTIONS_ORG + "
        "claim/observation insight types + schema 3.0 bump"
    )

    def plan(self, ctx: MigrationContext) -> str:
        """Summarise what apply() would touch — pure read, no writes."""
        files = list(_iter_gi_files(ctx.corpus_root))
        if not files:
            return "no .gi.json files under corpus — nothing to migrate"
        would_change = mp_total = mo_total = unparsable = 0
        for f in files:
            try:
                before = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                unparsable += 1
                continue
            after = migrate_gi_document_v3(copy.deepcopy(before))
            changed, mp, mo = _classify(before, after)
            if changed:
                would_change += 1
                mp_total += mp
                mo_total += mo
        return (
            f"GI v3 migration plan: {len(files)} files scanned, "
            f"{would_change} would change "
            f"({mp_total} MENTIONS→MENTIONS_PERSON, {mo_total} MENTIONS→MENTIONS_ORG), "
            f"{unparsable} unparsable (will be skipped)"
        )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Walk all .gi.json files; write only those whose content actually changes.

        Idempotent: files already at v3 are detected (before==after) and skipped
        without a write. Files that fail to parse are recorded in ``details`` but
        do NOT fail the migration — they're either gitignored/junk or a separate
        upstream bug, both of which a corpus-upgrade run shouldn't block on.
        """
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
        unchanged = mp_total = mo_total = 0
        unparsable: list[str] = []

        for f in files:
            try:
                before = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                unparsable.append(f"{f}: {exc.__class__.__name__}")
                continue
            after = migrate_gi_document_v3(copy.deepcopy(before))
            changed, mp, mo = _classify(before, after)
            if not changed:
                unchanged += 1
                continue
            mp_total += mp
            mo_total += mo
            changed_files.append(str(f.relative_to(ctx.corpus_root)))
            if ctx.dry_run:
                continue
            # Pretty-print + trailing newline, matching how the standalone
            # ``migrate_gi_to_v3.py`` writes (json.dumps(..., indent=2)).
            f.write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")

        message = (
            f"{'would write' if ctx.dry_run else 'wrote'} {len(changed_files)} files "
            f"({mp_total} MENTIONS_PERSON, {mo_total} MENTIONS_ORG); "
            f"{unchanged} already-current, {len(unparsable)} unparsable"
        )
        ctx.log(message)
        details: dict = {
            "files_scanned": len(files),
            "files_changed": len(changed_files),
            "files_unchanged": unchanged,
            "files_unparsable": len(unparsable),
            "mentions_person_typed": mp_total,
            "mentions_org_typed": mo_total,
        }
        # Only embed the lists when they're small — long path lists clutter
        # the ledger and CLI output without buying anything an operator reads.
        if len(changed_files) <= 50:
            details["changed_files"] = changed_files
        if unparsable:
            details["unparsable_samples"] = unparsable[:10]
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=ctx.dry_run,
            message=message,
            details=details,
        )
