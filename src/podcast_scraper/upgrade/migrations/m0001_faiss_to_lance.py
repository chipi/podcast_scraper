"""0001 — FAISS → two-tier LanceDB index (historical; FAISS retired in #995).

Originally migrated a corpus's FAISS index into the two-tier LanceDB layout. FAISS has
since been removed entirely (ADR-099 / #995): the LanceDB index is built directly by
``cli index`` and pre-LanceDB corpora are rebuilt, not migrated. This step is retained as a
recorded no-op so the migration chain and version history stay intact.
"""

from __future__ import annotations

from typing import Tuple

from ..migration import Migration, MigrationContext, MigrationResult


class FaissToLanceMigration(Migration):
    """Historical FAISS→LanceDB step, now a recorded no-op (FAISS retired, #995)."""

    id = "0001_faiss_to_lance"
    to_version = "2.7.0"
    description = "Historical FAISS→LanceDB migration; no-op since FAISS was retired (#995)"

    def plan(self, ctx: MigrationContext) -> str:
        return "FAISS retired (#995); nothing to migrate (no-op)."

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        ctx.log("FAISS retired (#995); nothing to migrate")
        return MigrationResult(
            self.id, applied=True, dry_run=ctx.dry_run, message="FAISS retired — no-op"
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        return True, "FAISS retired (#995); no-op"
