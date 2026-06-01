"""0001 — FAISS → two-tier LanceDB index (2.6 → 2.7, #858 under #862).

The first registered upgrade step. Wraps ``search.migration.migrate_faiss_to_lance``
behind the ``Migration`` interface so the runner sequences, records, and verifies it
like any other step. Idempotent: the underlying migration merge-inserts on id, and a
missing FAISS index is a clean no-op (a brand-new corpus has nothing to migrate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ...search.faiss_store import VECTORS_FILE
from ..migration import Migration, MigrationContext, MigrationResult


class FaissToLanceMigration(Migration):
    """Re-project the corpus FAISS index into the two-tier LanceDB layout."""

    id = "0001_faiss_to_lance"
    to_version = "2.7.0"
    description = "Build the two-tier LanceDB index from the existing FAISS index (RFC-090, #858)"

    def _faiss_dir(self, ctx: MigrationContext) -> Path:
        return Path(ctx.options.get("faiss_dir") or ctx.corpus_root / "search")

    def _lance_path(self, ctx: MigrationContext) -> Path:
        # Default: co-located with the corpus so the index travels with it.
        return Path(ctx.options.get("lance_path") or ctx.corpus_root / "search" / "lance_index")

    def plan(self, ctx: MigrationContext) -> str:
        """Describe the source FAISS index and the target LanceDB path."""
        faiss_dir = self._faiss_dir(ctx)
        if not (faiss_dir / VECTORS_FILE).is_file():
            return f"No FAISS index at {faiss_dir} — nothing to migrate (no-op)."
        return f"Migrate {faiss_dir}/{VECTORS_FILE} → two-tier LanceDB at {self._lance_path(ctx)}"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Run the FAISS→LanceDB migration (or report the plan when dry-run)."""
        faiss_dir = self._faiss_dir(ctx)
        lance_path = self._lance_path(ctx)

        if not (faiss_dir / VECTORS_FILE).is_file():
            ctx.log(f"no FAISS index at {faiss_dir}; nothing to migrate")
            return MigrationResult(
                self.id, applied=True, dry_run=ctx.dry_run, message="no FAISS index — no-op"
            )

        if ctx.dry_run:
            ctx.log(self.plan(ctx))
            return MigrationResult(self.id, applied=False, dry_run=True, message=self.plan(ctx))

        from ...search.migration import migrate_faiss_to_lance

        ctx.log(f"migrating {faiss_dir} → {lance_path}")
        stats = migrate_faiss_to_lance(faiss_dir, lance_path)
        details = {
            "segments": stats.segments,
            "insights": stats.insights,
            "skipped": stats.skipped,
            "lance_path": str(lance_path),
        }
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=False,
            message=f"migrated segments={stats.segments} insights={stats.insights}",
            details=details,
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Confirm the LanceDB index exists with non-empty tiers (skip pure no-op)."""
        faiss_dir = self._faiss_dir(ctx)
        if not (faiss_dir / VECTORS_FILE).is_file():
            return True, "no source FAISS index (no-op migration)"
        lance_path = self._lance_path(ctx)
        if not lance_path.exists():
            return False, f"LanceDB index missing at {lance_path}"
        try:
            from ...search.backends.lancedb_backend import LanceDBBackend

            health = LanceDBBackend(str(lance_path)).health()
        except Exception as exc:  # noqa: BLE001 - surface as a verify failure
            return False, f"LanceDB health check failed: {exc}"
        total = int(health.get("segments", 0)) + int(health.get("insights", 0))
        if total <= 0:
            return False, "LanceDB index has no rows"
        return (
            True,
            f"LanceDB ok: segments={health.get('segments')} insights={health.get('insights')}",
        )
