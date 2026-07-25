"""0004 — reindex to populate ``insight_type`` on the insight tier (Search v3 §S8).

``LANCE_SCHEMA_VERSION`` bumped 2 → 3: the insight tier gained an ``insight_type``
column (RFC-072 GIL v1.1) so the Search v3 §S8 compare ``insight_types`` filter can
scope by type. An index built at v2 has no such column, so the read path
(``hybrid_search``) reports ``no_index`` for it (fail-safe — it never serves an
index missing a column the read path expects).

Why a NEW migration rather than reusing 0002: the upgrade runner records each
migration by **id** in the per-corpus ledger and runs it exactly once
(``runner.pending`` → ``m.id not in applied``). Every already-upgraded corpus has
``0002_two_tier_native_reindex`` in its ledger, so 0002 will never re-fire for the
v3 bump. This step carries a fresh id so it applies once on the release that ships
schema v3, rebuilding any v2 index natively from corpus artifacts.

Idempotent: a corpus already at schema v3 (freshly built post-deploy) is a no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ..migration import Migration, MigrationContext, MigrationResult


class InsightTypeReindexMigration(Migration):
    """Rebuild the two-tier index when its schema predates the ``insight_type`` column."""

    id = "0004_insight_type_reindex"
    # Shares the 2.7.1 release marker with m0003 (both are unreleased, landing in
    # the same next train over the deployed 2.7.0.dev0). to_version is a ledger
    # LABEL + optional --to-version ceiling, NOT a run gate: the runner applies any
    # migration whose id is absent from the corpus ledger regardless of version, so
    # a dev0 → dev1 deploy still triggers this via `cli upgrade run`.
    to_version = "2.7.1"
    description = "Reindex to add insight_type (LANCE_SCHEMA_VERSION 3) for §S8 compare filter"

    def _lance_path(self, ctx: MigrationContext) -> Path:
        return Path(ctx.options.get("lance_path") or ctx.corpus_root / "search" / "lance_index")

    def plan(self, ctx: MigrationContext) -> str:
        """Return a human summary of whether the LanceDB index needs an insight_type reindex."""
        from ...search.backends.lancedb_backend import lance_index_is_stale

        lance_path = self._lance_path(ctx)
        if not lance_path.exists():
            return "No LanceDB index present — nothing to reindex (a first build emits v3)."
        if lance_index_is_stale(lance_path):
            return (
                "LanceDB index schema is stale (pre-insight_type / v2) — rebuild natively so "
                "the insight tier carries insight_type for the §S8 compare filter."
            )
        return "LanceDB index already at the current schema (insight_type present) — no-op."

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Rebuild the LanceDB index when stale so the insight tier carries insight_type."""
        from ...search.backends.lancedb_backend import lance_index_is_stale

        lance_path = self._lance_path(ctx)
        if not lance_path.exists():
            ctx.log(f"no LanceDB index at {lance_path}; nothing to reindex")
            return MigrationResult(
                self.id, applied=True, dry_run=ctx.dry_run, message="no index — no-op"
            )
        if not lance_index_is_stale(lance_path):
            ctx.log(f"LanceDB index at {lance_path} already at current schema; skipping")
            return MigrationResult(
                self.id, applied=True, dry_run=ctx.dry_run, message="schema current — no-op"
            )
        if ctx.dry_run:
            ctx.log(self.plan(ctx))
            return MigrationResult(self.id, applied=False, dry_run=True, message=self.plan(ctx))

        from ...search.two_tier_indexer import build_two_tier_index

        ctx.log(f"rebuilding two-tier index at {lance_path} to add insight_type")
        stats = build_two_tier_index(ctx.corpus_root, lance_path)
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=False,
            message=f"insight_type reindex: episodes={stats.episodes} segments={stats.segments} "
            f"insights={stats.insights}",
            details={
                "episodes": stats.episodes,
                "segments": stats.segments,
                "insights": stats.insights,
                "lance_path": str(lance_path),
            },
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Confirm the index is no longer schema-stale (or genuinely absent)."""
        lance_path = self._lance_path(ctx)
        if not lance_path.exists():
            return True, "no LanceDB index (empty corpus or nothing to index)"
        from ...search.backends.lancedb_backend import lance_index_is_stale

        if lance_index_is_stale(lance_path):
            return False, "LanceDB index still schema-stale after reindex"
        return True, "LanceDB index at current schema (insight_type column present)"
