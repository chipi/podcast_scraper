"""0002 — native two-tier reindex fallback (2.6 → 2.7, follow-up B under #862).

Completes the chain that 0001 starts. 0001 (FAISS→LanceDB) is a **no-op when a
corpus has no FAISS index** to migrate. This step covers exactly that gap: if no
LanceDB index exists yet, build it **natively** from corpus artifacts
(``build_two_tier_index``, follow-up B); if one already exists (0001 built it, or a
prior run), it is a no-op. Together the two guarantee every upgraded corpus ends
with a two-tier index via the cheapest available path, without double-building.

Why this and not the originally-speculated migrations: the cross-episode entity
canonical map (#852) is computed **live** at graph-build (``corpus_graph`` →
``build_entity_id_map``), not persisted, so there is nothing to migrate; and the
basic vector-index build is already 0001. Registering those would be busy-work — see
the upgrade guide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ..migration import Migration, MigrationContext, MigrationResult


class TwoTierNativeReindexMigration(Migration):
    """Build the two-tier LanceDB index natively when 0001 left none."""

    id = "0002_two_tier_native_reindex"
    to_version = "2.7.0"
    description = "Native two-tier index build for corpora without a FAISS index (RFC-090 B)"

    def _lance_path(self, ctx: MigrationContext) -> Path:
        return Path(ctx.options.get("lance_path") or ctx.corpus_root / "search" / "lance_index")

    def plan(self, ctx: MigrationContext) -> str:
        """Describe whether a native build is needed for this corpus."""
        from ...search.backends.lancedb_backend import lance_index_is_stale

        lance_path = self._lance_path(ctx)
        if lance_path.exists():
            if lance_index_is_stale(lance_path):
                return "LanceDB index present but schema-stale — rebuild natively."
            if not (lance_path.parent / "metadata.json").is_file():
                return (
                    "LanceDB index present but missing the metadata.json offset sidecar "
                    "(pre-#1010 index) — reindex natively to emit it for the GIL offset verifier."
                )
            return "LanceDB index already present — no-op."
        return (
            f"Build two-tier LanceDB index natively from {ctx.corpus_root} (no FAISS to migrate)."
        )

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Build the native index when none exists, or rebuild a schema-stale one."""
        from ...search.backends.lancedb_backend import lance_index_is_stale

        lance_path = self._lance_path(ctx)
        sidecar_present = (lance_path.parent / "metadata.json").is_file()
        if lance_path.exists() and not lance_index_is_stale(lance_path) and sidecar_present:
            ctx.log(f"LanceDB index already present at {lance_path}; skipping native build")
            return MigrationResult(
                self.id, applied=True, dry_run=ctx.dry_run, message="index already present — no-op"
            )
        if lance_path.exists() and not lance_index_is_stale(lance_path):
            # Healthy index, but it predates the metadata.json offset sidecar (#1010). Reindex
            # to emit it (build_two_tier_index upserts idempotently + writes the sidecar) so the
            # GIL chunk-offset verifier works on upgraded corpora.
            ctx.log(f"LanceDB index at {lance_path} lacks metadata.json sidecar; reindexing")
        elif lance_path.exists():
            ctx.log(f"LanceDB index at {lance_path} has a stale schema; rebuilding natively")
        if ctx.dry_run:
            ctx.log(self.plan(ctx))
            return MigrationResult(self.id, applied=False, dry_run=True, message=self.plan(ctx))

        from ...search.two_tier_indexer import build_two_tier_index

        ctx.log(f"building native two-tier index at {lance_path}")
        stats = build_two_tier_index(ctx.corpus_root, lance_path)
        return MigrationResult(
            self.id,
            applied=True,
            dry_run=False,
            message=f"native build: episodes={stats.episodes} segments={stats.segments} "
            f"insights={stats.insights}",
            details={
                "episodes": stats.episodes,
                "segments": stats.segments,
                "insights": stats.insights,
                "lance_path": str(lance_path),
            },
        )

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Confirm a LanceDB index exists (built here or by 0001)."""
        lance_path = self._lance_path(ctx)
        if not lance_path.exists():
            # Acceptable only if the corpus genuinely has no indexable content.
            return True, "no LanceDB index (empty corpus or nothing to index)"
        try:
            from ...search.backends.lancedb_backend import LanceDBBackend

            health = LanceDBBackend(str(lance_path)).health()
        except Exception as exc:  # noqa: BLE001 - surface as verify failure
            return False, f"LanceDB health check failed: {exc}"
        total = int(health.get("segments", 0)) + int(health.get("insights", 0))
        if total <= 0:
            return False, "LanceDB index present but empty"
        return (
            True,
            f"LanceDB present: segments={health.get('segments')} insights={health.get('insights')}",
        )
