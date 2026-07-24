# ADR-127: Search-index `insight_type` column + reindex-on-upgrade (Search v3 §S8)

- **Status**: Accepted
- **Date**: 2026-07-24
- **Authors**: Marko
- **Related RFCs**:
  - [RFC-107](../rfc/RFC-107-search-v3-query-workspace.md) — Search v3 §S8 Compare (2 subjects)
  - [RFC-072](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) — GIL v1.1 insight types
- **Related Documents**:
  - `src/podcast_scraper/search/backends/lancedb_backend.py` — `LANCE_SCHEMA_VERSION`, `_insight_schema`
  - `src/podcast_scraper/upgrade/migrations/m0004_insight_type_reindex.py` — the migration
  - [ADR-099](ADR-099-lancedb-first-single-index-search.md) — LanceDB single-index search

## Context & Problem

Search v3 §S8's `POST /api/search/compare` gained an optional `insight_types`
filter (RFC-072 GIL v1.1 — `claim` / `recommendation` / `observation` /
`question` / `unknown`) that narrows both compared subjects symmetrically to the
requested insight types. The filter reads `metadata.insight_type` on each
insight-tier search hit.

Real-corpus verification against the prod-v2 corpus (99 episodes) found the
filter was **inert-but-destructive**: the two-tier LanceDB index never stored
`insight_type`, even though every GI node carries it in its `properties`. With
no such field on the hit, the filter dropped **all** insight-tier rows for any
requested type, leaving only segment/aux rows — so enabling the filter made both
packs ungrounded instead of narrowing them.

Root cause: the index build (`indexer.py::_collect_docs_for_episode`) omitted
`insight_type` from the insight document metadata, and the document model
(`InsightDocument`) + LanceDB `_insight_schema` had no column for it, so the
read-path projection (`hybrid_search._to_search_result`) had nothing to surface.

## Decision

**Add `insight_type` as a first-class column on the insight tier and bump the
lance schema version, self-healing existing indexes via a new upgrade
migration.**

1. `InsightDocument` gains `insight_type: Optional[str]`; the indexer emits it
   from the GI node `properties` (legacy nodes with no type default to
   `unknown`, matching `gil_kg_identity_migrations`).
2. `_insight_schema` gains an `insight_type` string column;
   `LANCE_SCHEMA_VERSION` bumps **2 → 3**.
3. `hybrid_search._to_search_result` surfaces `insight_type` into hit metadata
   for insight rows.
4. A new upgrade migration, `m0004_insight_type_reindex`, rebuilds any
   schema-stale index natively from corpus artifacts.

**Why a new migration rather than reusing `m0002_two_tier_native_reindex`:** the
upgrade runner records each migration by **id** in the per-corpus ledger and runs
it exactly once (`runner.pending` → `m.id not in applied`); `to_version` is only
an optional ceiling. Every already-upgraded corpus has `0002` in its ledger, so
`0002` will never re-fire for the v3 bump. A fresh id (`0004`) fires once on the
release that ships schema v3.

## Consequences

**Positive**

- The §S8 `insight_types` filter narrows by real type on real data (verified on
  prod-v2: `claim` keeps all claim insights grounded; other types drop them).
- The schema bump is fail-safe: an index at v2 is reported as `no_index` by the
  read path (`hybrid_search` → `lance_index_is_stale`) rather than served with a
  missing column, so no wrong/partial results leak before the reindex.
- Existing corpora self-heal on the next `cli upgrade` / `make upgrade-corpus`.

**Negative / cost**

- **Every existing corpus must be reindexed** before the filter works. Until the
  reindex runs, `/api/search` reports `no_index` for that corpus (search is
  unavailable, not wrong). Operators must run `cli upgrade` (or
  `cli index-two-tier`) after deploying this change.
- Reindex cost is a full re-embed of all docs (prod-v2: ~10.5k docs on
  `all-MiniLM-L6-v2`, minutes on CPU). Not GPU-bound.

## Migration

The upgrade runner is **id-ledger-driven, not version-gated**: `cli upgrade run`
applies any migration whose id is absent from the corpus's `.upgrade` ledger,
regardless of the package version. `m0004`'s `to_version` (`2.7.1`, shared with
`m0003`) is only a ledger label + optional `--to-version` ceiling — it does not
decide whether the migration runs. So a `2.7.0.dev0 → 2.7.0.dev1` deploy triggers
`m0004` the same as any version bump: the new image carries it in the registry,
the prod corpus ledger doesn't, so it is pending and runs.

- **On deploy/restore (prod path):** `restore_corpus_from_tarball_host.sh` with
  `RESTORE_UPGRADE_MODE=auto` runs `upgrade status` (exit 2 = pending) then
  `upgrade run --yes` in the live post-boot api container and restarts the api so
  it picks up the rebuilt v3 index. The DR drill workflow exercises
  status → dry-run → run → verify.
- **Manual:** `make upgrade-corpus CORPUS_DIR=<corpus>` (or, low-level,
  `python -m podcast_scraper.cli index-two-tier --output-dir <corpus>`).
- **Idempotent:** a corpus already at v3 is a no-op (`lance_index_is_stale` false).
- **Rollback:** the schema addition is additive; to revert, redeploy the prior
  code (which ignores the extra column) — the index stays readable. No data
  migration to undo.

## Alternatives considered

- **Join `insight_type` at query time from GI artifacts by `source_id`** —
  rejected: an N-hit fan-out of artifact reads per query, defeating the
  single-index design (ADR-099).
- **Reuse `m0002`** — rejected: cannot re-fire (ledger is id-keyed, see above).
