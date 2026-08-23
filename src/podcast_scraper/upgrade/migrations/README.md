# Corpus-upgrade migrations — the one canonical home

**This directory is THE place for every migration** — search-index rebuilds, on-disk
artifact/schema shape changes (GI/KG/enrichment), corpus-format bumps, and any future
database/storage migration. If you are changing something a deployed corpus already has on disk,
the answer is: **add a migration here.** Do not write a one-off script and do not add a parallel
read-time-only shim.

A CI check (`.github/workflows/migration-guard.yml`, GitHub #1176) enforces this: a PR that touches
a schema **sentinel** file (`docs/architecture/gi/gi.schema.json`, `.../kg/kg.schema.json`,
`enrichment.schema.json`, `config/corpus_snapshot_format.json`, `corpus_version.py`) must either add
a migration here or carry an explicit `migration-not-required: <reason>` opt-out in the PR body.
Prefer adding the migration.

## What a migration is

An ordered, **idempotent**, ledger-recorded unit of upgrade work. The runner
(`cli upgrade run`) applies every registered migration whose `id` is not yet in the per-corpus
ledger, in `id` order, exactly once. "Idempotent" is load-bearing: running twice, or on a corpus
already at the target shape, must be a clean no-op (detect `before == after` and skip the write).

## How to add one

1. **Create `mNNNN_<snake_name>.py`** here (next free number; `id` must match `mNNNN_<snake_name>`
   and sort after the last one). Subclass `..migration.Migration` and set:
   - `id` — e.g. `"0005_gi_v3_1_route_and_tag"` (stable; lexicographic order == apply order)
   - `to_version` — the corpus version this step brings the corpus to (semver, e.g. `"2.7.1"`)
   - `description` — one line; shows in `status`/ledger
   - implement `apply(ctx) -> MigrationResult`; optionally override `plan(ctx)` (dry read summary)
     and `verify(ctx)` (checkable post-condition).
2. **Reuse the transform logic — don't re-implement it.** The document rewrite functions live in
   `podcast_scraper/migrations/gil_kg_identity_migrations.py` (`migrate_gi_document_v3`,
   `migrate_gi_document_v3_1`, `migrate_kg_document_v2`, …). A framework migration is a thin walker
   that globs the artifacts and applies the matching transform (see `m0003` and `m0005`). Keep pure
   transforms there; keep the walk/write/ledger concerns here.
3. **Write atomically.** tmp-file + `os.replace` — a kill mid-write must not leave a truncated,
   unparsable artifact. Record unparsable files in `details` but do not fail the run on them.
4. **Register it** in `registry.py` (import + add to `_MIGRATIONS`).
5. **Test it** with a `tests/unit/upgrade/test_migration_NNNN.py` (stamp/idempotent/dry-run/no-op/
   unparsable), and extend the migration-set assertion in `tests/unit/upgrade/test_cli_handlers.py`
   — it is a literal, and it is the one place adding a migration is *meant* to be a deliberate
   edit. `tests/integration/upgrade/test_end_to_end.py` derives its set from `get_migrations()` and
   needs no edit; it used to hold a literal too, and m0007 turned it red on main.

## Examples

- `m0001_faiss_to_lance`, `m0002_two_tier_native_reindex`, `m0004_insight_type_reindex` — **search
  index** rebuilds (Lance).
- `m0003_gi_v3_typed_mentions` — **GI artifact** rewrite (typed MENTIONS + schema 3.0).
- `m0005_gi_v3_1_route_and_tag` — **GI artifact** version stamp 3.0 → 3.1 (ADR-135/#1191). Minimal
  reference for "add a migration for a schema bump."
- `m0006_kg_v2_typed_entities` — **KG artifact** rewrite (typed Person/Organization + id/kind
  normalization, schema 2.0). Replaced the former standalone `scripts/migrate_kg_*.py` one-offs.
- `m0007_scope_bare_person_names` — **GI + KG together**: a single-token person id (`person:jensen`)
  identifies someone within one episode and nobody globally, so it is scoped per episode — or
  healed to the full name when that episode has exactly one candidate (#1685). The reference for
  "both artifacts of an episode must be rewritten as a pair, or neither."

See `docs/guides/CORPUS_UPGRADE.md` for the runner, ledger, and CLI details.
