# Corpus upgrade framework (#862)

Managed, idempotent migrations for moving a **deployed corpus** across releases. The
registered steps bring a 2.6 corpus up to the 2.7 two-tier **LanceDB** search index
(RFC-090 / #858, ADR-099). Use this when upgrading an on-disk corpus to a newer code
version — not for fresh corpora (those are produced at the current version by the
pipeline).

**Are you here because you're changing a schema or artifact shape?** Jump to
[When is a migration required?](#when-is-a-migration-required) — that section is the
decision guide. The trigger + full checklist is also in `AGENTS.md` →
"Migrations — WHEN to add one" so every agent editing an on-disk shape sees the
prompt before touching code.

## TL;DR

```bash
export CORPUS_DIR=/path/to/corpus            # parent of feeds/
make upgrade-status   CORPUS_DIR=$CORPUS_DIR  # what's pending? (exit 2 if pending)
make upgrade-dry-run  CORPUS_DIR=$CORPUS_DIR  # preview the plan, write nothing
make upgrade-corpus   CORPUS_DIR=$CORPUS_DIR  # apply (non-interactive, --yes)
make upgrade-verify   CORPUS_DIR=$CORPUS_DIR  # verify applied migrations
```

Or drive the CLI directly (interactive confirmation, JSON, version ceiling):

```bash
python -m podcast_scraper.cli upgrade status  --corpus-dir $CORPUS_DIR [--json]
python -m podcast_scraper.cli upgrade list    --corpus-dir $CORPUS_DIR
python -m podcast_scraper.cli upgrade run     --corpus-dir $CORPUS_DIR [--dry-run] [--yes] [--to 2.7.0]
python -m podcast_scraper.cli upgrade verify  --corpus-dir $CORPUS_DIR
```

## Manual vs automated

- **Manual / on-demand:** `upgrade status` to see what's pending, `upgrade run`
  (no `--yes`) prompts before applying, `--dry-run` previews.
- **Automated:** `upgrade status --json` exits **2** when migrations are pending
  (distinct from error=1), so a boot script or CI job can gate on it:

  ```bash
  python -m podcast_scraper.cli upgrade status --corpus-dir "$CORPUS_DIR" --json || \
    python -m podcast_scraper.cli upgrade run --corpus-dir "$CORPUS_DIR" --yes
  ```

## How it works

- **Version stamp** comes from the corpus's existing `corpus_manifest.json`
  (`produced_by.code_version`, #796). After a migration runs, the new version is
  recorded in `upgrade_ledger.json` at the corpus root, which then takes precedence.
- **Ledger-driven** (like a DB migrations table): a migration runs unless its id is
  already recorded. Re-runs are safe — steps are idempotent (the native reindex builds
  the index in place) and recorded steps are skipped.
- **Stop on failure:** if a step fails, earlier steps stay recorded; fix and re-run
  to resume from the failed step.

## Forward-compatibility with a database

The framework is deliberately storage-agnostic so it survives a future move from
files-on-disk to a database:

- *Where* version + ledger live is behind the `StateStore` protocol
  (`upgrade/state.py`). Today: JSON on disk. With a DB: add one `DbStateStore` over a
  `schema_migrations` table — the runner and CLI don't change.
- A `Migration` (`upgrade/migration.py`) is opaque about storage. A step may rewrite
  files today and target a DB tomorrow; the runner only sequences and records it.

## When is a migration required?

**You must add a framework migration whenever a code change would leave
already-deployed corpora unable to serve correctly with the new code.** In
practice that means any of:

- **On-disk artifact shape change** — anything under `<corpus>/feeds/**` that
  a reader deserialises: `*.gi.json`, `*.kg.json`, `*.metadata.json`,
  `corpus_manifest.json`, `run_summary.json`, `feeds.spec.yaml`, the enrichment
  sidecars under `<corpus>/metadata/enrichments/`, the search sidecars under
  `<corpus>/search/`. If an existing field is renamed, retyped, or removed, or
  the vocabulary of an enum shifts — that's a migration.
- **Schema JSON update** — `docs/architecture/gi/gi.schema.json`,
  `docs/architecture/kg/kg.schema.json`, or
  `src/podcast_scraper/enrichment/_schema/enrichment.schema.json` when the
  change is non-additive. Additive fields alone do not need a migration if the
  reader tolerates absence.
- **Index format change** — the two-tier LanceDB index layout, embedding
  dimensionality, or any content-addressed cache path an old corpus would
  point at incorrectly.
- **Producer/reader contract change** — the reader is now stricter about a
  field the producer used to emit loosely (or vice versa) → a migration
  normalises the corpus so the new reader accepts it.

**You do NOT need a migration when:** you add a new optional field the reader
tolerates absence of; you add a new derived sidecar the reader treats as
optional; you add a new enricher whose absence is a no-op; you change
internal implementation without touching on-disk shape.

**When in doubt, add one.** The `Migration` scaffold is cheap, and a no-op
migration recorded in the ledger is a permanent breadcrumb that "at version
X.Y this was checked" — future migrations can rely on that history.

## Adding a migration

1. **Write the migration.** Add `upgrade/migrations/mNNNN_<name>.py`
   subclassing `Migration` (set `id`, `to_version`, `description`; implement
   `apply`, optionally `verify`). Make `apply` idempotent — running it twice
   must be a no-op the second time. Include the retro-active detection (how
   do you know this migration has already been applied to a file that
   predates the ledger entry?).
2. **Register it.** Add to `upgrade/registry.py` — the runner sorts by id, so
   pick `mNNNN` = the next integer.
3. **Bump the producer format version.** Edit
   `config/corpus_snapshot_format.json` and increment `corpus_format_version`
   when the migration is user-visible in the on-disk corpus (any change a
   backup-repo consumer might see). Additive-only migrations that affect only
   the ledger can keep the format version.
4. **Extend the reader range.** Edit
   `config/corpus_snapshot_reader_support.json`. If the new format is
   backward-compatible, bump `supported_corpus_format_version_max`. If it is
   NOT backward-compatible (readers cannot serve older corpora), bump BOTH
   `min` and `max` and understand this will refuse to restore any snapshot
   older than the new min — coordinate this with the operator before landing.
5. **Test.** Add unit coverage in `tests/unit/upgrade/test_migration_NNNN.py`
   (mirror the shape of `test_migration_0002.py` — cover apply on unstamped,
   apply on already-applied, verify, and any error branches). Extend
   `tests/integration/upgrade/test_end_to_end.py` with the new step's row in
   the "no FAISS" walk.
6. **Update the read-time shim (if needed).** If serving legacy artifacts
   without migration is still supported, extend
   `src/podcast_scraper/migrations/gil_kg_identity_migrations.py` with a
   matching transform so an unmigrated corpus still reads correctly. Cross-
   reference the framework migration and the shim in each other's docstrings.
7. **Update docs.** Three files:
   - **This guide** — add a bullet under "Registered migrations" above.
   - `docs/api/MIGRATION_GUIDE.md` — add a section to the current version
     block (or start a new version block if you're crossing a release
     boundary) describing the on-disk change from the caller's perspective.
   - Any RFC / ADR you're implementing — link the migration id
     (`m0004_<name>`) back from the RFC.
8. **Consider a read-check.** If the migration is high-stakes (renames a
   load-bearing field), extend `src/podcast_scraper/corpus_version.py`'s
   `assess_corpus_version_compat` with a warning when the corpus predates
   the change — so operators who skip the framework path still get told.

Restore paths already run `podcast upgrade run --yes` post-extract
(`scripts/ops/restore_corpus_from_tarball_host.sh`, #1176), so the new step
lands automatically on every prod restore once merged. Local
`make restore-corpus` / `make import-corpus` require the operator to run
`make upgrade-corpus CORPUS_DIR=<path>` by hand — this is documented in the
airgap runbook, not enforced.

## Prod-state fixture maintenance (after every prod deploy)

The CI net includes a fixture-pinned test
(`tests/integration/upgrade/test_pinned_fixture_upgrade.py`) that proves the
gap between the last prod deploy and current `main` will apply cleanly.
The pin is at `tests/fixtures/upgrade/corpus_at_last_prod_release/` and its
identity comes from `config/last_deployed_prod_version.json`. **We only test
prod → HEAD** — the gap that matters for the next deploy. Older-than-prod
state is not tested; that gap is history.

**Steps 1–3 below are automated** by the `Bump prod-state marker + pinned fixture`
step in `.github/workflows/deploy-prod.yml`, which opens a PR
titled `chore(prod-marker): bump to sha-<short> after prod deploy` on every
green deploy. Step 4 is the human-in-the-loop step; the unit test in
`test_pinned_fixture_shape.py` fails the PR if you skip it.

1. `config/last_deployed_prod_version.json` — `code_version`, `sha`,
   `deployed_at`, and `applied_migrations` (from `registry.py` at the
   deployed SHA). **Automated.**
2. Fixture's `upgrade_ledger.json` — records every migration in
   `applied_migrations`, with `to_version` matching each migration's declared
   value. **Automated.**
3. Fixture's `corpus_manifest.json.produced_by.code_version` — bumped to
   match the marker. **Automated.**
4. On-disk artifact shapes in the fixture that changed as part of the
   migrations that just deployed. **Manual** — the auto-bump script cannot
   run migrations against the fixture (would need LanceDB and per-migration
   context). Fix by hand-editing the fixture in the same PR the workflow
   opened; `test_pinned_fixture_shape.py` tells you exactly which file drifted.

The workflow uses `secrets.PROD_MARKER_PR_TOKEN` when present, otherwise falls
back to `GITHUB_TOKEN` (which can open PRs but not push to protected `main`).
This bump is one commit per prod deploy. It does not need to happen between
deploys. See the fixture's own `README.md` for the detailed contract.

## Registered migrations

- `0001_faiss_to_lance` — **historical, now a recorded no-op.** It once migrated an
  existing FAISS index into the two-tier LanceDB layout; FAISS was retired in #995
  (ADR-099), so there is nothing to migrate. Kept registered so the version chain and
  ledger stay intact for corpora that already recorded it.
- `0002_two_tier_native_reindex` — build the two-tier LanceDB index **natively** from
  corpus artifacts when none exists. No-op when an index is already present. With 0001
  reduced to a no-op, this step is what actually (re)builds the index for a pre-2.7
  corpus.
- `0003_gi_v3_typed_mentions` — apply the RFC-097 v3 GI schema to every `*.gi.json`
  under the corpus root: bump `schema_version` `2.0` → `3.0`, rewrite legacy
  `MENTIONS` edges (Insight → Person/Org) as typed `MENTIONS_PERSON` /
  `MENTIONS_ORG`, and normalise the legacy `insight_type` vocab
  (`fact` / `opinion`) to v3 (`claim` / `observation`). Idempotent — a file already
  at 3.0 with typed edges passes through unchanged. Wraps the standalone
  `scripts/migrate_gi_to_v3.py` so a fresh operator can rely on
  `make upgrade-corpus` catching it. KG-side `MENTIONS` edges (Topic → Episode
  discovery) stay untouched by design (see
  `src/podcast_scraper/migrations/gil_kg_identity_migrations.py`).

**Not a migration:** the cross-episode entity canonical map (#852) is computed *live*
at graph-build (`search/corpus_graph.py` → `build_entity_id_map`), not persisted —
there is no artifact to rebuild, so it needs no migration.

## Other migration surfaces in the repo

The `upgrade/` framework above owns **on-disk corpus migrations**. Two other
migration surfaces exist and are easy to conflate:

- **Read-time schema shims** live in
  `src/podcast_scraper/migrations/gil_kg_identity_migrations.py`
  (`migrate_gil_document`, `migrate_kg_document`, `migrate_kg_document_v2`,
  `migrate_gi_document_v3`). These are pure functions used by the server + graph
  build at read time to accept legacy artifact shapes without requiring an
  in-place upgrade. The framework's `m0003` step calls the same
  `migrate_gi_document_v3` under the hood, so a corpus upgraded through the
  framework does not need the read-time shim. Prefer the framework path when
  you own the corpus; the shim is the fallback for legacy artifacts you cannot
  rewrite.
- **API surface changes** (endpoint additions, response-shape moves, config
  renames) live in [`docs/api/MIGRATION_GUIDE.md`](../api/MIGRATION_GUIDE.md).
  That guide's v2.6→v2.7 section spans the same release as this framework's
  registered migrations — read both when moving a deployed corpus across a
  release boundary.

## Loose migration scripts

Historical one-shot scripts under `scripts/migrate*` predate this framework.
Each carries a status line in its docstring — "historical one-shot; superseded
by mNNNN" or "pending frameworking, see #NNNN". None run automatically; each is
manual, idempotent, and safe to re-execute. When adding a new step, prefer a
framework migration (`upgrade/migrations/mNNNN_<name>.py`) so an operator can
rely on `make upgrade-corpus` and the ledger; loose scripts should be reserved
for genuinely one-shot data-shape corrections that never repeat.

## See also

- [API Migration Guide](../api/MIGRATION_GUIDE.md) — API surface changes per version.
- [Corpus snapshot manifest and restore](CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md) —
  SSOT for corpus backup / restore surfaces (each restore path pairs with
  `upgrade-check` — see [#1176](https://github.com/chipi/podcast_scraper/issues/1176)).
- [Corpus airgap runbook](CORPUS_AIRGAP_RUNBOOK.md) — post-import upgrade recipe.
