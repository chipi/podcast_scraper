# RFC-135: Corpus migrations on rows

- **Status**: Draft, 2026-11-26, for review by the database session and the operator
- **Release**: E06, the corpus cut-over to Postgres (with News 2.0, dark). Built before the
  cut-over; no row migration runs on prod until after it.
- **Authors**: Marko Dragoljevic (the ingestion session)
- **Stakeholders**: the operator; the database, infra and public sessions
- **Related ADRs**:
  - [ADR-169](../adr/ADR-169-the-database-is-the-source-of-truth.md) (the database is the source
    of truth; decision 7, migrations; Implementation Notes, a database `StateStore`)
  - [ADR-172](../adr/ADR-172-module-migration-histories-and-readiness.md) (expand-only histories)
  - [ADR-174](../adr/ADR-174-extension-corpus-migrations.md) (extension corpus migrations, one
    numbering)
  - [ADR-184](../adr/ADR-184-the-corpus-schema-and-its-histories.md) (the corpus schema; infra
    owns grants; no `downgrade()`)
- **Related Documents**:
  - [Corpus upgrade guide](../guides/CORPUS_UPGRADE.md), [Corpus change
    method](../guides/CORPUS_CHANGE_METHOD.md)
  - Common: CRFC-005 (the corpus database model; §7 roles and pools, §8 owners),
    `docs/wip/platform-plan/DECISIONS.md` (§4 leases and lock order, §5 the migrate step),
    `docs/wip/platform-plan/epics/E06-plan.md` (item 6, "the corpus migration framework")

## Abstract

- **What:** the `upgrade` framework gains a second kind of corpus migration, one that changes
  rows in the `corpus` schema instead of files. It keeps today's contracts (one numbering,
  pending by ledger, the outcome hook's regime, receipts and undo) and changes their storage:
  the ledger is `corpus.corpus_migrations`, the receipts are `corpus.migration_receipts`, a dry
  run is the apply inside a transaction that is rolled back, and the run connects as `migrator`.
- **Why:** after the cut-over the corpus is rows (ADR-169), and the prod fix ladder's rung 2 is "a
  deterministic data migration over the rows using evidence already stored (dry-run as a
  rolled-back transaction, verify by SQL assertions, undo from the receipts table, run under the
  `migrator` role)" (`CLAUDE.md`). Today only the two tables exist (`corpus_0002_operations`); no
  code reads or writes them.
- **Scope:** the framework, its schema additions, the `migrator` engine, the CLI and the tests.
  Not the change method's rewording for rows (E06 item 7, the operator approves it), and not any
  real data migration: the first one is written after the cut-over, against the tables then in
  place.

## Problem Statement

The file framework (`podcast_scraper/upgrade/`) is built around files:

- the ledger is `<corpus>/upgrade_ledger.json` (`state.FilesystemStateStore`);
- receipts are a JSONL file of `{relpath, sha_before, sha_after}`, backups are file copies
  (`file_rewrite.py`), and undo restores a file only while it still holds `sha_after`;
- a dry run is the migration's own code path that writes nothing, and the outcome hook's plan
  inputs are the sha256 of every file the plan read (`read_recorder.py`);
- the corpus lock is a file lock (`utils/corpus_lock.py`); `upgrade run` snapshots the corpus
  with `copytree` first.

None of that carries over to rows. What the schema has today leaves gaps the framework must
close (the survey, 2026-11-26):

1. `corpus_migrations.status` allows `'dry_run'`, but a dry run that is a rolled-back transaction
   leaves no row.
2. `corpus_migrations.natural_id` is `UNIQUE`, so apply, undo and apply again of one id have no
   defined rows; nor does a per-item rerun (files: `<receipts>.rerun-<stamp>.jsonl`).
3. `migration_receipts` has no run id, so the receipts of an apply, its undo and a second apply
   cannot be told apart, and there is no home for what the file era keeps beside the receipts
   (the outcome receipt `upgrade-outcomes/<id>.json`, the plan diff).
4. There is no `to_version` on the ledger, no `migrator` engine (`db/engines.py` has plane rw/ro
   and `corpus_ro` only; `ENV_MIGRATOR_URL` is read by nothing), and no lease resource named for a
   corpus migration (DECISIONS §4 replaces the file lock with `job_leases`).
5. `CORPUS_UPGRADE.md` promises "a `DbStateStore` over a `schema_migrations` table"; the table
   that exists is `corpus_migrations`.

## Goals

1. A row migration is a `Migration` (same `id`, `to_version`, `description`, `apply`, `verify`,
   `plan`), registered the same way (the static list or `Extension.corpus_migrations`), in the
   same numbering. The runner does not care which kind it runs, except for where state lives.
2. **Atomic:** a migration's row changes, its receipts, its ledger row and the `corpus_revision`
   bump commit in one transaction, or nothing does. No "applied but not recorded" state.
3. **Dry run = the apply, rolled back.** The same code path as the apply; what it would write is
   read from the receipts it wrote before the rollback, then recorded in a separate transaction.
4. **Undo from receipts, all or nothing.** Each changed row is restored only while it still holds
   its receipt's `after`; if any row has changed since, the undo refuses and writes nothing.
5. **Verify by SQL assertions**, on a read connection.
6. **Runs as `migrator`, in a one-shot**, holding the corpus write lease exclusively; refused, never
   waiting, while an ingest or enrichment holds it (DECISIONS §4's two-process test "migration
   refused during an ingest").
7. **The regime holds** (ids ≥ 0031): the shipped outcome, the code digest, first pending, the plan
   diff with exits 3 / 4 / 5 and `--accept-plan-diff`, with plan inputs hashed from rows.
8. **Derived surfaces in the same run:** the migration names the items it changed; the run
   refreshes what is built from them (rungs 3 and 4 of the ladder) before it reports done.

## Constraints & Assumptions

- Expand-only schema changes, no `downgrade()` (ADR-172 decision 6, ADR-184 decision 4). The
  corpus history has not shipped to prod, but other branches and the rehearsal databases are at
  the latest corpus revision, so this RFC adds a revision rather than editing `corpus_0002`.
- Infra owns grants (ADR-184 decision 3). `migrator` owns the `corpus` schema; the grant step
  after each migrate gives `pipeline_rw` and `corpus_ro` what this RFC's tables need (§3).
- The `migrator` URL goes only into one-shots, connection limit 2 (CRFC-005 §7).
- Migrations read stored evidence only: no LLM call, no model, no network (the prod ladder).
- The content tables are being written (`sources`, `content_items` and `item_cards` in
  `corpus_0003`; `documents` and hash-partitioned `segments` in `corpus_0004`, #2564; entities and
  intel to come, the database session's s-entities and s-intel). The framework and its tests use
  `content_items` and `item_cards`; it must not assume any other table's shape.
- `corpus_revision` (DECISIONS rule 7, bumped in every content transaction) is not in the schema
  yet. The database session adds it (one row, a bigint counter) in `corpus_0007`. The framework
  bumps it through one function that is a no-op until the table exists, and a test against
  `corpus_0007`'s table pins the bump, so it cannot stay a no-op.

## Design & Implementation

### 1. The migration kind

```python
class RowMigration(Migration):
    """A corpus migration over rows. ``apply_rows`` changes rows through ``ctx.rows``."""

    kind = "rows"

    def apply_rows(self, ctx: RowContext) -> RowResult: ...
    def verify_rows(self, conn: Connection) -> Tuple[bool, str]: ...  # SQL assertions
    def plan_rows(self, ctx: RowContext) -> Iterable[PlanRow]: ...     # for the outcome hook
```

- `RowContext` carries the open transaction's connection and `ctx.rows`, the only way a row
  migration writes: `rows.update(table, key, values)`, `rows.insert(table, values)`,
  `rows.delete(table, key)`. Each call reads the row's current state `FOR UPDATE`, writes, and
  appends a receipt (§2). A write outside `ctx.rows` is a review finding; a test asserts that
  every row a test migration changed has a receipt (the changed-row census in §Testing).
- `apply` (the base `Migration` method) is implemented by the framework: it opens the
  transaction, calls `apply_rows`, and commits or rolls back by `ctx.dry_run`. A row migration
  never implements `apply` itself.
- `RowResult` names the `item_ids` it changed, for the derived surfaces (§6).
- Idempotent: a second apply finds nothing to change and writes no receipt (as files).

### 2. Schema additions (`corpus_0006_migration_runs`)

| Object | Change |
| --- | --- |
| `migration_runs` (new) | `run_id` identity PK; `natural_id` text; `kind` (`dry_run`, `apply`, `undo`, `rerun`); `status` (`ok`, `refused`, `failed`); `code_digest`, `outcome_sha256` (nullable); `plan` and `result` JSONB; `started_at`, `finished_at` |
| `migration_receipts` | add `run_id` FK to `migration_runs` (nullable for expand-only; the framework always sets it) and `seq` int (order within the run) |
| `corpus_migrations` | add `to_version` text and `last_run_id` FK; the row is the ledger: one per migration id, `status` `applied` or `undone` |

- **The ledger stays one row per id** (`natural_id` UNIQUE keeps its meaning): apply inserts or
  sets `applied`, undo sets `undone`, a second apply after an undo sets `applied` again with a new
  `last_run_id`. Every attempt is a `migration_runs` row; a dry run is only that.
- **`'dry_run'` in `corpus_migrations.status` stays allowed and unused** (dropping a CHECK value is
  a contraction). The runner never writes it.
- `target_key` is the row's primary key as canonical JSON (`[42]`, or `["a", 3]` for a composite
  key); `before` and `after` are the whole row as JSONB (`to_jsonb(t)`), `before` null for an
  insert and `after` null for a delete.
- Grants (infra's step): `migrator` owns all three; `corpus_ro` reads them (status, the Workbench);
  `pipeline_rw` reads `corpus_migrations` (a writer refuses to start on a pending migration, as
  readiness does for schema heads) and has no privilege on the other two. Infra (2026-11-26): its
  default privileges (`infra/postgres/roles.sql`, `ALTER DEFAULT PRIVILEGES FOR ROLE migrator IN
  SCHEMA corpus`) give `pipeline_rw` full DML on every new corpus table, narrowed today only for
  `schema_history` and `alembic_version*`. So an infra PR adds the three tables to that narrowing,
  in `roles.sql` and in the corpus branch of `infra/deploy/pg_db_grants.sh`: revoke all from
  `pipeline_rw` on `migration_runs` and `migration_receipts`, and revoke INSERT, UPDATE, DELETE and
  TRUNCATE on `corpus_migrations`. A test in §Testing pins it.

The database session owns the corpus history and agreed to this table as written (2026-11-26):
the revision is `corpus_0006_migration_runs`, `down_revision` `corpus_0005_entities` (public
#2567); its `search_chunks` and `corpus_revision` revision takes `corpus_0007` on top. `schema.py`'s
`corpus_metadata` carries the same columns (`test_db_corpus.py` compares the metadata with the
migrated database); the grants are infra's step after the migrate.

### 3. The run

`upgrade run` (and `rerun`, `undo`) with `--database` (or when the corpus is configured as rows):

1. Connect as `migrator` (`db/engines.migrator_engine()`: `PODCAST_DATABASE_MIGRATOR_URL`, pool 1
   plus 0, no statement timeout as `migrate._engine`, `lock_timeout` 5 s so a held row lock fails
   fast instead of hanging).
2. Take the lease: `job_leases` resource `corpus`, mode `exclusive`, holder `upgrade:<run_id>`,
   with a heartbeat. Refused (exit 1, nothing written) when any holder has it, shared or
   exclusive. This replaces the file lock; the file lock is not taken in row mode.
3. Regime checks before the transaction (as today, exits 3 / 4): outcome, digest, first pending,
   the plan diff against the outcome's plan.
4. **One transaction:** `apply_rows`; receipts; `corpus_migrations` row; `corpus_revision` bump
   (last in the lock order, DECISIONS §4); `migration_runs` row with `result`; commit. A dry run
   reads its receipts back (what it would write, per table), then rolls back, then writes the
   `migration_runs` row (`kind = dry_run`, `plan`, `result`) in a second, short transaction.
5. Measure (the hook's exit 5 on an outcome mismatch keeps the apply, as files): scoreboard after,
   written to `migration_runs.result` instead of `.podcast_scraper/upgrade-outcomes/<id>.json`.
6. Derived surfaces (§6), then `verify_rows`, then release the lease.

**One transaction per migration** is the first version. A migration too large for one
transaction is not supported until one exists that needs it; the design for it (chunks, each its
own transaction, the lease re-checked per chunk, the run resumable from `migration_runs`) is
recorded here as the way forward, not built.

There is no `copytree` snapshot in row mode. The undo is the receipts; the backstop for a broken
migration is the pre-migration `pg_dump` of DECISIONS §5, taken by the one-shot's wrapper (the
`db-migrate` and `corpus-migration` skills), not by the framework.

### 4. Undo

`upgrade undo --migration <id>` (new subcommand, the operator's decision 2026-11-26; in row mode
only, since files keep each module's `undo(root)`). Row migrations define no undo of their own:
the receipts hold whole rows, so the undo is the same for every migration. It runs as `upgrade run`
does (§3: `migrator`, the exclusive lease, a `migration_runs` row, the derived surfaces, the same
exit codes) and carries the same guards: `--yes`, refused on a non-TTY without it, a
`--dry-run` that is the undo rolled back, and on prod the operator's go as for any apply. It:

- reads the receipts of the migration's `last_run_id`, in reverse `seq`;
- for each, locks the row and compares it with `after` (JSONB equality of the whole row); a row
  already equal to `before` counts as restored (the file rule); any other state refuses;
- restores `before` (update, re-insert, or delete for an insert);
- all in one transaction: one refusal rolls back the whole undo and lists every refused row;
- on success: `corpus_migrations.status = 'undone'`, a `migration_runs` row (`kind = undo`), the
  `corpus_revision` bump, then the derived surfaces for the restored items.

### 5. The outcome hook on rows

- Plan rows keep their shape (`meta`, `counts`, `inputs`). `inputs` becomes the sha256 of each row
  the plan read, keyed `table:key`, recorded by `ctx.rows` reads (`rows.get`, `rows.select`)
  instead of `read_recorder`'s file hooks.
- The clone is a restored copy of a prod `pg_dump` (the rehearsal's local copy), not a file
  clone; `corpus_change_gate.py` gets a `--database-url` mode. The rest of the regime (digest over
  the module closure plus `verdicts_file()` and `list_file()`, the accept file) is unchanged.

### 6. Derived surfaces

`RowResult.item_ids` drives two kinds of refresh (the operator's decision 2026-11-26, on the
database session's recommendation, Key Decision 7):

- **Inside the data transaction:** the per-item surfaces in Postgres that readers read directly,
  for the changed items: `item_cards`, the flags on `content_items`, and the `search_chunks` and
  embeddings rows. A failure rolls the whole migration back, so rows and these surfaces never
  disagree. A dry run refreshes them too, and rolls them back with the rest.
- **After the commit:** the LanceDB search index (outside Postgres until E09), the corpus-wide
  projections and identity builds (batch-built under a build id; readers see the previous build
  during a rebuild), and the deterministic non-LLM enrichers. Each step's result goes into
  `migration_runs.result`; a failed step exits **6** with the migration applied, and the message
  names the command that re-runs that step on its own (each step is idempotent). Not 4: on this
  path 4 already means "refused before starting, nothing ran" (infra, 2026-11-26: `cl-oneoff.sh`'s
  nightly and pipeline guards, which also pass a container's own exit code through; `run-step`;
  the corpus-migration skill's `migrate.py`; `upgrade run`'s plan diff), and the codes in use are
  2, 3, 4, 5, 10 and 137. A failed one-shot is re-run by hand on the box, never automatically.
  The file framework's `upgrade run` exits 4 today when the ledger refresh fails after an apply
  (`cli_handlers.py`, the `ledger_refresh` call), the same ambiguity; changing it is outside this
  RFC.
- **No automatic undo** on a failed step: it would be a second prod write nobody reviewed, made
  while something is already failing.
- The views of DECISIONS (`v_bridge`, `v_context`, `v_catalog`, `v_served_id_map`,
  `v_confidence_words`) need no refresh while they are plain views; a view made materialized
  joins the first list or the second when that is decided.
- **A precondition outside this RFC:** a stale LanceDB hit after the commit is safe only if search
  hydration drops a hit whose natural id no longer resolves; the Player's intents (r-intents) must
  do that and do not yet.
- **Not measured:** how long the in-transaction refresh takes for a migration touching many items.
  The test migration measures it on the rehearsal copy; a migration too slow for one transaction
  is the chunking case of §3.

### 7. Where it lives

Public `podcast_scraper/upgrade/` (`rows.py`, `db_state.py`) and `db/engines.py`. CRFC-005 §8
lists "the migration framework" under Common; the framework extends public's `Migration`,
runner, registry and outcome hook, which every extension's migrations already use (ADR-174), and
ADR-169's Implementation Notes put the database `StateStore` there. Common's row migrations, when
there are any, live in Common as its file migrations do.

## Key Decisions

1. **The ledger is one row per migration id; every attempt is a run row.** Keeps `natural_id`
   UNIQUE meaningful and gives apply, undo, re-apply and dry runs a history.
2. **One transaction per migration**, chunking deferred until a migration needs it.
3. **Undo is all or nothing.** The file undo restores what it can and refuses the rest; on rows a
   partial undo would leave a state no receipt describes.
4. **Writes only through `ctx.rows`**, so every changed row has a receipt by construction.
5. **No framework snapshot; the `pg_dump` before the one-shot is the backstop.**
6. **Undo is a subcommand, `upgrade undo`** (the operator, 2026-11-26, on the recommendation): one
   entry point with the apply's role, lease, run row, derived surfaces, exit codes and guards; no
   per-migration undo code. Cost: a new CLI surface, and two undo mechanisms while files remain.
7. **Derived surfaces: per-item Postgres surfaces inside the transaction, the rest after the
   commit with exit 6, no automatic undo** (the operator, 2026-11-26, on the database session's
   recommendation; §6).

## Alternatives Considered

- **A dry run that runs a separate read-only "plan" path** (the file era's way): two code paths
  that can disagree; the rolled-back apply is the ladder's wording and proves the writes.
- **Receipts via triggers** on every content table: catches writes outside `ctx.rows`, but also
  the pipeline's own writes, needs grants and DDL per table, and puts migration bookkeeping in
  every ingest. Rejected; the census test covers the gap for test migrations.
- **A `natural_id` suffix per attempt** (`0060_x.rerun-<stamp>`, as file receipts): turns the
  ledger into a history and breaks "pending = id not applied".
- **SAVEPOINT-based dry runs inside a longer transaction:** no benefit for a one-shot.

## Testing Strategy

Against the test database (`PODCAST_TEST_DATABASE_URL`, as `tests/unit/podcast_scraper/db/`), with
a test-only row migration over `content_items` and `item_cards`:

- apply then undo: every touched row equal (`to_jsonb`) to before; `corpus_migrations` `undone`;
  pending again;
- dry run: every corpus table's row hash equal before and after; a `migration_runs` row with the
  plan; receipts zero;
- second apply writes no receipt; apply, undo, apply gives the same rows as one apply;
- undo refused when one row changed after the apply: nothing restored, the row list names it;
- a failure inside `apply_rows` leaves no row, receipt or ledger change;
- the derived surfaces (§6): the changed items' `item_cards` equal a fresh build after the apply;
  a failure in an in-transaction refresh leaves no row, receipt or ledger change; a failure in an
  after-commit step exits 6 with the migration applied, the failure in `migration_runs.result`,
  and the step re-run on its own completes it;
- the changed-row census: every row that differs after the test migration has a receipt;
- two processes: a migration refused while an ingest holds the lease, and an ingest refused while
  a migration holds it;
- the regime on rows: digest mismatch exit 3, plan diff exit 4, accept file, outcome mismatch exit
  5 with the apply kept;
- grants: the framework's tables usable as `migrator`, readable as `corpus_ro`, not writable as
  `pipeline_rw` (with infra's grant step applied in the test helper).

## Rollout & Monitoring

- Built and merged into `feat/corpus-database-release` before the cut-over; shipped dark: nothing
  registers a row migration, so `upgrade status` on prod reports the file ledger as today until the
  cut-over switches the corpus to rows.
- Each `migration_runs` row is also an `emit_event` (`corpus_migration.run`, with kind, status and
  counts) for the observability stack.
- The rehearsal runs one test row migration on the restored copy, applies, undoes and dry-runs it
  (E06's "a corpus migration on the copy").

## Relationship to Other RFCs

Extends the file framework of the [corpus upgrade guide](../guides/CORPUS_UPGRADE.md) (#862);
RFC-118's delta backbone is how rung 4's "affected items only" is computed for the derived
surfaces.

## Open Questions

1. Answered 2026-11-26 by the database session: yes, as `corpus_0006_migration_runs` (§2).
2. The lease resource and mode the ingest and enrichers take (`corpus`, shared?), so the refusal
   rule of §3 step 2 matches theirs. Infra (2026-11-26): not infra's; `job_leases` is the database
   session's tracker row `w-leases` (todo), and infra has no lease code. Asked of the database
   session.
3. Answered 2026-11-26: undo is a subcommand (Key Decision 6).
4. Answered 2026-11-26: the operator took the database session's recommendation (Key Decision 7,
   §6). Infra had not answered when the operator decided; its retry practice for a failed one-shot
   is still asked, for the runbook, not for this decision.
