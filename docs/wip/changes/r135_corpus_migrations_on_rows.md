---
id: r135_corpus_migrations_on_rows
class: light
status: open
issue: none (E06 item 6, tracker row w-migrations; RFC-135, public #2568)
touches:
  - src/podcast_scraper/upgrade/rows.py
  - src/podcast_scraper/upgrade/cli_handlers.py
  - src/podcast_scraper/upgrade/runner.py
  - src/podcast_scraper/db/engines.py
  - tests/unit/podcast_scraper/upgrade/test_rows.py
---

# R135 — the corpus migration framework on rows

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4.

## 0. Why light

**What changes.** No stored file changes, and no migration changes or is added. The change adds
one module, `upgrade/rows.py`: the framework RFC-135 designs for migrations over rows in the
`corpus` schema (the `RowMigration` base, `Rows` that writes a receipt for every change, the
`migrator` engine, `run_migration` and `undo_migration` under the `corpus` lease, the database
ledger). Its tables are `corpus_0006_migration_runs`'s (committed before it, `db/schema.py`).

- **Nothing registers a row migration.** The registry and every file migration are unchanged;
  the file runner (`runner.py`) now leaves out a migration of kind `rows`, which none of the
  registered ones is, so `upgrade run`, `verify` and `rerun` do on the files what they do today;
  the only row migrations are test-only, in its tests. `migrator_engine` moves to
  `db/engines.py`, beside `unbounded_connect_args`; nothing else there changes.
- **The CLI's new paths are row mode only.** `upgrade run --database`, `upgrade undo` and
  `upgrade surfaces` are new in `upgrade/cli_handlers.py`; every file subcommand, and `run`
  without `--database`, takes the code path it takes today.
- **No digest moves from this change** (0051's is `5699a0209e74` on 35650421d, on 5e44cff02 and
  with this change: `digests_check.py`, every public migration with a shipped outcome). `rows.py`
  is a new module that no existing migration imports, so it is in no migration's code closure;
  nor is `cli_handlers.py` (`closure54.txt` lists no CLI module).
- **`runner.py` and `db/engines.py` are in 0054's closure** (`closure54.txt`), and 0054 is a
  Common migration still pending on prod. Its digest is taken over the installed public package
  (`outcome_hook.py:74`), which Common's image pins by `PUBLIC_REF` (e29fcc3ad), so these edits
  move 0054's digest only when `PUBLIC_REF` is bumped to a public commit carrying them. The
  database session and ingestion agreed (2026-11-26) to re-gate 0054 once, after the last corpus
  revision of this release is on public's release branch and `PUBLIC_REF` is bumped to it, and
  before the release image is chosen; these edits are under that re-gate. Rehearsal 1 runs
  `PUBLIC_REF` e29fcc3ad and is unaffected.
- **But `corpus_0006` moves 0051's digest, and 0054's.** `db/schema.py` is in both closures
  (`module_closure` of `m0051_kg_3_0`, 509 modules, and `closure54.txt`), and the separate commit
  `corpus_0006` (35650421d) edits it: 0051's digest on this tree is `5699a0209e74`, its shipped
  outcome's `ec6f3d57c6ce`. The release base d2e2cde76 is already stale for 0051 (`9bb72871e96d`:
  the database session's corpus_0004 and corpus_0005 edited `schema.py` after the outcome), so
  this branch adds a move, not the first one. 0051 is applied on prod (snapshot-prod-20261124's
  `upgrade_ledger.json`: 2026-11-23T20:25:25Z), so its move does not matter; 0054's is under the
  one re-gate above.
- **What a listener or reader is shown:** unchanged. The framework writes only when a row
  migration is run, and the first real one is written after the cut-over with a record of its own
  (full or light by what it changes).

**Proof:** its tests, against the test database: apply then undo restores every row (`to_jsonb`
equal); a dry run leaves every row as it was and records a `dry_run` run with zero receipts kept;
a second apply is refused; an undo refuses, and restores nothing, when a row changed after the
apply; a failure inside `apply_rows` leaves no row, receipt or ledger change; the changed-row
census (every changed row has a receipt); the `corpus` lease refuses a run while another holder
has it; the `corpus_revision` bump is a no-op while its table is absent.

Added with the CLI: a dry run leaves every corpus table's rows as they were and records its plan
(per item: receipts per table, the sha256 of each row read); apply, undo, apply gives the rows of
one apply but for the times the database stamps; an in-transaction surface sees the changes and
its failure undoes them; an after-commit failure keeps the apply, is recorded, and `upgrade
surfaces` completes it; an ingest is refused while a migration holds the lease; the revision bump
advances the counter when its table exists (made and rolled back in the test); every run is a
`corpus_migration.run` event; the regime refuses (exit 3) on a digest, applied-set, verdict or
dirty-code mismatch and without an outcome, stops (exit 4) on a plan difference with nothing
written, applies with an accept file naming the difference, and stops on one naming another; the
CLI refuses without a migrator URL and off a terminal without `--yes`, exits 6 with the re-run
command on stderr's last line, 1 for an undo refused with the changed rows listed, and leaves the
file commands as they were.

Added with the measure: plan rows are keyed by the item's natural id (the row scoreboard's key);
with no row scoreboard registered the measure is recorded skipped; with one, it scores before and
after `apply_rows` in the transaction, passes when the deltas are the clone's, keeps the apply and
names the failure (exit 5, with the undo command) when they are not, and rolls the apply back
when the scoreboard fails. The file runner leaves a migration on rows out.

The roles (not a repo test; `~/.cache/closelistening-ingestion-drill/rows-migrator/run.sh`, a
throwaway `postgres:18.6` with infra #120's `roles.sql` and `pg_db_grants.sh`'s corpus step,
removed on exit): the corpus plane migrated as `migrator` owns the three framework tables; a dry
run, apply and undo as `migrator` (content written as `pipeline_rw`) restore every row;
`pipeline_rw` cannot write `migration_runs` or `corpus_migrations` or read `migration_receipts`;
`corpus_ro` reads the runs and receipts and cannot write them. `ROLES: PASS`, 12 checks,
2026-11-26.
