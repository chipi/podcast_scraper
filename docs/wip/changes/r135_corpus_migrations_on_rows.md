---
id: r135_corpus_migrations_on_rows
class: light
status: open
issue: none (E06 item 6, tracker row w-migrations; RFC-135, public #2568)
touches:
  - src/podcast_scraper/upgrade/rows.py
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

- **Nothing registers a row migration.** The registry, the file runner and every file migration
  are unchanged, so `upgrade run`, `verify` and `rerun` do on the files what they do today; the
  only row migrations are test-only, in its tests.
- **No digest moves.** `rows.py` is a new module that no existing migration imports, so it is in
  no migration's code closure; `db/engines.py`, `db/settings.py`, `upgrade/migration.py`,
  `state.py`, `registry.py` and `runner.py`, which are in 0054's closure (`r37` drill
  `closure54_new.txt`, 2026-11-26), are not edited. (`db/schema.py` is in the closure and is
  edited by `corpus_0006`, a separate commit, as by the database session's own corpus revisions.)
- **What a listener or reader is shown:** unchanged. The framework writes only when a row
  migration is run, and the first real one is written after the cut-over with a record of its own
  (full or light by what it changes).

**Proof:** its tests, against the test database: apply then undo restores every row (`to_jsonb`
equal); a dry run leaves every row as it was and records a `dry_run` run with zero receipts kept;
a second apply is refused; an undo refuses, and restores nothing, when a row changed after the
apply; a failure inside `apply_rows` leaves no row, receipt or ledger change; the changed-row
census (every changed row has a receipt); the `corpus` lease refuses a run while another holder
has it; the `corpus_revision` bump is a no-op while its table is absent.
