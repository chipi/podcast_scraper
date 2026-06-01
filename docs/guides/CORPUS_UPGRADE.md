# Corpus upgrade framework (#862)

Managed, idempotent migrations for moving a **deployed corpus** across releases. The
first registered step is the 2.6 → 2.7 FAISS → two-tier LanceDB migration (RFC-090
/ #858). Use this when upgrading an on-disk corpus to a newer code version — not for
fresh corpora (those are produced at the current version by the pipeline).

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
podcast upgrade status  --corpus-dir $CORPUS_DIR [--json]
podcast upgrade list    --corpus-dir $CORPUS_DIR
podcast upgrade run     --corpus-dir $CORPUS_DIR [--dry-run] [--yes] [--to 2.7.0]
podcast upgrade verify  --corpus-dir $CORPUS_DIR
```

## Manual vs automated

- **Manual / on-demand:** `upgrade status` to see what's pending, `upgrade run`
  (no `--yes`) prompts before applying, `--dry-run` previews.
- **Automated:** `upgrade status --json` exits **2** when migrations are pending
  (distinct from error=1), so a boot script or CI job can gate on it:

  ```bash
  podcast upgrade status --corpus-dir "$CORPUS_DIR" --json || \
    podcast upgrade run --corpus-dir "$CORPUS_DIR" --yes
  ```

## How it works

- **Version stamp** comes from the corpus's existing `corpus_manifest.json`
  (`produced_by.code_version`, #796). After a migration runs, the new version is
  recorded in `upgrade_ledger.json` at the corpus root, which then takes precedence.
- **Ledger-driven** (like a DB migrations table): a migration runs unless its id is
  already recorded. Re-runs are safe — steps are idempotent (the FAISS→LanceDB step
  merge-inserts) and recorded steps are skipped.
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

## Adding a migration

1. Add `upgrade/migrations/mNNNN_<name>.py` subclassing `Migration` (set `id`,
   `to_version`, `description`; implement `apply`, optionally `verify`). Make `apply`
   idempotent.
2. Register it in `upgrade/registry.py`.
3. Add unit coverage in `tests/unit/upgrade/`.

Known 2.6 → 2.7 migrations still to inventory/register: vector index rebuilds and the
cross-episode entity canonical-map rebuild (#852).
