# Corpus upgrade framework (#862)

Managed, idempotent migrations for moving a **deployed corpus** across releases. The
registered steps bring a 2.6 corpus up to the 2.7 two-tier **LanceDB** search index
(RFC-090 / #858, ADR-099). Use this when upgrading an on-disk corpus to a newer code
version — not for fresh corpora (those are produced at the current version by the
pipeline).

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

## Adding a migration

1. Add `upgrade/migrations/mNNNN_<name>.py` subclassing `Migration` (set `id`,
   `to_version`, `description`; implement `apply`, optionally `verify`). Make `apply`
   idempotent.
2. Register it in `upgrade/registry.py`.
3. Add unit coverage in `tests/unit/upgrade/`.

## Registered migrations

- `0001_faiss_to_lance` — **historical, now a recorded no-op.** It once migrated an
  existing FAISS index into the two-tier LanceDB layout; FAISS was retired in #995
  (ADR-099), so there is nothing to migrate. Kept registered so the version chain and
  ledger stay intact for corpora that already recorded it.
- `0002_two_tier_native_reindex` — build the two-tier LanceDB index **natively** from
  corpus artifacts when none exists. No-op when an index is already present. With 0001
  reduced to a no-op, this step is what actually (re)builds the index for a pre-2.7
  corpus.

**Not a migration:** the cross-episode entity canonical map (#852) is computed *live*
at graph-build (`search/corpus_graph.py` → `build_entity_id_map`), not persisted —
there is no artifact to rebuild, so it needs no migration.
