# Corpus upgrade framework (#862)

Managed, idempotent migrations for moving a **deployed corpus** across releases. The steps
this repository registers (see [Registered migrations](#registered-migrations)) cover the
two-tier **LanceDB** search index (RFC-090 / #858, ADR-099), KG schema versions, the shared
artwork store, the feed-language backfill and transcript passage times. An installed extension
adds its own steps to the same sequence. Use this when upgrading an on-disk corpus to a newer
code version — not for fresh corpora (those are produced at the current version by the
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
make upgrade-check    CORPUS_DIR=$CORPUS_DIR  # same, as JSON (for boot scripts / CI)
make upgrade-dry-run  CORPUS_DIR=$CORPUS_DIR  # preview the plan, write nothing
make upgrade-corpus   CORPUS_DIR=$CORPUS_DIR  # apply (non-interactive, --yes)
                                              # add SNAPSHOT_DIR=/persistent/path on a deployed corpus
make upgrade-verify   CORPUS_DIR=$CORPUS_DIR  # verify applied migrations
```

Or drive the CLI directly (interactive confirmation, JSON, version ceiling):

```bash
python -m podcast_scraper.cli upgrade status  --corpus-dir $CORPUS_DIR [--json]
python -m podcast_scraper.cli upgrade list    --corpus-dir $CORPUS_DIR
python -m podcast_scraper.cli upgrade run     --corpus-dir $CORPUS_DIR [--dry-run] [--yes] [--to 2.7.0] [--snapshot-dir PATH]
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
2. **Register it.** Add to `upgrade/registry.py`. The runner sorts by id, so
   pick `mNNNN` = the next integer. An installed extension registers its own
   through `Extension.corpus_migrations` instead ([ADR-174](../adr/ADR-174-extension-corpus-migrations.md)).
   The numbering is one sequence across every package, so take the next number
   no installed package uses. The registry refuses a number used twice and the
   retired `0032`.
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
   the installed GIL extension's read-time shims with a
   matching transform so an unmigrated corpus still reads correctly. Cross-
   reference the framework migration and the shim in each other's docstrings.
7. **Update docs.** Three files:
   - **This guide** — add a bullet under "Registered migrations" above.
   - `docs/api/MIGRATION_GUIDE.md` — add a section to the current version
     block (or start a new version block if you're crossing a release
     boundary) describing the on-disk change from the caller's perspective.
   - Any RFC / ADR you're implementing — link the migration id
     (`m0004_<name>`) back from the RFC.
8. **Resync the ad cut when a migration re-renders transcript text.** The ad cut (the
   `<base>.adfree.*` files) is owned by an installed extension's ad-cleaning stage, and it is
   computed from the LABELLED screenplay: `Name:` markers, same-speaker turns merging and the
   host-read cross-promo detector all read the labels. A migration that renames voices and
   re-renders the raw transcript while carrying the stored ad map can leave the extension's ad-cut
   check failing where the new labels change the cut. The pipeline rebuilds the cut after a
   relabel (`_produce_transcript_sidecars` in `workflow/episode_processor.py` calls the
   `adfree_base` hook); a migration that writes such episodes must do the same, or end with the
   extension's resync step for them, and its clone measurement runs the full `upgrade verify` to
   show it did. A public-only install has no ad cut, so this step applies only where an extension
   is installed.
9. **Consider a read-check.** If the migration is high-stakes (renames a
   load-bearing field), extend `src/podcast_scraper/corpus_version.py`'s
   `assess_corpus_version_compat` with a warning when the corpus predates
   the change — so operators who skip the framework path still get told.

The operator restore path (not in this repository) runs
`podcast upgrade run --yes` post-extract (#1176), so the new step lands
automatically on every restore once merged. A corpus restored or copied by
hand needs `make upgrade-corpus CORPUS_DIR=<path>` run by hand — not enforced.

## Prod-state fixture maintenance (after every prod deploy)

The CI net includes a fixture-pinned test
(`tests/integration/upgrade/test_pinned_fixture_upgrade.py`) that proves the
gap between the last prod deploy and current `main` will apply cleanly.
The pin is at `tests/fixtures/upgrade/corpus_at_last_prod_release/` and its
identity comes from `config/last_deployed_prod_version.json`. **We only test
prod → HEAD** — the gap that matters for the next deploy. Older-than-prod
state is not tested; that gap is history.

**Steps 1–3 below are automated** by the `Bump prod-state marker + pinned fixture`
step of the deployment's deploy workflow (not in this repository), which opens a PR
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

This bump is one commit per prod deploy. It does not need to happen between
deploys. See the fixture's own `README.md` for the detailed contract.

## Registered migrations

The ids this repository registers (`upgrade/registry.py`, files under
`upgrade/migrations/`), in apply order. Gaps in the numbering belong to an installed
extension's migrations (`Extension.corpus_migrations`, ADR-174) or are retired (`0032`).

- `0001_faiss_to_lance` — **historical, now a recorded no-op.** It once migrated an
  existing FAISS index into the two-tier LanceDB layout; FAISS was retired in #995
  (ADR-099), so there is nothing to migrate. Kept registered so the version chain and
  ledger stay intact for corpora that already recorded it.
- `0002_two_tier_native_reindex` — build the two-tier LanceDB index **natively** from
  corpus artifacts when none exists. No-op when an index is already present. With 0001
  reduced to a no-op, this step is what actually (re)builds the index for a pre-2.7
  corpus.
- `0004_insight_type_reindex` — rebuild an index whose schema predates the
  `insight_type` column (`LANCE_SCHEMA_VERSION` 3), natively from corpus artifacts.
- `0006_kg_v2_typed_entities` — bring `.kg.json` envelopes to RFC-097 v2.0: typed
  `Person` / `Organization` nodes and normalised entity ids. Idempotent; a v2.0 artifact is
  skipped without a write. Wraps `migrate_kg_document_v2`
  (`podcast_scraper/migrations/kg_document_migrations.py`).
- `0008_object_entity_kind` — stamp `.kg.json` envelopes as schema 2.1 (the `Object` entity
  kind). It cannot reclassify nodes that were misclassified at extraction time.
- `0011_shared_artwork_store` — one artwork store per corpus at the corpus root, instead
  of a copy under each run directory.
- `0013_artwork_thumbnails` — every stored cover gets its list thumbnail on disk
  (derived, content-addressed, regenerable).
- `0021_backfill_feed_language` — fetch each show's declared RSS `<language>` and write it
  onto the show and its episodes (#2173). The one migration that touches the network.
- `0026_artwork_medium` — every stored cover gets its player-sized (up to 1024 px) copy.
- `0027_missing_covers_stored` — store covers the corpus only pointed at by URL, and record
  the store path on those episodes (backed up and receipted).
- `0028_transcript_chunk_times` — give transcript passages already in the search index the
  time they were said, in place, without re-embedding.

The GI v3 typed-mentions migration (`0003`) and the speaker, naming and ad-cut repairs are not
registered here: they ship with the installed extension that owns those artifacts.

**Not a migration:** the cross-episode entity canonical map (#852) is computed *live*
at graph-build (`search/corpus_graph.py` → `build_entity_id_map`), not persisted —
there is no artifact to rebuild, so it needs no migration.

## Other migration surfaces in the repo

The `upgrade/` framework above owns **on-disk corpus migrations**. Two other
migration surfaces exist and are easy to conflate:

- **Read-time schema shims** live in
  the installed GIL extension
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

When adding a new step, prefer a framework migration
(`upgrade/migrations/mNNNN_<name>.py`) so an operator can rely on
`make upgrade-corpus` and the ledger; a loose script is only for a genuinely one-shot
data-shape correction that never repeats.

## See also

- [API Migration Guide](../api/MIGRATION_GUIDE.md) — API surface changes per version.
- Every operator restore path pairs with `upgrade-check` — see
  [#1176](https://github.com/chipi/podcast_scraper/issues/1176). The backup and restore
  tooling and its runbooks are not in this repository.
