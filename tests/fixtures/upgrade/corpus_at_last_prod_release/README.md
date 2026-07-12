# Corpus at last prod release — pin for CI net C (#1176)

This fixture represents **the on-disk state of the production corpus after the
most recent successful prod deploy**. It is NOT arbitrary legacy state. Its
whole purpose is to prove that any migration currently sitting on `main` (that
has not yet been deployed) can move a real prod-shape corpus forward safely on
the next deploy.

The corpus code version + the list of migrations already-applied are pinned in
`config/last_deployed_prod_version.json`. This fixture's
`corpus_manifest.json.produced_by.code_version` and `upgrade_ledger.json`
must agree with that marker.

## What this fixture tests

- **Prod-deploy gap** — every migration in the current registry whose id is NOT
  in the marker's `applied_migrations` list represents work that the next prod
  deploy will do. The C tests apply exactly those migrations to this fixture
  and assert they succeed.
- **Nothing older than prod** — we intentionally do NOT test migrations that
  have already run in prod. That gap is history.
- **Framework consistency at prod state** — running the framework against this
  fixture must be a clean no-op when the registry equals the marker (i.e.
  nothing new since last deploy).

## Maintenance rule (after every prod deploy)

1. Update `config/last_deployed_prod_version.json`:
   - `code_version` — the version tag / pyproject.toml version that just
     deployed.
   - `sha` — the git SHA of the deploy.
   - `deployed_at` — the deploy date (YYYY-MM-DD, UTC).
   - `applied_migrations` — the list of migration ids in
     `src/podcast_scraper/upgrade/registry.py` at that SHA. Copy them
     verbatim so lexicographic order matches.
2. Update this fixture's `upgrade_ledger.json` to record every migration in
   `applied_migrations`, with `to_version` matching each migration's declared
   value.
3. Update this fixture's `corpus_manifest.json.produced_by.code_version` to
   match the marker's `code_version`.
4. Update any on-disk artifact shapes in this fixture that changed as part of
   the deployed migrations (e.g. after m0003 deployed, `.gi.json` here uses
   `schema_version: "3.0"` + typed `MENTIONS_PERSON` / `MENTIONS_ORG` edges).

The maintenance is one commit per prod deploy. It does not need to happen
between deploys.

## Not part of the fixture

- **No `search/lance_index/`** — the framework is ledger-driven, so recording
  m0002 in the ledger is enough to signal "already applied." Not carrying the
  binary index keeps the fixture cheap and portable.
- **No transcripts beyond the minimum** — the fixture is not a corpus quality
  test; it is a framework-behaviour pin. One episode is enough.
