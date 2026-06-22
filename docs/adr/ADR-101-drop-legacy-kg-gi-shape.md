# ADR-101: Drop legacy KG + GI shape support

- **Status**: Accepted
- **Date**: 2026-06-22
- **Authors**: Marko Dragoljevic, Claude (Opus 4.7)
- **Related RFCs**:
  [RFC-097](../rfc/RFC-097-unified-kg-gi-ontology-v2.md) (the unifying
  ontology); chunk 9 is this ADR.

## Context

RFC-097 chunk 2 shipped permissive KG v2.0 + GI v3.0 schemas that
accepted both legacy and new shape during the bake window. RFC-097
§161 declared a 2–4 week bake gate before flipping the schemas to
strict ("hard rejection of legacy lands in chunk 9, gated on 2–4
weeks of production operation under v2").

The bake gate is the right default for guidance written assuming
multi-tenant production with external corpus consumers. **This
project has neither.** Concretely:

- No external party reads our corpus shape; the only consumers are
  the viewer (`web/gi-kg-viewer/`) and downstream Python pipelines,
  both in-repo.
- All corpora are regenerable from sources: RSS → audio →
  transcripts → KG/GI artifacts. Migration scripts (chunk 6) handle
  one-pass legacy → v2/v3 conversion for any cached corpus on disk.
- The risk the gate mitigates — silent legacy-emit paths surfacing in
  production after strict validation removes the safety net — has no
  load-bearing scenario here. The test-prod corpus
  (`.test_outputs/manual/prod-v2/corpus`, 99 episodes) can be
  re-migrated in minutes.

The operator confirmed 2026-06-22:

> "we only have test prod and can migrate all data, no real users
> who are legacy, how to think here? just do it now and be ready to
> migrate / reprocess all we have"

This ADR records the bake-gate-dropped decision and the strict-shape
contract that lands instead.

## Decision

**Make KG v2.0 and GI v3.0 the only accepted schemas at the validator
layer.** Specifically:

### 1. KG validator strict on `schema_version == "2.0"`

`src/podcast_scraper/kg/schema.py:_minimal_validate` now raises
`ValueError` on any artifact whose `schema_version` is not exactly
`"2.0"`. Legacy 1.0 / 1.1 / 1.2 shape is no longer accepted by
`validate_artifact`.

### 2. GI validator strict on `schema_version == "3.0"`

`src/podcast_scraper/gi/schema.py:_minimal_validate` raises on any
artifact whose `schema_version` is not exactly `"3.0"`. Legacy 1.0 /
2.0 shape is no longer accepted.

### 3. GI pipeline emits 3.0 at the source

`src/podcast_scraper/gi/pipeline.py` `build_artifact()` (two emit
sites) sets `"schema_version": "3.0"`. KG pipeline (chunk 3) already
emits `"2.0"`.

### 4. Migration scripts stay legacy-readable

The migration scripts (`scripts/migrate_kg_entity_to_person_org.py`,
`scripts/migrate_gi_to_v3.py`, `scripts/compute_gi_position_hints.py`)
read input as raw `json.load`, not via `validate_artifact`. The strict
validator therefore does not block legacy-corpus migration. Verified
by `test_migrate_kg_v2_output_passes_strict_schema` +
`test_migrate_gi_v3_output_passes_strict_schema` in
`tests/unit/podcast_scraper/migrations/test_gil_kg_identity_migrations.py`
— they build legacy dicts in memory, run the migration function, then
strict-validate the output.

### 5. Existing checked-in fixtures bumped to v2/v3

All checked-in `.kg.json` / `.gi.json` fixtures under `tests/fixtures/`
were bumped to the new `schema_version` value:

- KG: 35 files migrated from 1.0/1.1/1.2 → 2.0.
- GI: 34 files migrated from 1.0/2.0 → 3.0.

Inline fixtures in unit-test Python files (~22 files) had the
embedded JSON `schema_version` literal bumped to the v2/v3 value.
Two GI test files updated assertion literals (`out["schema_version"]
== "2.0"` → `"3.0"`) to match the new emit value.

### 6. Migration-test cases retain legacy input as test-data

Tests in
`tests/unit/podcast_scraper/migrations/test_gil_kg_identity_migrations.py`
that build legacy dicts in memory and exercise the migration
functions are unchanged. Their purpose is to test legacy → v2/v3
conversion; the strict validator is irrelevant to them because they
never call `validate_artifact` on legacy input.

## Consequences

### Positive

- A single canonical on-disk shape (`schema_version: "2.0"` for KG,
  `"3.0"` for GI) — no dual-shape branches in viewer / search /
  enrichment code going forward.
- Future v2 emit-side regressions caught immediately by the
  validator. Without the strict gate, a silent legacy-emit could go
  unnoticed for months.
- Schema-as-contract: future contributors can't accidentally re-add
  Optional `insight_type` or re-introduce `Entity(kind=*)` —
  validators throw.

### Negative

- A consumer outside this repo that expected legacy shape would
  break. **Acceptable**: no such consumer exists, and this ADR
  records the project-specific waiver of RFC-097's bake gate.
- Anyone with a cached corpus on disk that wasn't migrated needs to
  run the migration scripts before the next pipeline pass:

  ```bash
  for f in <corpus_dir>/feeds/*/run_*/metadata/*.kg.json; do
    .venv/bin/python scripts/migrate_kg_entity_to_person_org.py --in "$f" --out "$f"
  done
  for f in <corpus_dir>/feeds/*/run_*/metadata/*.gi.json; do
    .venv/bin/python scripts/migrate_gi_to_v3.py --in "$f" --out "$f"
    .venv/bin/python scripts/compute_gi_position_hints.py --in "$f" --out "$f"
  done
  ```

  If migration fails on any individual episode, re-emit by re-running
  the GI/KG pipeline on the source transcript.

### Risk

- **Risk**: a code path somewhere still emits legacy shape. Mitigated
  by the strict validator — such a path would fail loudly the first
  time its output is validated, surfacing the regression rather than
  hiding it.
- **Risk**: a test fixture was missed by the bulk-bump. Mitigated by
  running the full unit-test suite at the chunk-9 commit (`make
  ci-fast` worth of coverage); any reader-of-fixture test fails
  loudly with the new validator's error message.

## Why no bake gate here

The RFC-097 §161 bake gate exists to catch hidden legacy-emit paths
during production operation before the strict validator removes the
safety net. The gate's risk-mitigation value is proportional to the
cost of a missed legacy path. In this project:

| Dimension | Project value | Gate value |
| --- | --- | --- |
| External corpus consumers | 0 | gate prevents user-facing breakage |
| Cost of re-migration on regression | minutes | gate avoids the operational cost |
| Visibility into emit paths | full source access | gate substitutes for opacity |
| Bake-window monitoring infrastructure | not built | gate assumes it exists |

All four are weak here. The gate's protection has no upside that
isn't already covered by (a) the strict validator failing loudly
the first time a regression emits legacy, and (b) the migration
scripts being a one-command fix.

Recording the rationale rather than silently dropping the gate so a
future contributor reading RFC-097 §161 sees the project-specific
choice.

## Non-goals

- **Re-fingerprinting historical eval runs.** Old `fingerprint.json`
  artifacts under `data/eval/runs/` are not retroactively edited.
  Their reports document which configs each run was under.
- **A dual-mode validator with a feature flag.** Per
  `feedback_no_lint_check_weakening`: code paths that validators
  exist to enforce don't grow opt-out knobs. If the strict shape
  becomes a problem, this ADR gets superseded — not toggled off
  per-call.
- **Migrating `data/eval/runs/` predictions.** Those are read-only
  scoring artifacts; downstream scoring doesn't validate them
  through `validate_artifact`.

## Verification

After this ADR's PR merges:

1. `make ci-fast` passes (unit tests + lint + docs strict).
2. Synthetic legacy artifact gets rejected:

   ```bash
   .venv/bin/python -c "
   from podcast_scraper.kg.schema import validate_artifact
   try:
       validate_artifact({
           'schema_version': '1.2', 'episode_id': 'x',
           'extraction': {'model_version': 'x', 'extracted_at': 'x', 'transcript_ref': 'x'},
           'nodes': [], 'edges': []
       })
   except ValueError as e:
       print('OK strict validator rejected 1.2:', e)
   "
   ```

3. Migration-script output validates strictly (already exercised by
   `test_migrate_kg_v2_output_passes_strict_schema` +
   `test_migrate_gi_v3_output_passes_strict_schema`).

## References

- [RFC-097](../rfc/RFC-097-unified-kg-gi-ontology-v2.md) — the
  unified KG+GI ontology v2; chunk 9 is the legacy-drop step.
- `docs/wip/RFC097_CHUNK9_PLAN.md` — the implementation plan
  (pre-execution; this ADR records the executed decision).
- `src/podcast_scraper/kg/schema.py` — KG validator (strict).
- `src/podcast_scraper/gi/schema.py` — GI validator (strict).
- `src/podcast_scraper/gi/pipeline.py:900` and `:1495` — GI emit
  sites pinned to `"3.0"`.
- `src/podcast_scraper/migrations/gil_kg_identity_migrations.py` —
  legacy-input migration functions, unchanged behaviour.
