# RFC-097 chunk 9 — strict schemas + drop legacy support (plan, 2026-06-22)

Chunk 9 lands in a follow-up PR after PR #1039 (RFC-097 chunks 0–8)
merges. The original RFC §161 bake gate (2–4 weeks of v2 production
operation) is **dropped** per operator 2026-06-22: this project has no
external legacy consumers and all corpora are regenerable, so the gate
adds no value here. The deferral is purely a scoping decision: chunk 9
touches ~15 files with case-by-case judgement calls (delete obsolete
test vs. rewrite to v2), and that work is cleaner as its own PR.

## What chunk 9 does

1. Make KG schema v2.0 strict-only — reject 1.0 / 1.1 / 1.2 in
   `_minimal_validate`.
2. Make GI schema v3.0 strict-only — reject 1.0 / 2.0 in
   `_minimal_validate`.
3. Tighten the JSON schemas:
   - KG `docs/architecture/kg/kg.schema.json` — remove `entity_node`
     definition (replaced by `person_node` + `organization_node` in v2)
   - GI `docs/architecture/gi/gi.schema.json` — make `insight_type`
     + `position_hint` required fields under `Insight.properties`
4. Migrate all legacy-shape fixtures + tests to v2/v3 shape.
5. Write ADR-101 documenting the decision.

## Surface area (the actual blast radius)

### Validator changes (~10 lines)

- `src/podcast_scraper/kg/schema.py:48-50` — flip
  `if sv not in ("1.0", "1.1", "1.2", "2.0")` to `if sv != "2.0"`.
- `src/podcast_scraper/gi/schema.py:52-54` — flip
  `if sv not in ("1.0", "2.0", "3.0")` to `if sv != "3.0"`.

Both validators must keep their behaviour for migration scripts: the
migration code reads input as raw JSON (`json.load`), not via
`validate_artifact`, so the strict gate doesn't block legacy → v2
conversion. Verify this assumption holds by reading
`src/podcast_scraper/migrations/gil_kg_identity_migrations.py` before
making the change.

### JSON schema tightening

- `docs/architecture/kg/kg.schema.json`:
  - Remove the `entity_node` `oneOf` branch (lines ~175-225)
  - Drop `1.0` / `1.1` / `1.2` from the `schema_version.enum`
  - Drop or rewrite the `enum: ["MENTIONS", "RELATED_TO",
    "HAS_EPISODE"]` to remove legacy-only edges if any apply
- `docs/architecture/gi/gi.schema.json`:
  - Drop `1.0` / `2.0` from the `schema_version.enum`
  - Insight: add `insight_type` and `position_hint` to the
    `properties.required` array

### Test files to update (10 files surveyed 2026-06-22)

Mechanical edits — per file, decide for each test case: KEEP (bump
schema_version 1.x → 2.0), DELETE (the legacy-shape feature being
tested no longer exists in v2), or REWRITE (case still applies but
needs v2 shape). Estimated 1.5–2 hours.

- `tests/unit/podcast_scraper/kg/test_kg_schema.py` (335 lines, 9
  test cases — DELETE the v1.1/v1.2-specific cases; KEEP the missing-
  key + bad-version + v2 cases; rewrite the legacy-coexists test as
  "rejects legacy under strict v2")
- `tests/unit/podcast_scraper/gi/test_schema.py` (243 lines, similar
  structure)
- `tests/unit/podcast_scraper/kg/test_corpus.py` — bump fixture
  schema_version
- `tests/unit/podcast_scraper/kg/test_kg_io.py` — bump fixture
- `tests/unit/podcast_scraper/kg/test_kg_contracts.py` — bump fixture
- `tests/unit/podcast_scraper/gi/test_io.py` — bump fixture
- `tests/unit/podcast_scraper/gi/test_explore.py` — bump fixture
- `tests/unit/podcast_scraper/builders/test_bridge_builder.py` —
  bump fixture
- `tests/unit/podcast_scraper/utils/test_corpus_graph_bullet_sync.py`
  — bump fixture
- `tests/unit/podcast_scraper/workflow/test_metadata_generation.py` —
  bump fixture
- `tests/unit/podcast_scraper/workflow/test_run_summary.py` — bump fixture
- `tests/unit/podcast_scraper/test_metrics.py` — bump fixture
- `tests/unit/podcast_scraper/test_corpus_cost_cli.py` — bump fixture
- `tests/unit/gi/test_relational_edges.py` — bump fixture
- `tests/integration/gi/test_gi_explore_uc5_integration.py` — bump fixture

Plus the migration tests, which legitimately use legacy shape as INPUT
(that's the whole point of testing migration). Those should be left
alone — the validator change only affects `validate_artifact`, and the
migration code reads input as raw JSON. Verify:

- `tests/unit/podcast_scraper/migrations/test_gil_kg_identity_migrations.py`
  — confirm the migration test code path does NOT call
  `validate_artifact` on input. If it does, switch to raw `json.load`
  on the input side.

### Fixture files (9 checked-in JSON artifacts)

```text
tests/fixtures/multi-run-corpus/feeds/rss_feeds_invalid_a9e2330b/run_20260103-120000_latest00/metadata/
├── 0001 - Episode_3_of_Fixture_Feed_A_*.kg.json
├── 0002 - Episode_4_of_Fixture_Feed_A_*.{kg,gi}.json
├── 0003 - Episode_5_of_Fixture_Feed_A_*.{kg,gi}.json
├── 0004 - Episode_6_of_Fixture_Feed_A_*.{kg,gi}.json
└── 0005 - Episode_7_of_Fixture_Feed_A_*.{kg,gi}.json
```

These feed `tests/integration/test_multi_run_corpus_fixture.py`.
Migrate in place via chunk 6 scripts. Suggested commands (run from
project root, after the validator change is committed):

```bash
for f in tests/fixtures/multi-run-corpus/feeds/*/run_*/metadata/*.kg.json; do
  .venv/bin/python scripts/migrate_kg_entity_to_person_org.py --in "$f" --out "$f"
done

for f in tests/fixtures/multi-run-corpus/feeds/*/run_*/metadata/*.gi.json; do
  .venv/bin/python scripts/migrate_gi_to_v3.py --in "$f" --out "$f"
  .venv/bin/python scripts/compute_gi_position_hints.py --in "$f" --out "$f"
done

# Verify the new fixtures load via the strict validators
.venv/bin/python -m pytest tests/integration/test_multi_run_corpus_fixture.py -x
```

If the migration scripts don't support in-place writes, write to a
temp file and `mv`. (Worth verifying via `--help` first.)

### ADR-101 outline (to be written)

```text
# ADR-101: Drop legacy KG + GI shape support (RFC-097 chunk 9)

## Status
Accepted — 2026-06-22

## Context
RFC-097 §161 declared a 2-4 week bake gate before dropping legacy
schema support. The gate exists to catch hidden legacy-emit paths
during production operation BEFORE the strict validator removes the
safety net. This project has no external corpus consumers; all data
is regenerable from sources + migration scripts. The gate's
risk-mitigation has no upside here.

## Decision
- KG schema validator (`src/podcast_scraper/kg/schema.py`) accepts
  only `schema_version: "2.0"`. Reads of any 1.0/1.1/1.2 artifact
  raise ValueError.
- GI schema validator (`src/podcast_scraper/gi/schema.py`) accepts
  only `schema_version: "3.0"`. Reads of any 1.0/2.0 artifact raise.
- JSON Schemas at `docs/architecture/{kg,gi}/*.schema.json` updated
  to declare only the v2/v3 shapes.
- All checked-in fixtures migrated to v2/v3 in-place via chunk 6
  migration scripts.
- Migration scripts retain ability to READ legacy as raw JSON (their
  whole purpose); the strict validator only affects `validate_artifact`
  call sites.

## Consequences
- Future v2 emit-side regressions caught immediately by validator
- One canonical shape on disk; no dual-shape branches in viewer / KG
  / GI consumers
- A consumer outside this repo expecting legacy shape would break —
  acceptable here because no such consumer exists
- The bake gate was the right default for RFC-097 generic guidance;
  this ADR records the project-specific waiver

## Migration
For any cached corpus on disk (`.test_outputs/manual/...`, ad-hoc
working dirs), operator runs:
```
.venv/bin/python scripts/migrate_kg_entity_to_person_org.py --in <kg.json>
.venv/bin/python scripts/migrate_gi_to_v3.py --in <gi.json>
.venv/bin/python scripts/compute_gi_position_hints.py --in <gi.json>
```
or re-runs the v2 pipeline against the source transcripts.
```

## Verification

After chunk 9 lands, the following must all pass:

```bash
make ci-fast              # full test sweep
make docs                 # mkdocs strict
# Confirm strict validator throws on a synthetic legacy artifact:
.venv/bin/python -c "
from podcast_scraper.kg.schema import validate_artifact
try:
    validate_artifact({'schema_version': '1.2', 'episode_id': 'x',
                       'extraction': {'model_version': 'x',
                                      'extracted_at': 'x',
                                      'transcript_ref': 'x'},
                       'nodes': [], 'edges': []})
    print('LEAKED — strict validator failed to reject 1.2'); exit(1)
except ValueError as e:
    print('OK — strict validator rejected 1.2:', e)
"
```

## Test-prod corpus migration (operator-run)

After chunk 9 merges, migrate the test-prod corpus once:

```bash
# 99-episode corpus at .test_outputs/manual/prod-v2/corpus
for f in .test_outputs/manual/prod-v2/corpus/feeds/*/run_*/metadata/*.kg.json; do
  .venv/bin/python scripts/migrate_kg_entity_to_person_org.py --in "$f" --out "$f"
done
for f in .test_outputs/manual/prod-v2/corpus/feeds/*/run_*/metadata/*.gi.json; do
  .venv/bin/python scripts/migrate_gi_to_v3.py --in "$f" --out "$f"
  .venv/bin/python scripts/compute_gi_position_hints.py --in "$f" --out "$f"
done
```

If anything throws during the migration, regenerate from source by
re-running the pipeline on the affected episodes.

## Why not in PR #1039

The PR for chunks 1–8 was already at 15 commits + 9-candidate
scoreboard re-baseline + extensive doc consolidation. Chunk 9 is
mechanically distinct (test/fixture migration, JSON schema tightening,
ADR write-up) and benefits from being its own bisectable PR. The
operator's "do it now" direction was accepted; the only reason chunk 9
is its own PR instead of being squashed into #1039 is the case-by-case
test rewrite work needs its own diff review.
