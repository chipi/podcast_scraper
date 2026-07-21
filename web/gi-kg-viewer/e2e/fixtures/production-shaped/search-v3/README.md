# production-shaped/search-v3/ — search-v3-specific mock content

**Owner:** Search v3 arc (epic #1229 · plan `docs/wip/SEARCH-V3-IMPLEMENTATION-PLAN.md`)
**Consumed by:** Tier-2 Playwright specs under `web/gi-kg-viewer/e2e/search-production/*.spec.ts` (specs land in slices S2–S8).
**Sibling of:** `../search/results-by-query.json` (basic per-query hit responses; predates Search v3).

## Why this directory exists

The parent `production-shaped` fixture caches API responses in the shape the
server produced when it was regenerated (last run against a real operator
corpus). Search v3 adds response shapes that either did not exist yet
(`operator_result` from `/api/search?operator=cluster|consensus`, `enriched`
from `?enrich_results=true`) or that the last regeneration did not exercise
(`lifted` compound-block, `query_type` intent). This directory carries those
shapes so Tier-2 specs have something to mock against WITHOUT needing a
regenerated fixture.

## `mocks.json` — 6 scenarios

Keyed by scenario name; each entry carries a request (endpoint + params /
body) and a response (the JSON the server would return). Playwright specs
read these directly and set them up via `page.route(...)` handlers. As of
schema_version 2 (2026-07-21), every scenario matches the **shipped**
`/api/search` and `/api/search/compare` shapes.

| Scenario | Exercises |
| --- | --- |
| `compound-lift` | RFC-072 KL1 — transcript hit with `lifted` block (Insight + speaker + topic + quote timestamps) |
| `enriched-answer` | RFC-088 chunk 5 (Search v3 §S5) — per-hit `metadata.query_enrichments.related_topics[]` (`topic_id`, `topic_label`, `similarity`); no top-level answer text (the shipped QueryEnricher chain does not synthesize one yet) |
| `operator-cluster` | RFC-107 §6.1 (Search v3 §S4b) — top-level `operator: "cluster"` + `clusters[]` in the `SearchClusterGroupModel` shape (`cluster_id`, `cluster_kind`, `label`, `size`, `hit_indices`); results carry the hit rows the indices reference |
| `operator-consensus` | RFC-107 §6.5 + ADR-108 (Search v3 §S4b) — top-level `operator: "consensus"` + `consensus_pairs[]` in the flat `SearchConsensusPairModel` shape (`topic_id`, `topic_label`, `person_a_*`, `person_b_*`, `insight_*_id`, `insight_*_text`, `contradiction_score`, `cosine_similarity`); 2 cross-speaker corroboration pairs |
| `operator-compare` | RFC-107 §S8 (Search v3 §S8) — `POST /api/search/compare` returning `{pack_a, pack_b, judge_summary}`; each pack is one `build_briefing_pack` output (RFC-093) with `subject`, `top_insight_*`, `coverage_summary`, `confidence_p50`, `grounded` fields; judge summary is deterministic (no LLM) and muted when either side reports `grounded=false` |
| `temporal-intent` | RFC-092 taxonomy — `query_type: "temporal_tracking"`; IntentChip in `WorkspaceHeader` renders "Temporal tracking" |

Not covered here (deferred to when the operator regenerates against a real
corpus via `--search-slice`):

- Realistic embedding scores (mock scores are hand-picked to make the sort
  order deterministic).
- `supporting_quotes` block per hit (the parent fixture already has examples;
  Search v3 UX inherits UXS-005 rendering).
- Rail-launch scope pre-fills (specs mock these client-side via the search
  store, not via HTTP).

## Anchoring

Every `episode_id` referenced in the mocks appears in the parent
`manifest.json` so hit-card handoffs to Library / Graph resolve cleanly
against the same fixture. The Odd Lots feed
(`sha256:1cd6aecdd31c0ae82c2c76efdb6327b54e6026924b47852c51ffb406a9f2b19c`)
carries 5 episodes in the fixture — used as the anchor for Cluster (theme
members) and Consensus (cross-speaker pairs) scenarios where we need multiple
same-show episodes.

## Regeneration

Hand-authored today (S0 baseline). To refresh against a real corpus:

```bash
python scripts/build_production_shaped_fixture.py \
    --corpus /abs/path/to/your/real-corpus \
    --api http://localhost:8000 \
    --output web/gi-kg-viewer/e2e/fixtures/production-shaped \
    --search-slice
```

`--search-slice` (added in the S0(e) commit that lands this directory)
appends the search-v3 queries to the fetched-query list AND writes back to
this file. Any manual edits made here are preserved as long as the scenario
key exists — the regenerator merges, not overwrites, per-scenario.

Idempotent + deterministic: same corpus + same pick ⇒ byte-identical output
(the parent fixture's ADR-095 invariant, extended).

## Shape contract sources

- **RFC-107 §2** — `SearchRequest` / `SearchResponse` types (Workspace store shape).
- **RFC-107 §6** — result-set operator table (Cluster / Consensus / Compare / Timeline / OnGraph).
- **RFC-072 §6** — `lifted` block shape (compound-lift).
- **RFC-088 chunk 5** + **RFC-088 §Reimagining note** + **ADR-108** — enriched-answer + `topic_consensus` tuple shape.
- **RFC-092** — intent taxonomy (5 classes).
- **UXS-016** — the primary UX doc; §Header, §Results, §Operators reference these mocks.

## Follow-ups (not this commit)

- `--search-slice` build-script mode currently appends to
  `search/results-by-query.json` (basic queries); merging per-scenario into
  `search-v3/mocks.json` is a follow-up when the operator first regenerates.

## Schema history

- **schema_version: 2** (2026-07-21) — refreshed post-S4/S5/S8 landing.
  Every scenario now matches the shipped shape: top-level `operator` +
  `clusters[]` / `consensus_pairs[]` for S4b; per-hit
  `metadata.query_enrichments.related_topics[]` for S5; `{pack_a, pack_b,
  judge_summary}` from `POST /api/search/compare` for S8. `compound-lift`
  and `temporal-intent` were unchanged (already matched). Existing Tier-2
  specs that inline-mock (e.g. `search-production/workspace.spec.ts`) may
  now migrate to consuming this fixture; not a blocking follow-up.
- **schema_version: 1** (2026-07-20) — hand-authored against the
  RFC-107 planned shape (`enriched.answer`, `operator_result.clusters[]`,
  `operator_result.pairs[]`). Did not match shipped code; superseded by
  schema_version 2.
