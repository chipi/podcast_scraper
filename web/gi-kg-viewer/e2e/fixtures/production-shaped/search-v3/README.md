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

## `mocks.json` — 5 scenarios

Keyed by scenario name; each entry carries a request (endpoint + params) and a
response (the JSON the server would return). Playwright specs read these
directly and set them up via `page.route(...)` handlers.

| Scenario | Exercises |
| --- | --- |
| `compound-lift` | RFC-072 KL1 — transcript hit with `lifted` block (Insight + speaker + topic + quote timestamps) |
| `enriched-answer` | RFC-088 chunk 5 — enriched-answer hero (`enriched.answer` + `sources[]` with `grounded: true`); source→hit tie-in via `doc_id` |
| `operator-cluster` | RFC-107 §6.1 — `operator_result.clusters[]`, one cluster with 6 members, one with 5 (both ≥5 per S0 acceptance) |
| `operator-consensus` | RFC-107 §6.5 + ADR-108 — `topic_consensus` output tuple `(topic_id, person_a_id, person_b_id, insight_a_id, insight_b_id, contradiction_score)`; 2 cross-speaker corroboration pairs |
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
- **Shape refresh needed (post-S4/S5 landing, 2026-07-21):** the original
  `mocks.json` was written against the RFC-107 fixture spec, which planned
  `enriched.answer` + `enriched.sources[]` at the top level and
  `operator_result.clusters[]` / `operator_result.consensus[]` under an
  `operator_result` wrapper. The **shipped** shape is different: S4b puts
  `clusters[]` and `consensus_pairs[]` directly on the response object, and
  S5 decorates each hit with `metadata.query_enrichments.related_topics[]`
  instead of returning a synthesized answer (the shipped
  QueryEnricher chain doesn't ship a text answer yet). Until the fixture
  is regenerated:
  - Tier-2 specs under `web/gi-kg-viewer/e2e/search-production/*.spec.ts`
    mock inline against the **shipped** shape (see `workspace.spec.ts` for
    the reference).
  - `compound-lift` and `temporal-intent` scenarios in `mocks.json` DO still
    match the shipped shape and can be consumed as-is.
  - `enriched-answer`, `operator-cluster`, and `operator-consensus` need a
    regeneration pass before consumers can trust them.
