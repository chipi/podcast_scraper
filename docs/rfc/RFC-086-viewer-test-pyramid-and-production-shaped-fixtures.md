# RFC-086: Viewer three-tier test pyramid + production-shaped fixtures

- **Status**: Completed
- **Authors**: Marko Dragoljevic
- **Stakeholders**: Viewer maintainers, anyone touching graph navigation,
  release gate operators
- **Related ADRs**:
  - `docs/adr/ADR-094-graph-handoff-orchestrator-fsm.md` (graph handoff
    orchestrator — what the matrix tests)
  - `docs/adr/ADR-066-playwright-for-ui-e2e-testing.md` (Playwright is the
    chosen UI E2E framework)
  - `docs/adr/ADR-095-viewer-test-pyramid.md` (locks the three-tier
    decision; written as a companion to this RFC)
- **Related RFCs**:
  - `docs/rfc/RFC-085-graph-handoff-orchestrator-retrospective.md` (the
    work that surfaced the gap this RFC closes)
  - `docs/rfc/RFC-062-gi-kg-viewer-v2.md` (the viewer itself)
- **Related Documents**:
  - `docs/architecture/VIEWER_GRAPH_SPEC.md` §"Matrix assertion layers (L0–L6)"
  - `web/gi-kg-viewer/e2e/HANDOFF_MATRIX.md` (the 41-row coverage contract)
  - `web/gi-kg-viewer/e2e/validation/real-corpus.spec.ts` (the validation
    walk that surfaced V5)

## Abstract

The graph handoff matrix is 41 rows, 52 tests, all green. But during
real-corpus validation against a 333-node graph, **three real bugs surfaced
that the matrix mocks cannot catch**: a hot-state Library→Library
handoff hits the 15s stuck-timeout (the FSM regression we built the matrix
to prevent), a Digest topic pill leaves the subject set without an
auto-selected cy node, and Search "Show on graph" can't be exercised
without a real vector index. The matrix passes because its fixture has 1
episode, 1 topic, sub-100ms layouts; production has 100+ episodes, multiple
KG second-wave loads, 1–3s layouts.

This RFC proposes a **three-tier test pyramid** plus a checked-in
**production-shaped fixture**, so scale-sensitive and timing-sensitive
bugs surface in CI rather than only in production. The institutional rule
this RFC locks in: **every real-corpus bug must land with a matrix row
that reproduces it, so the bug becomes structurally impossible to
regress**.

**Architecture Alignment:** Extends the existing matrix-as-coverage-contract
pattern from ADR-094 with a second tier sized to expose timing races and
scale-dependent state transitions that the fast mock fixture collapses.

## Problem Statement

The matrix established a six-point user-visible contract per handoff
(selection + camera zoom + camera pan-center + subject store + invariant +
no console errors). It catches every contract regression the assertion
layer covers, in ~30 seconds.

It cannot catch bugs that depend on:

1. **Graph scale** — a 333-node graph runs layout in 1–3s; a 6-node mock
   runs it in <100ms. Bugs that race against layoutstop (V5 stuck-timeout)
   only fire on the slow layout.
2. **KG second-wave loads** — production paths often load GI first, then
   merge KG ~1–2s later. The merge fires another `redrawing_full` →
   `applying` cycle that the mock skips entirely.
3. **Real subject → cy id resolution** — the mock's topic node is
   `g:topic:ci-policy`. Production's are `topic:ci-policy` (no prefix),
   resolved via `resolveCyNodeId`'s 10-variant candidate list. The mock
   shortcuts this resolution.
4. **Real search hits** — search results in production carry `doc_type:
   'insight' | 'quote' | 'kg_topic' | 'kg_entity'` from FAISS; the mock
   fixture uses contrived metadata that may not match what `Show on graph`
   filters on (V3 surfaced this).
5. **Compound mount ordering** — topic-cluster compound parents
   (`tc:*`) need to mount before their member topics resolve; the mock has
   a single cluster with one member, so ordering doesn't matter.

The canonical illustration is **V5**: hot-state Library→Library
supersession passes in mocks (because the second handoff completes before
the first one's layoutstop fires), fails in real corpus (because the
second handoff fires while the first's redraw is still running, and the
supersession path doesn't cancel cleanly).

**Use Cases:**

1. **Pre-PR confidence**: a developer changes `loadEpisodeSliceForTerritoryStrip`
   and wants to know within minutes whether the change breaks the
   second-Library-click path on a real-sized graph.
2. **Regression capture**: when validation surfaces a bug like V5, the fix
   PR adds a matrix row that reproduces it under the production-shaped
   fixture. The bug becomes mechanically impossible to regress.
3. **Release-gate confidence**: before pushing to main, the operator runs a
   real-corpus validation walk and knows the handoff matrix has been
   exercised against scale + timing conditions that match production.

## Goals

1. **Mechanically prevent V5-class bugs**: scale + timing-sensitive
   regressions must surface in CI, not in production.
2. **Make every real-corpus bug a permanent matrix row**: when a bug is
   found by the validation walk, the fix lands a matrix row that
   reproduces it. The bug is structurally impossible to regress.
3. **Keep the fast loop fast**: the existing 30s matrix run stays as the
   PR-blocking gate. The slower tier runs on demand or in `ci-ui-full`.
4. **Make scale-shaped fixtures cheap to author**: a single helper
   generates a representative fixture from a real corpus snapshot.
5. **Surface drift between fixtures and reality**: if production behaviour
   diverges from the fixture, the validation walk catches it before push.

## Constraints & Assumptions

**Constraints:**

- The fast matrix tier (~30 s) must remain unchanged and stay green on
  every PR. This RFC adds, never subtracts.
- The production-shaped fixture must be checked into the repo (no
  external-network dependency).
- Total fixture size must stay under ~5 MB (for repo hygiene).
- The slow tier must finish in under 5 minutes locally; under 10 minutes in
  CI.
- The real-backend validation must work against `make serve` without extra
  setup beyond pointing at an on-disk corpus.

**Assumptions:**

- An operator-supplied real-corpus snapshot (multi-feed, multi-episode,
  GI + KG populated) is representative of production complexity for the
  test surfaces that matter.
- The 41 matrix rows are the right coverage contract; the new tier mirrors
  them rather than introducing new rows.
- Future surfaces (e.g. F1.6-style new entry points) will land matrix
  rows in BOTH tiers as part of the migration recipe.

## Proposal — three-tier test pyramid

```text
TIER 1 — Fast matrix        ~30 s    Every PR (blocking)
  Fixture:   web/gi-kg-viewer/e2e/handoff/_handoff-helpers.ts
             1 episode, 1 topic cluster, trivial GI fixture
  Catches:   contract regressions, FSM transition-table breaks,
             envelope-shape drift
  Runs in:   make ci-ui-fast

TIER 2 — Production-shaped matrix   ~3–5 min   ci-ui-full only
  Fixture:   web/gi-kg-viewer/e2e/fixtures/production-shaped/
             25 episodes × 5 feeds, 5 topic clusters (3–6 members each),
             GI + KG both populated, 200–400 cy nodes after merge,
             layout times in the 1–3 s range
  Catches:   scale-sensitive bugs, timing races, KG-second-wave,
             compound-mount ordering, subject↔cy resolution drift
  Runs in:   make ci-ui-full
  Spec:      web/gi-kg-viewer/e2e/handoff-production/*.spec.ts

TIER 3 — Real-backend validation    ~5–10 min  Manual + weekly cron
  Fixture:   any on-disk corpus the operator points at
  Catches:   drift between Tier-2 fixture and the real backend
             (vector index, real LLM-generated topic clusters, etc.)
  Runs in:   make ci-ui-validation  +  scheduled GHA against a
             checked-in corpus snapshot
  Spec:      web/gi-kg-viewer/e2e/validation/real-corpus.spec.ts
```

### What Tier 2 looks like — the production-shaped fixture

A checked-in static fixture under `web/gi-kg-viewer/e2e/fixtures/production-shaped/`:

```text
production-shaped/
├── manifest.json                     // index of all files + counts
├── corpus/
│   ├── episodes.json                 // 25 entries across 5 feeds
│   ├── digest.json                   // 3 recent rows, 4 topic bands
│   ├── feeds.json                    // 5 feeds with metadata
│   ├── topic-clusters.json           // 5 clusters (3–6 members)
│   ├── stats.json                    // dashboard
│   ├── coverage.json
│   ├── persons-top.json
│   ├── runs-summary.json
│   └── index-stats.json
├── artifacts/
│   ├── ep-01.gi.json                 // realistic GI per episode
│   ├── ep-01.kg.json                 // realistic KG per episode
│   ├── ... (×25)
└── search/
    └── results-by-query.json         // pre-recorded {query: hits[]}
```

**Sizing rationale:**

- 25 episodes × 5 feeds: enough to exercise multi-feed topic clusters and
  the digest topic-band view. 5 episodes per feed gives non-trivial digest
  rows without exploding fixture size.
- 5 topic clusters with 3–6 members each: triggers compound mount + member
  resolution for Tier-2 H1.7 (NodeDetail Load) at real complexity.
- GI + KG both populated per episode: triggers the KG second-wave merge
  path that hits the `redrawing_full → applying` transition twice per
  episode.
- 200–400 cy nodes after merge: the threshold where layout time becomes
  measurable and the V5-class timing race becomes reproducible.

**Authoring approach:**

A script `scripts/build_production_shaped_fixture.py` takes an
operator-supplied real corpus path and:

1. Picks 5 episodes per feed (deterministic — first 5 by publish_date).
2. Copies their `*.metadata.json`, `*.gi.json`, `*.kg.json` into the
   fixture directory.
3. Aggregates `digest.json`, `episodes.json`, `feeds.json`,
   `topic-clusters.json` to the fixture's reduced episode set.
4. Records the corpus stats / coverage / persons endpoints' current
   responses.
5. Optionally records search results for a fixed query list.

Re-running the script regenerates the fixture from a fresher real corpus
when behaviour drifts. Idempotent + deterministic so PR diffs are minimal.

### What Tier 2 tests look like

Mirror the 41 fast-matrix rows under `e2e/handoff-production/`:

- `cold-start.spec.ts`, `hot-state.spec.ts`, `repeat-click.spec.ts`,
  `cross-entry.spec.ts`, `concurrency.spec.ts`, `failure.spec.ts`,
  `lifecycle.spec.ts`, `filters.spec.ts`
- Same `assertHandoffApplied` helper
- Same 6-point assertion contract (L0+L1+L2+L3+L5+L6 for UI-driven rows)
- New helper `setupProductionShapedMocks(page)` replaces
  `setupHandoffMatrixMocks(page)`
- Expected: V5-class bugs surface as failures during initial migration —
  fix them or mark `test.fail()` with a tracking issue.

### What Tier 3 looks like

`web/gi-kg-viewer/e2e/validation/real-corpus.spec.ts` (already exists,
from the V1–V5 walk) becomes a permanent fixture. Promoted to:

- `playwright.validation.config.ts` (already exists) targets
  `localhost:5173` (i.e. `make serve`) instead of its own webServer.
- `make ci-ui-validation` target runs it against an operator-supplied
  corpus path (`make ci-ui-validation CORPUS=/path/to/my-corpus`).
- A scheduled GHA workflow runs weekly against a checked-in corpus
  snapshot (a "golden" corpus, separate from the fixture above).
  Regression auto-files an issue.

## The institutional rule

**Every bug surfaced by Tier 3 (real-corpus validation) lands a Tier 2
matrix row that reproduces it.** No exceptions. Fix PRs that don't add a
row are not mergeable. This makes bugs structurally non-regressable.

The PR template adds three checkboxes:

```text
- [ ] Tier 1 fast matrix green (`make ci-ui-fast`)
- [ ] Tier 2 production-shaped matrix green (`make ci-ui-full` or `make ci-ui-prod-matrix`)
- [ ] Tier 3 validation walk: ran against {corpus_name} on {date}
```

## Alternatives Considered

### A. Increase the fast-matrix fixture size

Bump the mock fixture from 1 to 25 episodes inside `setupHandoffMatrixMocks`.
**Rejected:** the existing mock-fixture spec files would all need to update
because their assertions reference specific cy ids. Also slows the fast
loop from 30s to ~3 minutes — wrong end of the speed/coverage tradeoff.
Separating tiers preserves the fast loop.

### B. Snapshot-test the matrix against a real running backend

Use `make serve` as the backend for every matrix test. **Rejected:** ties
CI to a backend bootstrap that's slow (~10s warmup) and stateful (corpus
loading). The fast tier MUST run without any out-of-process dependency.

### C. Property-based testing of FSM transitions

Use fast-check or similar to generate envelope sequences randomly and
assert FSM invariants. **Rejected (for now):** the 75 unit tests cover the
transition table at the right level; what we're missing is *scale* and
*timing*, not transition coverage.

### D. Visual regression testing with screenshot diffs

Take screenshots after every matrix row and diff against a baseline.
**Rejected (for now):** brittle to font / theme / animation timing
changes. Camera-pan center assertion already catches the "user-visible
wrong-place" bug class without the maintenance burden.

## Consequences

- **Positive:**
  - V5-class scale-sensitive bugs surface in CI, not in production.
  - The fast loop stays fast (~30s); slower coverage opt-in.
  - The institutional rule makes every real bug a permanent regression
    barrier.
  - Future entry surfaces inherit Tier 1 + Tier 2 coverage by recipe.
  - Real-backend validation becomes a documented release gate, not a
    one-off "I clicked around for 5 minutes."

- **Negative:**
  - +5 days initial investment to author the fixture, mirror the matrix,
    fix V2/V3/V5.
  - Fixture maintenance: when the corpus schema evolves, the fixture must
    be regenerated. Mitigation: the build script is deterministic.
  - The Tier 2 + Tier 3 mirror the assertion logic of Tier 1; if the
    contract changes, three places update. Mitigation: shared
    `assertHandoffApplied` helper across all tiers.

- **Neutral:**
  - `ci-ui-full` gets slower by ~3 minutes. Already an opt-in gate; matches
    operator expectations for substantial refactors.

## Implementation Plan

### Phase 1 — Production-shaped fixture (~1 day)

Build `scripts/build_production_shaped_fixture.py`. Extract a deterministic
slice from an operator-supplied real-corpus snapshot into
`web/gi-kg-viewer/e2e/fixtures/production-shaped/`. Verify total size <5 MB.
Verify all referenced files exist. Commit the fixture + the script.

### Phase 2 — Production-shaped matrix (~1.5 days)

Author `web/gi-kg-viewer/e2e/handoff-production/*.spec.ts` mirroring the
41 fast-matrix rows. Author `setupProductionShapedMocks(page)` helper.
Mark currently-failing rows `test.fail()` with tracking refs. Wire into
`make ci-ui-full`.

### Phase 3 — Fix V2/V3/V5 and convert tracking rows to green (~2 days)

For each row marked `test.fail()` in Phase 2:

- Diagnose root cause in `GraphCanvas.vue` / FSM / store
- Land fix
- Flip row to `test()`

Headline targets: V5 hot-state stuck-timeout, V2 subject-without-selection,
V3 search index integration.

### Phase 4 — Tier 3 promoted to permanent gate (~0.5 day)

Move `web/gi-kg-viewer/e2e/validation/` and `playwright.validation.config.ts`
out of "session artefact" status. Add `make ci-ui-validation` target.
Document in `web/gi-kg-viewer/e2e/validation/README.md` how to point at
arbitrary corpora.

### Phase 5 — Institutional muscle (~0.5 day)

- Author ADR-095 (the locked decision) — separate companion file to this
  RFC.
- Memory rule: "real-corpus bug → matrix row reproducing it before fix
  merges."
- Update PR template with the three checkboxes.
- (Optional, defer to follow-up): scheduled GHA cron + auto-issue on
  Tier 3 regression.

### Stop-and-ship checkpoints

| After | Ship-able? | Why |
| --- | --- | --- |
| Phase 1 | ❌ | No new coverage yet |
| Phase 2 | ⚠️ | Bugs documented but not fixed; can ship if known regressions acceptable |
| Phase 3 | ✅ | Headline target — all tiers green |
| Phase 4 | ✅ | Permanent gate in place |
| Phase 5 | ✅ | Self-reinforcing loop locked in |

**Total cost**: ~5.5 working days. Minimum-viable to ship: Phases 1+2+3 = ~4.5 days.

## Open Questions

1. **Where does the "golden corpus" for scheduled Tier 3 live?** Checked
   into the repo (size?) or pulled from a dedicated S3 bucket on cron run?
   Deferred to Phase 5 cron implementation; not blocking Phases 1–4.
2. **Should Tier 2 also run on every PR?** Or only `ci-ui-full`?
   Recommendation: only `ci-ui-full` for now; promote to PR-blocking if
   it proves stable and fast enough.
3. **Multi-corpus Tier 3**: should the validation walk run against
   multiple operator-supplied corpora (e.g. a multi-feed run plus a
   single-feed run)? Deferred to Phase 5; start with one corpus and
   expand if drift is observed.

## References

- [RFC-085: Graph handoff orchestrator retrospective](RFC-085-graph-handoff-orchestrator-retrospective.md) — the work that surfaced this gap
- [ADR-094: Graph handoff orchestrator FSM](../adr/ADR-094-graph-handoff-orchestrator-fsm.md) — what the matrix tests
- [ADR-095: Three-tier viewer test pyramid](../adr/ADR-095-viewer-test-pyramid.md) — locked decision (companion to this RFC)
- [VIEWER_GRAPH_SPEC.md §"Matrix assertion layers (L0–L6)"](../architecture/VIEWER_GRAPH_SPEC.md)
- `web/gi-kg-viewer/e2e/HANDOFF_MATRIX.md` — Tier 1 contract
- `web/gi-kg-viewer/e2e/validation/real-corpus.spec.ts` — Tier 3 starting point
