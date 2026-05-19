# ADR-095: Three-tier viewer test pyramid + production-shaped fixtures

- **Status**: Accepted
- **Date**: 2026-05-16
- **Authors**: Marko Dragoljevic
- **Related RFCs**: [RFC-086](../rfc/RFC-086-viewer-test-pyramid-and-production-shaped-fixtures.md) (the origin of this decision)
- **Related ADRs**:
  - [ADR-094](ADR-094-graph-handoff-orchestrator-fsm.md) (graph handoff
    orchestrator FSM — what the matrix tests)
  - [ADR-066](ADR-066-playwright-for-ui-e2e-testing.md) (Playwright for
    UI E2E)
- **Related Documents**:
  - [VIEWER_GRAPH_SPEC.md §"Matrix assertion layers"](../architecture/VIEWER_GRAPH_SPEC.md)
  - `web/gi-kg-viewer/e2e/HANDOFF_MATRIX.md` (Tier 1 coverage contract)

## Context & Problem Statement

The graph handoff matrix (41 rows, 52 tests) catches every contract
regression in ~30 seconds. But real-corpus validation against a 333-node
graph surfaced three production bugs that the matrix mocks cannot reach:

- **V5**: hot-state Library→Library handoff hits the 15s stuck-timeout in
  production. Passes in mocks because the 1-episode fixture's layout
  finishes in <100ms before the second handoff fires.
- **V2**: Digest topic pill sets `subject.kind=topic` but no cy node
  auto-selects in a 333-node graph (the resolver doesn't find a matching
  prefix variant); rail looks right, graph doesn't.
- **V3**: Search "Show on graph" can't be exercised without a real vector
  index — the mock's contrived `doc_type` doesn't match the production
  filter set.

The shared root cause: **the mock fixture collapses scale and timing**
to zero. A 1-episode, 1-cluster fixture with sub-100ms layout cannot
exercise the supersession path that breaks at 200+ nodes with 1–3s layout
time. The matrix is necessary but not sufficient.

## Decision

Adopt a **three-tier test pyramid** for the viewer:

| Tier | Fixture | Speed | When it runs | What it catches |
| --- | --- | --- | --- | --- |
| **1 — Fast matrix** | Mocked (1 ep, 1 cluster) | ~30 s | Every PR | Contract regressions, FSM transitions, envelope shape |
| **2 — Production-shaped** | Checked-in fixture (25 eps, 5 feeds, 5 clusters, GI+KG, 200–400 cy nodes) | ~3–5 min | `make ci-ui-full` | Scale + timing-sensitive bugs, KG second-wave, compound mount ordering |
| **3 — Real-backend validation** | Operator-supplied corpus + `make serve` | ~5–10 min | `make ci-ui-validation` + scheduled cron | Drift between Tier-2 fixture and reality |

All three tiers share the same `assertHandoffApplied` helper and the
same 6-point user-visible contract (selection + camera zoom + camera
pan-center + subject store + invariant + no console errors). Tier 2
mirrors Tier 1's 41 rows row-for-row.

**Institutional rule (locked):** every bug surfaced by Tier 3 (real-corpus
validation) lands a Tier 2 matrix row that reproduces it before the fix
PR merges. Bugs become structurally non-regressable.

## Rationale

- **Tier separation preserves the fast loop.** Bumping the Tier 1 fixture
  to 25 episodes would slow PR validation from 30 s to ~3 min. The fast
  tier MUST stay fast; the slow tier handles scale.
- **Shared assertion helper means shared contract evolution.** When the
  6-point contract grows (e.g. adding L7 selection-dim), it lands once and
  all three tiers benefit. No drift between layers.
- **Production-shaped fixture is reproducible.** A deterministic
  `scripts/build_production_shaped_fixture.py` regenerates the fixture
  from a real corpus snapshot. PR diffs stay small when behaviour evolves.
- **Tier 3 catches what Tier 2 can't.** A checked-in fixture is a
  point-in-time snapshot; real corpus behaviour drifts as the pipeline
  evolves (new topic-cluster shapes, new vector index, new metadata
  fields). Tier 3 keeps Tier 2's fixture honest.
- **The institutional rule makes regression impossible.** Without it,
  Tier 3 bugs would surface, get fixed, then re-surface 3 months later
  because no one wrote the test. With it, every bug is encoded as a
  permanent regression barrier.

## Alternatives Considered

1. **Bump Tier 1 fixture size to 25 episodes**. Rejected — slows PR
   validation by 6×; existing spec files would all need to update for new
   cy ids; the fast loop is the wrong place to spend that latency.

2. **Snapshot-test the matrix against `make serve`**. Rejected — ties CI
   to backend bootstrap (~10s warmup) and stateful corpus loading. The
   fast tier MUST run without out-of-process dependency.

3. **Property-based testing of FSM transitions**. Rejected for now — 75
   unit tests already cover the transition table at the right level; what
   we're missing is *scale* and *timing*, not transition coverage.

4. **Visual regression with screenshot diffs**. Rejected for now —
   brittle to font / theme / animation changes. The camera-pan center
   assertion (L3) already catches "user-visible wrong-place" without the
   maintenance burden.

5. **Stop at Tier 1 + manual checklist for Tier 3**. Rejected — the
   manual checklist is what we have today; it didn't catch V5 until
   accidentally running validation. The institutional rule is only
   enforceable when Tier 2 exists.

## Consequences

- **Positive**:
  - V5-class scale-sensitive bugs surface in CI, not in production.
  - The fast loop stays fast (~30 s); deeper coverage opt-in via
    `ci-ui-full`.
  - Every Tier-3 bug becomes a permanent Tier-2 regression barrier.
  - Future entry surfaces inherit Tier 1 + Tier 2 coverage by recipe;
    the migration recipe in VIEWER_GRAPH_SPEC.md grows a Tier-2 step.
  - Real-backend validation becomes a documented release gate.

- **Negative**:
  - +5 days initial investment to author the fixture, mirror the matrix,
    fix the bugs Tier 2 will surface.
  - Fixture maintenance: corpus schema changes require regeneration via
    the build script. Mitigation: deterministic + idempotent script.
  - Three places assert the same contract. Mitigation: shared
    `assertHandoffApplied` helper across all tiers; no duplication of
    assertion logic.

- **Neutral**:
  - `make ci-ui-full` gets ~3 minutes slower. Matches operator
    expectations for substantial-refactor gates.
  - PR template grows three checkboxes.

## Implementation Notes

- **Fixture**: `web/gi-kg-viewer/e2e/fixtures/production-shaped/` (checked
  in, <5 MB total).
- **Build script**: `scripts/build_production_shaped_fixture.py`
  (deterministic; regenerable from a real corpus snapshot).
- **Tier 2 specs**: `web/gi-kg-viewer/e2e/handoff-production/*.spec.ts`
  mirror Tier 1's 8 spec files row-for-row.
- **Tier 2 mock helper**: `setupProductionShapedMocks(page)` in
  `web/gi-kg-viewer/e2e/handoff-production/_helpers.ts`.
- **Tier 3 spec**: `web/gi-kg-viewer/e2e/validation/real-corpus.spec.ts`
  (already exists; promoted from session artefact to permanent gate).
- **Tier 3 config**: `web/gi-kg-viewer/playwright.validation.config.ts`
  (already exists).
- **Make targets**:
  - `make ci-ui-fast` — unchanged (Tier 1 only)
  - `make ci-ui-full` — adds Tier 2
  - `make ci-ui-validation CORPUS=/path/to/corpus` — Tier 3
- **PR template**: three checkboxes for the three tiers.
- **Pattern**: Test pyramid (Mike Cohn) adapted for UI-heavy applications,
  with scale + timing as the discriminating dimension between Tier 1 and
  Tier 2.

## References

- [RFC-086: Three-tier viewer test pyramid + production-shaped fixtures](../rfc/RFC-086-viewer-test-pyramid-and-production-shaped-fixtures.md)
- [ADR-094: Graph handoff orchestrator FSM](ADR-094-graph-handoff-orchestrator-fsm.md)
- [ADR-066: Playwright for UI E2E testing](ADR-066-playwright-for-ui-e2e-testing.md)
- `web/gi-kg-viewer/e2e/HANDOFF_MATRIX.md` — Tier 1 contract
- [VIEWER_GRAPH_SPEC.md](../architecture/VIEWER_GRAPH_SPEC.md) — operational reference
