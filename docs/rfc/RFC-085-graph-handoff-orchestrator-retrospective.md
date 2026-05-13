# RFC-085: Graph handoff orchestrator — stabilization retrospective

- **Status**: Completed
- **Authors**: Marko Dragoljevic
- **Stakeholders**: Viewer maintainers, anyone touching graph navigation
- **Related ADRs**:
  - `docs/adr/ADR-094-graph-handoff-orchestrator-fsm.md` (canonical decisions)
  - `docs/adr/ADR-066-playwright-for-ui-e2e-testing.md`
- **Related RFCs**:
  - `docs/rfc/RFC-062-gi-kg-viewer-v2.md`
  - `docs/rfc/RFC-080-graph-visualization-extensions.md`
- **Related UX specs**:
  - `docs/uxs/UXS-004-graph-exploration.md`
- **Related Documents**:
  - `docs/architecture/VIEWER_GRAPH_SPEC.md` (operational reference)
  - `docs/architecture/VIEWER_ASYNC_STABILITY.md` (adjacent async patterns)
  - `web/gi-kg-viewer/e2e/HANDOFF_MATRIX.md` (test coverage contract)
  - `docs/wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md` (pre-fix audit)

## Abstract

This RFC is a **retrospective** on the graph handoff orchestrator
stabilization shipped on `chore/final-bits` for the v2.6 release. It
documents the journey from a 13-surface entry-point asymmetry that produced
the canonical "second Library G does nothing" bug to a single 8-state
finite-state-machine orchestrator with deterministic supersession, visible
failure UX, and self-healing reconciliation. The design decisions live in
[ADR-094](../adr/ADR-094-graph-handoff-orchestrator-fsm.md); this RFC
captures the *why* and the *lessons learned*.

**Architecture Alignment:** Extends the stale-run / single-flight pattern
documented in [VIEWER_ASYNC_STABILITY.md](../architecture/VIEWER_ASYNC_STABILITY.md)
to the multi-surface graph navigation handoff. Single orchestrator owns
state; consumers project from it.

## Problem Statement

Graph navigation in the GI/KG viewer has 13 entry surfaces (Library row,
Digest pill, Search "Show on graph", Dashboard topic landscape, Episode
panel, NodeDetail Load, GraphConnections neighbour, SubjectRail, StatusBar,
Explore, App-shell tab, plus canvas direct interactions: single-tap,
double-tap expand). Pre-fix, each surface mutated some combination of three
Pinia stores (`subject`, `graphNavigation`, `artifacts`) in slightly
different orders. `GraphCanvas.vue` (~3,400 lines) had 16 top-level
`watch()` blocks plus 48 `nextTick`/`setTimeout`/`raf` synchronisation
points reacting to those store mutations. **No single owner enforced the
invariant** that `filteredArtifact` containing node N implies
`cy.core.$id(N).nonempty()` before focus / camera asserts.

The canonical user-visible bug was "second Library G does nothing": a
second Library "Open in graph" click triggered an incremental artifact
append while `artifacts.currentLoadSource === null` (set asynchronously by
the territory auto-load watcher, not at click time), hit an early-return in
the `filteredArtifact` watcher, and Cytoscape never redrew. The asymmetry
was load-bearing-wrong: every other entry surface set `loadSource`
synchronously at the click site; Library did not.

**Use Cases:**

1. **Open Library episode in graph (twice in a row).** Pre-fix: second
   click was a no-op or showed stale state. Required deterministic
   "newest wins" semantics.
2. **Rapid Digest pill clicks during corpus load.** Pre-fix: race-prone
   apply phases could double-select or apply stale focus. Required
   generation-token concurrency.
3. **Failed handoff (network 404, model unavailable).** Pre-fix: silently
   swallowed at `GraphCanvas.vue:901-903`; users saw "I clicked, nothing
   happened". Required a visible failure surface.
4. **Inconsistent state between logical artifact and rendered canvas.**
   Pre-fix: silent — broken graph, no detection. Required self-healing
   reconciliation.

## Goals & Non-Goals

**Goals:**

- Single orchestrator owning the handoff lifecycle from click to settled
  selection across all 13 surfaces.
- Deterministic supersession of stale events via generation tokens.
- Visible failure UX; no silent swallow.
- Self-healing reconciliation when logical artifact diverges from rendered
  canvas.
- Mechanically enforced architectural invariants (per-surface contract
  tests; no accidental rollback by future "while I'm here" refactors).
- 100% E2E coverage matrix as a living test contract.

**Non-Goals:**

- Replace Cytoscape. The orchestrator integrates with Cytoscape; the cy
  rendering is unchanged.
- New entry surfaces (e.g. people-as-graph-target). Reserved for 2.7;
  the FSM design leaves `kind: 'person'` available without rewriting the
  contract.
- URL-based deep-linking. Out of scope for this release.

## Approach

The fix is an explicit finite-state machine orchestrator with envelope
dispatch. The 16 design decisions are listed in
[ADR-094](../adr/ADR-094-graph-handoff-orchestrator-fsm.md); the headline
points:

1. **8-state flat FSM**:
   `idle → loading_fetch → loading_bootstrap → loading_merge →
   redrawing_incremental | redrawing_full → applying → ready`. Each state
   has one defined entry condition, exit condition, and cancel-point.
2. **Single envelope** replaces 7+ caller-specific store mutation patterns.
   `GraphHandoffEnvelope = { kind, cyId?, metadataPath?, episodeId?, source,
   loadSource, camera, highlights?, suppressCamera?, generation }`. Each
   entry surface builds an envelope and fires one of three FSM events:
   `handoffRequested` (cross-surface), `canvasTapped` (in-canvas direct),
   `expansionRequested` (additive load).
3. **Generation tokens** are a strict-monotonic counter the FSM bumps on
   every supersession-eligible event. Async work captures the generation at
   start, checks before mutating UI state. Stale work returns early without
   side effects. The bare-await contract requires every `await` in
   orchestrator code to be paired with an `isStale()` check.
4. **Per-event re-entrance policies** (per-event, not per-state):
   `handoffRequested` always supersedes; `canvasTapped` queues if
   same-target / supersedes if different; `expansionRequested` always
   queues (additive — cancelling loses user work); `focusCleared`
   supersedes with empty envelope; `tabReturned` drops if not in `ready`;
   `corpusReloaded` resets to `idle`.
5. **Self-healing** runs as a set-difference invariant on every
   `layoutstop`. Violations get one targeted `core.add()` retry, then
   accept divergence + log if still violated.
6. **Visible failure UX**: `data-testid="handoff-error-strip"` renders
   when `lastResult.status === 'failed'`. Replaces silent swallow at
   `GraphCanvas.vue:901-903`.
7. **15s wall-clock stuck-handoff timeout** (originally 5s; raised to 15s
   after real-world handoffs that include a topic-cluster bootstrap step
   tripped the timer): the only timer in the orchestrator. Carved out from
   concern #3 ("no time-based gates") because timeouts are real-time
   bounds, not synchronisation primitives.

The Pinia store at
`web/gi-kg-viewer/src/stores/graphHandoff.ts`
wraps the pure FSM at
`web/gi-kg-viewer/src/services/graphHandoffFsm.ts`
and exposes reactive bindings + dev-only `window.__GIKG_FSM__` for E2E
inspection.

## Implementation phases

- **Phase 1 (C0–C8, ~5 working days)**: matrix scaffold, Library load-source
  patch (the canonical-bug cheap win), debug log strip, FSM scaffold + 75
  unit tests, partial entry-point migration, error strip UX, dev-only
  invariant warning.
- **Phase 2 (F1–F6, ~3.5 working days actually expended; ~11.5 days
  planned budget)**: complete entry-point migration across 13 surfaces,
  make FSM authoritative through the load → apply pipeline (remove
  `recordApplied` shortcut, add `advanceState` + `notifyLayoutStop` calls),
  watcher rework + bare-await contract, production self-healing
  reconciliation, manual validation + ci-ui-full gate, PostHog telemetry.
- **Phase 3 (G1–G5 + T1–T5, ~1.4 working days)**: documentation hardening
  (ADR-094, VIEWER_GRAPH_SPEC.md addendum, VIEWER_ASYNC_STABILITY.md
  cross-link, deferred-mock tracking issue, this RFC) + test coverage
  hardening (architectural-invariant contract tests, FSM state-walking
  integration, self-healing reconciliation predicate extraction + 14 unit
  tests, error strip render test, telemetry firing unit test).

Total: ~10 working days actually expended over a multi-week wall-clock
window. The plan budget of ~16 days assumed less leverage from the existing
codebase (e.g. the pure episode resolver was already extracted — saving
1.5 days planned).

## Decisions log

The 16 locked decisions are tabulated in
[ADR-094](../adr/ADR-094-graph-handoff-orchestrator-fsm.md). Highlighting
the pivot points:

- **Decision #1 (sync `setLoadSource` rule)**. Walked through "fix L1
  only" vs "make it a rule for all 13 surfaces" vs "fix the watcher
  early-return instead". Picked the all-surfaces rule — closing the
  asymmetry rather than the bug.
- **Decision #5 (FSM with events as methods)**. Considered: passive
  observer with hooks, RxJS-style streams, XState. Picked hand-rolled FSM:
  small enough to be readable, no runtime dependency, generation tokens
  fit cleanly without `AbortController` plumbing.
- **Decision #14 (8 flat states)**. A pressure-test pass found that
  `loading` and `redrawing` each conflated operations with different
  barrier semantics. Splitting them was non-obvious until that audit.
- **Decision #16 (15s wall-clock stuck timeout)**. Almost rejected because
  it violated the spirit of "no time-based gates"; carved out as
  exception because timeouts are categorically different from
  synchronisation. Document the carve-out explicitly to prevent future
  cargo-cult removal.

## Test coverage strategy

Three layers:

- **Pure FSM unit tests** (`services/graphHandoffFsm.test.ts`): 75 tests
  covering every (state × event) cell in the transition table. No DOM, no
  Pinia, no Cytoscape — just `applyEvent`, `validateEnvelope`, `isStale`,
  `advanceState` against in-memory FSM instances.
- **Architectural-invariant contract tests** (`e2e/handoff/contracts.spec.ts`,
  T3): 7 Playwright tests, one per migrated surface, that mechanically
  verify "surface X fires FSM event Y with envelope.source === Z and
  loadSource === W". Reads the dev-only `__GIKG_FSM_EVENT_LOG__`
  populated by the store. Catches the regression where a future "while
  I'm here" refactor drops the FSM event call from a surface — the legacy
  triplet still fires (so user-visible behaviour passes) but the FSM
  contract breaks.
- **End-to-end coverage matrix** (`e2e/handoff/HANDOFF_MATRIX.md` + 7
  spec files): 28 rows across cold-start / hot-state / repeat-click /
  cross-entry / concurrency / failure / lifecycle. 11 rows currently
  pass with real assertions; 17 are documented `test.skip()` rows tracked
  under
  [GH #754](https://github.com/chipi/podcast_scraper/issues/754) for
  the heavier mock infrastructure they need (Search, NodeDetail
  TopicCluster, full Digest topic-band).

Plus targeted unit / Playwright tests for state-walking (T1), the
self-healing reconciliation predicate (T2 — 14 unit tests on the
extracted helper at `utils/graphHandoffInvariant.ts`), error-strip
rendering (T4 — 2 Playwright tests), and PostHog telemetry firing (T5 —
8 Vitest tests on the store with `vi.mock('posthog-js')`).

## Lessons learned

- **Audit before designing**. The pre-fix entry-point catalogue (13
  surfaces, ~63 call sites) was the single most useful artifact. Without
  it the fix would have been "patch L1" rather than "fix the asymmetry
  class". Investing a half day on read-only exploration paid off many
  times over.
- **Pressure-test the design before writing code**. A second-pass design
  review surfaced gaps (3-state FSM was undercount; need 8) that would
  have been costly to find after implementation. The plan-mode workflow
  forced this.
- **Stop-and-ship checkpoints matter**. The plan's per-checkpoint
  "stop-and-ship?" annotations let the work compress when needed without
  leaving half-finished features. Several checkpoints (G3, G4) ended up
  trivial; others (F2, F3, T3) ended up high-value. Hard to predict
  ratios in advance.
- **Generation tokens > cancellation tokens**. Tried the AbortController
  approach in an early sketch; abandoned because it doesn't compose with
  Pinia reactivity. Generation tokens (capture-at-start, check-before-write)
  are simpler and work across the Pinia/Cytoscape boundary.
- **Visible failure UX must be in the spec**. The pre-fix silent-swallow
  was the result of "what should we do on error?" being unanswered at
  design time. Decision #15 (visible error strip) is small in code but
  large in user trust.
- **Bare-await contracts need lint enforcement, not just docs**. The
  current "every await must have a paired `isStale()` check" rule is
  documented in `services/graphHandoffFsm.ts` header but not
  mechanically enforced. A future custom ESLint rule would close that
  gap; review-only is the trade-off until then.
- **Test the contract, not just the behaviour**. The architectural
  contract tests (T3) catch a different regression class than user-flow
  E2E tests. A future PR can pass all behaviour tests while silently
  reverting the architecture; T3 is the only thing that goes red.

## Open follow-up

- [GH #754](https://github.com/chipi/podcast_scraper/issues/754) —
  complete the matrix's deferred 17 `test.skip()` rows that need heavier
  mock infrastructure.
- 2.7 candidate: ESLint rule for bare-await in `services/graphHandoff.ts`
  (currently review-only).
- 2.7 candidate: extend FSM with `kind: 'person'` envelope variant if
  product decides people should become graph navigation targets directly.
- 2.7 candidate: migrate the 3 documented `setTimeout` fallback recenters
  in `animateCameraToFocusedNode` to event-driven barriers if a reliable
  Cytoscape resize event becomes available.

## References

- [ADR-094](../adr/ADR-094-graph-handoff-orchestrator-fsm.md)
- [VIEWER_GRAPH_SPEC.md](../architecture/VIEWER_GRAPH_SPEC.md)
- [VIEWER_ASYNC_STABILITY.md](../architecture/VIEWER_ASYNC_STABILITY.md)
- [docs/wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md](../wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md)
- GH issues closed by this work: #748, #749, #750
- GH issue tracking deferred matrix work: #754
