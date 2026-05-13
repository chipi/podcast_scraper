# ADR-094: Graph handoff orchestrator FSM

- **Status**: Accepted
- **Date**: 2026-05-10
- **Authors**: Marko Dragoljevic
- **Related RFCs**: [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md), [RFC-080](../rfc/RFC-080-graph-visualization-extensions.md)
- **Related PRDs**: —

## Context & Problem Statement

The GI/KG viewer's graph navigation handoff (the path users take from Library /
Digest / Search / Dashboard / Episode panel / NodeDetail / canvas-direct
interactions to a focused node in the Cytoscape canvas) was a structural-debt
hotspot.

A pre-fix audit catalogued **13 entry surfaces** across `LibraryView`,
`DigestView`, `SearchPanel`, `DashboardView`, `EpisodeDetailPanel`,
`NodeDetail`, `GraphConnectionsSection`, `SubjectRail`, `StatusBar`, the
`ExplorePanel`, the App-shell tab activation, and `GraphCanvas`'s direct
interactions (single-tap, double-tap expand, mini-map). Each surface mutated
some combination of three Pinia stores (`subject`, `graphNavigation`,
`artifacts`) in slightly different orders. `GraphCanvas.vue` (~3,400 lines)
contained 16 top-level `watch()` blocks plus 48 `nextTick`/`setTimeout`/`raf`
synchronisation points. **No single owner enforced the invariant** that
`filteredArtifact` containing node N implies `cy.core.$id(N).nonempty()`
before focus / camera asserts.

The user-visible symptom was the canonical "second Library G does nothing"
bug — a second Library "Open in graph" click triggered an incremental
artifact append while `artifacts.currentLoadSource === null` (set
asynchronously by territory auto-load, not at click time), hit an early-return
in the `filteredArtifact` watcher, and Cytoscape never redrew. The
asymmetry was load-bearing-wrong: every other entry point set `loadSource`
synchronously at the click site; Library did not.

The deeper analysis is in
[docs/wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md](../wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md).

After diagnosis, the user explicitly broadened the scope from "fix the bugs"
to addressing **six architectural concerns**: robustness (bounded outcomes,
no stuck states), centralised core logic (single orchestrator across all 13
surfaces), removal of time-based flakiness (no `setTimeout` for
synchronisation; only for real-time bounds), concurrency (deterministic
supersession of stale events), self-healing (detect and reconcile
divergence between logical artifact and rendered canvas), and full E2E
coverage matrix.

## Decision

Land a finite-state-machine orchestrator that owns the graph handoff
lifecycle from click to settled selection. The 16 design points below are
the locked decisions; together they form the contract.

| # | Decision | Enforced at |
| --- | --- | --- |
| 1 | Synchronous `setLoadSource` rule. Every entry surface sets a load-source synchronously at click time. | All 13 surfaces (see §"Critical files" in plan) |
| 2 | Medium granularity load-source enum. `subject-external`, `digest-external`, `graph-internal`. `library-external` renamed to `subject-external`. | `web/gi-kg-viewer/src/stores/artifacts.ts` |
| 3 | `graph-internal` semantics = "expansion that preserves layout" (Definition X). | `web/gi-kg-viewer/src/services/graphHandoffFsm.ts` |
| 4 | Single envelope replaces the `requestFocusNode` multi-arity API. `pendingFocusNodeId` triple collapses into one `pendingHandoff` field. | `services/graphHandoffFsm.ts`, `web/gi-kg-viewer/src/stores/graphHandoff.ts` |
| 5 | Explicit FSM with events as methods. Generation tokens enforce concurrency; invalid transitions rejected. | `services/graphHandoffFsm.ts:applyEvent` |
| 6 | Mini-map / neighbour click → `canvasTapped({ source: 'minimap', suppressCamera: true })`. Selection without camera chase. | `web/gi-kg-viewer/src/components/graph/GraphConnectionsSection.vue` |
| 7 | `tabReturned` policy = reconcile-only. No re-apply on consistent state. | `stores/graphHandoff.ts:tabReturned` |
| 8 | `restoreFromPreference` is an FSM init step (not a self-triggering watcher). | `components/graph/GraphCanvas.vue` (restoreEpisodeCyId block) |
| 9 | Search dual-action (S1 navigates, S2 rail-only) preserved as designed. | `App.vue`, `SearchPanel.vue` |
| 10 | Highlights reset every apply phase from envelope.highlights. Stateless rule. | `services/graphHandoffFsm.ts` (envelope shape) |
| 11 | Typed `CameraStrategy` discriminated union (`center` / `center-on-target` / `fit` / `preserve` / `none`). Replaces `setRequestFitAfterLoad` + `pendingFocusCameraIncludeRawIds`. | `services/graphHandoffFsm.ts:CameraStrategy` |
| 12 | People stay rail-only. FSM design leaves `kind: 'person'` reserved for 2.7. | `services/graphHandoffFsm.ts:EnvelopeKind` |
| 13 | `activateGraphTab` double-fire folds into FSM `loading` state (await topic-cluster bootstrap as part of artifact-ready barrier). | `App.vue:activateGraphTab` |
| 14 | 8 flat states: `idle` → `loading_fetch` → `loading_bootstrap` → `loading_merge` → `redrawing_incremental` or `redrawing_full` → `applying` → `ready`. | `services/graphHandoffFsm.ts:FsmState` |
| 15 | Visible failure UX. Failed handoffs render `data-testid="handoff-error-strip"`. Replaces silent swallow. | `web/gi-kg-viewer/src/components/graph/HandoffErrorStrip.vue` |
| 16 | 15s wall-clock stuck-handoff timeout (originally 5s; raised to 15s after slow real-world handoffs — large topic-cluster bootstraps — were tripping it). The "no time-based gates" rule (concern #3) carves out timeouts (real-time bounds) vs synchronisation (use promises/events). | `stores/graphHandoff.ts:STUCK_TIMEOUT_MS` |

## Rationale

**FSM over passive observer.** A passive observer pattern (events emitted,
multiple consumers react in any order) was the status quo and produced the
"second Library G" bug class. The FSM forces a single owner of state
transitions, making generation tokens and self-healing structurally enforceable
rather than relying on careful coding at every site.

**Explicit states (8 flat) over hierarchical or implicit.** Each state has
a single defined entry condition, exit condition, and cancel-point. A pressure-test
pass (documented in the plan file's "FSM design specification") found that
`loading` and `redrawing` each conflated operations with different barrier
semantics — `loading_fetch` (HTTP) vs `loading_bootstrap` (topic-cluster
compound mount) vs `loading_merge` (Pinia append) have different cancel
characteristics. Splitting them avoids the "where in loading are we" ambiguity
that plagued the prior nextTick/raf-based watchers.

**Generation tokens over cancellation tokens.** Generation tokens are a
strict-monotonic counter the FSM bumps on every supersession-eligible event.
Async work captures the generation at start; checks before mutating UI state.
Stale work returns early without side effects. Simpler than Promise cancellation
(no `AbortController` plumbing) and works across the Pinia/Cytoscape boundary.
8 documented check points (see `services/graphHandoffFsm.ts` header) — the
**bare-await contract** mandates a paired `isStale()` check after every async
await in orchestrator code.

**Per-event re-entrance policy** instead of per-state. Different event types
have inherently different re-entrance semantics: `handoffRequested` always
supersedes (newer user intent wins); `canvasTapped` queues if same target /
supersedes if different (defends against double-clicks); `expansionRequested`
always queues (additive — cancelling loses user work); `focusCleared` always
supersedes with empty envelope; `tabReturned` drops if not in `ready`;
`corpusReloaded` resets to `idle`.

**Self-healing as set-difference invariant.** The invariant "every node in
`viewWithEgo(focusNodeId)` exists in `cy.core`" is checkable on every
`layoutstop`. Violations get one targeted `core.add()` retry; failure of that
retry accepts divergence + logs structured warning. Bounded recovery — no
infinite loops, no silent corruption.

**Wall-clock stuck timeout as carved-out exception.** Concern #3 (no
time-based gates) is about **synchronisation** (use promises/events).
Timeouts are about **real-time bounds** — they're a different category. The
15s `STUCK_TIMEOUT_MS` is the only timer in the orchestrator and is explicitly
documented as such.

## Alternatives Considered

1. **Targeted patches without architectural change**. Fix the L1 sync
   `setLoadSource` and stop. Rejected: would close the canonical bug but
   not the broader class — the next entry-point asymmetry would re-surface
   the same failure mode. The audit catalogued 13 distinct surfaces with
   slightly different mutation orders; a one-bug fix doesn't address the
   structural debt.

2. **Passive observer with explicit hooks**. Promote `subject` /
   `graphNavigation` mutations to typed events; let consumers (the existing
   watchers in `GraphCanvas.vue`) react. Rejected: keeps the multi-orchestrator
   problem (the prior 3 paths — `activateGraphTab`, `tryApplyPendingFocus`,
   `applyEpisodeRepresentativeFocusIfNeeded` — all kept agency over apply
   state). Generation tokens become coordination overhead rather than a
   structural property.

3. **RxJS-style event streams**. Reactive streams over the envelope. Rejected:
   adds a dependency for a problem that's solvable with a 200-line pure
   FSM module. Stream debugging is also less tractable for the kind of
   "two clicks racing" scenarios that motivate generation tokens.

4. **State-machine-as-data (XState or similar)**. XState gives visual
   debugging, declarative transitions, and well-defined semantics. Rejected:
   the orchestrator's transition table is small enough (8 states × 9 events
   ≈ 72 cells; ~42 valid) that hand-rolled is more readable than a JSON
   spec, and avoids a runtime dependency for a single use case.

## Consequences

- **Positive**:
  - **Single source of truth** for graph handoff intent (the envelope) and
    state (the FSM). Adding a new entry surface is a 3-step recipe (build
    envelope, fire event, write contract test).
  - **Deterministic supersession**. Rapid clicks resolve "newest wins" by
    construction; no UI flicker from racing apply phases.
  - **Visible failure UX**. The error strip surface ends the silent-swallow
    bug class permanently; users always know when a handoff failed and why.
  - **Mechanically testable**. 75 unit tests cover the transition table; 11
    Playwright contract tests verify per-surface envelope payloads;
    architectural invariants (set-difference predicate, no-bare-await rule)
    are enforced rather than reviewed.
  - **Self-healing in production**. Targeted `core.add()` reconciliation +
    1-shot retry budget catches divergence between logical artifact and
    rendered canvas without blowing up; user sees recovery, not breakage.

- **Negative**:
  - **New module surface**. `services/graphHandoffFsm.ts` (~400 lines pure
    TS) + `stores/graphHandoff.ts` (~200 lines Pinia wrapper) + a
    `HandoffErrorStrip.vue` component + ~80 lines of Dockerfile/compose
    plumbing for the timeout knob. Adds cognitive load for contributors
    who haven't seen the FSM pattern in this codebase before.
  - **Bare-await contract is review-only**. No automated lint enforces the
    `await ... isStale(gen)` pattern; relies on the file header docstring
    plus code review until a custom ESLint rule lands.
  - **Some time-based gates remain documented as safety nets**. Three
    `setTimeout` fallback recenters in `animateCameraToFocusedNode`
    (400/900/1800ms) survived the cleanup as INTENTIONAL TIME-BASED SAFETY
    NETS; ResizeObserver doesn't fire reliably for every layout shift.
    Documented in code, not converted.

- **Neutral**:
  - **Migration completed**. All 13 entry surfaces migrated to the FSM in
    F1; `recordApplied` shortcut removed in F2; restore-from-preference
    routed through the orchestrator in F3b.
  - **PostHog telemetry added**. Four events
    (`graph_handoff_started/applied/failed/stuck`) emit on every transition;
    wrapped in try/catch so telemetry can never affect runtime.

## Implementation Notes

- **Pure FSM**: `web/gi-kg-viewer/src/services/graphHandoffFsm.ts`
  (state types, events, `applyEvent`, `validateEnvelope`, `advanceState`,
  `isStale`, `STUCK_TIMEOUT_MS`).
- **Pinia wrapper**: `web/gi-kg-viewer/src/stores/graphHandoff.ts`
  (reactive state, stuck timer, dev-only `window.__GIKG_FSM__` hook,
  PostHog telemetry).
- **Service-level README**: `web/gi-kg-viewer/src/services/README.md`
  (FSM contract for code-local audience).
- **Failure UX**: `web/gi-kg-viewer/src/components/graph/HandoffErrorStrip.vue`.
- **Pure resolver**: `web/gi-kg-viewer/src/utils/graphEpisodeMetadata.ts`
  (3-layer episode identity: metadata path / episode_id / `__unified_ep__:UUID`).
- **E2E matrix**: `web/gi-kg-viewer/e2e/handoff/` +
  `web/gi-kg-viewer/e2e/HANDOFF_MATRIX.md` coverage contract.
- **Pattern**: 8-state FSM + generation tokens + envelope dispatch (similar
  in spirit to `XState` / `Boost.SML` / `redux-saga`'s state machine
  middleware, but hand-rolled for this surface).

## References

- [docs/wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md](../wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md) — pre-fix audit
- [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- [RFC-080: Graph visualization extensions](../rfc/RFC-080-graph-visualization-extensions.md)
- [ADR-066: Playwright for UI E2E testing](ADR-066-playwright-for-ui-e2e-testing.md)
- [VIEWER_GRAPH_SPEC.md §"Graph handoff orchestrator"](../architecture/VIEWER_GRAPH_SPEC.md) — operational reference
- [VIEWER_ASYNC_STABILITY.md](../architecture/VIEWER_ASYNC_STABILITY.md) — adjacent async stability patterns
- GH issues closed by this work: #748, #749, #750
