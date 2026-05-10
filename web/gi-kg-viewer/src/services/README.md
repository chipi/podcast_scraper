# `services/` — graph handoff orchestrator

This directory contains the orchestration layer the GI/KG viewer uses to coordinate
graph navigation handoffs across all entry surfaces (Library, Digest, Search, Dashboard,
Episode panel, NodeDetail, GraphConnectionsSection, SubjectRail, StatusBar, Explore,
plus canvas direct interactions and lifecycle events).

## Files

- **`graphHandoffFsm.ts`** — pure finite state machine. Types, transitions, generation
  tokens, no Vue / Pinia / Cytoscape dependencies. Easy to unit-test (75 tests in
  `graphHandoffFsm.test.ts`).
- **`graphHandoffFsm.test.ts`** — Vitest unit tests covering the transition table,
  re-entrance policies, generation supersession, camera-strategy preservation, envelope
  validation, and stuck-detector wall clock.

The Pinia store wrapping the FSM lives at **`stores/graphHandoff.ts`**. The store
exposes reactive state for components, manages the 5-second stuck-handoff wall-clock
timer, and stamps the dev-only `window.__GIKG_FSM__` hook for E2E inspection.

## Architectural decisions

The plan that produced this module is at
`/Users/markodragoljevic/.claude/plans/in-this-b-tanch-gentle-pillow.md`. The 13 locked
design decisions and FSM specification live in that file.

## States (8 flat)

```text
idle
  → loading_fetch       // HTTP fetchCorpusEpisodeDetail
  → loading_bootstrap   // maybeBootstrapGraphFromTopicClusterOnly + ensureTopicClusterCompoundVisible
  → loading_merge       // appendRelativeArtifacts into Pinia
  → redrawing_incremental   // incremental layout on added subset only
  → redrawing_full          // full layout
  → applying            // selection + dimming + camera
  → ready
```

Each state has a single defined entry condition, exit condition, and cancel-point.
Errors in any state collapse to `ready` with `HandoffResult.status = 'failed'` and the
prior selection preserved.

## Events (9 first-class)

| Event | Source | Re-entrance policy |
| --- | --- | --- |
| `handoffRequested(envelope)` | Library, Digest, Search, Dashboard, Episode panel, NodeDetail, SubjectRail, StatusBar, App-shell tab activation, restore-from-preference | Always supersede (bump generation, cancel in-flight, restart from `loading_*`) |
| `canvasTapped(envelope)` | Single-tap on canvas, mini-map / neighbour click (with `suppressCamera: true`) | Supersede different-target / queue same-target |
| `expansionRequested(envelope)` | Double-tap expand, NodeDetail Load | Always queue (additive; cancelling loses user work) |
| `focusCleared()` | Escape key | Always supersede with empty envelope; → `ready` |
| `tabReturned()` | onActivated (KeepAlive return) | Drop if not in `ready`; reconcile-only |
| `corpusReloaded()` | Corpus path watcher | Always full reset; bump generation, drop envelope, → `idle` |
| `handoffFailed(reason)` | territory fetch errors, resolution failures | Drives `failed` outcome of `HandoffResult`; preserves prior selection |
| `layoutstop` | Cytoscape layout completion | `redrawing_* → applying`, `applying → ready` |
| `layoutstart` | Cytoscape layout start | `applying → redrawing_full` if mid-apply |

## Generation tokens

Every envelope is stamped with the generation at orchestration entry. Every async step
checks `graphHandoff.isStale(envelope.generation)` before mutating UI state. Mismatch →
return early; the in-flight work was superseded.

Documented check points (per FSM spec § 8+ sites):

1. Before `await fetchCorpusEpisodeDetail`
2. After `fetchCorpusEpisodeDetail`, before `subject.setEpisodeId`
3. Before `await appendRelativeArtifacts`
4. After `appendRelativeArtifacts`, before Pinia writes
5. In `layoutstop` listener, before `finishLayoutPass`
6. At entry of `tryApplyPendingFocus`
7. Before `cy.animate` in `animateCameraToFocusedNode`; in animation `complete` callback
8. Inside `nextTick` chains in lifecycle hooks

Currently wired: 1, 2, 3, 4 inside `loadEpisodeSliceForTerritoryStrip`. The remaining
sites are tracked under follow-up F3 (deferred for the deeper watcher rework).

## Self-healing reconciliation

After every `layoutstop`, `finishLayoutPass` runs the set-difference invariant:

```text
expected = {n.id : n in viewWithEgo(focusNodeId).visNodes}
actual   = {n.id() : n in core.nodes()}
missing  = expected − actual
extra    = actual − expected
invariant: missing.length === 0 && extra.length === 0
```

Violations trigger a **single targeted reconciliation**: when `0 < missing.length < 20`,
`core.add(toCytoElements(missing))` brings the canvas in line with the logical view.
If the next `layoutstop` still violates, the divergence is accepted and logged as a
structured warning. The retry budget is per envelope generation.

## Stuck detection

If a `pendingEnvelope` does not reach `ready` within `STUCK_TIMEOUT_MS` (5 seconds wall
clock), the FSM clears the envelope, emits `handoffStuck` to listeners, and surfaces a
visible failure via `HandoffErrorStrip.vue`.

The "no time-based gates" rule (architectural concern #3) applies to *synchronization*
(use promises/events instead of timers); it does NOT apply to *timeouts* (use timers
for real-time bounds). Stuck detection without a wall clock is impossible.

## Dev hook

In dev builds, the store stamps `window.__GIKG_FSM__` for E2E inspection:

```ts
window.__GIKG_FSM__ = {
  state,        // current FsmState
  pending,      // current GraphHandoffEnvelope or null
  generation,   // monotonic counter
  lastResult,   // most recent HandoffResult or null
}
```

Used by `e2e/handoff/_handoff-helpers.ts:readFsmState()` to assert FSM state from
Playwright tests.

## Adding a new entry surface

When a new UI surface needs to navigate to graph:

1. Build a `GraphHandoffEnvelope` matching the surface's intent (see `EnvelopeSource`
   union for the source enum).
2. Choose `loadSource`: `'subject-external'` for cross-surface handoffs, `'digest-external'`
   for Digest paths, `'graph-internal'` for in-graph expansions.
3. Choose `camera`: `'center'` with explicit cyId; `'center-on-target'` when the cyId
   resolves during loading; `'fit'` for multi-node loads; `'preserve'` for canvas taps
   and mini-map clicks; `'none'` for explicit no-op.
4. Call the right event method on the store: `handoffRequested` (cross-surface),
   `canvasTapped` (in-canvas selection), `expansionRequested` (additive load).
5. Add a corresponding row to `e2e/handoff/HANDOFF_MATRIX.md` and a real assertion to
   the matching `*.spec.ts` under `e2e/handoff/`.

## Following the contract

- Every entry surface fires exactly one FSM event per user action.
- The orchestrator runtime in `GraphCanvas.vue` advances the FSM through the load → apply
  pipeline by calling `graphHandoff.advanceState()` at barrier points and emitting
  `notifyLayoutStop()` when Cytoscape signals layout completion.
- `recordApplied(cyId)` marks the terminal `applying → ready` transition and clears the
  stuck timer.
- Tests assert FSM state via `window.__GIKG_FSM__` (dev hook).
