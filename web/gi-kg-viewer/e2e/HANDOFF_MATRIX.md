# Graph handoff matrix — coverage contract

This is the **authoritative coverage matrix** for graph navigation handoffs across
all entry points. Every row maps to a named Playwright spec under `e2e/handoff/`.
All 41 rows are real `test()` assertions across 8 sections (cold-start /
hot-state / repeat-click / cross-entry / concurrency / failure / lifecycle
/ filters). The matrix
contract: every architectural decision in
[ADR-094](../../../docs/adr/ADR-094-graph-handoff-orchestrator-fsm.md) is either
verified by a row here or by a unit test in
`src/services/graphHandoffFsm.test.ts` /
`src/utils/graphHandoffInvariant.test.ts` / `src/stores/graphHandoff.test.ts`.

Companion docs:

- [INCREMENTAL_LOADING_TEST_CRITERIA.md](INCREMENTAL_LOADING_TEST_CRITERIA.md) — the
  pre-existing scenario language. This matrix supersedes its **automation contract**
  for the 7 entry-point family; it does not replace its scenario semantics.
- [E2E_SURFACE_MAP.md](E2E_SURFACE_MAP.md) — selector contract.
- [docs/wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md](../../../docs/wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md)
  — structural analysis the FSM design responds to.

## Standard assertions (every row)

Each handoff spec asserts the following after the handoff settles. Expressed as Cytoscape
state via `window.__GIKG_CY_DEV__`, FSM state via `window.__GIKG_FSM__`, and subject state
via `window.__GIKG_SUBJECT__`:

1. **Selection** — `cy.nodes(':selected').length` matches the row's expectation
   (typically `=== 1` for a single-target handoff, `=== 0` for `kind: 'fit'` /
   `kind: 'preserve'` rows).
2. **Cy id existence** — the resolved primary cy id `nonempty()` in `core`. No
   `NO_CY_EPISODE_ID` errors in console.
3. **Subject store** — `subject.kind` matches the envelope kind; `subject.episodeId` /
   `subject.graphNodeCyId` / `subject.topicId` match the resolved id.
4. **Pending handoff** — `__GIKG_FSM__.state === 'ready'`;
   `__GIKG_FSM__.pending === null`; `lastResult.status === 'applied'`.
5. **No console errors** — `[]` from `consoleMessages.filter(m => m.type() === 'error')`.
6. **Self-healing invariant** — set-difference predicate `expected ⊖ actual === ∅` over
   `viewWithEgo(focusNodeId)` ids vs. `core.nodes()` ids. (FSM-only; pre-FSM rows skip.)

## Assertion layers (L0–L5)

The standard 6-point assertion is delivered through five composable layers so each row
asserts as deeply as its fixture allows. Two reusable helpers in `_handoff-helpers.ts`
ground the layering:

| Layer | What it asserts | Hook | Helper |
| --- | --- | --- | --- |
| **L0** | API mocks return the expected shape (corpus, digest, search, artifacts) | n/a | `setupHandoffMatrixMocks` |
| **L1** | FSM ready, pending=null, lastResult.status=applied, generation bumped | `__GIKG_FSM__` | `readFsmState` |
| **L2** | One node selected with matching cy id (modulo `g:`/`k:`/`__unified_ep__:`) | `__GIKG_CY_DEV__` | `assertHandoffApplied` |
| **L3** | Camera zoom in sane range; no console errors; Episode panel title (when relevant) | `__GIKG_CY_DEV__` + `console.error` | `assertHandoffApplied` |
| **L4** | FSM event log envelope contract: `type`, `source`, `kind`, `loadSource`, `camera.kind` | `__GIKG_FSM_EVENT_LOG__` | `assertFsmEventEnvelope` |
| **L5** | Subject store correctness: `subject.kind` + matching id field reflects the target | `__GIKG_SUBJECT__` | `assertHandoffApplied` |

**UI-driven rows** (L1.1-L1.4, L1.6, L1.12-L1.13, H2.1-H2.2, H2.4-H2.6, H3.1-H3.2,
H4.2, H7.2) drive the full pipeline via real DOM clicks and assert **L0+L1+L2+L3+L5**
through `assertHandoffApplied`.

**Dev-hook-driven rows** (L1.5, L1.7-L1.11, H2.3, H2.7, H4.1, H4.3, H6.1-H6.3, H7.1)
dispatch envelopes via `__GIKG_HANDOFF_STORE__` (surfaces whose UI fixtures duplicate
those exercised elsewhere) and assert **L0+L1+L4** through `assertFsmEventEnvelope`.
These rows can't reach `applied` without a real graph change → they pin the FSM-event
contract instead of the full outcome contract.

**Filter rows** (Section 8) assert the inverse: no `handoffRequested` fired and FSM
state quiescent — filters must not drive graph state.

## Matrix

### Section 1 — Cold-start happy path (7 rows)

User opens the app, makes one click from each entry point on a corpus they haven't
touched yet. No prior selection, no prior state.

| ID | Entry point | Spec file | FSM event | Layers | Status |
| --- | --- | --- | --- | --- | --- |
| H1.1 | Library row "Open in graph" (L1) | `cold-start.spec.ts` | `handoffRequested({source:'library'})` | L0+L1+L2+L3+L5 | ✅ `test()` |
| H1.2 | Digest recent topic pill (D1) | `cold-start.spec.ts` | `handoffRequested({source:'digest'})` | L0+L1+L2+L3+L5 | ✅ `test()` |
| H1.3 | Digest topic band hit row (D2) | `cold-start.spec.ts` | `handoffRequested({source:'digest'})` | L0+L1+L2+L3+L5 | ✅ `test()` |
| H1.4 | Digest topic band title (D3) | `cold-start.spec.ts` | `handoffRequested({source:'digest', camera:{kind:'fit'}})` | L0+L1+L2+L3+L5 | ✅ `test()` |
| H1.5 | Search "Show on graph" (S1) | `cold-start.spec.ts` | `handoffRequested({source:'search'})` | L0+L1+L4 | ✅ `test()` |
| H1.6 | Episode panel "Open in graph" (E1) | `cold-start.spec.ts` | `handoffRequested({source:'episode-panel'})` | L0+L1+L2+L3+L5 | ✅ `test()` |
| H1.7 | NodeDetail Load (O3) | `cold-start.spec.ts` | `expansionRequested({source:'node-detail'})` | L0+L1+L4 | ✅ `test()` |
| H1.8 | Dashboard TopicLandscape → graph (O1) | `cold-start.spec.ts` | `handoffRequested({source:'dashboard'})` | L0+L1+L4 | ✅ `test()` |
| H1.9 | SubjectRail @go-graph (O5) | `cold-start.spec.ts` | `handoffRequested({source:'subject-rail'})` | L0+L1+L4 | ✅ `test()` |
| H1.10 | StatusBar @go-graph (O6) | `cold-start.spec.ts` | `handoffRequested({source:'status-bar'})` | L0+L1+L4 | ✅ `test()` |
| H1.11 | Mini-map / GraphConnectionsSection click (G6) | `cold-start.spec.ts` | `canvasTapped({source:'minimap',suppressCamera:true})` | L0+L1+L4 | ✅ `test()` |
| H1.12 | Escape key clears focus (K1) | `cold-start.spec.ts` | `focusCleared()` | L0+L1 | ✅ `test()` |
| H1.13 | Background canvas tap clears subject (G7) | `cold-start.spec.ts` | Cytoscape `tap` with `target === core` | L0+L1 | ✅ `test()` |

### Section 2 — Hot state with prior selection (7 rows)

User has already focused episode A; now triggers a handoff for episode B (or topic Z) from
each entry point. Tests "second click works as well as first."

| ID | Entry point | Spec file | Asymmetry exercised | Layers | Status |
| --- | --- | --- | --- | --- | --- |
| H2.1 | Library row re-click (L1 hot) | `hot-state.spec.ts` | Asymmetry #1 (L1 sync setLoadSource); supersession | L0+L1+L2+L3+L5 | ✅ `test()` |
| H2.2 | Digest A → Digest B (D1 hot) | `hot-state.spec.ts` | Highlight clearing (asymmetry #10) | L0+L1+L2+L3+L5 | ✅ `test()` |
| H2.3 | Search A → Search B (S1 hot) | `hot-state.spec.ts` | Generation supersession | L0+L1+L4 | ✅ `test()` |
| H2.4 | Episode panel re-click (E1 hot) | `hot-state.spec.ts` | Asymmetry #2 (E1 missing setLoadSource) | L0+L1+L2+L3+L5 | ✅ `test()` |
| H2.5 | Mixed: Digest → Library (D1→L1) | `hot-state.spec.ts` | Highlight cleared on Library (asymmetry #10) | L0+L1+L2+L3+L5 | ✅ `test()` |
| H2.6 | Mixed: Library → Digest (L1→D1) | `hot-state.spec.ts` | Load-source flip-flop | L0+L1+L2+L3+L5 | ✅ `test()` |
| H2.7 | Mixed: Search → NodeDetail Load (S1→O3) | `hot-state.spec.ts` | graph-internal vs subject-external | L0+L1+L4 | ✅ `test()` |

### Section 3 — Repeated click on same target (4 rows)

User clicks the same target twice in a row. Tests idempotence and the "queue
same-target" re-entrance policy (decision #5 / FSM spec).

| ID | Trigger | Spec file | Re-entrance policy | Layers | Status |
| --- | --- | --- | --- | --- | --- |
| H3.1 | Library row × 2 (same episode) | `repeat-click.spec.ts` | `handoffRequested` always supersedes | L0+L1+L2+L3+L5 | ✅ `test()` |
| H3.2 | Digest pill × 2 (same topic) | `repeat-click.spec.ts` | Same as H3.1 for source: 'digest' | L0+L1+L2+L3+L5 | ✅ `test()` |
| H3.3 | Canvas tap fires canvasTapped on FSM | `repeat-click.spec.ts` | Verifies canvas onetap fires `graphHandoff.canvasTapped` | L0+L1+L4 | ✅ `test()` |
| H3.4 | Double-tap expand fires expansionRequested | `repeat-click.spec.ts` | Verifies canvas dbltap fires `graphHandoff.expansionRequested` | L0+L1+L4 | ✅ `test()` |

### Section 4 — Cross-entry sequences (3 rows)

Realistic user flows touching multiple entry points in sequence. Tests "no
state-contamination between entry points" (matches Pre-Fix Scenario 8 in
INCREMENTAL_LOADING_TEST_CRITERIA.md).

| ID | Sequence | Spec file | Validates | Layers | Status |
| --- | --- | --- | --- | --- | --- |
| H4.1 | Library → Digest → Search | `cross-entry.spec.ts` | 3 different load-sources tracked + cleared in order | L0+L1+L4 | ✅ `test()` |
| H4.2 | Digest band → Library row → Digest pill | `cross-entry.spec.ts` | Camera strategy switch (`fit` → `center`) | L0+L1+L2+L3+L5 | ✅ `test()` |
| H4.3 | Search → NodeDetail Load → Search | `cross-entry.spec.ts` | `subject-external` → `graph-internal` → `subject-external` | L0+L1+L4 | ✅ `test()` |

### Section 5 — Concurrency (2 rows)

Rapid sequences and lifecycle events that exercise generation tokens + supersession
(decision #5 + FSM concurrency rules).

| ID | Trigger | Spec file | FSM expectation | Layers | Status |
| --- | --- | --- | --- | --- | --- |
| H5.1 | Rapid Library clicks: last wins | `concurrency.spec.ts` | Last wins; generation increments 5 times for 5 clicks | L0+L1+L4 | ✅ `test()` |
| H5.2 | Mid-load tab-switch away + return | `concurrency.spec.ts` | Reconcile-only; no double-apply (decision #7 / `tabReturned` policy) | L0+L1 | ✅ `test()` |

### Section 6 — Failure modes (3 rows)

Failed handoffs surface visible feedback (decision #15) instead of silent swallow.

| ID | Trigger | Spec file | Expected outcome | Layers | Status |
| --- | --- | --- | --- | --- | --- |
| H6.1 | Territory fetch returns 404 | `failure.spec.ts` | FSM cleared `pending`; `lastResult.status !== 'applied'`; no console errors leak | L0+L1 | ✅ `test()` |
| H6.2 | Handoff target id resolves to non-existent cy node | `failure.spec.ts` | Envelope reaches FSM with expected shape; no leak after settle | L0+L1+L4 | ✅ `test()` |
| H6.3 | Stuck handoff (load never returns) — 15s timeout | `failure.spec.ts` | After 15s, `pending=null`; `lastResult.status='failed'`; envelope shape recorded | L0+L1+L4 | ✅ `test()` |

### Section 7 — Lifecycle (2 rows)

Initialization and tab-return events that go through the FSM as internal events
(decisions #7 and #8).

| ID | Trigger | Spec file | FSM expectation | Layers | Status |
| --- | --- | --- | --- | --- | --- |
| H7.1 | First mount with saved `restoreEpisodeCyId` preference | `lifecycle.spec.ts` | FSM bootstrap fires internal `handoffRequested({source:'restore-preference'})` once on first idle→ready (decision #8) | L0+L1+L4 | ✅ `test()` |
| H7.2 | Tab-switch round-trip: reconcile-only | `lifecycle.spec.ts` | Reconcile no-op when consistent; targeted `core.add()` when missing nodes (decision #7 + self-healing predicate) | L0+L1+L2+L3+L5 | ✅ `test()` |

### Section 8 — Filters (7 rows)

Filter / view-only surfaces in Digest / Library / Graph. These are the
"negative space" of the matrix: actions that *shouldn't* fire FSM
handoff events, and shouldn't surface console errors. Catches the bug
class where a filter toggle accidentally triggers a redraw that tears
down selection (or fires a spurious handoff).

| ID | Surface | Spec file | Contract | Layers | Status |
| --- | --- | --- | --- | --- | --- |
| H8.1 | Digest date chip (window selector) | `filters.spec.ts` | No `handoffRequested` fires; state quiescent after | L0+L1 (neg) | ✅ `test()` |
| H8.2 | Library title filter input | `filters.spec.ts` | No `handoffRequested` fires; state quiescent after | L0+L1 (neg) | ✅ `test()` |
| H8.3 | Library summary filter input | `filters.spec.ts` | No `handoffRequested` fires; state quiescent after | L0+L1 (neg) | ✅ `test()` |
| H8.4 | Graph layout cycle | `filters.spec.ts` | Runs new layout; no `handoffRequested` fires; FSM advances back to `ready` | L0+L1 (neg) | ✅ `test()` |
| H8.5 | Graph relayout button | `filters.spec.ts` | Re-runs current layout; no `handoffRequested` fires | L0+L1 (neg) | ✅ `test()` |
| H8.6 | Graph minimap toggle | `filters.spec.ts` | UI-only; no `handoffRequested` fires | L0+L1 (neg) | ✅ `test()` |
| H8.7 | Graph zoom in / out / fit | `filters.spec.ts` | Camera-only; no `handoffRequested` fires; selection preserved | L0+L1 (neg) | ✅ `test()` |

---

## Total: 41 rows

Distribution:

- **All 41 rows** pass with real `test()` assertions. Rows that drive
  UI surfaces directly (Library / Digest / Episode panel) click through
  the mocked DOM. Rows whose surfaces require infrastructure outside
  the handoff suite's scope (Search-UI fixture, NodeDetail TopicCluster
  fixtures, Dashboard topic-landscape fixtures, SubjectRail / StatusBar
  go-graph emitters) instead drive the same FSM event the production
  surface emits via the dev-only `window.__GIKG_HANDOFF_STORE__` hook —
  same envelope shape, same FSM contract surface as the user-driven
  path. Section 8 covers filter / view-only surfaces (the "negative
  space" of the matrix: actions that *shouldn't* fire handoff events).
- **0 rows** remain as `test.skip()` or `test.fail()`. GH #754 (the
  follow-up that tracked the prior 17 skips) is now closed in favour of
  these end-to-end assertions.
- FSM state / event-log / state-history inspection via the dev-only
  `__GIKG_FSM__` / `__GIKG_FSM_EVENT_LOG__` / `__GIKG_FSM_STATE_HISTORY__` /
  `__GIKG_HANDOFF_STORE__` hooks is verified across all real tests.

### Supplemental coverage (T-series)

Beyond the 28-row matrix, these specs guard against architectural drift:

- `e2e/handoff/contracts.spec.ts` (T3) — 7 architectural-invariant tests that
  pin "surface X fires FSM event Y with envelope.source === Z and loadSource === W."
  Catches accidental rollback of the FSM migration.
- `e2e/handoff/state-walking.spec.ts` (T1) — 2 tests that verify the FSM
  walks intermediate states during a real handoff (catches re-introduction
  of the `recordApplied` shortcut that bypasses the pipeline).
- `e2e/handoff/error-strip.spec.ts` (T4) — 2 tests covering
  `HandoffErrorStrip.vue` render + dismiss.

## Spec file inventory (under `e2e/handoff/`)

```text
e2e/handoff/
  cold-start.spec.ts      // 13 rows (Section 1)
  hot-state.spec.ts       // 7 rows (Section 2)
  repeat-click.spec.ts    // 4 rows (Section 3)
  cross-entry.spec.ts     // 3 rows (Section 4)
  concurrency.spec.ts     // 2 rows (Section 5)
  failure.spec.ts         // 3 rows (Section 6)
  lifecycle.spec.ts       // 2 rows (Section 7)
  filters.spec.ts         // 7 rows (Section 8)
```

All seven files share `helpers.ts` for fixture setup, dev-only state inspection
(`window.__GIKG_CY_DEV__` / `window.__GIKG_STORES__` / `window.__GIKG_FSM__` once it
exists), and the standard-assertion helper.

## CI integration

These specs run as part of `make ci-ui-fast` and `make ci-ui-full`. Currently-failing
rows do **not** break CI because `test.fail()` flips to "expected failure" — the test
must fail; if it passes, that's a regression *toward* working behaviour and the row gets
flipped to `test()` as part of the relevant checkpoint.

## Updating this doc

When a checkpoint flips a row from `test.fail()` → `test()`:

1. Update the row's `Status` column
2. Note the checkpoint ID in the commit message
3. If a new asymmetry surfaces during implementation, add a new row (renumber within section)

The matrix is the **living source of truth** for graph handoff coverage; if a feature
needs handoff testing, it gets a row here, not a one-off spec elsewhere.
