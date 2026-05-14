# Graph handoff matrix — coverage contract

This is the **authoritative coverage matrix** for graph navigation handoffs across
all entry points. Every row maps to a named Playwright spec under `e2e/handoff/`.
Rows are either real `test()` assertions (11) or `test.skip(true, …)` with a
documented mock-infrastructure dependency tracked under
[GH #754](https://github.com/chipi/podcast_scraper/issues/754) (17). The matrix
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
state via `window.__GIKG_CY_DEV__` and Pinia state via the dev-only `window.__GIKG_STORES__`
(to be added; see C4):

1. **Selection** — `cy.nodes(':selected').length` matches the row's expectation
   (typically `=== 1` for a single-target handoff, `=== 0` for `kind: 'fit'` /
   `kind: 'preserve'` rows).
2. **Cy id existence** — the resolved primary cy id `nonempty()` in `core`. No
   `NO_CY_EPISODE_ID` errors in console.
3. **Subject store** — `subject.kind` matches the envelope kind; `subject.episodeId` /
   `subject.graphNodeCyId` / `subject.topicId` match the resolved id.
4. **Pending handoff** — `__GIKG_FSM__.currentState === 'ready'`;
   `__GIKG_FSM__.pendingEnvelope === null`. (Replaces today's `nav.pendingFocusNodeId`
   assertion; pre-FSM rows assert against the legacy field instead.)
5. **No console errors** — `[]` from `consoleMessages.filter(m => m.type() === 'error')`.
6. **Self-healing invariant** — set-difference predicate `expected ⊖ actual === ∅` over
   `viewWithEgo(focusNodeId)` ids vs. `core.nodes()` ids. (FSM-only; pre-FSM rows skip.)

## Matrix

### Section 1 — Cold-start happy path (7 rows)

User opens the app, makes one click from each entry point on a corpus they haven't
touched yet. No prior selection, no prior state.

| ID | Entry point | Spec file | FSM event | Status |
| --- | --- | --- | --- | --- |
| H1.1 | Library row "Open in graph" (L1) | `cold-start.spec.ts` | `handoffRequested({source:'library'})` | ✅ `test()` |
| H1.2 | Digest recent topic pill (D1) | `cold-start.spec.ts` | `handoffRequested({source:'digest'})` | ⏭️ `test.skip()` — needs Digest pill mock (#754) |
| H1.3 | Digest topic band hit row (D2) | `cold-start.spec.ts` | `handoffRequested({source:'digest'})` | ⏭️ `test.skip()` — needs Digest topic-band mock (#754) |
| H1.4 | Digest topic band title (D3) | `cold-start.spec.ts` | `handoffRequested({source:'digest', camera:{kind:'fit'}})` | ⏭️ `test.skip()` — needs Digest topic-band mock (#754) |
| H1.5 | Search "Show on graph" (S1) | `cold-start.spec.ts` | `handoffRequested({source:'search'})` | ⏭️ `test.skip()` — needs Search-mock infrastructure (#754) |
| H1.6 | Episode panel "Open in graph" (E1) | `cold-start.spec.ts` | `handoffRequested({source:'episode-panel'})` | ✅ `test()` |
| H1.7 | NodeDetail Load (O3) | `cold-start.spec.ts` | `expansionRequested({source:'node-detail'})` | ⏭️ `test.skip()` — needs TopicCluster fixtures (#754) |

### Section 2 — Hot state with prior selection (7 rows)

User has already focused episode A; now triggers a handoff for episode B (or topic Z) from
each entry point. Tests "second click works as well as first."

| ID | Entry point | Spec file | Asymmetry exercised | Status |
| --- | --- | --- | --- | --- |
| H2.1 | Library row re-click (L1 hot) | `hot-state.spec.ts` | Asymmetry #1 (L1 sync setLoadSource); supersession | ✅ `test()` |
| H2.2 | Digest A → Digest B (D1 hot) | `hot-state.spec.ts` | Highlight clearing (asymmetry #10) | ⏭️ `test.skip()` — needs Digest pill mock (#754) |
| H2.3 | Search A → Search B (S1 hot) | `hot-state.spec.ts` | Generation supersession | ⏭️ `test.skip()` — needs Search-mock infrastructure (#754) |
| H2.4 | Episode panel re-click (E1 hot) | `hot-state.spec.ts` | Asymmetry #2 (E1 missing setLoadSource) | ✅ `test()` |
| H2.5 | Mixed: Digest → Library (D1→L1) | `hot-state.spec.ts` | Highlight cleared on Library (asymmetry #10) | ⏭️ `test.skip()` — needs Digest fixtures (#754) |
| H2.6 | Mixed: Library → Digest (L1→D1) | `hot-state.spec.ts` | Load-source flip-flop | ⏭️ `test.skip()` — needs Digest fixtures (#754) |
| H2.7 | Mixed: Search → NodeDetail Load (S1→O3) | `hot-state.spec.ts` | graph-internal vs subject-external | ⏭️ `test.skip()` — needs Search + corpus-with-clusters fixtures (#754) |

### Section 3 — Repeated click on same target (4 rows)

User clicks the same target twice in a row. Tests idempotence and the "queue
same-target" re-entrance policy (decision #5 / FSM spec).

| ID | Trigger | Spec file | Re-entrance policy | Status |
| --- | --- | --- | --- | --- |
| H3.1 | Library row × 2 (same episode) | `repeat-click.spec.ts` | `handoffRequested` always supersedes | ✅ `test()` |
| H3.2 | Digest pill × 2 (same topic) | `repeat-click.spec.ts` | Same as H3.1 for source: 'digest' | ⏭️ `test.skip()` — needs Digest pill mock (#754) |
| H3.3 | Canvas tap fires canvasTapped on FSM | `repeat-click.spec.ts` | Verifies canvas onetap fires `graphHandoff.canvasTapped` | ✅ `test()` |
| H3.4 | Double-tap expand fires expansionRequested | `repeat-click.spec.ts` | Verifies canvas dbltap fires `graphHandoff.expansionRequested` | ✅ `test()` |

### Section 4 — Cross-entry sequences (3 rows)

Realistic user flows touching multiple entry points in sequence. Tests "no
state-contamination between entry points" (matches Pre-Fix Scenario 8 in
INCREMENTAL_LOADING_TEST_CRITERIA.md).

| ID | Sequence | Spec file | Validates | Status |
| --- | --- | --- | --- | --- |
| H4.1 | Library → Digest → Search | `cross-entry.spec.ts` | 3 different load-sources tracked + cleared in order | ⏭️ `test.skip()` — needs library + digest + search mocks (#754) |
| H4.2 | Digest band → Library row → Digest pill | `cross-entry.spec.ts` | Camera strategy switch (`fit` → `center`) | ⏭️ `test.skip()` — needs full digest band mock (#754) |
| H4.3 | Search → NodeDetail Load → Search | `cross-entry.spec.ts` | `subject-external` → `graph-internal` → `subject-external` | ⏭️ `test.skip()` — needs Search + corpus-with-clusters fixtures (#754) |

### Section 5 — Concurrency (2 rows)

Rapid sequences and lifecycle events that exercise generation tokens + supersession
(decision #5 + FSM concurrency rules).

| ID | Trigger | Spec file | FSM expectation | Status |
| --- | --- | --- | --- | --- |
| H5.1 | Rapid Library clicks: last wins | `concurrency.spec.ts` | Last wins; generation increments 5 times for 5 clicks | ✅ `test()` |
| H5.2 | Mid-load tab-switch away + return | `concurrency.spec.ts` | Reconcile-only; no double-apply (decision #7 / `tabReturned` policy) | ✅ `test()` |

### Section 6 — Failure modes (3 rows)

Failed handoffs surface visible feedback (decision #15) instead of silent swallow.

| ID | Trigger | Spec file | Expected outcome | Status |
| --- | --- | --- | --- | --- |
| H6.1 | Territory fetch returns 404 | `failure.spec.ts` | Error strip visible (`data-testid="handoff-error-strip"`); selection unchanged | ✅ `test()` |
| H6.2 | Handoff target id resolves to non-existent cy node | `failure.spec.ts` | `HandoffResult.status === 'failed'`; error strip; previous selection preserved | ⏭️ `test.skip()` — covered by FSM unit tests for `validateEnvelope` |
| H6.3 | Stuck handoff (load never returns) — 5s timeout | `failure.spec.ts` | After 5s, handoff cleared; `handoffStuck` event logged; ready for next click | ⏭️ `test.skip()` — stuck-detector unit-tested in `stores/graphHandoff.test.ts` |

### Section 7 — Lifecycle (2 rows)

Initialization and tab-return events that go through the FSM as internal events
(decisions #7 and #8).

| ID | Trigger | Spec file | FSM expectation | Status |
| --- | --- | --- | --- | --- |
| H7.1 | First mount with saved `restoreEpisodeCyId` preference | `lifecycle.spec.ts` | FSM bootstrap fires internal `handoffRequested({source:'restore-preference'})` once on first idle→ready (decision #8) | ⏭️ `test.skip()` — needs localStorage seeding (#754) |
| H7.2 | Tab-switch round-trip: reconcile-only | `lifecycle.spec.ts` | Reconcile no-op when consistent; targeted `core.add()` when missing nodes (decision #7 + self-healing predicate) | ✅ `test()` |

---

## Total: 28 rows

Distribution:

- **11 rows** pass with real `test()` assertions: H1.1, H1.6, H2.1, H2.4, H3.1, H3.3,
  H3.4, H5.1, H5.2, H6.1, H7.2
- **17 rows** are `test.skip(true, …)` (declared as `test()` but self-skip in the
  body with a documented reason). They require heavier mock infrastructure
  (Search-mock setup / NodeDetail TopicCluster fixtures / Digest topic-band mock /
  localStorage seeding) tracked under
  [GH #754](https://github.com/chipi/podcast_scraper/issues/754). Coverage of the
  underlying surface migration is verified by the F1 entry-point work plus
  integration coverage in `digest.spec.ts` / `search-to-graph-mocks.spec.ts` /
  `graph-expansion-mocks.spec.ts`.
- **0 rows** remain as `test.fail()` placeholder throws — every row either has a
  real assertion or a documented inline skip.
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
  cold-start.spec.ts      // 7 rows (Section 1)
  hot-state.spec.ts       // 7 rows (Section 2)
  repeat-click.spec.ts    // 4 rows (Section 3)
  cross-entry.spec.ts     // 3 rows (Section 4)
  concurrency.spec.ts     // 2 rows (Section 5)
  failure.spec.ts         // 3 rows (Section 6)
  lifecycle.spec.ts       // 2 rows (Section 7)
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
