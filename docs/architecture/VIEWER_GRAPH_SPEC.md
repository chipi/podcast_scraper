# Viewer graph — implementation specification

> **Single place** for Cytoscape load, lens window + episode cap, status line,
> default node types, stylesheet, gesture overlay, and graph-focus entry points.
> **User-visible** behaviour and test IDs: [UXS-004](../uxs/UXS-004-graph-exploration.md).

## Graph initial load

**Status:** Ready for implementation
**Author:** Design session (Marko + Claude), April 2026
**Repo:** `chipi/podcast_scraper`
**Target area:** `web/gi-kg-viewer/src/`
**Related docs:** UXS-001, UXS-004, RFC-062, RFC-076
**Scope:** Load strategy, graphLens store, default node type visibility,
status line. No Cytoscape stylesheet changes (those are in the Graph
Styling spec).

---

### Initial load — 1. Problem

The graph currently has no defined default load strategy. In online mode
with a corpus path set, loading all GI/KG artifacts produces a hairball
for any non-trivial corpus. There is no time lens on the Graph tab, no
episode cap, and no visual indication of what is loaded or how to expand.

Simultaneously, the shared `corpusLens` (Digest + Library) is the only
time filter in the app. If Graph naively inherits it, a user who set
Library to "all time" to browse the full catalog would trigger a full
corpus graph load — defeating any load cap.

---

### 2. Direction

Three connected decisions:

1. **graphLens** — Graph owns its own time lens, seeded from `corpusLens`
   at first open but diverging independently thereafter.

2. **Episode cap** — Graph loads at most N episodes (default 15) from
   the `graphLens` window. RFC-076 progressive expand is the mechanism

   for loading more.

3. **Default node type visibility** — High-noise, low-signal node types
   (Quote, Speaker, Episode) are hidden by default. The canvas opens on

   the intelligence layer.

These three together produce a first graph load that is readable,
fast, and coherent with the rest of the viewer.

---

### 3. graphLens

#### Concept

`graphLens` is a time filter owned exclusively by the Graph tab. It is
**not** `corpusLens` and does not share state with Digest or Library.

#### Initialisation

On first Graph tab open per session:

```text

If `corpusLens` has a date set (e.g. user set Library to "30d"), Graph
seeds from that. If `corpusLens` is unset (all time), Graph defaults to
`7d` — it does not inherit "all time" from Library.

After initial seed, `graphLens` is independent. Changing `corpusLens`
in Digest/Library does not update `graphLens`. Changing `graphLens` in
Graph does not affect Digest/Library.

```

#### Values

Same set as `corpusLens`:

| Value | Meaning |
| --- | --- |
| `7d` | Last 7 calendar days (default) |
| `30d` | Last 30 days |
| `90d` | Last 90 days |
| `all` | All time (user must opt in explicitly) |
| `since:YYYY-MM-DD` | Since a specific date |

#### Store

Add to `graphExplorer.ts` (or a new `graphLensStore.ts` if preferred —
follow existing store granularity pattern):

```typescript

interface GraphLensState {
  lens: '7d' | '30d' | '90d' | 'all' | string  // string = 'since:YYYY-MM-DD'
  seeded: boolean  // true once initialised from corpusLens
}

// Actions
function seedFromCorpusLens(corpusLensValue: string | null): void
function setLens(value: string): void
function reset(): void  // called on corpus path change

```

On corpus path change: call `reset()` — clear the seeded flag so next
Graph tab open re-seeds from the new corpus context.

---

### 4. Episode Cap

#### Rule

When loading GI/KG artifacts for the graph, select at most **N episodes**
from the candidate pool defined by `graphLens`.

**Candidate pool selection (implemented in `web/gi-kg-viewer/src/utils/graphEpisodeSelection.ts`):**

1. Build one **episode stem** per GI/KG/bridge basename group; use listing `publish_date` (YYYY-MM-DD) per stem.
2. **Filter** by `graphLens`: dated window = `publish_date >= sinceYmd`; **all time** = all stems.
3. **Score** each stem: `recency_weight + cluster_bonus` (tunables in [UXS-001](../uxs/UXS-001-gi-kg-viewer.md)). Recency is linear from **0.2** (oldest in the dated pool, or older than the trailing **90**-day window from the newest publish for **all time**) to **1.0** at the newest. **Cluster bonus** (+0.4 default) applies when the stem matches an `episode_id` on a **topic cluster** member (`topic_clusters.json`), if that doc was loaded.
4. Sort by **score descending**, then **publish date descending**, then stem id (stable tie-break).
5. Take the **top N** stems; load all GI/KG/bridge paths for those stems only.

The cap controls load time and canvas density, not node count directly.

#### Default cap value

**N = 15**

Add to UXS-001 tunable parameters table:

| Parameter | Current value | Status | Notes |
| --- | --- | --- | --- |
| Graph default episode cap | 15 | Open | Tunable; increase for denser corpora |

#### Artifact load implementation

In the artifact loading flow (`artifactsStore` or wherever GI/KG paths
are listed and loaded):

1. `GET /api/artifacts` returns all available GI/KG paths (with `publish_date` per row).
2. `GET /api/corpus/topic-clusters` (when present) supplies cluster membership for the scoring bonus.
3. Run the **score + cap** selection above, then load only the chosen paths into the merged graph.

**Future:** per-episode GI Insight counts from a coverage-style API can add a third term to the score (see UXS-001 **GI density max weight**).

---

### 5. Graph Status Line

A muted status line sits **above the graph canvas** (below the toolbar
chrome, above the Cytoscape surface). Always visible when graph is
loaded. Updates reactively.

#### Content

```text

Examples:

Showing all time · 15 episodes (capped) · 934 nodes   [All ▾]

```

- Node count: use `k` suffix for counts ≥ 1000 (e.g. `1.2k`)
- Episode count rounds down: show loaded count, not candidate pool size

#### Styling

- `text-[10px]`, `muted` token — unobtrusive, same density as toolbar
- Left-aligned text, right-aligned lens selector
- Background: `bg-canvas` in the canvas column (embedded in **`GraphBottomBar`** centre zone under the main graph canvas)
- Height: ~24px, same compact scale as toolbar rows

#### Lens selector

Compact inline control — same visual style as the Digest Window
presets but smaller. Options: **7d · 30d · 90d · All · Since…**

Active option uses `primary` ring or underline (same pattern as
Digest preset active state). Selecting a new value:

1. Updates `graphLens`
2. Re-runs artifact load with new window
3. Clears current graph (replaces, does not append)
4. RFC-076 expansion state is reset

"Since…" opens a small date input (same pattern as `corpusLens` date
field — reuse that component).

#### data-testid values

```text

data-testid="graph-status-since-input"
data-testid="graph-status-episode-count"
data-testid="graph-status-node-count"

```

`graph-status-episode-count` and `graph-status-node-count` wrap **numeric text only** (node count may use the `k` suffix from formatting).

---

### 6. Default Node Type Visibility

#### Problem

The graph toolbar has a Types section with per-type checkboxes. Currently
all types are on by default. Quote, Speaker, and Episode nodes create
visual noise without adding to the first-read intelligence value — their
information is available in the subject rail when you need it.

#### Default state

| Node type | Default | Rationale |
| --- | --- | --- |
| Insight | On | Primary intelligence carrier |
| Topic | On | Conceptual anchor |
| Person | On | Attribution — high value |
| TopicCluster | On | Meta-level grouping |
| Entity | On | Organisation/concept — useful context |
| Quote | Off | Evidence layer — available in subject rail |
| Speaker | Off | Pre-canonical person — superseded by Person |
| Episode | Off | Structural container — available in status bar |

#### Default node-type implementation

In `graphFilters.ts` store, set initial checked state per the table
above. Existing per-type checkbox UI in the toolbar is unchanged —
users can turn any type on/off freely. This only changes the defaults.

#### Visual indicator

When any non-default type visibility is chosen (vs graph defaults), a
**filters active — reset** chip (**muted**) appears next to the **Types**
heading in the top toolbar. Clicking it resets all types to defaults.

`data-testid="graph-types-reset"`

---

### 7. Interaction with RFC-076 Progressive Expand

The episode cap and `graphLens` define the **starting graph**. RFC-076
expand is the mechanism for growing it.

When the user double-clicks an eligible node:

- `POST /api/corpus/node-episodes` returns additional episode paths for
  that identity

- Those paths are appended to the merged graph regardless of `graphLens`
  window — expand deliberately goes beyond the current lens

- The status line updates: episode count and node count increase
- The expanded node shows the blue ring (already in RFC-076 spec)
- The status line does NOT show "(capped)" for expanded content — the
  cap only applies to the initial auto-load

When `graphLens` is changed (user selects new window from lens selector):

- All RFC-076 expanded content is cleared
- Fresh load from new window with cap applied
- This is expected — changing the lens is a deliberate context switch

Any other **full graph reload** from the artifacts store (Digest/Library handoff to graph, Dashboard **Load into graph**, corpus sync, etc.) uses the same rule: **`loadSelected()`** resets RFC-076 expansion by default. Only **append/remove** reload paths preserve expansion so expand/collapse stay consistent (see RFC-076 *Expansion reset vs full reload*).

---

### 8. Interaction with Gesture Overlay

The gesture overlay (separate spec) teaches the dbl-click expand gesture.
The status line complements it with persistent context — "15 episodes
loaded" tells the user the graph is a slice, not the whole corpus, and
that there's more to explore via expand.

The overlay's "Dbl-click → Load more episodes for this node" hint
directly addresses the affordance the status line implies. The two
features are designed to work together.

---

### 9. Files to Touch

#### Initial load — modified files

```text

web/gi-kg-viewer/src/stores/graphFilters.ts
  — update default node type visibility (Quote, Speaker, Episode OFF)

web/gi-kg-viewer/src/stores/artifacts.ts (or graph loading path)
  — apply graphLens filter + episode cap before loading GI/KG paths

web/gi-kg-viewer/src/components/graph/GraphCanvas.vue
  — embed GraphStatusLine in GraphBottomBar (centre zone under canvas)
  — wire graphLens changes to reload trigger

web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md
  — add graph-status-line, lens selector, episode count, node count

```

#### Initial load — new component

```text

web/gi-kg-viewer/src/components/graph/GraphStatusLine.vue
  — status line row (lens selector, counts, since input); embedded in bottom bar under canvas

```

#### Initial load — UXS amendment (after implementation)

```text

docs/uxs/UXS-004-graph-exploration.md
  — Document graphLens, episode cap, status line, default type visibility
  — Add graphLens lens selector to toolbar description

docs/uxs/UXS-001-gi-kg-viewer.md
  — Add graph default episode cap to Tunable parameters table

```

---

### 10. Checkpoints

**Checkpoint 1 — graphLens store**

- `graphLens` exists in store, seeds from `corpusLens` on first Graph tab
  open, resets on corpus path change

- Changing `corpusLens` in Digest/Library does not affect `graphLens`
  after seed

**Checkpoint 2 — Episode cap**

- Graph loads at most 15 episodes from `graphLens` window (scored selection, not pure recency)
- "(capped)" indicator appears when pool was truncated
- "all time" lens still caps at 15 **highest-scoring** episodes across the corpus (not the entire corpus)

**Checkpoint 3 — Status line**

- Status line shows correct episode count, node count, active lens
- Lens selector changes `graphLens` and triggers reload
- Counts update after RFC-076 expand

**Checkpoint 4 — Default type visibility**

- Fresh graph load: Quote, Speaker, Episode nodes hidden
- Types toolbar checkboxes reflect defaults correctly
- "filters active" chip appears when user deviates from defaults
- Reset chip restores defaults

---

### 11. Open Questions (Not Blocking)

1. **Publish date on artifact listing** — If `GET /api/artifacts` does
   not include publish date, the lens filter requires loading metadata

   first. Check API response shape; if missing, add it as a follow-up
   API improvement (not blocking — can filter post-parse for now).

2. **Re-seed behaviour** — If user changes `corpusLens` in Digest to
   "30d" and then opens Graph (which has never been opened this session),

   should Graph seed to "30d"? Current spec says yes. If user has already
   opened Graph this session, no re-seed. This is the correct behaviour
   but worth validating during implementation.

3. **Cap value** — 15 is the starting point. After implementation, check
   canvas density with a real corpus. May want to increase to 20 or

   decrease to 10 depending on average node count per episode.

---

## Graph handoff orchestrator

**Status:** Active — see [ADR-094](../adr/ADR-094-graph-handoff-orchestrator-fsm.md)
for the locked decisions and rationale.
**Repo:** `chipi/podcast_scraper`
**Target area:** `web/gi-kg-viewer/src/services/graphHandoffFsm.ts`,
`stores/graphHandoff.ts`, every entry-point component (Library / Digest /
Search / Dashboard / Episode panel / NodeDetail / GraphConnectionsSection /
SubjectRail / StatusBar / Explore / App-shell tab / GraphCanvas direct
interactions).
**Related docs:** [ADR-094](../adr/ADR-094-graph-handoff-orchestrator-fsm.md)
(canonical decisions), [VIEWER_ASYNC_STABILITY.md](VIEWER_ASYNC_STABILITY.md)
(adjacent async patterns),
`web/gi-kg-viewer/src/services/README.md`
(code-local FSM contract),
`web/gi-kg-viewer/e2e/HANDOFF_MATRIX.md`
(coverage contract).
**Scope:** Operational reference for adding new entry surfaces and debugging
handoff bugs. The decisions and rationale live in ADR-094; this section is
"how the orchestrator integrates with the rest of the graph viewer."

---

### Why the orchestrator exists

Pre-fix, navigating to graph from any of 13 entry surfaces (Library row,
Digest pill, Search "Show on graph", Dashboard topic landscape, Episode
panel, NodeDetail Load, GraphConnections neighbour, SubjectRail @go-graph,
StatusBar @go-graph, Explore focus, App-shell tab, plus canvas direct
interactions) mutated some combination of three Pinia stores (`subject`,
`graphNavigation`, `artifacts`) in slightly different orders. No single
owner enforced the invariant that `filteredArtifact` containing node N
implies `cy.core.$id(N).nonempty()` before focus / camera asserts. The
canonical user-visible bug was "second Library G does nothing" — a load-source
asymmetry let the `filteredArtifact` watcher early-return without scheduling
a redraw. See
[docs/wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md](../wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md)
for the full audit.

The fix: a finite-state machine that owns the handoff lifecycle from
click to settled selection.

---

### Entry-point catalog (compressed)

Reference for each surface's FSM event + envelope shape. Full per-row test
coverage in
`web/gi-kg-viewer/e2e/HANDOFF_MATRIX.md`.

| ID | Surface | FSM event | source | loadSource | camera |
| --- | --- | --- | --- | --- | --- |
| L1 | Library row "Open in graph" | `handoffRequested` | `library` | `subject-external` | `center-on-target` |
| D1 | Digest recent topic pill | `handoffRequested` | `digest` | `digest-external` | `center-on-target` |
| D2 | Digest topic-band hit row | `handoffRequested` | `digest` | `digest-external` | `center` (topicFocus / eid) |
| D3 | Digest topic-band title | `handoffRequested` | `digest` | `digest-external` | `fit` (multi-episode) |
| S1 | Search "Show on graph" | `handoffRequested` (dispatched directly from `SearchPanel.onFocusHit`) | `search` | `subject-external` | `center-on-target` |
| E1 | Episode panel "Open in graph" | `handoffRequested` | `episode-panel` | `subject-external` | `center-on-target` |
| O3 | NodeDetail Load + neighbour go-graph | `expansionRequested` (dispatched directly from `NodeDetail.focusTopicClusterMember` / `focusNeighborOnGraph`) | `node-detail` | `graph-internal` | `center-on-target` |
| O1/O2/O4–O6 | Dashboard / Explore / SubjectRail / StatusBar @go-graph | `handoffRequested` (via `activateGraphTab(source: …)`) | (per surface) | `subject-external` | `center-on-target` |
| G1/G2 | Canvas single-tap (incl. onetap rail) | `canvasTapped` | `canvas-tap` | `graph-internal` | `center` (cyId) |
| G3 | Canvas double-tap expand | `expansionRequested` | `double-tap-expand` | `graph-internal` | `preserve` |
| G6 | Mini-map / GraphConnections neighbour | `canvasTapped` (with `suppressCamera: true`) | `minimap` | `graph-internal` | `preserve` |
| K1 | Escape (clear graph focus) | `focusCleared` | — | — | — |
| — | Corpus path change | `corpusReloaded` | — | — | — |
| — | First mount with saved preference | internal init `handoffRequested({ source: 'restore-preference' })` | `restore-preference` | `subject-external` | `center` |

The `loadSource` enum has 3 values (decision #2): `subject-external` for
non-graph surfaces opening a single subject, `digest-external` for Digest
paths, `graph-internal` for in-graph expansions and direct interactions
(Definition X — graph-internal = "expansion that preserves layout").

The `camera` strategy has 5 variants
(decision #11): `center` (explicit cyId + optional bbox includes),
`center-on-target` (resolve cyId during apply phase from envelope's
`cyId | metadataPath | episodeId`), `fit` (multi-node loads), `preserve`
(canvas tap, mini-map), `none` (explicit no-op).

---

### State-walking lifecycle

A typical Library "Open in graph" handoff walks all 8 states:

```text
idle / ready
   │
   │ handoffRequested(envelope)
   ▼
loading_fetch                         (territory auto-load HTTP starts)
   │
   │ advanceState (in loadEpisodeSliceForTerritoryStrip)
   ▼
loading_merge                         (artifacts.appendRelativeArtifacts)
   │
   │ filteredArtifact watcher → scheduleRedraw → runRelayout
   │ runRelayout: graphHandoff.advanceState('redrawing_full')
   ▼
redrawing_full                        (cy layout running)
   │
   │ lo.one('layoutstop'): graphHandoff.notifyLayoutStop()
   ▼
applying                              (finishLayoutPass: select + dim + camera)
   │
   │ end of finishLayoutPass: graphHandoff.recordApplied(cyId)
   ▼
ready                                 (stuck timer disarmed; lastResult = 'applied')
```

For canvas direct interactions (G1/G2/G3/G6), the FSM jumps `idle/ready →
applying → ready` without going through `loading_*` (no artifacts to
fetch, no layout to run for a tap).

For canvas double-tap expand (G3), the FSM walks `loading_fetch →
loading_merge → redrawing_incremental → applying → ready` (incremental
layout preserves positions of unchanged nodes).

---

### Generation-token check points (the bare-await contract)

Every async `await` inside orchestrator code (notably
`loadEpisodeSliceForTerritoryStrip` in
`web/gi-kg-viewer/src/components/graph/GraphCanvas.vue`)
**must** be paired with an `isStale(envelope.generation)` check on the
next non-empty line. Without it, an in-flight handoff can mutate UI state
for an envelope that has already been superseded by a newer click —
producing the exact "old episode wins after rapid clicks" race the FSM is
designed to prevent.

The 8 documented check points are listed in the
`web/gi-kg-viewer/src/services/graphHandoffFsm.ts` header.
Adding a new `await` to an orchestrator path requires a matching
`isStale()` check; the contract is currently review-only (no automated
ESLint rule yet).

---

### Adding a new entry surface (recipe)

1. **Build the envelope** with the right `kind` / `cyId | metadataPath |
   episodeId` / `source` / `loadSource` / `camera` (see catalog above for
   the right combination per surface type).
2. **Fire the right FSM event** at click time:
   - cross-surface handoff (most cases) → `graphHandoff.handoffRequested(envelope)`
   - canvas direct selection → `graphHandoff.canvasTapped(envelope)`
   - additive expansion (load more on existing graph) → `graphHandoff.expansionRequested(envelope)`
3. **Synchronously set the load-source** on `artifacts`
   (`artifacts.setLoadSource(envelope.loadSource)`). Decision #1: this
   prevents the canonical "second G does nothing" bug class.
4. **Add a row to**
   `web/gi-kg-viewer/e2e/HANDOFF_MATRIX.md`
   under the right section (cold-start / hot-state / repeat-click /
   cross-entry / concurrency / failure / lifecycle).
5. **Write a Playwright contract test** asserting the surface fires the
   right event with the right envelope payload (see existing examples in
   `web/gi-kg-viewer/e2e/handoff/contracts.spec.ts`). This mechanically
   enforces the migration so a future refactor can't silently undo it.

---

### Tab-switch during in-flight handoff — institutional contract

**The rule:** when `GraphCanvas.onActivated` fires and `graphHandoff.pending`
is non-null with FSM in any in-flight state (`loading_fetch` / `loading_bootstrap` /
`loading_merge` / `redrawing_*` / `applying`), the helper
`tryApplyPendingFsmEnvelopeFromTabReturn(cy)` must drive the FSM forward to
`applied` (when the target resolves in cy) or leave it for the stuck-timer
to surface as `failed`.

**Why this exists:** without the helper, an L1 / E1 / D1 click followed by an
immediate tab-switch leaves the FSM in `loading_fetch` with no path to
terminal — the user returns to the Graph tab to find an empty canvas + a
15-second-delayed "stuck-timeout" error strip. The natural redraw-driven
apply path (`finishLayoutPass` → `recordApplied`) only fires after a real
`layoutstop`, which may not happen if the load completed while the tab was
inactive.

**Don't regress:** if you refactor `GraphCanvas.onActivated`, preserve the
call to `tryApplyPendingFsmEnvelopeFromTabReturn` on every activation where
`graphHandoff.pending` is non-null. The Tier-3 P5.2 spec at
`web/gi-kg-viewer/e2e/validation/handoff-matrix-real-corpus.spec.ts` catches
removal of this hook end-to-end.

**Related fix in click handlers:** the L1 (`LibraryView.openEpisodeInGraph`)
and D1 (`DigestView.openDigestRecentTopicPillInGraph`) handlers also call
`artifacts.loadSelected({ preserveExpansion: true })` after
`appendRelativeArtifacts` to force a redraw when the target's artifacts are
already in the store (so the append is a no-op). Without this, two
in-rapid-succession handoffs targeting the same episode stuck-timeout.
Tier-2 `e2e/handoff-production/cross-entry.spec.ts::P2.5` pins this
regression deterministically; Tier-3 `handoff-matrix-real-corpus.spec.ts::P2.5/P2.6`
covers it end-to-end.

---

### Debugging

- **Inspect the FSM in dev**: `window.__GIKG_FSM__` exposes
  `{ state, pending, generation, lastResult }` (read-only). Useful in DevTools
  console while reproducing a stuck handoff.
- **Inspect Cytoscape directly**: `window.__GIKG_CY_DEV__` (existing
  pattern, predates the FSM). Combine with `__GIKG_FSM__` to see whether
  the FSM thinks the handoff completed but cy.core doesn't have the
  expected node — that's a self-healing reconciliation case.
- **Read error-strip messages**: failed handoffs render
  `data-testid="handoff-error-strip"` with reason. Stuck-timeout failures
  carry `reason: "stuck-timeout after 15000ms"`; HTTP failures carry the
  `Error.message` of the originating fetch.
- **Self-healing warnings**: invariant violations log
  `[graphHandoff invariant] expected nodes missing from cy core: count=N
  sample=...` on every `layoutstop`. After a targeted reconciliation:
  `reconciled missing=N via targeted core.add (retry budget consumed for
  gen=G)`. After exhausted retry: `divergence accepted: missing=N retried=true`.
- **PostHog telemetry**: `graph_handoff_started/applied/failed/superseded/stuck`
  events fire from `stores/graphHandoff.ts`. Use to track handoff success
  rates per source in production.

---

### Matrix assertion layers (L0–L6)

The 41-row `web/gi-kg-viewer/e2e/HANDOFF_MATRIX.md` drives the 6-point
standard contract through six composable assertion layers. Each row asserts as deeply as its fixture allows; the layer profile
per row is recorded in the matrix doc's "Layers" column.

| Layer | What it asserts | Hook | Helper |
| --- | --- | --- | --- |
| **L0** | API mocks return expected shape | n/a | `setupHandoffMatrixMocks` |
| **L1** | FSM `ready` + `pending=null` + `lastResult.status=applied` + generation bumped | `__GIKG_FSM__` | `readFsmState` |
| **L2** | Exactly 1 cy node selected with id matching expected (modulo `g:`/`k:`/`__unified_ep__:` prefix) | `__GIKG_CY_DEV__` | `assertHandoffApplied` |
| **L3** | Camera zoom in `[0.2, 5]` AND target node `renderedPosition` within inner `cameraCenterTolerance × viewport` (default 0.35) box; no console errors; Episode panel title (when relevant) | `__GIKG_CY_DEV__` + `console.error` | `assertHandoffApplied` |
| **L4** | FSM event envelope shape: `type` + `source` + `kind` + `loadSource` + `camera.kind` | `__GIKG_FSM_EVENT_LOG__` | `assertFsmEventEnvelope` |
| **L5** | Subject store: `subject.kind` + matching id field reflects target | `__GIKG_SUBJECT__` | `assertHandoffApplied` |
| **L6** | Self-healing invariant: `viewWithEgo(focusNodeId)` ⊖ `core.nodes()` is empty after most recent `finishLayoutPass` | `__GIKG_FSM__.lastInvariant` | `assertHandoffApplied` |

**UI-driven rows** (Library / Digest pill / Search "Show on graph" /
Episode panel / NodeDetail Load / Dashboard topic-cluster chip / hot-state /
repeat / cross-entry / lifecycle tab-reconcile) assert **L0+L1+L2+L3+L5+L6**
through `assertHandoffApplied`. L6 is advisory (accepts `null` when no
layoutstop fired since the handoff — typical when the target was already
in the graph and no redraw was needed).

**Dev-hook-driven rows** (SubjectRail / StatusBar / Mini-map no-target
funnels, composite multi-envelope sequences, failure modes, lifecycle
restore) assert **L0+L1+L4** through `assertFsmEventEnvelope`. They
dispatch envelopes via `__GIKG_HANDOFF_STORE__` and can't reach the
"applied" outcome without a real graph change, so they pin the FSM-event
contract instead of the full outcome contract.

**Filter rows** (Section 8) assert the inverse: no `handoffRequested`
fires and FSM state stays quiescent — filters must not drive graph state.

The **camera-center check** (part of L3) addresses the GH #771 failure
class where `cy.zoom()` is fine but pan misaligned, leaving the user
looking at empty space or wrong nodes. Implementation polls
`node.renderedPosition()` until it stabilizes (2-consecutive identical
reads, cap 2.5 s — waits for `cy.animate` to settle since `recordApplied`
marks `ready` before animation completes), then asserts pixel distance
from viewport center. Skip via `skipCameraCenter: true` for envelopes
with `camera: 'fit'` / `'preserve'`.

The **L6 self-healing invariant** is stashed by `finishLayoutPass` on
every layoutstop (in production, not just dev) via
`graphHandoff.recordInvariant(missing, extra)`. The same predicate drives
production reconciliation: `< 20` missing nodes triggers a targeted
`core.add()`; subsequent layoutstop reruns the invariant; second violation
accepts divergence with a logged warning (single retry budget per
envelope generation).

---

## Graph visual styling

**Status:** Active — implementation tracks [GitHub #608](https://github.com/chipi/podcast_scraper/issues/608).
**Author:** Design session (Marko + Claude), April 2026
**Repo:** `chipi/podcast_scraper`
**Target area:** `web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts`,
`cyGraphLabelTier.ts`, `cyCoseLayoutOptions.ts`, `parsing.ts`, `GraphCanvas.vue`,
`GraphNeighborhoodMiniMap.vue`
**Related docs:** UXS-001, UXS-004, RFC-062
**Scope:** Pure visual — Cytoscape stylesheet (`cyGraphStylesheet.ts`), shared
zoom-tier helper (`cyGraphLabelTier.ts`), COSE options (`cyCoseLayoutOptions.ts`),
node data enrichment (`parsing.ts` → `toCytoElements`), and **main graph wiring**
in `GraphCanvas.vue` (selection dim classes, zoom listener → label tiers,
post-layout Topic degree heat). No store logic changes, no API changes. Works
independently of the Initial Load spec.

---

### 1. Objective

Transform the graph from a data dump into an **intelligence communication
surface**. The visual system should do cognitive work for the user:

- Important things attract the eye before any label is read
- Relationship types are readable at a glance from edge style alone
- The topology communicates corpus structure — topic islands, hubs, recency
- Selection focuses the view — noise recedes, signal comes forward

Styles and parsed node fields live in the files above; **interaction wiring**
(selection dim, zoom tier sync, degree heat batch) lives in **`GraphCanvas.vue`**
(and the Episode subject rail **`GraphNeighborhoodMiniMap.vue`** preview reuses the same
stylesheet + tier sync). No store changes, no API changes.

---

### 2. Phasing

**Phase 1 — Stylesheet (low risk, ship together):**
Node size hierarchy, edge visual language, selection dim, zoom-responsive
labels, TopicCluster compound fill, drop shadows, default edge arrows.
Pure stylesheet values — no data dependencies.

**Phase 2 — Computed properties (medium effort, ship after Phase 1):**
Recency tint, grounding confidence tint, degree heat. These require
node property enrichment at parse time in `parsing.ts`.

Phase 1 is complete and coherent without Phase 2. Phase 2 is additive.

---

### 3. Phase 1 — Stylesheet Changes

#### 3.1 Node size by tier

Nodes render at different base sizes by type. Size = cognitive importance.

| Node type | Diameter | Tier |
| --- | --- | --- |
| Insight | 44px | 1 |
| Topic | 40px | 1 |
| TopicCluster | 48px (compound, **logical** tier — see note below) | 2 |
| Person | 34px | 2 |
| Entity | 26px | 3 |
| Quote | 22px | 3 |
| Speaker | 18px | 4 |
| Episode | 18px | 4 |

Implementation in `cyGraphStylesheet.ts`:

```typescript

// Example — adapt to existing stylesheet structure
{
  selector: 'node[type = "Insight"]',
  style: { width: 44, height: 44 }
},
{
  selector: 'node[type = "Topic"]',
  style: { width: 40, height: 40 }
},
// ... etc

```

**TopicCluster sizing (implementation):** The compound uses **padding** (WIP §3.6)
and Cytoscape’s **auto compound** geometry so member topics lay out correctly.
The stylesheet **does not** force a fixed **48×48** compound width/height; treat
**48px** as the **visual weight / tier** of the cluster chrome, not a hard layout
box.

---

#### 3.2 Edge visual language

Each edge type gets a distinct visual treatment so relationship type
is readable without hovering.

| Edge type | Width | Style | Colour | Arrow |
| --- | --- | --- | --- | --- |
| `HAS_INSIGHT` | 2px | solid | `primary` token | target arrowhead |
| `ABOUT` | 2px | solid | `gi` token | none (undirected) |
| `SUPPORTED_BY` | 1px | dashed | `muted` token | target arrowhead |
| `RELATED_TO` | 1px | solid | `kg` token | none |
| `MENTIONS` | 1px | dotted | `muted` token | none |
| `SPOKE_IN` | 2px | solid | `primary` token | target arrowhead |
| `HAS_MEMBER` | 1.5px | solid | `kg` at 60% opacity | none |

**Opacity:** All edges at 60% opacity base. Selected neighbourhood
edges at 90%. Unrelated edges (when something is selected) at 20%.

**Curve style:** `bezier` for all edges. Prevents straight lines
overlapping node borders at dense cluster areas.

**Label:** No edge labels by default. Edge type is communicated by
visual style alone. Labels add too much canvas noise.

Cytoscape edge data uses **`edgeType`** (see `toCytoElements` in `parsing.ts`), not `type`. Example selectors:

```typescript

{
  selector: 'edge[edgeType = "ABOUT"]',
  style: {
    width: 2,
    'line-color': 'var(--ps-gi)',
    'line-style': 'solid',
    'target-arrow-shape': 'none',
    opacity: 0.6,
    'curve-style': 'bezier'
  }
},
{
  selector: 'edge[edgeType = "SUPPORTED_BY"]',
  style: {
    width: 1,
    'line-color': 'var(--ps-muted)',
    'line-style': 'dashed',
    'target-arrow-shape': 'triangle',
    'target-arrow-color': 'var(--ps-muted)',
    opacity: 0.6,
    'curve-style': 'bezier'
  }
},
// ... per known edgeType; explicit `edge[edgeType = "(unknown)"]` plus generic `edge` base

```

---

#### 3.3 Selection focus — dim unrelated nodes

When a node is selected, unrelated nodes dim to **40%** opacity (times
**`recencyWeight`** when present — see §4.1). The **focused** selected node uses
**full** opacity × recency. **1-hop neighbours** (closed neighbourhood) use
**~85%** opacity × recency in the shipped stylesheet so they read slightly softer
than the focus node while still clearly in the in-group (Cytoscape classes
**`graph-neighbour`**).

Stylesheet rules use prefixed classes on **nodes** and **edges** (e.g. `graph-dimmed`, `graph-focused`, `graph-neighbour`, `graph-edge-dimmed`, `graph-edge-neighbour`) so they do not clash with other graph classes.

Wire in `GraphCanvas.vue` on Cytoscape `select` and `unselect` events (and clear on destroy).

**Single selection:** The viewer interaction model keeps **at most one** selected
graph node at a time (tap handlers clear others). Dimming logic assumes a single
**focused** node; **multi-select is not supported** and is not a target scenario.

Example:

```typescript

cy.on('select', 'node', (e) => {
  const node = e.target
  cy.nodes().addClass('graph-dimmed')
  cy.edges().addClass('graph-edge-dimmed')
  node.addClass('graph-focused').removeClass('graph-dimmed')
  node.closedNeighborhood().nodes().addClass('graph-neighbour').removeClass('graph-dimmed')
  node.closedNeighborhood().edges().addClass('graph-edge-neighbour').removeClass('graph-edge-dimmed')
})

cy.on('unselect', 'node', () => {
  cy.nodes().removeClass('graph-dimmed graph-focused graph-neighbour')
  cy.edges().removeClass('graph-edge-dimmed graph-edge-neighbour')
})

```

Transition: add `transition-opacity 0.15s` via Cytoscape transition
properties so the dim/brighten animates smoothly rather than jumping.

`prefers-reduced-motion`: skip transition, apply instantly.

---

#### 3.4 Drop shadows on Tier 1 nodes

Insight and Topic nodes get a subtle drop shadow — creates visual lift,
signals importance tier without changing colour.

```typescript

{
  selector: 'node[type = "Insight"], node[type = "Topic"]',
  style: {
    'shadow-blur': 8,
    'shadow-color': 'var(--ps-border)',
    'shadow-offset-x': 0,
    'shadow-offset-y': 2,
    'shadow-opacity': 0.6
  }
}

```

Tier 2 (Person, TopicCluster): no shadow.
Tier 3/4: no shadow.

---

#### 3.5 Zoom-responsive labels

Labels show/hide and truncate based on zoom level. Cytoscape **does not** support CSS-like `min-zoom` / `max-zoom` stylesheet selectors on graph elements.

**Implementation:** `GraphCanvas.vue` listens to the existing Cytoscape **`zoom`** event and assigns **one** of three mutually exclusive class families on every node (e.g. `graph-label-tier-none`, `graph-label-tier-short`, `graph-label-tier-full`). The stylesheet maps those classes to `label` / `text-opacity` (Tier 1+2 use `data(shortLabel)` in the short tier; low tiers hide text in that band). Thresholds match the table below.

**shortLabel** — add to node data at parse time in `parsing.ts`:

```typescript

shortLabel: label.length > 18 ? label.slice(0, 16) + '…' : label

```

**Label placement (main graph today):** The default **merged graph** uses **side**
labels (`text-halign: center` + dynamic `text-margin-x` from
`cytoscapeSideLabelMarginXCallback` in `GraphCanvas.vue`’s `buildCyStyle`), with
canvas-tinted halo styles from the shared stylesheet. **`buildGiKgCyStylesheet`**
also supports **`above`** / **`below`** for previews or future shell changes.
UXS-004 describes the halo requirement; **default horizontal placement is side**,
not above the disc.

**Zoom thresholds:**

| Zoom | Visible labels |
| --- | --- |
| < 0.5 | None |
| 0.5 – 1.0 | Tier 1 + 2 only, shortLabel |
| > 1.0 | All visible nodes, full label |

---

#### 3.6 TopicCluster compound fill tint

TopicCluster compound parent nodes get a very faint `kg` token fill —
defines cluster territory visually without dominating the canvas.

```typescript

{
  selector: 'node[type = "TopicCluster"]',
  style: {
    'background-color': 'var(--ps-kg)',
    'background-opacity': 0.06,  // very faint — territory, not colour
    'border-style': 'dashed',
    'border-color': 'var(--ps-kg)',
    'border-opacity': 0.4,
    'border-width': 1.5,
    padding: '18px'  // space between compound border and member nodes
  }
}

```

The compound outline was already dashed per UXS-004 — this adds the
faint fill tint and ensures the padding gives member nodes breathing room.

---

#### 3.7 COSE layout parameter tuning

Tune spring strength by edge type to produce semantic gravity —
Insights cluster around their Topics, Quotes stay close to their
Insights without pulling them off cluster.

In the COSE layout config (wherever it is called in `GraphCanvas.vue`
or `graphExplorer.ts`):

```typescript

const layoutOptions = {
  name: 'cose',
  // ... existing options
  idealEdgeLength: (edge) => {
    switch (edge.data('edgeType')) {
      case 'HAS_INSIGHT': return 60   // episode–insight anchor (also scaled in compact profile)
      case 'ABOUT':      return 80   // Insight close to Topic
      case 'SUPPORTED_BY': return 40 // Quote close to Insight
      case 'RELATED_TO': return 120  // Topics spread laterally
      case 'SPOKE_IN':   return 100  // Person near but not inside cluster
      case 'MENTIONS':   return 150  // Loose reference — far
      default:           return 100
    }
  },
  edgeElasticity: (edge) => {
    switch (edge.data('edgeType')) {
      case 'HAS_INSIGHT': return 180
      case 'ABOUT':      return 200  // strong spring
      case 'SUPPORTED_BY': return 150
      case 'RELATED_TO': return 100
      case 'SPOKE_IN':   return 120
      case 'MENTIONS':   return 60   // weak spring
      default:           return 100
    }
  }
}

```

These values are starting points — visual testing with a real corpus
will require tuning. Add them to UXS-001 tunable parameters table as
Open (not Frozen) so they can be adjusted without a UXS revision.

---

### 4. Phase 2 — Computed Property Tints

Phase 2 adds three visual dimensions that require node property data
computed at parse time. All three enrich the node `data()` object in
`parsing.ts` and are then read by the stylesheet.

#### 4.1 Recency tint

Nodes from more recent episodes render at higher opacity / saturation.
Older nodes are muted.

**At parse time** (`parsing.ts`), for each node compute:

```typescript

// recencyWeight: 1.0 = current week, 0.4 = 90+ days ago
const daysSince = (Date.now() - episodePublishDate) / 86400000
const recencyWeight = Math.max(0.4, 1.0 - (daysSince / 90) * 0.6)
node.data.recencyWeight = recencyWeight

```

**`recencyWeight`** with **selection-dim** factors (dimmed **0.4×**,
neighbour **0.85×**, focused or default **1×**). Cytoscape could map
`opacity: 'data(recencyWeight)'` alone, but that would not compose with §3.3;
the callback keeps one combined path. **`recencyWeight`** is clamped to
**[0.4, 1]** in `parsing.ts` and the same floor is applied when reading the
field on the node so bad values cannot drop below **0.4**.

Requires episode publish date to be available on the parsed node when
recency should vary; otherwise **`recencyWeight` defaults to 1.0** (see
`toCytoElements`).

---

#### 4.2 Grounding confidence tint — Insight nodes only

Insight nodes with high confidence score render at full `gi` colour.
Low-confidence Insights are slightly desaturated — a subtle signal to
treat them with more scepticism.

**At parse time:**

```typescript

// confidence: 0.0 – 1.0 from GI schema
// confidenceOpacity: maps to fill opacity
const confidence = node.properties?.confidence ?? 0.7
node.data.confidenceOpacity = 0.5 + confidence * 0.5  // range 0.5–1.0

```

```typescript

{
  selector: 'node[type = "Insight"]',
  style: {
    'background-opacity': 'data(confidenceOpacity)'
  }
}

```

When `confidence` is absent (field not populated), default to 0.7
(neutral — neither highlights nor suppresses).

---

#### 4.3 Degree heat — Topic nodes only

High-degree Topic nodes (many connections) get a slightly warmer, more
prominent visual treatment — they are the conceptual hubs of the corpus.

**At parse time** (or post-layout in `GraphCanvas.vue` after Cytoscape
has computed degree):

```typescript

// degree: number of connected edges
// Cap normalisation to top of expected range
const degree = cy.$(`#${nodeId}`).degree()
const maxDegree = 30  // tune based on real corpus
const heat = Math.min(1.0, degree / maxDegree)
cy.$(`#${nodeId}`).data('degreeHeat', heat)

```

**In stylesheet — two effects:**

1. Border width scales with heat (thicker border = higher degree). **No border**
   when **`degreeHeat`** is **0** (isolates / low connectivity in the current slice)

   so the Topic disc does not compete with **RFC-076** rings or **search-hit**;
   ramp from **1px** once heat is positive, up to **4px** at heat **1.0** (main profile):

```typescript

{
  selector: 'node[type = "Topic"]',
  style: {
    'border-width': (ele) => {
      const h = Number(ele.data('degreeHeat'))
      if (!Number.isFinite(h) || h <= 0) return 0
      return 1 + Math.min(1, h) * 3  // 0, then ~1px–4px
    }
  }
}

```

```typescript

{
  selector: 'node[type = "Topic"][?highDegree]',  // flag set at parse
  style: {
    'shadow-blur': 12,
    'shadow-color': 'var(--ps-kg)',
    'shadow-opacity': 0.5
  }
}

```

**Note:** Degree heat should be computed post-layout (after COSE runs),
not at parse time, since degree depends on which nodes are currently
in the merged graph. Wire it as a post-layout callback in `GraphCanvas.vue`.

---

### 5. Minimap Contrast

The minimap (lower-left, already in spec) becomes more useful as a
navigation aid if Tier 1 nodes render with stronger contrast in it.

In Cytoscape, the minimap renders the same stylesheet. No special minimap
styling is needed — the size hierarchy and shadow from Phase 1 already
make Insight and Topic nodes more visible in the minimap than Quote and
Episode nodes. No additional change required.

However: ensure minimap background uses `canvas` token so the faint
TopicCluster compound fill tint is visible in the minimap. If minimap
background is currently hardcoded, update to use the CSS variable.

---

### 6. Light Mode Considerations

All colour values should use CSS variables (`var(--ps-gi)`,
`var(--ps-kg)`, etc.) not hardcoded hex. `cyGraphStylesheet.ts` likely
already does this for domain colours — extend to all new properties.

The `text-background-color` for label halos should use `var(--ps-canvas)`
so it matches the page background in both light and dark mode.

Drop shadow colours (`rgba(0,0,0,0.4)`) work in dark mode but are too
strong in light mode where the canvas is light. Use:

```typescript

'shadow-color': 'var(--ps-border)'  // adapts to mode

```

---

### 7. Files to Touch

#### Visual styling — modified files

```text

  — All Phase 1 and Phase 2 stylesheet rules

web/gi-kg-viewer/src/utils/parsing.ts
  — shortLabel generation (Phase 1)
  — recencyWeight computation (Phase 2)
  — confidenceOpacity computation (Phase 2)

web/gi-kg-viewer/src/components/graph/GraphCanvas.vue
  — Selection dim/focus class application (Phase 1)
  — Post-layout degree heat computation (Phase 2)
  — COSE layout parameter update (Phase 1)

web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md
  — Note stylesheet changes (no new selectors needed for E2E;
    these are visual-only changes)

```

#### Visual styling — UXS amendment (after implementation)

```text

  — Add section: ## Selection focus behaviour
  — Update Toolbar section: COSE parameters noted as tunable
  — Note Phase 1 vs Phase 2 separation

docs/uxs/UXS-001-gi-kg-viewer.md
  — Add COSE spring parameters to Tunable parameters table (Open)
  — Add degree heat maxDegree to Tunable parameters table (Open)
  — Add recency decay window (90 days) to Tunable parameters table (Open)

```

---

### 8. Tunable Parameters Added to UXS-001

| Parameter | Default | Status | Notes |
| --- | --- | --- | --- |
| COSE ABOUT edge ideal length | 80px | Open | Tune with real corpus |
| COSE MENTIONS edge ideal length | 150px | Open | Tune with real corpus |
| Recency decay window | 90 days | Open | Full decay at 90 days |
| Recency minimum opacity | 0.4 | Open | Floor for oldest nodes |
| Degree heat max degree | 30 | Open | Normalisation ceiling |
| Label zoom threshold (none) | 0.5 | Open | Below = no labels |
| Label zoom threshold (full) | 1.0 | Open | Above = full labels |
| Compound fill opacity | 0.06 | Open | Very faint territory tint |

---

### 9. Phase 1 Checkpoints

1. Node sizes reflect tier table — Insight largest, Episode/Speaker smallest
2. Edge types visually distinct — ABOUT solid green, SUPPORTED_BY dashed muted,
   SPOKE_IN solid primary with arrowhead

3. Selecting a node dims unrelated nodes to 40% opacity; deselect restores
4. Tier 1 nodes have drop shadow; Tier 3/4 have none
5. Labels hidden at zoom < 0.5; Tier 1/2 short labels at 0.5–1.0; full at > 1.0
6. TopicCluster compound has faint kg fill tint
7. Light mode and dark mode both render correctly (CSS variables used throughout)

### 10. Phase 2 Checkpoints

1. Nodes from last 7 days at full opacity; 90+ day nodes at ~40% opacity
2. High-confidence Insights brighter; low-confidence slightly faded
3. High-degree Topics have thicker border than low-degree Topics
4. Post-layout callback runs degree heat computation without blocking layout
5. All computed properties fall back gracefully when source data is absent

---

### 11. What This Does Not Change

- Node shapes (already defined in existing stylesheet)
- GI/KG domain token colours (gi green, kg purple) — Phase 2 modulates
  within these colours but does not change them

- RFC-076 ring colours (teal = expandable, blue = expanded) — untouched
- TopicCluster compound outline style (dashed) — Phase 1 only adds fill
- Any store logic, API calls, or component behaviour
- The gesture overlay (separate spec)
- The initial load strategy (separate spec)

---

## Graph gesture overlay

**Status:** Implemented in viewer (keep doc in sync with code when behaviour changes)
**Author:** Design session (Marko + Claude), April 2026
**Repo:** `chipi/podcast_scraper`
**Target area:** `web/gi-kg-viewer/src/components/graph/`
**Related docs:** UXS-004, RFC-062, [RFC-076](../rfc/RFC-076-progressive-graph-expansion.md) (progressive expand / ring semantics)
**Scope:** Small, self-contained. No store changes. No API changes.

---

### Gesture overlay — problem statement

The graph has 5 non-obvious interaction gestures and 2 visual ring cues that
were easy to miss when they only appeared in a single line of small
`text-[10px]` copy on the top toolbar row.

Current gesture inventory a user needs to learn:

| Gesture | Effect |
| --- | --- |
| Single click (`onetap`) | Open subject rail for that node |
| Shift + double-click | 1-hop ego / neighbourhood expand |
| Shift + drag | Box zoom / selection |
| Plain double-click (`dbltap`) | Progressive expand — loads more episodes for eligible nodes (RFC-076) |
| Second plain double-click | Collapse that expansion |

Visual ring cues:

| Ring colour | Meaning |
| --- | --- |
| Teal border | Node is expandable (has episodes not yet in the graph) |
| Blue border | Node is currently expanded |

None of these are in a menu. None are labelled on the canvas. A developer
opening the graph for the first time will discover single-click and pan/zoom
immediately (mouse-native), but will miss neighbourhood expand, progressive
expand, and the ring semantics entirely unless they open the gesture overlay
(**Gestures** / first visit) or stumble on them by accident.

---

### 2. Solution

A **one-time dismissible gesture hint overlay** that appears on the graph
canvas the **first time per browser profile** the user sees a non-empty merged
graph (`localStorage` key `ps_graph_hints_seen` unset). After dismissal the
flag is set to `1` and the overlay does not auto-open again until storage is
cleared.

**Dismiss:** **Got it**, **backdrop** click outside the card (`@click.self` on
the overlay root; the card uses `@click.stop` so row clicks do not dismiss),
and **Escape** as the primary keyboard dismiss. Do **not** treat arbitrary
keypresses or canvas drags as dismiss (avoids accidental close and matches
graph chrome focus rules).

This is a standard pattern for non-obvious gestures in data tools (Grafana
new feature hints, Figma gesture tours). The goal is one moment of teaching,
not persistent UI noise.

---

### 3. Visual Spec

#### Overlay container

- Positioned: `absolute inset-0` over the graph canvas host
  (`.graph-canvas` or the same `overflow-hidden` region as the minimap)

- Background: `rgba` of `canvas` token at ~60% opacity —
  `color-mix(in srgb, var(--ps-canvas) 65%, transparent)` when supported; otherwise a neutral translucent veil (see scoped `.ps-gesture-overlay-root` in `GraphGestureOverlay.vue`)

- `z-index`: above the main **`.graph-canvas`** Cytoscape layer; **below** the minimap panel when it is open (minimap uses a higher stacking order inside the same canvas host). **`GraphBottomBar`** is mounted **below** `canvasHost`, so it is **not** covered by this overlay and stays usable while the gesture card is open.

- The dimmed backdrop covers only the **canvas host** region (not the bottom bar)

#### Hint card

- Centered in the overlay: implementation uses **flex** on the backdrop root (`items-center justify-center`); an equivalent pattern is `absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2` on the card
- Background: `elevated` token
- Border: `border` token, `rounded-sm` (UXS-001 radius)
- Padding: `p-4`
- Max width: `20rem` — stays compact, doesn't fill the canvas
- Shadow: subtle (`shadow-md`) — gives depth against the dimmed canvas

#### Card header

- Text: **"Graph gestures"**
- Scale: `text-sm font-semibold`, `surface-foreground` token
- Bottom margin: `mb-3`

#### Gesture rows

Five rows, one per gesture. Layout per row:

```text

(icon | label | effect) — horizontal row; see bullets below

```

- Single click → `MousePointer`
- Shift + double-click → `Expand` or `Network`
- Shift + drag → `Scan` or `BoxSelect`
- Double-click → `PlusCircle`
- Second double-click → `MinusCircle`
- Gesture label: `text-xs font-mono`, `surface-foreground` — e.g.
  **"Click"**, **"Shift + dbl-click"**, **"Shift + drag"**,

  **"Dbl-click"**, **"Dbl-click again"**

- Effect: `text-xs`, `muted` — short plain language description

Row gap: `gap-2` between rows. Within each row: `gap-3` between icon,
label, effect.

Full row content:

| Icon | Label | Effect |
| --- | --- | --- |
| MousePointer | Click | Open node details |
| Network | Shift + dbl-click | Expand 1-hop neighbourhood |
| BoxSelect | Shift + drag | Box zoom / select |
| PlusCircle | Dbl-click | Load more episodes for this node |
| MinusCircle | Dbl-click again | Collapse loaded episodes |

#### Ring legend

Below the gesture rows, a `border-t border-border mt-3 pt-3` divider,
then two ring legend rows:

```json

[●  blue ring]   Episodes loaded (expanded)

```javascript

- Dot: 10px circle, inline SVG or `w-2.5 h-2.5 rounded-full`
- Teal dot: `border-2` with teal colour matching the RFC-076 expandable
  ring colour (use the same CSS var or value used in `cyGraphStylesheet.ts`)

- Blue dot: `border-2` with the RFC-076 expanded ring colour
- Label: `text-xs`, `muted`

#### Dismiss control

Below the ring legend, `mt-3`, right-aligned:

- Button: **"Got it"** — `text-xs`, `primary` token (same as secondary
  action buttons elsewhere), `px-3 py-1`

- `aria-label`: "Dismiss graph gesture hints"
- `data-testid`: `graph-gesture-overlay-dismiss`

#### Dismiss behaviour

Overlay dismisses when **any** of these occur:

- Clicking **Got it**
- Clicking the **dimmed backdrop** outside the hint card (not clicks on the Cytoscape layer beneath — the dimmed layer receives the hit; card uses `@click.stop`)
- **Escape**, when focus is inside the overlay (e.g. **Got it**) or when focus is on the graph canvas host after open — **not** when focus is in other graph chrome (**⚙** filters popover, **Since** in **`graph-status-line`**, **Fit** / zoom / **Gestures** in **`graph-bottom-bar`**, etc.) so users are not surprised while using those controls

On dismiss: set `localStorage` key `ps_graph_hints_seen` = `"1"`.
The overlay does not auto-open again; users can still use the optional **Gestures** reopen control (Section 7) without clearing `localStorage`.

#### Accessibility

- Overlay root: `data-testid="graph-gesture-overlay"` (backdrop + dim layer); **Escape** containment uses a Vue **template ref** on that root (`contains(activeElement)`), not `document.querySelector`, so multiple graph roots would not cross-wire (only one graph is mounted today).
- Card: `role="dialog"`, `aria-modal="true"`, `aria-labelledby` referencing the **Graph gestures** heading.
- Initial focus: **Got it** after open; on dismiss, return focus to the graph canvas host (`tabindex="-1"`) for keyboard continuity.

#### Motion

Default: hint card fades in over `150ms` (`opacity-0` → `opacity-100`),
using the same transition pattern as existing modals in the codebase.

`prefers-reduced-motion`: skip transition, render static at full opacity
immediately.

---

### 4. Trigger Condition

The overlay renders when **all** of:

1. `mainTab === 'graph'`
2. The merged graph has at least one node (i.e. artifacts are loaded —
   don't show on an empty canvas)

3. `localStorage.getItem('ps_graph_hints_seen')` is null or absent

Check `hasNodes` and `localStorage` when the component mounts and when
`hasNodes` becomes true. If the flag is already set, do not auto-open.

**Do not** auto-show the overlay:

- On the empty "no corpus loaded" graph state
- If the user has previously dismissed it (`localStorage` flag)
- When `hasNodes` is false (filters / ego view with zero nodes)

**Implementation note:** Today the **Graph** tab is the only place `GraphCanvas`
mounts: [`App.vue`](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/src/App.vue) uses `v-if="mainTab === 'graph'"` on
`GraphTabPanel`. The overlay therefore does **not** require a separate `mainTab`
prop unless the shell is refactored (then pass e.g. `isGraphTabActive`).

---

### 5. Component

#### `GraphGestureOverlay.vue`

Location: `web/gi-kg-viewer/src/components/graph/GraphGestureOverlay.vue`

Props: **`{ hasNodes: boolean }` only** (parent derives from Cytoscape node count).

The parent (`GraphCanvas.vue`) passes `hasNodes` and renders the overlay inside
the same `absolute inset-0` canvas host as `.graph-canvas`, with `z-index`
above the Cytoscape layer and **below** minimap / zoom / layout chrome so
chrome stays usable while the overlay is open.

**Sketch (behaviour, not copy-paste):** `visible` is driven by `hasNodes` +
`localStorage` for auto-open; optional `reopen()` sets `visible` without
clearing `localStorage`. **Escape** handling is registered while `visible` and
ignores keys when focus is outside the overlay (graph chrome).

---

### 6. Toolbar Hint Row — Simplify

With the overlay handling first-time teaching, the **top** toolbar row is
**Types** + **⚙** filters only — **no** persistent Shift / double-click / ring prose.
The **search highlight** chip (when applicable) sits in the **bottom bar** centre
zone under the canvas (see [UXS-004 Graph chrome](../uxs/UXS-004-graph-exploration.md#graph-chrome-toolbar-bottom-bar-filters-popover)).

**Requirement:** Removing that copy **requires** either the optional **Gestures**
reopen control (Section 7) or another minimal affordance (e.g. a **?** entry) so
users who dismissed quickly are not stranded without a path back to the legend.

---

### 7. Re-open Affordance

A **Gestures** (or **?** / **Shortcuts**) control in the **graph bottom bar** right zone
(next to **Export PNG**) calls `reopen()` so the card appears **without** clearing
`localStorage` (manual open only; auto-open still suppressed after dismiss).

`data-testid`: `graph-gesture-overlay-reopen`

**Shipped:** persistent gesture prose was removed from the top toolbar in favour of this overlay; **Gestures** in **`GraphBottomBar`** is the deliberate path back to the overlay copy.

---

### 8. Files to Touch

#### Gesture overlay — new component

```text

web/gi-kg-viewer/src/components/graph/GraphGestureOverlay.vue
  — dismissible overlay card + backdrop; localStorage gate

```

#### Gesture overlay — modified files

```text

  — render GraphGestureOverlay inside canvas host
  — pass hasNodes computed prop
  — shorten or remove toolbar gesture hint text

web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md
  — add: graph-gesture-overlay, graph-gesture-overlay-dismiss

```

#### Gesture overlay — UXS amendment (after implementation)

```text

  — Add section: ## Gesture discovery overlay
  — Reference visual spec (tokens, layout, dismiss behaviour)
  — Note toolbar hint row simplified after overlay ships
  — Note localStorage key: ps_graph_hints_seen

```

---

### 9. E2E Contract

Add to `e2e/E2E_SURFACE_MAP.md`:

| Surface | Selector / role | Notes |
| --- | --- | --- |
| Overlay container | `data-testid="graph-gesture-overlay"` | Present on first graph load with nodes |
| Dismiss button | `data-testid="graph-gesture-overlay-dismiss"` | Click to dismiss |
| Re-open button | `data-testid="graph-gesture-overlay-reopen"` | **Gestures** in **`graph-bottom-bar`** right zone |

Playwright spec: `e2e/graph-gesture-overlay.spec.ts`

**Vitest:** The viewer package does not ship `@vue/test-utils`; keep this overlay locked with Playwright for v1 rather than adding an SFC unit harness.

Test cases:

1. Overlay appears when graph has nodes and localStorage flag is absent
2. Clicking "Got it" dismisses overlay and sets localStorage flag
3. Clicking outside card (on overlay backdrop) dismisses overlay
4. Reloading page after dismiss: overlay does not appear
5. `hasNodes = false`: overlay does not appear
6. Re-open button (if implemented): overlay re-appears without resetting flag

---

### 10. What This Does Not Change

- Cytoscape event handlers (`onetap`, `dbltap`, Shift+drag) — unchanged
- RFC-076 progressive expansion logic — unchanged
- Ring colours in `cyGraphStylesheet.ts` — unchanged (overlay references
  them visually but does not own them)

- All other graph chrome (**layout cycle**, **Re-layout**, minimap, **Fit** / zoom / **Export PNG** in **`GraphBottomBar`**) — behaviour unchanged by the overlay; only placement vs legacy canvas overlays evolved per [UXS-004 Graph chrome](../uxs/UXS-004-graph-exploration.md#graph-chrome-toolbar-bottom-bar-filters-popover)
- Token system — uses only existing UXS-001 tokens

---

*Small, contained, one new component. No store changes, no API changes.*

---

## Graph focus entry points

**Status:** Draft checklist (not indexed). **Purpose:** align every “open graph” path with
the **node id vocabulary** the merged GI+KG graph actually contains (`topic:`, `tc:`,
episode-scoped slugs from bullets, search hit payloads, and so on).

### Phase 0 (landed)

- **Code:** `web/gi-kg-viewer/src/utils/cilGraphFocus.ts` — maps a `CilDigestTopicPill` +
  optional episode id to `graphNavigation.requestFocusNode` (primary `topic:…`, fallback

  episode id, optional `pendingFocusCameraIncludeRawIds` for `tc:…` when
  `in_topic_cluster`). Used from **Digest Recent** (`DigestView.vue`) and **Episode subject rail**
  canonical topic pills (`EpisodeDetailPanel.vue`).

- **Tests:** `web/gi-kg-viewer/src/utils/cilGraphFocus.test.ts` (Vitest).
- **Docs:** [Development Guide — Viewer v2](../guides/DEVELOPMENT_GUIDE.md#viewer-v2-rfc-062-489) (bullet **CIL pill to graph focus**).

### Why this exists

CIL and digest **pills** unify **bridge** `topic:` ids with optional RFC-075
**`topic_cluster_compound_id`** (`tc:`) for the viewer. The **cytoscape merge** still
contains multiple node kinds and legacy paths (for example bullet-derived `topic:` slugs).
If one surface passes the wrong token, focus can **miss** or highlight the **wrong**
vertex.

### Entry surfaces to audit (code + UX)

| Surface | Expected focus token | Notes |
| ------- | -------------------- | ----- |
| Digest — CIL chip | `topic:` and/or `tc:` from `cil_digest_topics[]` | Server builds pills; viewer must pass compound when clustered. |
| Digest — topic band / semantic row | Topic ids from digest API / GI load | Confirm same slugging as graph merge. |
| Search — open graph from hit | Hit payload (`lifted`, anchors, …) | Transcript lift vs insight hit may differ. |
| Library — Episode subject rail CIL | Same pill shape as digest detail | List rows omit pills; detail only. |
| Graph — double-tap cross-episode expand | Canonical `node_id` from graph | Already constrained by API contract. |
| Person / org drill-ins | `person:` / `org:` | Less overlap with `tc:` but worth a row in tests. |

### Suggested engineering outcomes (remaining)

1. ~~**Single helper** for CIL pills~~ — done for Digest + Episode subject rail (`cilGraphFocus.ts`).
2. **Extend or reuse** the same contract for other surfaces (topic band rows, explore,
   any future chips) and optionally dedupe SearchPanel’s `topic_cluster` camera logic

   with one shared primitive.

3. **Playwright** assertion that clustered pill passes camera ids (optional: spy on
   store or assert zoom behaviour if stable in mocks).

4. Optional: **dev-only log** when focus id is not found in the loaded graph (guarded so
   production builds stay quiet).

### When to fold into UXS / Development Guide

After the audit is done, fold the **contract** into [UXS-002 Corpus
Digest](../uxs/UXS-002-corpus-digest.md) / [UXS-003 Corpus
Library](../uxs/UXS-003-corpus-library.md) or the [Development Guide — GI / KG
viewer](../guides/DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) and trim or archive this checklist.

### GitHub issue

Tracked as [#596](https://github.com/chipi/podcast_scraper/issues/596) (body template:
[GitHub #596](https://github.com/chipi/podcast_scraper/issues/596).

---

## Cluster sibling episode load

**Status:** Design locked for implementation phase (2026-04-15 iteration).

### Clarification: what the earlier “options” meant

Three **UX triggers** were discussed for *when* to pull in sibling episodes:

| Option | Meaning |
| -------- | -------- |
| **Button only** | User clicks something like “Load cluster episodes” — nothing loads until they click. |
| **Confirm dialog** | After an action (e.g. Open in graph), a dialog: “Load N related episodes?” Confirm / Skip. |
| **Automatic with cap** | The app loads siblings without a click, but **stops at a max** (configurable) to avoid huge graphs by accident. |

### Locked product decision (this iteration)

- **Trigger:** **Automatic** in the **Graph** context: when a topic cluster applies and sibling episodes can be resolved from `topic_clusters.json` + catalog, **merge-load** their GI/KG into the current graph selection up to a **safety cap** on how many **additional** episodes to pull in per merge (not unlimited).
- **Safety cap — default `10`:** At most **10** sibling episodes are auto-merged per trigger (tune if needed). Order: deterministic (e.g. catalog sort, or first N unresolved ids) — specify in implementation.
- **Configurable “higher up”:** One central place (e.g. viewer env `VITE_CLUSTER_SIBLING_EPISODE_CAP` with default `10`, or a small `viewerConfig` module parsing `import.meta.env` + fallback). Document in Development / Polyglot guide and optional `.env.example` so power users can raise the cap without code edits.
- **Scope:** **Graph / artifact merge path only** for this phase — **no change** to **Digest** or **Library** behavior or layout (those can be revisited later if we want parity).
- **Transparency:** When the cap trims the list, UI should indicate **“Loaded N of M sibling episodes (cap …)”** or similar so users know more exist.

### Is this a “good” answer?

**Yes.** A default of **10** with a **configurable** cap balances full-cluster intent with predictable worst-case load. Raising the default or env for large monitors / fast machines is easy; lowering it protects laptops and huge corpora.

**Technically:** Same as before: `episode_ids` on cluster members, resolve to paths via catalog, **merge** (not replace) `selectedRelPaths`, apply cap when selecting which sibling paths to add.

### Implementation reminders (unchanged core)

1. **Store:** `appendRelativeArtifacts` / merge load — required because `loadRelativeArtifacts` replaces selection today.
2. **Server:** Resolve `episode_id` → `gi_relative_path` / `kg_relative_path` (catalog scan per request or cached server-side if needed).
3. **Client:** From loaded graph + `topicClustersDoc`, compute sibling episode ids → resolve → merge load **automatically** when conditions are met (define exact hook: e.g. after `loadSelected` completes when graph tab is active and cluster has unresolved siblings).

### Out of scope (this phase)

- Digest “open in graph” flows.
- Library episode detail / “Open in graph” (unless we later wire the same merge helper behind an explicit control there).

### See also (layout, separate workstream)

- [Graph layout — topic cluster](#graph-layout-topic-cluster) — tighter TopicCluster footprint on the main canvas; neighborhood minimap COSE/2D instead of breadthfirst strip.

---

## Graph layout — topic cluster {#graph-layout-topic-cluster}

**Status:** Cluster compaction tuned (section 1); minimap layout still pending (section 2).

### 1. Cluster compound: tighter footprint on main canvas

**Issue:** For 2-4 nodes inside a TopicCluster compound, the cluster **region** can occupy
~1/4 of the whole graph -- poor use of space and visually "pulls" the layout.

**Applied (cyCoseLayoutOptions.ts):**

- **Member repulsion** cut from 420k to 180k (main) / 57k to 24k (compact) -- members
  pack much tighter inside the compound.

- **Intra-cluster ideal edge length** cut from 58 to 36 (main) / 32 to 20 (compact) --
  connected members pull closer together.

- **Gravity** raised from 0.15 to 0.18 (main) / 0.28 to 0.32 (compact) -- slightly
  stronger pull toward center reduces overall sprawl.

- **Nesting factor** raised from 1.38 to 1.52 -- cross-boundary edges stretch more
  relative to intra-cluster edges, keeping external nodes further from the compound

  while internals stay tight.

- **numIter** set to 2500 (explicit) -- ensures the simulation converges well at the
  lower repulsion values.

- Compound **padding** (`cyGraphStylesheet.ts`) left at 6px/3px -- already minimal; the
  issue was internal node spacing, not border padding.

### 2. Neighborhood minimap: 2D layout, not a single line

**Issue:** In the **local neighborhood** preview, nodes feel **all on one horizontal
line** -- too tight 1D; want a **normal COSE** (or equivalent) feel and **more 2D** spread.

**Current code:** `web/gi-kg-viewer/src/components/graph/GraphNeighborhoodMiniMap.vue`
uses **`breadthfirst`** with `directed: true` and `roots` -- that tends to produce
**tree / strip** layouts (often one row or column).

**Direction:**

- Switch minimap layout to **`cose`** (align with main graph's `layoutOptionsFor('cose')`
  in `web/gi-kg-viewer/src/components/graph/GraphCanvas.vue`) or **`fcose`** if available

  in the bundled cytoscape build -- tune `nodeRepulsion`, `idealEdgeLength`, `gravity`
  for **small** element counts so the preview is 2D without huge spread.

- Keep `fit` after layout; preserve selection/highlight behavior.
- Re-check TopicCluster neighborhood path (`topicClusterNeighborhood` prop) vs generic
  ego slice so both look reasonable.

### 3. Relationship to sibling-episode auto-load

Independent: episode merge behavior is data volume; this doc is **pure
layout/presentation**. They can ship in either order.

### 4. UX / E2E

When layout strings or minimap behavior become E2E-visible, update
`web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md` and the relevant UXS if the visual contract
changes.

---
