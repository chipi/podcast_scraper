# Graph Initial Load — Implementation Spec

**Status:** Ready for implementation  
**Author:** Design session (Marko + Claude), April 2026  
**Repo:** `chipi/podcast_scraper`  
**Target area:** `web/gi-kg-viewer/src/`  
**Related docs:** UXS-001, UXS-004, RFC-062, RFC-076  
**Scope:** Load strategy, graphLens store, default node type visibility,
status line. No Cytoscape stylesheet changes (those are in the Graph
Styling spec).

---

## 1. Problem

The graph currently has no defined default load strategy. In online mode
with a corpus path set, loading all GI/KG artifacts produces a hairball
for any non-trivial corpus. There is no time lens on the Graph tab, no
episode cap, and no visual indication of what is loaded or how to expand.

Simultaneously, the shared `corpusLens` (Digest + Library) is the only
time filter in the app. If Graph naively inherits it, a user who set
Library to "all time" to browse the full catalog would trigger a full
corpus graph load — defeating any load cap.

---

## 2. Direction

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

## 3. graphLens

### Concept

`graphLens` is a time filter owned exclusively by the Graph tab. It is
**not** `corpusLens` and does not share state with Digest or Library.

### Initialisation

On first Graph tab open per session:

```
graphLens.value = corpusLens.value ?? '7d'
```

If `corpusLens` has a date set (e.g. user set Library to "30d"), Graph
seeds from that. If `corpusLens` is unset (all time), Graph defaults to
`7d` — it does not inherit "all time" from Library.

After initial seed, `graphLens` is independent. Changing `corpusLens`
in Digest/Library does not update `graphLens`. Changing `graphLens` in
Graph does not affect Digest/Library.

### Values

Same set as `corpusLens`:

| Value | Meaning |
|---|---|
| `7d` | Last 7 calendar days (default) |
| `30d` | Last 30 days |
| `90d` | Last 90 days |
| `all` | All time (user must opt in explicitly) |
| `since:YYYY-MM-DD` | Since a specific date |

### Store

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

## 4. Episode Cap

### Rule

When loading GI/KG artifacts for the graph, select at most **N episodes**
from the candidate pool defined by `graphLens`.

**Candidate pool selection:**

```
if graphLens === 'all':
    candidates = most recent N episodes by publish_date (desc)
else:
    candidates = episodes where publish_date >= window_start, sorted desc
    if candidates.length > N: truncate to N
```

The cap is a ceiling on **episodes**, not on nodes. A 15-episode corpus
slice may produce 200+ nodes (many Insights, Topics, Persons per episode).
The cap controls load time and canvas density, not node count directly.

### Default cap value

**N = 15**

Add to UXS-001 tunable parameters table:

| Parameter | Current value | Status | Notes |
|---|---|---|---|
| Graph default episode cap | 15 | Open | Tunable; increase for denser corpora |

### Implementation

In the artifact loading flow (`artifactsStore` or wherever GI/KG paths
are listed and loaded):

1. `GET /api/artifacts` returns all available GI/KG paths
2. Before loading, filter paths by `graphLens` window using episode
   publish date (available in artifact path or metadata)
3. Sort by publish date descending
4. Take first N paths
5. Load those N GI/KG JSON files into the merged graph

If the API does not expose publish date on artifact listing, fall back
to loading all paths and filtering by episode metadata after parsing.
Note this in implementation — a future API improvement could expose
date on the artifact list endpoint to avoid loading then discarding.

---

## 5. Graph Status Line

A muted status line sits **above the graph canvas** (below the toolbar
chrome, above the Cytoscape surface). Always visible when graph is
loaded. Updates reactively.

### Content

```
Showing [lens label] · [N] episodes · [M] nodes   [Lens selector]
```

Examples:
```
Showing last 7 days · 12 episodes · 847 nodes    [7d ▾]
Showing last 30 days · 15 episodes (capped) · 1.2k nodes   [30d ▾]
Showing all time · 15 episodes (capped) · 934 nodes   [All ▾]
```

- **"(capped)"** appears only when the candidate pool exceeded N and was
  truncated. Omitted when all candidates fit within the cap.
- Node count: use `k` suffix for counts ≥ 1000 (e.g. `1.2k`)
- Episode count rounds down: show loaded count, not candidate pool size

### Styling

- `text-[10px]`, `muted` token — unobtrusive, same density as toolbar
- Left-aligned text, right-aligned lens selector
- Background: `bg-canvas` in the canvas column (reads as part of the canvas stack above Cytoscape)
- Height: ~24px, same compact scale as toolbar rows

### Lens selector

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

### data-testid values

```
data-testid="graph-status-line"
data-testid="graph-status-lens-selector"
data-testid="graph-status-since-input"
data-testid="graph-status-episode-count"
data-testid="graph-status-node-count"
```

`graph-status-episode-count` and `graph-status-node-count` wrap **numeric text only** (node count may use the `k` suffix from formatting).

---

## 6. Default Node Type Visibility

### Problem

The graph toolbar has a Types section with per-type checkboxes. Currently
all types are on by default. Quote, Speaker, and Episode nodes create
visual noise without adding to the first-read intelligence value — their
information is available in the subject rail when you need it.

### Default state

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

### Implementation

In `graphFilters.ts` store, set initial checked state per the table
above. Existing per-type checkbox UI in the toolbar is unchanged —
users can turn any type on/off freely. This only changes the defaults.

### Visual indicator

When any non-default type is enabled (e.g. user turns on Episode nodes),
a small **"filters active"** muted chip appears next to the Types heading
in the toolbar — same pattern already referenced in UXS-004 for the
Sources row. Clicking it resets all types to defaults.

`data-testid="graph-types-reset"`

---

## 7. Interaction with RFC-076 Progressive Expand

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

## 8. Interaction with Gesture Overlay

The gesture overlay (separate spec) teaches the dbl-click expand gesture.
The status line complements it with persistent context — "15 episodes
loaded" tells the user the graph is a slice, not the whole corpus, and
that there's more to explore via expand.

The overlay's "Dbl-click → Load more episodes for this node" hint
directly addresses the affordance the status line implies. The two
features are designed to work together.

---

## 9. Files to Touch

### Modified:
```
web/gi-kg-viewer/src/stores/graphExplorer.ts (or new graphLensStore.ts)
  — add graphLens state, seedFromCorpusLens, setLens, reset actions

web/gi-kg-viewer/src/stores/graphFilters.ts
  — update default node type visibility (Quote, Speaker, Episode OFF)

web/gi-kg-viewer/src/stores/artifacts.ts (or graph loading path)
  — apply graphLens filter + episode cap before loading GI/KG paths

web/gi-kg-viewer/src/components/graph/GraphCanvas.vue
  — render GraphStatusLine above canvas
  — wire graphLens changes to reload trigger

web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md
  — add graph-status-line, lens selector, episode count, node count
```

### New:
```
web/gi-kg-viewer/src/components/graph/GraphStatusLine.vue
```

### UXS amendment (after implementation):
```
docs/uxs/UXS-004-graph-exploration.md
  — Add section: ## Default graph load
  — Document graphLens, episode cap, status line, default type visibility
  — Add graphLens lens selector to toolbar description

docs/uxs/UXS-001-gi-kg-viewer.md
  — Add graph default episode cap to Tunable parameters table
```

---

## 10. Checkpoints

**Checkpoint 1 — graphLens store**
- `graphLens` exists in store, seeds from `corpusLens` on first Graph tab
  open, resets on corpus path change
- Changing `corpusLens` in Digest/Library does not affect `graphLens`
  after seed

**Checkpoint 2 — Episode cap**
- Graph loads at most 15 episodes from `graphLens` window
- "(capped)" indicator appears when pool was truncated
- "all time" lens loads 15 most recent, not everything

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

## 11. Open Questions (Not Blocking)

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
