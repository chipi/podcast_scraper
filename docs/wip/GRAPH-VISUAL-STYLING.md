# Graph Visual Styling — Implementation Spec

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

## 1. Objective

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

## 2. Phasing

**Phase 1 — Stylesheet (low risk, ship together):**
Node size hierarchy, edge visual language, selection dim, zoom-responsive
labels, TopicCluster compound fill, drop shadows, default edge arrows.
Pure stylesheet values — no data dependencies.

**Phase 2 — Computed properties (medium effort, ship after Phase 1):**
Recency tint, grounding confidence tint, degree heat. These require
node property enrichment at parse time in `parsing.ts`.

Phase 1 is complete and coherent without Phase 2. Phase 2 is additive.

---

## 3. Phase 1 — Stylesheet Changes

### 3.1 Node size by tier

Nodes render at different base sizes by type. Size = cognitive importance.

| Node type | Diameter | Tier |
|---|---|---|
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

### 3.2 Edge visual language

Each edge type gets a distinct visual treatment so relationship type
is readable without hovering.

| Edge type | Width | Style | Colour | Arrow |
|---|---|---|---|---|
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

### 3.3 Selection focus — dim unrelated nodes

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

### 3.4 Drop shadows on Tier 1 nodes

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

### 3.5 Zoom-responsive labels

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
|---|---|
| < 0.5 | None |
| 0.5 – 1.0 | Tier 1 + 2 only, shortLabel |
| > 1.0 | All visible nodes, full label |

---

### 3.6 TopicCluster compound fill tint

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

### 3.7 COSE layout parameter tuning

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

## 4. Phase 2 — Computed Property Tints

Phase 2 adds three visual dimensions that require node property data
computed at parse time. All three enrich the node `data()` object in
`parsing.ts` and are then read by the stylesheet.

### 4.1 Recency tint

Nodes from more recent episodes render at higher opacity / saturation.
Older nodes are muted.

**At parse time** (`parsing.ts`), for each node compute:

```typescript
// recencyWeight: 1.0 = current week, 0.4 = 90+ days ago
const daysSince = (Date.now() - episodePublishDate) / 86400000
const recencyWeight = Math.max(0.4, 1.0 - (daysSince / 90) * 0.6)
node.data.recencyWeight = recencyWeight
```

**In stylesheet (shipped):** Node **`opacity`** is a **style function** on
`node` (`graphNodeOpacity` in `cyGraphStylesheet.ts`) that multiplies
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

### 4.2 Grounding confidence tint — Insight nodes only

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

**In stylesheet:**

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

### 4.3 Degree heat — Topic nodes only

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

2. Optional glow effect for very high degree nodes (heat > 0.7):
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

## 5. Minimap Contrast

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

## 6. Light Mode Considerations

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

## 7. Files to Touch

### Modified:
```
web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts
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

### UXS amendment (after implementation):
```
docs/uxs/UXS-004-graph-exploration.md
  — Add section: ## Node visual hierarchy
  — Add section: ## Edge visual language
  — Add section: ## Selection focus behaviour
  — Update Toolbar section: COSE parameters noted as tunable
  — Note Phase 1 vs Phase 2 separation

docs/uxs/UXS-001-gi-kg-viewer.md
  — Add COSE spring parameters to Tunable parameters table (Open)
  — Add degree heat maxDegree to Tunable parameters table (Open)
  — Add recency decay window (90 days) to Tunable parameters table (Open)
```

---

## 8. Tunable Parameters Added to UXS-001

| Parameter | Default | Status | Notes |
|---|---|---|---|
| COSE ABOUT edge ideal length | 80px | Open | Tune with real corpus |
| COSE MENTIONS edge ideal length | 150px | Open | Tune with real corpus |
| Recency decay window | 90 days | Open | Full decay at 90 days |
| Recency minimum opacity | 0.4 | Open | Floor for oldest nodes |
| Degree heat max degree | 30 | Open | Normalisation ceiling |
| Label zoom threshold (none) | 0.5 | Open | Below = no labels |
| Label zoom threshold (full) | 1.0 | Open | Above = full labels |
| Compound fill opacity | 0.06 | Open | Very faint territory tint |

---

## 9. Phase 1 Checkpoints

1. Node sizes reflect tier table — Insight largest, Episode/Speaker smallest
2. Edge types visually distinct — ABOUT solid green, SUPPORTED_BY dashed muted,
   SPOKE_IN solid primary with arrowhead
3. Selecting a node dims unrelated nodes to 40% opacity; deselect restores
4. Tier 1 nodes have drop shadow; Tier 3/4 have none
5. Labels hidden at zoom < 0.5; Tier 1/2 short labels at 0.5–1.0; full at > 1.0
6. TopicCluster compound has faint kg fill tint
7. Light mode and dark mode both render correctly (CSS variables used throughout)

## 10. Phase 2 Checkpoints

1. Nodes from last 7 days at full opacity; 90+ day nodes at ~40% opacity
2. High-confidence Insights brighter; low-confidence slightly faded
3. High-degree Topics have thicker border than low-degree Topics
4. Post-layout callback runs degree heat computation without blocking layout
5. All computed properties fall back gracefully when source data is absent

---

## 11. What This Does Not Change

- Node shapes (already defined in existing stylesheet)
- GI/KG domain token colours (gi green, kg purple) — Phase 2 modulates
  within these colours but does not change them
- RFC-076 ring colours (teal = expandable, blue = expanded) — untouched
- TopicCluster compound outline style (dashed) — Phase 1 only adds fill
- Any store logic, API calls, or component behaviour
- The gesture overlay (separate spec)
- The initial load strategy (separate spec)
