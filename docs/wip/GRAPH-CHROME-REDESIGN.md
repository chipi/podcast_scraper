# Graph Chrome Redesign — UXS-004 Amendment

**Status:** Ready for implementation
**Author:** Design session (Marko + Claude), April 2026
**Repo:** `chipi/podcast_scraper`
**Amends:** `docs/uxs/UXS-004-graph-exploration.md`
**Related:** GRAPH-INITIAL-LOAD-V2, GRAPH-VISUAL-STYLING, VIEWER-SHELL-RESTRUCTURE
**Scope:** Graph toolbar, canvas overlays, bottom bar, minimap toggle.
No changes to Cytoscape event handling, node detail rail, or stylesheet.

---

## 1. Problem

The current graph chrome has accumulated controls without a spatial logic:

- **5 toolbar rows** — gesture hints, Sources, Minimap checkbox, Edges,
  Types. Too much chrome above the canvas.
- **Cramped upper-right overlay** — Re-layout, layout algorithm select,
  degree filter buckets, and Clear squeezed into ~6.75rem width.
- **Minimap checkbox far from minimap** — checkbox is in the top toolbar,
  minimap is in the lower-left canvas corner. They control the same thing
  from opposite ends of the screen.
- **No visual logic** — visibility filters, arrangement controls, and
  navigation aids are mixed across rows and overlays with no grouping
  principle.

---

## 2. Design principle

Three distinct jobs → three distinct areas:

| Job | Question | Location |
|---|---|---|
| Visibility | What am I looking at? | Top toolbar |
| Arrangement | How is it organised? | Bottom bar left zone |
| Navigation | Where am I / how do I move? | Bottom bar right zone |

Context (graphLens, episode count, Reset) sits in the bottom bar centre
zone — already specced in GRAPH-INITIAL-LOAD-V2 as the graph status line,
now formally part of the bottom bar.

---

## 3. Top toolbar — one row

Replaces the current 5-row toolbar chrome entirely.

**Contents:**
```
Types: [Ep] [Topic] [Person] [Cluster] [Insight] [Entity]  ·  All  None  ·  ⚙
```

- **Types checkboxes** — one per node type, same as today. Compact pill
  style, swatch colour matches node fill. Count badge on each (number of
  that type currently in the merged graph). Full width of the toolbar.
- **All / None** — quick reset shortcuts. `text-xs muted`.
- **⚙ icon button** — rightmost. Opens the filters popover (Section 4).
  `aria-label` "More graph filters". Shows a `warning` dot when any
  non-default filter is active inside the popover.

**What moves out:**
- Sources row → filters popover
- Minimap checkbox row → bottom bar left zone (self-contained on minimap)
- Edges row → filters popover
- Gesture hint row → gesture overlay (already specced in
  GRAPH-GESTURE-OVERLAY spec). No permanent toolbar row needed.

**Height:** same `text-[10px]` compact row as today. Single row only.

`data-testid="graph-toolbar-types"`
`data-testid="graph-toolbar-more-filters"`

---

## 4. Filters popover (⚙)

Anchored to the ⚙ button. `elevated` background, `border`, `rounded-sm`,
`p-3`, `w-56`. Closes on outside click or Escape.

**Contents:**

**Sources section:**
- GI toggle (show/hide GI nodes)
- KG toggle (show/hide KG nodes)
- Hide ungrounded checkbox

**Edges section:**
- Per-edge-type checkboxes (same as current Edges row)
- All / None shortcuts

**Degree filter section:**
- Degree buckets in a 2-column grid (same as current upper-right overlay)
- Clear button when active (`text-xs primary`)

Each section has a `text-[10px] muted tracking-wider` label heading.
Dividers between sections: `border-t border-border`.

When any non-default filter is active: the ⚙ button in the toolbar shows
a 6px `warning` dot in its top-right corner — same pattern as the
"filters active" chip in UXS-003 Library filters.

`data-testid="graph-filters-popover"`
`data-testid="graph-filters-sources"`
`data-testid="graph-filters-edges"`
`data-testid="graph-filters-degree"`

---

## 5. Bottom bar

A thin persistent bar at the bottom of the graph canvas. Visually extends
the shell status bar pattern into the graph surface.

**Visual:**
- Height: ~36px — same as shell status bar
- Background: `canvas` token
- Top edge: `border-t border-border`
- Three zones separated by `border-r border-border` hairlines

**Layout:**
```
┌──────────────────┬──────────────────────────────────────┬────────────────────────┐
│  LEFT ZONE       │  CENTRE ZONE                         │  RIGHT ZONE            │
│  arrangement     │  context (graphLens status line)     │  navigation            │
│                  │                                      │                        │
│ [⊞][Re-layout]   │  Showing 7d · 12 ep · 134n · 4c     │  [fit][-][+][100%][⬇] │
│ [⟲ cose]        │  [7d ▾]  [Reset]                     │                        │
└──────────────────┴──────────────────────────────────────┴────────────────────────┘
```

---

### 5.1 Left zone — arrangement

Three controls:

**Minimap toggle `⊞`:**
- Icon button. When minimap is visible: active state (`primary` tint).
  When hidden: inactive (`muted`).
- `aria-label` "Toggle minimap"
- `data-testid="graph-minimap-toggle"`
- Replaces the Minimap checkbox row in the old toolbar. The toggle is now
  directly below the minimap — maximum proximity.

**Re-layout button:**
- Text button: "Re-layout". `text-xs`.
- Triggers a fresh COSE layout run on the current graph.
- Same behaviour as the current Re-layout button in the upper-right overlay.
- `data-testid="graph-relayout"`

**Layout cycle `⟲`:**
- Icon button with current algorithm name as a text label next to it:
  e.g. `⟲ cose` / `⟲ grid` / `⟲ circle` / `⟲ breadthfirst`
- Clicking cycles through: cose → breadthfirst → circle → grid → cose
- Each click also triggers a Re-layout (cycles and applies immediately)
- Native `title` tooltip shows the full algorithm name and what's next:
  "Current: COSE force-directed. Click to switch to Breadthfirst."
- `data-testid="graph-layout-cycle"`

**Replaces:** the entire upper-right canvas overlay
(`graph-layout-controls` region). That overlay is removed entirely.

---

### 5.2 Centre zone — context (graphLens status line)

Already specced in GRAPH-INITIAL-LOAD-V2 Section 5. Reproduced here
for completeness as part of the bottom bar.

```
Showing [lens label]  ·  [N] episodes  ·  [M] nodes  ·  [K] components    [lens ▾]  [Reset]
```

- Lens selector: compact inline — All / 7d / 30d / 90d / Since
- "(capped)" when episode pool was truncated
- Reset: appears only when graph has diverged from initial load state
- `text-[10px] muted` for all text

`data-testid="graph-status-line"` (existing from GRAPH-INITIAL-LOAD-V2)

---

### 5.3 Right zone — navigation

Contents from the current bottom-right canvas overlay, moved into the bar:

- **Fit** — fit graph to canvas. `aria-label` "Fit graph"
- **−** — zoom out
- **+** — zoom in
- **100%** — reset zoom to 1.0 (pan unchanged)
- **⬇ PNG** — export graph as PNG at 2× resolution

Same controls, same behaviour, new location. The separate bottom-right
canvas overlay is removed — these controls now live in the bar.

`data-testid="graph-zoom-fit"`
`data-testid="graph-zoom-out"`
`data-testid="graph-zoom-in"`
`data-testid="graph-zoom-reset"`
`data-testid="graph-export-png"`

---

## 6. Minimap — self-contained

The minimap (lower-left of canvas) gains its own collapse affordance.
The top-toolbar Minimap checkbox is removed entirely.

**When minimap is visible:**
- A small `×` button sits in the top-right corner of the minimap frame.
  `text-[10px] muted`, hover `surface-foreground`.
  `aria-label` "Hide minimap". Clicking hides it.
- The bottom bar `⊞` toggle shows active state.

**When minimap is hidden:**
- Bottom bar `⊞` toggle shows inactive state.
- Clicking `⊞` shows the minimap.

No other minimap chrome changes. Dimensions, position, and rendering
are unchanged.

`data-testid="graph-minimap"` (existing)
`data-testid="graph-minimap-close"` (new — the × button)

---

## 7. Upper-right overlay — removed

The current `graph-layout-controls` upper-right overlay is removed
entirely. Its contents are redistributed:

| Was in upper-right overlay | Now lives in |
|---|---|
| Re-layout button | Bottom bar left zone |
| Layout algorithm select | Bottom bar left zone (cycle button) |
| Degree filter buckets | Filters popover (⚙) |
| Clear degree button | Filters popover (⚙) |

The `role="region"` `aria-label="Graph layout, re-layout, and degree filter"`
region is deprecated. Remove from `GraphCanvas.vue`.

---

## 8. Search highlight chip

Currently in the primary toolbar row. After this redesign, the primary
toolbar row is the Types row only.

The search highlight chip ("Showing N results for: [query]") moves to
the centre zone of the bottom bar, appearing between the status line text
and the lens selector when a search highlight is active. It sits on its
own line within the centre zone (the zone expands slightly in height when
the chip is present).

When no search highlight is active: centre zone is single-line as normal.
When active: centre zone shows status line on top, search chip below.

`data-testid="graph-search-highlight-chip"` (existing testid preserved)

---

## 12. Before / after summary

**Before:**
```
[toolbar row 1]  gesture hints · search chip
[toolbar row 2]  Sources: GI · KG · Hide ungrounded
[toolbar row 3]  Minimap checkbox
[toolbar row 4]  Edges checkboxes
[toolbar row 5]  Types checkboxes · all/none · counts

[upper-right overlay]  Re-layout · Layout select · Degree buckets · Clear

[canvas lower-left]   Minimap
[canvas bottom-right] Fit · − · + · 100% · PNG
```

**After:**
```
[toolbar row 1]  Types checkboxes · all/none · ⚙

[bottom bar left]    ⊞ · Re-layout · ⟲ cose
[bottom bar centre]  Showing 7d · 12ep · 134n · 4c  [7d▾]  [Reset]
[bottom bar right]   Fit · − · + · 100% · ⬇PNG

[canvas lower-left]  Minimap (× to collapse)
```

5 toolbar rows → 1 toolbar row.
1 cramped overlay → gone.
Minimap toggle → collocated with minimap.

---

## 10. Bottom bar collapse

The bottom bar can be collapsed to maximise canvas space.

### Toggle

A `⌄` chevron button sits at the far right of the bottom bar, after
the Export PNG button. Last element in the right zone.

- `aria-label` "Collapse graph bar"
- `data-testid="graph-bottom-bar-toggle"`
- `text-[10px] muted`, hover `surface-foreground`

### Collapsed state

Bar shrinks to a **4px strip** at the bottom of the canvas. The strip
shows only the `⌃` chevron centred — enough target to click to restore.

- Strip background: `canvas` token
- Strip top border: `border-t border-border`
- Chevron centred in the strip

### Expanded state

Full 36px bar, all three zones visible. Chevron shows `⌄`.

### Persistence

`localStorage` key `ps_graph_bottom_bar_collapsed`. Restored on next
session. Default: expanded.

### Minimap when bar is collapsed

Minimap lives in the canvas, not in the bar — stays visible when bar is
collapsed. The minimap `×` close button still works. The `⊞` toggle is
inaccessible while collapsed — acceptable since `×` works independently.

### Auto-expand on search highlight

When a search highlight becomes active and the bar is collapsed: bar
auto-expands so the search chip is visible. User must know a highlight
is applied.

When the search highlight clears: bar does **not** auto-collapse. Only
auto-expand is triggered, never auto-collapse. Explicit user preference
is respected.

### Keyboard

`Alt+B` toggles collapsed/expanded. Add to gesture overlay hint list
as low-priority item.

`data-testid="graph-bottom-bar"` on the bar element, with
`aria-expanded="true/false"` reflecting state.

---

## 13. Files to touch

```
web/gi-kg-viewer/src/components/graph/GraphCanvas.vue
  — remove upper-right overlay (graph-layout-controls region)
  — add GraphBottomBar component below canvas
  — top toolbar: keep Types row only, remove all other rows
  — minimap: add × close button to minimap frame

web/gi-kg-viewer/src/components/graph/GraphBottomBar.vue  (new)
  — three-zone bottom bar
  — left: minimap toggle, re-layout, layout cycle
  — centre: graphLens status line (from GRAPH-INITIAL-LOAD-V2)
  — right: fit, zoom, export, collapse chevron
  — collapsed state: 4px strip with chevron only
  — localStorage: ps_graph_bottom_bar_collapsed
  — auto-expand when search highlight becomes active

web/gi-kg-viewer/src/components/graph/GraphFiltersPopover.vue  (new)
  — ⚙ popover: Sources, Edges, Degree sections

web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md
  — add: graph-minimap-toggle, graph-minimap-close
  — add: graph-relayout (moved from overlay)
  — add: graph-layout-cycle
  — add: graph-filters-popover, graph-filters-sources,
          graph-filters-edges, graph-filters-degree
  — add: graph-bottom-bar (aria-expanded)
  — add: graph-bottom-bar-toggle (chevron)
  — remove: graph-layout-controls (deprecated region)
```

---

## 14. UXS-004 sections to rewrite

After implementation, update `docs/uxs/UXS-004-graph-exploration.md`:

- **Toolbar (primary row)** → rewrite as "Top toolbar — Types + filters"
- **Canvas overlay (upper-right)** → mark as removed, reference this spec
- **Canvas overlay (bottom-right)** → rewrite as "Bottom bar right zone"
- **Toolbar (chrome below primary)** → mark as removed
- **Minimap** → add self-contained toggle behaviour
- Add new section: **Bottom bar** with three-zone layout
