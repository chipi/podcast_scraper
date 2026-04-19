# Graph Gesture Discovery Overlay — Implementation Spec

**Status:** Implemented in viewer (keep doc in sync with code when behaviour changes)  
**Author:** Design session (Marko + Claude), April 2026  
**Repo:** `chipi/podcast_scraper`  
**Target area:** `web/gi-kg-viewer/src/components/graph/`  
**Related docs:** UXS-004, RFC-062, [RFC-076](../rfc/RFC-076-progressive-graph-expansion.md) (progressive expand / ring semantics)  
**Scope:** Small, self-contained. No store changes. No API changes.

---

## 1. Problem

The graph has 5 non-obvious interaction gestures and 2 visual ring cues that
were easy to miss when they only appeared in a single line of small
`text-[10px]` copy on the top toolbar row.

Current gesture inventory a user needs to learn:

| Gesture | Effect |
|---|---|
| Single click (`onetap`) | Open subject rail for that node |
| Shift + double-click | 1-hop ego / neighbourhood expand |
| Shift + drag | Box zoom / selection |
| Plain double-click (`dbltap`) | Progressive expand — loads more episodes for eligible nodes (RFC-076) |
| Second plain double-click | Collapse that expansion |

Visual ring cues:
| Ring colour | Meaning |
|---|---|
| Teal border | Node is expandable (has episodes not yet in the graph) |
| Blue border | Node is currently expanded |

None of these are in a menu. None are labelled on the canvas. A developer
opening the graph for the first time will discover single-click and pan/zoom
immediately (mouse-native), but will miss neighbourhood expand, progressive
expand, and the ring semantics entirely unless they open the gesture overlay
(**Gestures** / first visit) or stumble on them by accident.

---

## 2. Solution

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

## 3. Visual Spec

### Overlay container

- Positioned: `absolute inset-0` over the graph canvas host
  (`.graph-canvas` or the same `overflow-hidden` region as the minimap)
- Background: `rgba` of `canvas` token at ~60% opacity —
  `color-mix(in srgb, var(--ps-canvas) 65%, transparent)` when supported; otherwise a neutral translucent veil (see scoped `.ps-gesture-overlay-root` in `GraphGestureOverlay.vue`)
- `z-index`: above Cytoscape canvas, below the graph chrome overlays
  (layout controls, fit/zoom toolbar, minimap)
- Does **not** block the chrome overlays — user can still see them

### Hint card

- Centered in the overlay: implementation uses **flex** on the backdrop root (`items-center justify-center`); an equivalent pattern is `absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2` on the card
- Background: `elevated` token
- Border: `border` token, `rounded-sm` (UXS-001 radius)
- Padding: `p-4`
- Max width: `20rem` — stays compact, doesn't fill the canvas
- Shadow: subtle (`shadow-md`) — gives depth against the dimmed canvas

### Card header

- Text: **"Graph gestures"**
- Scale: `text-sm font-semibold`, `surface-foreground` token
- Bottom margin: `mb-3`

### Gesture rows

Five rows, one per gesture. Layout per row:

```
[icon]  [gesture label]     [effect description]
```

- Icon: 16×16px, `muted` token. Use **inline SVG** (viewer does not ship Lucide as a dependency):
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
|---|---|---|
| MousePointer | Click | Open node details |
| Network | Shift + dbl-click | Expand 1-hop neighbourhood |
| BoxSelect | Shift + drag | Box zoom / select |
| PlusCircle | Dbl-click | Load more episodes for this node |
| MinusCircle | Dbl-click again | Collapse loaded episodes |

### Ring legend

Below the gesture rows, a `border-t border-border mt-3 pt-3` divider,
then two ring legend rows:

```
[●  teal ring]   More episodes available
[●  blue ring]   Episodes loaded (expanded)
```

- Dot: 10px circle, inline SVG or `w-2.5 h-2.5 rounded-full`
- Teal dot: `border-2` with teal colour matching the RFC-076 expandable
  ring colour (use the same CSS var or value used in `cyGraphStylesheet.ts`)
- Blue dot: `border-2` with the RFC-076 expanded ring colour
- Label: `text-xs`, `muted`

### Dismiss control

Below the ring legend, `mt-3`, right-aligned:

- Button: **"Got it"** — `text-xs`, `primary` token (same as secondary
  action buttons elsewhere), `px-3 py-1`
- `aria-label`: "Dismiss graph gesture hints"
- `data-testid`: `graph-gesture-overlay-dismiss`

### Dismiss behaviour

Overlay dismisses when **any** of these occur:
- Clicking **Got it**
- Clicking the **dimmed backdrop** outside the hint card (not clicks on the Cytoscape layer beneath — the dimmed layer receives the hit; card uses `@click.stop`)
- **Escape**, when focus is inside the overlay (e.g. **Got it**) or when focus is on the graph canvas host after open — **not** when focus is in graph chrome above the overlay (layout combobox, zoom toolbar, etc.) so users are not surprised while using those controls

On dismiss: set `localStorage` key `ps_graph_hints_seen` = `"1"`.
The overlay does not auto-open again; users can still use the optional **Gestures** reopen control (Section 7) without clearing `localStorage`.

### Accessibility

- Overlay root: `data-testid="graph-gesture-overlay"` (backdrop + dim layer); **Escape** containment uses a Vue **template ref** on that root (`contains(activeElement)`), not `document.querySelector`, so multiple graph roots would not cross-wire (only one graph is mounted today).
- Card: `role="dialog"`, `aria-modal="true"`, `aria-labelledby` referencing the **Graph gestures** heading.
- Initial focus: **Got it** after open; on dismiss, return focus to the graph canvas host (`tabindex="-1"`) for keyboard continuity.

### Motion

Default: hint card fades in over `150ms` (`opacity-0` → `opacity-100`),
using the same transition pattern as existing modals in the codebase.

`prefers-reduced-motion`: skip transition, render static at full opacity
immediately.

---

## 4. Trigger Condition

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

## 5. Component

### `GraphGestureOverlay.vue`

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

## 6. Toolbar Hint Row — Simplify

With the overlay handling first-time teaching, the **top** toolbar primary row
holds **only** the search highlight chip when applicable — **no** persistent
Shift / double-click / ring prose.

**Requirement:** Removing that copy **requires** either the optional **Gestures**
reopen control (Section 7) or another minimal affordance (e.g. a **?** entry) so
users who dismissed quickly are not stranded without a path back to the legend.

---

## 7. Re-open Affordance

A **Gestures** (or **?** / **Shortcuts**) control in the bottom-right zoom cluster
(next to **Export PNG**) calls `reopen()` so the card appears **without** clearing
`localStorage` (manual open only; auto-open still suppressed after dismiss).

`data-testid`: `graph-gesture-overlay-reopen`

Ship this when the primary toolbar gesture line is removed (Section 6), so
reviewers always have a deliberate path back to the overlay copy.

---

## 8. Files to Touch

### New:
```
web/gi-kg-viewer/src/components/graph/GraphGestureOverlay.vue
```

### Modified:
```
web/gi-kg-viewer/src/components/graph/GraphCanvas.vue
  — render GraphGestureOverlay inside canvas host
  — pass hasNodes computed prop
  — shorten or remove toolbar gesture hint text

web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md
  — add: graph-gesture-overlay, graph-gesture-overlay-dismiss
```

### UXS amendment (after implementation):
```
docs/uxs/UXS-004-graph-exploration.md
  — Add section: ## Gesture discovery overlay
  — Reference visual spec (tokens, layout, dismiss behaviour)
  — Note toolbar hint row simplified after overlay ships
  — Note localStorage key: ps_graph_hints_seen
```

---

## 9. E2E Contract

Add to `e2e/E2E_SURFACE_MAP.md`:

| Surface | Selector / role | Notes |
|---|---|---|
| Overlay container | `data-testid="graph-gesture-overlay"` | Present on first graph load with nodes |
| Dismiss button | `data-testid="graph-gesture-overlay-dismiss"` | Click to dismiss |
| Re-open button | `data-testid="graph-gesture-overlay-reopen"` | **Gestures** in bottom-right toolbar |

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

## 10. What This Does Not Change

- Cytoscape event handlers (`onetap`, `dbltap`, Shift+drag) — unchanged
- RFC-076 progressive expansion logic — unchanged
- Ring colours in `cyGraphStylesheet.ts` — unchanged (overlay references
  them visually but does not own them)
- All other graph chrome (layout controls, minimap, zoom toolbar) — unchanged
- Token system — uses only existing UXS-001 tokens

---

*Small, contained, one new component. No store changes, no API changes.*
