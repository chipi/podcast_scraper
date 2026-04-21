# Viewer Information Architecture

**Status:** Active
**Lives in:** `docs/uxs/VIEWER_IA.md`
**Scope:** Structural layer only — where things live, what each area is for,
and how navigation works. This document does **not** cover visual tokens
(UXS-001), per-surface layout detail (UXS-002 through UXS-010), or
behavioral rules such as animation timing and debounce intervals (RFC-062).

**Canonical shell IA:** This file is the single source for **regions, axes,
persistence, and clearing** after the shell restructure
([GitHub #606](https://github.com/chipi/podcast_scraper/issues/606)). Feature UXSs
(UXS-002–010) link here for “where in the shell” context; they own surface
density and controls. **Timing, debounce, and keyboard implementation**
details live in [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md).

**Single canonical IA:** For the GI/KG viewer, **information-architecture decisions** (regions, axes, persistence, clearing, first-run) live **only** in this document. **[UXS-001](UXS-001-gi-kg-viewer.md)** and **[UXS-002](UXS-002-corpus-digest.md)** through **[UXS-010](UXS-010-person-profile.md)** defer here for shell questions and own **tokens plus per-surface** visual contracts.

**No placeholder UI:** The shipped app must not show controls or sections for
capabilities that are not implemented. First-run copy and disabled states
describe only what exists today; speculative “attention” chrome (e.g. pulsing
borders) is out of scope until a dedicated implementation ships.

UXS files reference this document for shell structure. RFC-062 references
it for the navigation model that behavioral rules are built on top of.

---

## Mental model

The viewer is a **query-driven intelligence tool**. The left panel is always
available for querying the corpus. The main area provides four perspectives
on the same corpus. The right rail shows detail about whatever entity is
currently in focus. The status bar keeps the operational context visible at
all times.

**Subject panel as single context layer:** The right rail is the **one**
context column. Every entity type (episode, topic, person, graph node)
surfaces there the same way, no matter which tab or query opened it. There
is no parallel “detail drawer” per tab — the subject store drives the rail.

---

## Shell layout

```text
┌─────────────────────────────────────────────────────────────────────┐
│  HEADER: App title · version · main tab buttons · theme toggle       │
├────────────────┬────────────────────────────────┬────────────────────┤
│                │                                │                    │
│  LEFT PANEL    │         MAIN AREA              │   RIGHT RAIL       │
│                │                                │                    │
│  Search        │  Digest | Library |            │  Subject Panel     │
│  (default)     │  Graph  | Dashboard            │                    │
│  ↔ Explore     │                                │  (empty when       │
│  mode          │  (tab content)                 │   nothing          │
│                │                                │   selected)        │
│                │                                │                    │
├────────────────┴────────────────────────────────┴────────────────────┤
│  STATUS BAR: [ corpus path _________________________ ]  ● OK  ⚡     │
└─────────────────────────────────────────────────────────────────────┘
```

Four independent areas. Each has one job:

| Area | Job |
| --- | --- |
| Left panel | Query the corpus (**Search** default; **Explore** as a deliberate mode) |
| Main area | Browse four perspectives on the corpus (tabs) |
| Right rail | Understand the current subject (entity detail) |
| Status bar | Operational context (corpus path + server health) |

---

## Navigation axes

The app has no URL routing. Navigation is tab and store state.

**Three independent axes operate simultaneously:**

| Axis | Values | Who owns it |
| --- | --- | --- |
| Main tab | `digest` / `library` / `graph` / `dashboard` | `App.vue` local ref |
| Subject (right rail) | episode / topic / person / graph-node / none | `subjectStore` |
| Left panel surface | `search` (semantic UI + results) / `explore` (GI explore UI) | `shellStore.leftPanelSurface` + `searchStore`, `exploreStore` |

Changing one axis does not reset the others. Switching from Library to Graph
does not clear the subject rail. Running a search does not change the active tab.

---

## Left panel — Query interface

**Always visible.** Permanent on the left regardless of which main tab is active.

**Two modes, one column width (`w-72`):**

- **Search** (default): semantic FAISS-based corpus search — query form, run controls, scrollable
  results. Quiet footer control **Explore corpus →** (`text-xs`, muted, arrow) slides the column to
  **Explore** mode (CSS transform on a two-pane strip inside `LeftPanel.vue`; **`data-testid="left-panel-slide-host"`**).
- **Explore**: GI cross-episode filters and natural-language / preset queries. **← Search** at the
  top (same quiet treatment) slides back; **search query and results are preserved** (`searchStore`
  untouched). **`data-testid="left-panel-enter-explore"`** / **`left-panel-back-search`**.

**Keyboard:** **`/`** expands the column if collapsed, sets surface to **Search**, then focuses
**`#search-q`** (`useViewerKeyboard` + `LeftPanel` `focusQuery`). In **Explore**, **Enter** in
**Topic contains** or **Speaker contains** runs **Explore**; **Limit** / **Min confidence** use
the same **Enter** → **Explore** behavior inside the **Advanced explore** dialog. **Enter** in
**Quick question** runs **Run quick question**; **Shift+Enter** there inserts a newline (IME
composition does not trigger submit).

**Purpose:** Search stays the default mental model (like a map app’s search field). Explore is a
deliberate “browse corpus” mode, not a second always-visible stack.

**Visual spec:** UXS-005 (Search), UXS-001 (tokens)

### Collapse (Option B)

The left column may be **collapsed** to a narrow strip so the main area gains
width (especially on the Graph tab). **Behaviour:**

- **Control:** A single **collapse/expand** control on the **top edge** of
  the left column (first row of that column), not inside `SearchPanel`.
- **Expanded:** Column width **`w-72`** (288px) — **Search** or **Explore** mode (one visible at a time).
- **Collapsed:** Column width **`w-8`** (32px) — strip shows the toggle plus a vertical **Search**
  affordance; activating it expands the column to **Search** mode and focuses the query field.
  **Explore** is opened only from the expanded column (**Explore corpus →**), or via graph hand-offs
  that set **Explore** mode programmatically (`App.vue` + `shell.setLeftPanelSurface('explore')`).
- **Persistence:** Preferred open/closed state is stored under
  **`localStorage` key `ps_left_panel_open`** (`"true"` / `"false"`). Default
  when the key is missing: **expanded**.
- **Slash (`/`):** When focus is not in an editable control, `/` **expands**
  the left column if it was collapsed, switches to **Search** mode, then focuses
  the semantic search query field (`#search-q`) — see `useViewerKeyboard` + `App.vue` + `LeftPanel.vue`.

**Implementation map:** `web/gi-kg-viewer/src/App.vue` (column chrome,
`leftOpen`), `LeftPanel.vue` (Search + Explore body). Broader module map:
[VIEWER_FRONTEND_ARCHITECTURE.md](../architecture/VIEWER_FRONTEND_ARCHITECTURE.md).

---

## Right rail — Subject panel

**Context-driven.** Shows detail about the current subject. Empty when
nothing is selected.

### Subject types

| Subject | Entry points | Component |
| --- | --- | --- |
| **Episode** | Library row, Digest row, Graph episode node, Search L button | `EpisodeDetailPanel` |
| **Topic** | Graph topic node, Search result TEV entry, Digest topic band, Dashboard Intelligence | `TopicEntityView` (UXS-007) |
| **Person** | Graph person node, Search speaker name, Enriched Search source | `PersonLanding` (UXS-010) |
| **GIL node** | Graph insight / quote / entity / speaker node | `GraphNodeRailPanel` |
| *(empty)* | Nothing selected, corpus not loaded | Empty state |

### Persistence rule

The subject persists when switching main tabs. If an episode is open in the
rail and the user switches from Library to Graph, the episode rail stays
open. On the Graph tab, the corresponding episode node is highlighted and
centred in the canvas if it exists in the merged graph.

### Clearing rule

The subject clears on:

1. Explicit close — user clicks the × button in the rail header
2. Selecting a different subject — new subject replaces the current one
3. Corpus path change — path change invalidates subject context

Tab switching does **not** clear the subject.

### Close button

Every subject panel mode has a × button in the rail header.

- Position: right-aligned in the rail header row
- Token: `muted`, hover `surface-foreground`
- `aria-label`: "Close [subject type] detail" (e.g. "Close episode detail")
- `data-testid`: `subject-rail-close`

### Empty state

When no subject is selected: centred `muted` text —
"Select an episode, topic, or graph node to see details here."

**Visual spec:** UXS-001 (tokens, shared component rules), per-type detail
in UXS-003, UXS-004, UXS-007, UXS-009, UXS-010

---

## Main tabs

Four tabs. Each answers a different question about the same corpus.

| Tab | Question it answers | Default? |
| --- | --- | --- |
| **Digest** | What's new in my corpus recently? | ✓ Yes (first load) |
| **Library** | What episodes and feeds have been processed? | |
| **Graph** | How do entities, topics, and insights connect? | |
| **Dashboard** | Is my corpus healthy, and what should I do? | |

**Tab order:** Digest · Library · Graph · Dashboard (left to right)

**Default on first load:** Digest

### Digest

Discovery surface. Rolling-window recent episodes and semantic topic bands.
Shares a date lens (`corpusLens`) with Library. See UXS-002.

### Library

Catalog. Answers "what's in the corpus?" with full filtering. Feed list +
paginated episode list + episode subject panel. See UXS-003.

### Graph

Topology exploration. Cytoscape canvas with GI/KG merged graph. Loads a
time-bounded episode slice by default (see `graphLens` in [Viewer graph spec](../architecture/VIEWER_GRAPH_SPEC.md#graph-initial-load)).
The graph is another perspective on the same entities that Library
and Digest surface — not a separate information domain. See UXS-004.

**Chrome (where controls live):** A **stats** strip at the top of the graph card (episode / node / component counts, plus **Gestures** when a full merged graph is loaded), optional **search-highlight** chip row, then a compact **Types** row and **⚙** filters popover. A **collapsible bottom bar** under the canvas holds minimap / re-layout / layout cycle, the **graph time lens** (when applicable), **Fit** / zoom / **PNG**, and **Gestures** only when the stats strip is absent. Details: [UXS-004 — Graph chrome](UXS-004-graph-exploration.md#graph-chrome-toolbar-bottom-bar-filters-popover).

### Dashboard

Operational intelligence. Permanent briefing card (last run, corpus health,
and action items) sits above three sub-tabs: Coverage, Intelligence, Pipeline.

See [UXS-006](UXS-006-dashboard.md) (includes the full dashboard implementation specification).

---

## Status bar

**Always visible.** Permanent at the bottom of the shell, below the main
content area. Single row, ~36px tall.

**Contents (left to right):**

- **Corpus path input**: always-visible editable text field. Placeholder
  "Set corpus path…". Changing the value triggers artifact listing + graph
  auto-load cascade. Value persisted in `localStorage` key `ps_corpus_path`
  and restored on next session.
- **Folder picker button** `[📁]`: appears in offline mode (no server).
  Opens local file picker for loading GI/KG artifacts without a server.
- **Health dot** `●`: 8px circle. `success` = server OK, `warning` =
  degraded, `danger` = unreachable. Clicking opens an anchored popover with
  capability rows from **`GET /api/health`** (including **`feeds_api`**,
  **`operator_config_api`**, **`jobs_api`** when present), Retry health button,
  and last error message.
- **Feeds** / **Operator YAML** (aka **Config**; when **`feeds_api`** / **`operator_config_api`** are true on health): compact triggers on the status bar open the shared **Corpus sources** modal (`data-testid="status-bar-sources-dialog"`) with tabs **Feeds** (canonical **`feeds.spec.yaml`** — JSON editor plus optional one-line-per-URL merge into **`feeds`**) and **Operator YAML** (packaged **`profile:`** picker from **`available_profiles`**; **GET** may seed **`profile: cloud_balanced`** when the operator file is missing/empty if that preset exists — plus monospace **overrides** YAML; not a second place to edit feed URLs). Tokens, `data-testid`s, and field layout: [UXS-001 — Corpus sources dialog](UXS-001-gi-kg-viewer.md#corpus-sources-dialog).
- **Rebuild indicator** `⚡`: appears only when `reindex_recommended` is
  true from index stats. `warning` token. Clicking opens health popover
  scrolled to the index section.

**Visual spec:** [UXS-001 — Status bar](UXS-001-gi-kg-viewer.md#status-bar) (tokens, height, `data-testid`)

---

## Viewport — three-column widths (1024px baseline)

The viewer assumes a **minimum ~1024px** width (UXS-001). With **default**
Tailwind shell widths at that viewport (both side columns **expanded**):

| Region | Approx. width | Notes |
| --- | --- | --- |
| Left column | **288px** (`w-72`) | Collapsed: **32px** (`w-8`) |
| Main area | **~400px** | Remainder after header padding, borders, and both side columns — tight for Library filters + list; acceptable for Graph + Digest |
| Right subject column | **384px** (`w-96`) | Collapsed: **32px** (`w-8`) |

**Right rail collapse:** Expanded/collapsed is toggled in `App.vue` (`rightOpen`) for
more canvas width on Graph. **Persistence:** not stored in `localStorage` in the
current build (resets to expanded on full page reload); only the **left** column uses
`ps_left_panel_open`.

If usability testing shows the main strip is too narrow at 1024px, tune
`w-72` / `w-96` in `App.vue` and record new numbers here (see also UXS-001
breakpoints / tunables).

---

## First-time / empty corpus state

When no corpus path has been set (first session, or localStorage cleared):

- Status bar: corpus path field shows placeholder only; health dot shows
  `danger` (server may be reachable but no corpus is configured)
- Briefing card: "Set a corpus path in the status bar below to begin."
- All main tabs: consistent empty states — short `muted` instruction per tab
- Left panel search: input disabled, tooltip "Set corpus path to enable search"
- Right rail: standard empty state

The viewer should feel intentionally empty — clear next steps — not broken.

Do **not** add speculative motion (e.g. pulsing border on the path field) or
other “draw attention” chrome until that behaviour is implemented in the same
change as its spec.

---

## Key cross-surface flows

These flows show how the navigation axes interact in common workflows.

### Open an episode from any surface

```text
Any entry point (Library row / Digest row / Search L / Graph episode node)
  → subjectStore.focusEpisode(metadataPath)
  → Right rail: EpisodeDetailPanel
  → If mainTab === 'graph': episode node highlighted + centred in canvas
  → Main tab: unchanged
```

### Open a topic from any surface

```text
Any entry point (Graph topic node / Search result / Digest band / Dashboard)
  → subjectStore.focusTopic(topicId)
  → Right rail: TopicEntityView
  → If mainTab === 'graph': topic node highlighted + centred
  → Main tab: unchanged
```

### Search result → subject

```text
Search result: user clicks G (graph focus)
  → subjectStore.focusGraphNode(cyId)
  → mainTab switches to 'graph'
  → Right rail: GraphNodeRailPanel for that node

Search result: user clicks L (library)
  → subjectStore.focusEpisode(path)
  → Right rail: EpisodeDetailPanel
  → Main tab: unchanged (user stays on current tab)
```

### Corpus path change

```text
User edits corpus path in status bar
  → shellStore.setCorpusPath(newPath)  [writes localStorage]
  → subjectStore.clearSubject()        [path change invalidates context]
  → Cascade: artifacts / library / digest / dashboard / index refresh
```

---

## Boundaries — what this document does not cover

| Topic | Where it lives |
| --- | --- |
| Visual tokens, colours, typography | UXS-001 |
| Per-surface layout and component detail | UXS-002 through UXS-010 |
| Animation timing, debounce intervals | RFC-062 |
| Graph-specific interaction model | RFC-062, UXS-004 |
| API endpoints and data contracts | [Server guide](../guides/SERVER_GUIDE.md), RFC files |
| Implementation component tree | VIEWER_FRONTEND_ARCHITECTURE.md |

---

## Revision history

| Date | Change |
| --- | --- |
| 2026-04-19 | Initial draft; shell restructure; subject rules; status bar; flows |
| 2026-04-19 | Opening principle (single context layer); collapse Option B + `ps_left_panel_open`; 1024px width table; no-placeholder / first-run policy |
| 2026-04-19 | Single canonical IA policy; visual spec link to UXS-001 `#status-bar` |
| 2026-04-19 | Boundaries table: link Server guide path (`../guides/SERVER_GUIDE.md`) |
| 2026-04-20 | Status bar: Feeds / Operator YAML open Corpus sources modal; profile + feeds split (RFC-077 / UXS-001 `#corpus-sources-dialog`) |
| 2026-04-21 | Operator YAML aka **Config**; empty `available_profiles` note; preset list cwd+repo union (RFC-077) |
| 2026-04-21 | Corpus sources Feeds: optional one-URL-per-line merge; operator **GET** may seed `profile: cloud_balanced` when file missing/empty |
| 2026-04-21 | Left panel: Search default vs Explore mode (slide); **`shell.leftPanelSurface`**; collapsed strip **Search** only; **`/`** → Search + focus |
| 2026-04-21 | Explore: **Advanced explore** dialog (Limit, Sort, Min confidence, checkboxes); **Enter** on main topic/speaker + in dialog |
| 2026-04-21 | Explore: **Enter** on filter fields → **Explore**; **Enter** on quick question → **Run quick question**; **Shift+Enter** newline |
| 2026-04-21 | **Graph** subsection: **Chrome** paragraph (stats strip, Types + **⚙**, bottom bar) links UXS-004 graph chrome section |
