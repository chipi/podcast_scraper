# Viewer Information Architecture

**Status:** Active
**Lives in:** `docs/uxs/VIEWER_IA.md`
**Scope:** Structural layer only — where things live, what each area is for,
and how navigation works. This document does **not** cover visual tokens
(UXS-001), per-surface layout detail (UXS-002 through UXS-010), or
behavioral rules such as animation timing and debounce intervals (RFC-062).

UXS files reference this document for shell structure. RFC-062 references
it for the navigation model that behavioral rules are built on top of.

---

## Mental model

The viewer is a **query-driven intelligence tool**. The left panel is always
available for querying the corpus. The main area provides four perspectives
on the same corpus. The right rail shows detail about whatever entity is
currently in focus. The status bar keeps the operational context visible at
all times.

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
│  ──────────    │  Graph  | Dashboard            │                    │
│  Explore       │                                │  (empty when       │
│                │  (tab content)                 │   nothing          │
│                │                                │   selected)        │
│                │                                │                    │
├────────────────┴────────────────────────────────┴────────────────────┤
│  STATUS BAR: [ corpus path _________________________ ]  ● OK  ⚡     │
└─────────────────────────────────────────────────────────────────────┘
```

Four independent areas. Each has one job:

| Area | Job |
| --- | --- |
| Left panel | Query the corpus (Search + Explore) |
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
| Left panel content | search results / explore results | `searchStore`, `exploreStore` |

Changing one axis does not reset the others. Switching from Library to Graph
does not clear the subject rail. Running a search does not change the active tab.

---

## Left panel — Query interface

**Always visible.** Permanent on the left regardless of which main tab is active.

**Contents:**

- **Search** (primary): semantic FAISS-based corpus search. Full-width query
  form, results list below. The primary entry point into the corpus from any
  context. Keyboard shortcut `/` focuses the search input.
- **Explore** (secondary): GI cross-episode explore and natural-language
  queries. Collapsible section below Search.

**Purpose:** The left panel is the global query surface. Users do not need
to navigate away from what they are doing to search — Search is always
adjacent to whatever tab or subject is active.

**Visual spec:** UXS-005 (Search), UXS-001 (tokens)

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
time-bounded episode slice by default (see `graphLens` in GRAPH-INITIAL-LOAD
spec). The graph is another perspective on the same entities that Library
and Digest surface — not a separate information domain. See UXS-004.

### Dashboard

Operational intelligence. Permanent briefing card (last run, corpus health,
and action items) sits above three sub-tabs: Coverage, Intelligence, Pipeline.

See UXS-006 and DASHBOARD-SPEC.

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
  all health flag rows, Retry health button, and last error message.
- **Rebuild indicator** `⚡`: appears only when `reindex_recommended` is
  true from index stats. `warning` token. Clicking opens health popover
  scrolled to the index section.

**Visual spec:** UXS-001 status bar section

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
| API endpoints and data contracts | SERVER_GUIDE.md, RFC files |
| Implementation component tree | VIEWER_FRONTEND_ARCHITECTURE.md |

---

## Revision history

| Date | Change |
| --- | --- |
| 2026-04-19 | Initial draft; shell restructure; subject rules; status bar; flows |
