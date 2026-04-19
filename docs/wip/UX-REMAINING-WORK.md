# UX Design — Remaining Work & Gap Tracker

**Created:** April 2026 (post design session)
**Purpose:** Delegate remaining UX spec work after the main design session.
**Context:** A full-day design session produced 6 implementation specs
(shell restructure, graph gesture overlay, graph initial load, graph visual
styling, digest/library improvements, dashboard). This document tracks what
still needs to be done to close the loop — spec gaps, cross-spec
inconsistencies, and UXS-001 consolidation work.

---

## How to use this document

Work items are grouped into three tiers:

- **Tier 1 — Must do before any implementation starts.** Agents building
  against the current UXS files will build the wrong thing without these.
- **Tier 2 — Close loose ends.** Small amendments and decisions that are
  unresolved but not blocking Phase 1 implementation.
- **Tier 3 — Worth doing, not blocking.** Design improvements that came
  out of the session but don't have specs yet.

Each item has enough context to delegate independently. Cross-references
to the relevant spec files are included.

---

## Output files from the design session

These exist as downloadable specs ready for implementation:

| File | What it covers |
|---|---|
| `VIEWER-SHELL-RESTRUCTURE.md` | Right rail → Subject Panel; Left panel → Search/Explore; Status bar for corpus path + health |
| `GRAPH-GESTURE-OVERLAY.md` | One-time dismissible gesture hint overlay on graph canvas |
| `GRAPH-INITIAL-LOAD.md` | graphLens, 15-episode cap, GraphStatusLine, default node type visibility |
| `GRAPH-VISUAL-STYLING.md` | Phase 1 stylesheet changes + Phase 2 computed property tints |
| `DIGEST-LIBRARY-UX-IMPROVEMENTS.md` | 9 improvements across Digest and Library |
| `DASHBOARD-SPEC.md` | Full dashboard rewrite: briefing card + 3 tabs + Tufte charts + 2 new endpoints |

---

## Tier 1 — Must do before implementation starts

### T1-A: UXS-001 new shell sections

**What:** UXS-001 is the shared design system and the single source of truth.
After the shell restructure, it still describes the old architecture (Corpus
tab, API·Data tab, mixed right rail). An implementing agent reading UXS-001
without the new specs would build the wrong thing. Three new sections need
to be added and two old sections deprecated.

**Sections to ADD to UXS-001:**

1. `## Shell layout` — three-column layout diagram:
   - Left panel = Search + Explore (always visible, primary query interface)
   - Main area = Digest | Library | Graph | Dashboard (tabs)
   - Right rail = Subject Panel (episode, topic, person, GIL node)
   - Status bar = corpus path + health (bottom of shell, always visible)
   - ASCII or prose diagram; same format as existing UXS-001 layout sections

2. `## Right rail — Subject Panel` — the new conceptual model for the rail:
   - Opening principle: "The subject panel is the single context layer.
     Every entity in the platform surfaces through it consistently
     regardless of which tab or query produced it."
   - Subject type table (Episode / Topic / Person / GIL node — entry points
     and component per type)
   - Persistence rule: subject stays in rail when switching main tabs
   - Clearing rule: explicit close (×), selecting different subject,
     corpus path change
   - × close button spec: placement (rail header, right-aligned), size,
     token (muted, hover surface-foreground), aria-label "Close [subject
     type] detail"
   - Empty state when nothing selected: muted text — "Select an episode,
     topic, or graph node to see details here."

3. `## Status bar` — visual spec:
   - Corpus path: always-visible editable text input, placeholder "Set
     corpus path…", `surface` background, `border`, full left width
   - Folder picker icon button: right of path input, offline mode only
   - Health dot: 8px filled circle, `success`/`warning`/`danger` token
     depending on health status. Clicking opens a popover.
   - Health popover: anchored to dot, `elevated` background, lists all
     health flag rows (same as current API section), plus Retry health
     button and last error when health fails
   - Rebuild indicator ⚡: appears only when `reindex_recommended` is
     true. `warning` token. Clicking opens health popover scrolled to
     index section.
   - Status bar height: ~36px. Background: `canvas`. Top border: `border`.
   - Tunable parameter: status bar height 36px (Open)

4. `## Left panel — Query Interface` — brief description:
   - Left panel contains Search (primary, always visible) and Explore
     (secondary, collapsible section below search)
   - No tabs, no switcher — replaced the old Corpus | API·Data tab structure
   - Reference to UXS-005 for Search form spec (with note about new
     left-panel placement context — see T1-C)

**Sections to DEPRECATE in UXS-001:**

5. `## API · Data (left panel)` — mark as Deprecated. Add note:
   "Superseded by the status bar (health + corpus path) and the
   VIEWER-SHELL-RESTRUCTURE spec. This section describes the old
   left panel design. Do not implement."

6. `## Corpus path (left Corpus tab)` — mark as Deprecated. Add note:
   "Superseded by the status bar corpus path input. See Shell layout
   and Status bar sections above."

**Files to edit:** `docs/uxs/UXS-001-gi-kg-viewer.md`

---

### T1-B: UXS-001 tunable parameters table update

**What:** 15 new tunable parameters were introduced across today's specs.
They need to be added to the tunable parameters table in UXS-001 so they're
in one place and can be frozen when values are confirmed.

**Parameters to add:**

From `GRAPH-INITIAL-LOAD.md`:
| Parameter | Default | Status | Source |
|---|---|---|---|
| Graph default episode cap | 15 | Open | GRAPH-INITIAL-LOAD §4 |
| Graph recency seed default | 7d | Open | GRAPH-INITIAL-LOAD §3 |

From `GRAPH-VISUAL-STYLING.md`:
| Parameter | Default | Status | Source |
|---|---|---|---|
| COSE ABOUT edge ideal length | 80px | Open | GRAPH-VISUAL-STYLING §3.7 |
| COSE MENTIONS edge ideal length | 150px | Open | GRAPH-VISUAL-STYLING §3.7 |
| Recency decay window | 90 days | Open | GRAPH-VISUAL-STYLING §4.1 |
| Recency minimum opacity | 0.4 | Open | GRAPH-VISUAL-STYLING §4.1 |
| Degree heat max degree | 30 | Open | GRAPH-VISUAL-STYLING §4.3 |
| Label zoom threshold (hide all) | 0.5 | Open | GRAPH-VISUAL-STYLING §3.5 |
| Label zoom threshold (full) | 1.0 | Open | GRAPH-VISUAL-STYLING §3.5 |
| Compound fill opacity | 0.06 | Open | GRAPH-VISUAL-STYLING §3.6 |

From `DIGEST-LIBRARY-UX-IMPROVEMENTS.md`:
| Parameter | Default | Status | Source |
|---|---|---|---|
| Similarity strong threshold | 0.85 | Open | DIGEST-LIBRARY §2.3 |
| Similarity good threshold | 0.70 | Open | DIGEST-LIBRARY §2.3 |
| Recency dot window | 24h | Open | DIGEST-LIBRARY §4 |

From `DASHBOARD-SPEC.md`:
| Parameter | Default | Status | Source |
|---|---|---|---|
| GI coverage warning threshold | 50% | Open | DASHBOARD §3.4 |
| Index coverage warning threshold | 60% | Open | DASHBOARD §3.4 |
| Dashboard action items max | 3 | Open | DASHBOARD §3.5 |
| Top voices limit | 5 | Open | DASHBOARD §5.3 |

**Also update:** Remove the word "optional" from the insight line rule
in UXS-001. It currently reads "optional insight line under each chart
when the data supports a clear takeaway." Change to: "Mandatory insight
line under every chart. If the data does not support a clear takeaway,
the chart does not belong on the dashboard."

**Files to edit:** `docs/uxs/UXS-001-gi-kg-viewer.md`

---

### T1-C: UXS-005 placement update

**What:** UXS-005 (Semantic Search) was written for the right-rail context.
After the shell restructure, Search lives in the left panel as the primary
permanent query interface. The form layout spec may need adjustments for
wider available width. The current UXS-005 doesn't know about this.

**Changes needed:**

1. Add a `## Placement` section at the top of UXS-005:
   "The Search panel lives in the **left panel** (permanent, always visible)
   after the shell restructure (VIEWER-SHELL-RESTRUCTURE.md). It was
   previously in the right rail. The form layout below was designed for a
   narrow rail context; implementers should verify that form element sizing
   is appropriate for the wider left panel at its current width."

2. Note that the `/` keyboard shortcut focuses the search input in the
   left panel (not the right rail as before). Update any reference to
   "right rail tools mode" or "paneKind = tools" — these are deprecated.

3. The `SearchPanel.vue` component moves from
   `src/components/search/SearchPanel.vue` (unchanged) to being rendered
   inside `LeftPanel.vue` — no rename needed but the import location changes.

**Files to edit:** `docs/uxs/UXS-005-semantic-search.md`

---

### T1-D: Briefing card empty state for no corpus path

**What:** After the shell restructure, corpus path lives in the status bar.
The DASHBOARD-SPEC briefing card doesn't specify what shows when no corpus
path is set. If the briefing card tries to load coverage/runs data without
a corpus path, it will fail. The empty state needs to point to the status
bar, not to a "Corpus tab" that no longer exists.

**Change needed:**

Add to `DASHBOARD-SPEC.md` Section 3 (Briefing Card), under a new
`### No corpus path state` heading:

"When `shellStore.corpusPath` is null or empty, the briefing card renders
a single `elevated` card with muted text:

```
Set a corpus path in the status bar below to begin.
```

No Last run / Corpus health / Action items sections are rendered. The
message is centred in the card area, same height as the loaded card.
`data-testid='briefing-no-corpus'`"

**Files to edit:** `DASHBOARD-SPEC.md` (in the spec output files, not repo)

---

### T1-E: subjectStore dependency note for Digest/Library line clamp

**What:** `DIGEST-LIBRARY-UX-IMPROVEMENTS.md` specifies that the summary
preview unclamps when a row "has bg-overlay / is the active subject." This
was written before the shell restructure was finalised. After the restructure,
the "active subject" is determined by `subjectStore.episodeMetadataPath ===
row.metadata_path` (not by `episodeRail.paneKind` as before). The implementing
agent must use `subjectStore`, not the deprecated `episodeRail`.

**Change needed:**

Add a note to `DIGEST-LIBRARY-UX-IMPROVEMENTS.md` Section 2.5 and 3.1
(both line clamp sections):

"**Shell restructure dependency:** 'Selected' row means
`subjectStore.episodeMetadataPath === row.metadata_path` after the shell
restructure (VIEWER-SHELL-RESTRUCTURE.md). Do not use `episodeRail.paneKind`
— that store is deprecated. Import and read from `subjectStore` instead."

**Files to edit:** `DIGEST-LIBRARY-UX-IMPROVEMENTS.md` (in spec output files)

---

### T1-F: Amber/orange token clarification in UXS-001

**What:** `DIGEST-LIBRARY-UX-IMPROVEMENTS.md` changes CIL topic pills from
amber/orange fill to `kg` violet (to align with graph TopicCluster nodes).
After this change, amber/orange (`warning` token fill) is used for exactly
one thing in the viewer: search hit emphasis on graph Quote/search nodes.
UXS-001 needs a clarification to prevent future confusion.

**Change needed:**

In UXS-001 under Domain tokens section, add a note after the `gi`/`kg` token
definitions:

"**`warning` token and cluster colours:** After UXS-002 update (CIL pill
alignment), `warning` fill is used for search hit emphasis on graph nodes
only. Topic cluster membership is always `kg` token — in Digest CIL pills,
TopicCluster compound nodes in Graph, and Intelligence tab cluster cards.
Do not use `warning` fill for anything cluster-related."

**Files to edit:** `docs/uxs/UXS-001-gi-kg-viewer.md`

---

## Tier 2 — Close loose ends

### T2-A: position_hint tooltip — UXS-009 amendment

**What:** UXS-009 (Position Tracker) specifies the `position_hint` bar as
"a small horizontal bar, tooltip shows the numeric value." The numeric value
(0.0–1.0) is meaningless to a user. The tooltip should show a human-readable
position label.

**Change needed:**

In `docs/uxs/UXS-009-position-tracker.md`, update the `position_hint`
tooltip spec:

"Tooltip: if episode `duration_ms` is available, compute position as
`position_hint × duration_ms` and display as `~[M]m [S]s into episode`.
If duration is unavailable, display a tier label:
- 0.0–0.33: "Early in episode"
- 0.34–0.66: "Mid episode"
- 0.67–1.0: "Late in episode"

Never show the raw 0.0–1.0 float value in the tooltip."

**Files to edit:** `docs/uxs/UXS-009-position-tracker.md`

---

### T2-B: Enhanced chip error state — UXS-008 amendment

**What:** UXS-008 (Enriched Search) specifies that the "Enhanced" chip appears
when `enriched_search_available: true` from health. When an enrichment call
fails (timeout/provider error), the panel shows an error state but the chip
stays in its normal `gi` border style. Users who configured enrichment but get
failures have no chip-level signal that something is wrong.

**Change needed:**

In `docs/uxs/UXS-008-enriched-search.md`, update the Enhanced chip spec:

"When the most recent enrichment call failed (timeout or provider error,
visible from the panel's error state), the "Enhanced" chip shows a subtle
`warning` border instead of the normal `gi` border. Background and text are
unchanged. Native `title` tooltip: 'Enrichment configured but last call
failed.' On successful call, chip reverts to `gi` border. This gives users
who configured an LLM enricher a signal when it's broken, without requiring
them to scroll to the panel error state."

**Files to edit:** `docs/uxs/UXS-008-enriched-search.md`

---

### T2-C: Intelligence tab enricher availability signal

**What:** The Intelligence tab (DASHBOARD-SPEC.md) has topic momentum and
emerging connections sections that degrade gracefully when enricher data is
absent. But the spec doesn't say how the viewer knows enricher data is
available. Topic clusters uses a 404/200 check on `GET /api/corpus/topic-
clusters`. The same pattern should apply to enricher data.

**Change needed:**

Add to `DASHBOARD-SPEC.md` Section 5 (Intelligence Tab) under a new
`### Enricher data availability` heading:

"Topic momentum and emerging connections check for their data via endpoint
existence, same pattern as topic clusters:

- `temporal_velocity`: `GET /api/corpus/enrichments/temporal-velocity` —
  200 = data available, 404 = not built. When 200: render momentum section.
  When 404: render degraded state.
- `topic_cooccurrence`: `GET /api/corpus/enrichments/topic-cooccurrence` —
  same pattern.

These endpoint paths are placeholders pending RFC-073 implementation. Update
paths when RFC-073 ships. The health check pattern (404 = graceful degrade)
is frozen; the paths are not."

**Note:** This requires coordination with RFC-073 to confirm the actual
endpoint paths when enrichers ship. Flag this as an open item in RFC-073.

**Files to edit:** `DASHBOARD-SPEC.md` (in spec output files)

---

### T2-D: Left panel collapse decision

**What:** The left panel (Search + Explore) is now a permanent primary
interface. On the Graph tab, a developer may want the full viewport width
for the canvas. Whether the left panel is collapsible is unspecified.

**Decision needed (product choice — Marko to decide):**

Option A: Left panel is always visible, never collapsible. Simplest
implementation. Graph canvas is narrower but consistent.

Option B: Left panel is collapsible via a toggle button (chevron/arrow)
on the panel edge. When collapsed: panel shrinks to ~40px showing only
a search icon and explore icon. Clicking expands. State persisted in
localStorage.

Option C: Left panel auto-collapses when Graph tab is active, auto-expands
on other tabs. Behaviour driven by main tab state.

**Recommendation:** Option B. The left panel is a query surface — users
need it most when browsing Digest/Library/Search, least when exploring the
graph. A manual toggle respects user intent without automatic surprises.

Once decided, add to `VIEWER-SHELL-RESTRUCTURE.md` Phase 1 and create
a UXS-001 amendment.

---

### T2-E: Three-column viewport measurement

**What:** VIEWER-SHELL-RESTRUCTURE notes "Left panel width — check after
Phase 3" as an open question. After Phase 3, measure actual rendered widths
at 1024px minimum viewport and check whether the Search form in the left
panel has sufficient width for comfortable use.

**Action needed (post Phase 3 implementation):**

At 1024px viewport:
- Left panel: current `w-72` = 288px
- Right rail: similar width
- Main area: remainder = 1024 - 288 - 288 - (padding) ≈ 400px

400px main area is tight for Library (which has a filter section + episode
list). If this is too narrow:
- Option A: reduce left panel to `w-64` (256px) — acceptable for Search
- Option B: reduce right rail to `w-64` — acceptable for subject panel
- Option C: set 1280px as the minimum viewport (tighter than current 1024px)

**Track as:** post-Phase-3 implementation task. Add to UXS-001 breakpoints
section once measured.

---

## Tier 3 — Worth doing, not blocking

### T3-A: Day one developer experience

**What:** A developer who has never run the pipeline opens the viewer.
Status bar: empty corpus path. Health dot: unreachable. Briefing card:
empty state. All tabs: empty states. Left panel search: nothing to search.
This is a cold blank screen. No spec exists for this first-time experience.

**Proposed approach:**

Add a `## First-time / empty corpus state` section to UXS-001 that specifies:

- Status bar empty state: path field shows placeholder "Set corpus path…"
  with a pulsing subtle border to draw attention (one-time, disappears once
  any path is set)
- Briefing card empty state: single card reading "Set a corpus path in the
  status bar below to begin. Then run `podcast scrape` to populate your
  corpus."
- All main tabs show a consistent empty state with the same message plus
  the specific thing that tab needs ("No episodes yet" for Library,
  "No graph artifacts yet" for Graph, etc.)
- Left panel search input disabled with tooltip: "Set corpus path to enable
  search"
- Overall: the viewer should feel intentionally empty, not broken. The
  distinction is whether there are clear next steps. There should be.

**Effort:** Small — mostly empty state strings and a few conditional
renders. High value for first impressions.

---

### T3-B: Left panel Search as primary query interface — UXS-005 redesign

**What:** Moving Search to the left panel is not just a location change —
it's a product statement. Search is now the permanent global entry point
into the corpus. UXS-005 was written for a narrow right-rail tool. It needs
to be rethought at a slightly higher level:

- The heading "Semantic search" should be prominent, not muted — it's the
  primary panel label now, not a section within a rail
- The query textarea should be full-width of the left panel
- "Advanced search" link feels more important in the primary position —
  consider surfacing it as a visible icon button rather than an underlined
  text link
- The results list is now permanent — it doesn't disappear when you select
  a result (because the subject opens in the right rail, not replacing the
  search). Results and subject panel coexist. This is a new interaction
  pattern not currently specified in UXS-005.

**Effort:** Medium. This is a UXS-005 rewrite focused on the new context,
not a ground-up redesign. Most of the detailed spec (advanced filters,
result card chips, lifted GI insight) stays the same.

---

### T3-C: Bad corpus path health messaging

**What:** With localStorage persistence for corpus path, the viewer can
cold-start pointing at an old or moved corpus path. Two different failure
modes need different messages:

1. "Server unreachable" — `podcast serve` is not running at all
2. "Server reachable, corpus path invalid/moved" — server is running but
   returns errors for the specified path

Currently the health dot just shows `danger` for both. The messages in the
health popover should distinguish:

- Server unreachable: "Server not reachable at [host]. Start `podcast serve`."
- Bad path: "Server is reachable but corpus path [path] returned an error.
  Check that the path exists and contains corpus data."

**Effort:** Small. Server-side: the health endpoint already distinguishes
these via HTTP status and error fields. Client-side: add conditional message
logic in the health popover.

---

### T3-D: RFC-062 update

**What:** RFC-062 is the umbrella viewer RFC. The shell restructure
(subjectStore replacing episodeRail, left panel restructure, status bar) is
a significant architectural change. RFC-062 should be updated to reflect the
new architecture, or a new RFC should be created that references it.

**Recommended:** Create `RFC-076-viewer-shell-restructure.md` that:
- References VIEWER-SHELL-RESTRUCTURE.md as the implementation spec
- Records the behavioral rules (debounce, transitions, stash removal) that
  belong in RFC territory
- Updates the delivered scope table in RFC-062 to reference RFC-076

The VIEWER-SHELL-RESTRUCTURE spec explicitly notes that behavioral timing
belongs in RFC-062. Until RFC-062 is updated or RFC-076 exists, those rules
are unspecified.

**Effort:** Medium. More writing than design work.

---

### T3-E: CLAUDE.md and .cursorrules update

**What:** CLAUDE.md and `.cursorrules` load mandatory context for every
Claude Code session. After the shell restructure, the component tree changes
significantly:
- `subjectStore` replaces `episodeRail`
- `SubjectRail.vue` is new
- `LeftPanel.vue` is new
- `StatusBar.vue` is new
- `episodeRail.ts` is deleted

If CLAUDE.md still references `episodeRail`, every Claude Code agent will
get confused context about the architecture.

**Action:** After shell restructure ships (Phase 3 complete), update CLAUDE.md
and `.cursorrules` to reflect new store names, component names, and remove
deprecated references. This is operational, not UX, but affects all future
agent work on the viewer.

---

## Summary table

| ID | Tier | Area | Files to edit | Effort |
|---|---|---|---|---|
| T1-A | 1 | UXS-001 new shell sections | UXS-001 | Medium |
| T1-B | 1 | UXS-001 tunable params | UXS-001 | Low |
| T1-C | 1 | UXS-005 placement update | UXS-005 | Low |
| T1-D | 1 | Briefing card empty state | DASHBOARD-SPEC.md | Low |
| T1-E | 1 | subjectStore note for line clamp | DIGEST-LIBRARY-UX-IMPROVEMENTS.md | Low |
| T1-F | 1 | Amber/orange clarification | UXS-001 | Low |
| T2-A | 2 | position_hint tooltip | UXS-009 | Low |
| T2-B | 2 | Enhanced chip error state | UXS-008 | Low |
| T2-C | 2 | Intelligence tab enricher signal | DASHBOARD-SPEC.md | Low |
| T2-D | 2 | Left panel collapse decision | VIEWER-SHELL-RESTRUCTURE.md + UXS-001 | Low (decision) |
| T2-E | 2 | Three-column viewport measurement | Post Phase 3 task | Low (measurement) |
| T3-A | 3 | Day one experience | UXS-001 new section | Low |
| T3-B | 3 | UXS-005 redesign for left panel | UXS-005 | Medium |
| T3-C | 3 | Bad corpus path messaging | UXS-001 + health popover | Low |
| T3-D | 3 | RFC-062 / RFC-076 update | New RFC-076 | Medium |
| T3-E | 3 | CLAUDE.md + .cursorrules update | CLAUDE.md, .cursorrules | Low |

**Tier 1 total:** 6 items, all low-medium effort. Do these before implementation.
**Tier 2 total:** 5 items, all low effort or decision-only. Do alongside Phase 1.
**Tier 3 total:** 5 items, none blocking. Schedule after implementation phases.

---

## Spec dependencies map

If delegating to multiple agents, this shows which specs depend on each other:

```
UXS-001 (T1-A, T1-B, T1-F)
  └─ everything references UXS-001; update this first

VIEWER-SHELL-RESTRUCTURE.md
  └─ DIGEST-LIBRARY line clamp (T1-E) — needs subjectStore note
  └─ UXS-005 (T1-C) — placement update
  └─ T2-D collapse decision
  └─ T3-E CLAUDE.md update (post Phase 3)

DASHBOARD-SPEC.md
  └─ T1-D briefing card empty state
  └─ T2-C enricher signal

UXS-008, UXS-009 — independent, no cross-dependencies
RFC-062 / RFC-076 (T3-D) — independent, can be done any time
```

Safe to parallelise: T1-B, T1-C, T1-D, T1-E, T1-F, T2-A, T2-B all touch
different files with no dependencies on each other.

T1-A (UXS-001 shell sections) should ideally be done first since multiple
other items reference it, but it doesn't block the others from starting.
