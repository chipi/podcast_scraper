# UXS-010: Person Profile

- **Status**: Draft
- **Authors**: Marko
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states
- **Related PRDs**:
  - [PRD-029: Person Profile](../prd/PRD-029-person-profile.md) -- full requirements
    and API contract
- **Related RFCs**:
  - [RFC-072: Canonical Identity Layer](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) --
    bridge artifact, Flagship 2, query Pattern B
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Related UX specs**:
  - [UXS-004: Graph Exploration](UXS-004-graph-exploration.md) -- "View in graph"
    handoff
  - [UXS-005: Semantic Search](UXS-005-semantic-search.md) -- search entry point
    for speaker click
  - [UXS-007: Topic Entity View](UXS-007-topic-entity-view.md) -- cross-linked from
    each topic row
  - [UXS-008: Enriched Search](UXS-008-enriched-search.md) -- enriched search entry
    point; speaker names in enriched sources open Person Landing
  - [UXS-009: Position Tracker](UXS-009-position-tracker.md) -- "Track positions"
    handoff from each topic row; hosted within the shared Person Landing defined
    by this UXS
- **Implementation paths**:
  - **Shipped (#672)**:
    `web/gi-kg-viewer/src/components/subject/PersonLandingView.vue` (rail panel
    with **Profile** / **Positions** ARIA tablist),
    `SubjectTimelineChart.vue` (Chart.js monthly mentions bar — same component
    TEV uses), `subjectMentionsTimeline.ts` (SPOKEN_BY edge walk for Person
    subjects). Subject store entry: `focusPerson(speaker_id)`. Mounts inside
    `SubjectRail` for `subject.kind === 'person'`. Profile tab holds aliases /
    description / edge counts / timeline; Positions tab lists SPOKEN_BY
    quotes with episode context (capped at `PERSON_LANDING_POSITIONS_CAP = 50`).
  - **Entry points (#674 item 4)**: Search supporting-quote speaker name
    (`search-result-speaker-link`) and Explore Top speakers rollup
    (`explore-top-speaker-link`) call `focusPerson(speaker_id)` and
    auto-load the corpus graph baseline so the panel has data to render.
  - **Existing graph-rail surface stays separate**:
    `GraphNodeRailPanel` + `NodeDetail` continue to render for graph clicks
    on Person nodes (`subject.kind === 'graph-node'`).
- **Shell IA:** [VIEWER_IA.md](VIEWER_IA.md) — Person subject in subject rail; navigation axes

---

## Summary

For shell layout, the three navigation axes, subject rail persistence and clearing, status bar, and first-run empty corpus behavior, see **[VIEWER_IA.md](VIEWER_IA.md)**. This document specifies **Person Profile** / Person Landing in the subject rail only.

The Person Profile is a person-anchored navigable surface that shows a person's
corpus presence at a glance: identity metadata, the list of topics they have
discussed (with Insight counts), and navigation links to drill deeper via the
Position Tracker or Topic Entity View. This UXS defines the visual contract for the
panel layout, section density, and graceful degradation states. All tokens reference
[UXS-001](UXS-001-gi-kg-viewer.md). Functional requirements are in
[PRD-029](../prd/PRD-029-person-profile.md).

---

## Principles

- **At-a-glance overview**: The profile is designed for quick scanning. The user
  should grasp the person's topic footprint within 5 seconds of opening the panel.
- **Navigation hub**: The profile is the starting point for deeper exploration.
  Every topic links to the Position Tracker and Topic Entity View.
- **Honest degradation**: When data is missing, the panel says so in plain language.
  No broken layouts, no spinners that never resolve.

---

## Scope

**In scope:**

- Person Profile right rail panel (graph and search entry points)
- Person Profile full-width layout when a **corpus-wide person browse** host exists
  (placement undecided; see [Full-width browse](#person-landing-fullwidth-browse))
- Person header (display name, slug, appearances, date range)
- Topic Overview section (topic list with Insight counts and navigation)
- Shared Person Landing component (tab bar hosting Profile and Position Tracker)
- All four degradation states (FR3.1 -- FR3.4)

**Non-goals:**

- Audio playback from timestamps (display only)
- Cross-corpus person views
- Analytical ranking, quote curation, or summary statistics -- out of scope

**Boundary note:** This UXS covers the **static visual contract** (tokens, layout,
component appearance, accessibility targets). Behavioral rules (animation timing,
debounce intervals, data-fetching strategies) belong in the related RFC. See the
[UXS vs RFC boundary](index.md#uxs-vs-rfc-boundary) guidance.

---

## Theme support

- **Mode:** both (follows system)
- **Primary palette:** dark -- the mode used as the design baseline
- **Breakpoints:** desktop only (right rail minimum width 360px)

---

## Placement

The Person Profile panel has two presentation modes:

- **Right rail** (from graph or search entry point): same panel slot used for graph
  node detail and episode detail. Panel width follows the existing rail width
  (UXS-001).
- **Full-width** (only once browse ships; same contract as
  [UXS-009](UXS-009-position-tracker.md) Placement): occupies the main content area
  instead of the right rail. Layout is the same vertical stack but with wider cards
  and side-by-side topic rows when space permits. **There is no browse host in the
  v2 main tab strip today** (see [Full-width browse](#person-landing-fullwidth-browse)).

The panel header shows "Person Profile" with the `gi` domain token color, reflecting
that the primary data source is GIL Insights.

---

## Entry points

- Clicking a Person node in the Cytoscape graph opens the Person Landing in the
  right rail.
- Clicking a speaker name in a search result card (including lifted results with
  `lifted.speaker` or enriched sources from UXS-008) opens the Person Landing in
  the right rail.

### Full-width browse (planned; shell gap) {#person-landing-fullwidth-browse}

**Implemented today:** Person Landing opens in the **right rail** from Graph Person
nodes, Semantic Search speaker affordances, and related handoffs from UXS-007 /
UXS-008 when those surfaces ship. The v2 viewer **main** tabs remain **Digest**,
**Library**, **Graph**, and **Dashboard** only; none of them hosts a corpus-wide
person roster or “open Person Landing full-width” control yet.

**Why this UXS still describes full-width:** Layout, density, and token rules for
Person Landing in the main column are part of the visual contract so work is not
blocked when browse exists.

**Where browse should live (product decision before implementation):** Pick one
primary host and record it in PRD-028, PRD-029, and the RFC-062 viewer checklist
(selectors and `E2E_SURFACE_MAP.md` follow the real control). Representative options:

1. **Fifth main tab** (working name **People**) -- corpus-wide list or search of
   `person:{slug}` that opens Person Landing full-width.
2. **Library drill-in** -- e.g. “People in this corpus” (or feed-scoped index) from
   the catalog surface.
3. **Graph- or shell-level affordance** -- e.g. filtered Person-node list, command
   palette, or header action that navigates with the same Person Landing component
   in `main` layout instead of the rail.

Until one of these (or an equivalent) ships, **automated E2E** should target **rail**
entry points only; do not add Playwright coverage for a top-level People tab that
does not exist.

---

## Person Landing component

This UXS **owns the shared Person Landing** -- the entry-point surface that hosts
both the Person Profile and the Position Tracker ([UXS-009](UXS-009-position-tracker.md)).

### Structure

- The Person Landing is a container that shows the **person header** (shared between
  both views) and a **tab bar** to switch between "Profile" and "Positions".
- Default tab: "Profile" (Person Profile).
- When the entry point is topic-specific (e.g. a "Track positions" link from
  PRD-026 with a preselected topic), the "Positions" tab is preselected with that
  topic active in the Position Tracker's topic selector.

### Tab bar

- Two tabs: "Profile" and "Positions".
- Tab style: `surface` background, `border` bottom border. Active tab uses `primary`
  bottom border (2px) and `primary` text. Inactive tab uses `muted` text.
- Tab bar sits below the person header and above the view-specific content.
- Keyboard-navigable: left/right arrow keys switch tabs, Enter/Space activates.

### Shared person header

The person header is rendered once in the Person Landing container (not duplicated
per view). Both the Person Profile and Position Tracker inherit the same header:

- Person display name (`text-lg font-semibold`) and canonical `person:{slug}` ID
  (`muted`, `text-xs`, monospace).
- Appearance count badge (`surface` chip, `text-sm`): "N episodes".
- Date range (first and last appearance) in `muted` `text-xs`.

### Implementation

- New: `web/gi-kg-viewer/src/components/person/PersonLanding.vue`
- Hosts: `PersonProfile.vue` (this UXS) and `PositionTracker.vue` (UXS-009)
- Both child components omit their own person headers and rely on the landing's
  shared header.

---

## Section layout

The panel is a single scrollable column on `surface` background. Sections stack
vertically with `border` dividers between them.

### Person header

- Person display name (`text-lg font-semibold`) and canonical `person:{slug}` ID
  (`muted`, `text-xs`, monospace).
- Appearance count badge (`surface` chip, `text-sm`): "N episodes".
- Date range (first and last appearance) in `muted` `text-xs`.

### Action buttons

Below the header, a row of action buttons:

- **View in graph** -- navigates to the Graph tab and focuses on this person's node.
  Uses `primary` token.
- **Track positions** -- navigates to the Position Tracker (PRD-028) for this person
  (topic selector opens there). Uses `primary` token.

### Topic Overview section

Section header: "Topics" (`text-base font-semibold`).

- A vertical list of topic rows, sorted by Insight count descending (most-discussed
  topic first).
- Each topic row (`surface` background, `border` bottom border):
  - Topic display name (`text-sm font-medium`).
  - Insight count badge (`surface` chip, `text-xs`): "N insights".
  - Two action links (`link` token, `text-xs`):
    - "Track positions" -- navigates to Position Tracker for this person + topic.
    - "View topic" -- navigates to Topic Entity View (PRD-026).
- When the person has no topics: the section shows "No grounded insights found for
  this person. Insights appear when the pipeline runs with GIL extraction enabled."
  (`muted`, `text-sm`, centered).

---

## Graceful degradation

All degradation states use `muted` text and honest language. No broken layouts.

- **Person has 0 Insights** (FR3.1): Person header shows with KG metadata. Topic
  Overview shows: "No grounded insights found for this person. Insights appear when
  the pipeline runs with GIL extraction enabled." (`muted`, `text-sm`, centered).
- **Topic has 0 Insights** (FR3.2): Topic row appears with "0 insights" badge.
- **Lift fails** (FR3.3): Search result card shows raw chunk without speaker link.
  No Person Profile navigation offered. (This state is in the search panel, not the
  Person Profile panel itself.)
- **No bridge.json for all episodes** (FR3.4): Profile area shows: "No cross-layer
  data available. Run the pipeline with bridge generation enabled." (`muted`,
  `text-sm`, centered). Person header still shows KG-derived metadata if available.

---

## E2E contract

New visible labels and selectors require updates to the
[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
before or with implementation. Key surfaces:

- Right rail panel with `aria-label="Person Profile"` (use `exact: true`)
- Full-width Person Landing host (after browse placement is chosen and built)
- Person header (display name, appearance count)
- Topic Overview section: topic rows, Insight count badges, "Track positions" and
  "View topic" links
- "View in graph" and "Track positions" buttons
- All four degradation empty states

---

## Accessibility

- **Focus:** Visible focus ring on all interactive elements (topic rows, buttons,
  links). Uses `primary` token, 2px solid, 2px offset.
- **Contrast:** WCAG AA for all text on `surface` backgrounds. Badge text meets
  contrast requirements against badge backgrounds.
- **Keyboard:** Topic rows and links are focusable. Tab bar is keyboard-navigable
  (left/right arrow keys, Enter/Space).
- **Screen reader:** Topic rows use `role="listitem"` within a `role="list"`
  container. Person header uses semantic heading hierarchy.

---

## Revision history

| Date       | Change                                                         |
| ---------- | -------------------------------------------------------------- |
| 2026-04-13 | Initial draft (PRD-029 companion)                              |
| 2026-04-14 | Rewritten as Person Profile (commercial content extracted)     |
| 2026-04-19 | Shell gap: full-width browse not in main tabs yet              |
