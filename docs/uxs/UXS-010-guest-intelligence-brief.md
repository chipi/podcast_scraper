# UXS-010: Guest Intelligence Brief

- **Status**: Draft
- **Authors**: Marko
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states
- **Related PRDs**:
  - [PRD-029: Guest Intelligence Brief](../prd/PRD-029-guest-intelligence-brief.md) --
    full requirements, API contract, Phase 6 contradiction detection vision
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
    each topic group
  - [UXS-008: Enriched Search](UXS-008-enriched-search.md) -- enriched search entry
    point; speaker names in enriched sources open Person Landing
  - [UXS-009: Position Tracker](UXS-009-position-tracker.md) -- "Track positions"
    handoff from each topic group; hosted within the shared Person Landing defined
    by this UXS
- **Implementation paths**:
  - New: `web/gi-kg-viewer/src/components/person/GuestBrief.vue`
  - Existing: `web/gi-kg-viewer/src/components/graph/GraphNodeRailPanel.vue` (rail
    host)
  - Existing: `web/gi-kg-viewer/src/stores/` (shared `person.ts` store with
    Position Tracker)

---

## Summary

The Guest Intelligence Brief is a person-anchored navigable surface that shows a
structured dossier of a person's corpus presence: known positions grouped by topic,
best quotes ranked by quality, and corpus-wide summary statistics. This UXS defines
the visual contract for the panel layout, section density, card components, Phase 6
placeholder regions, and graceful degradation states. All tokens reference
[UXS-001](UXS-001-gi-kg-viewer.md). Functional requirements are in
[PRD-029](../prd/PRD-029-guest-intelligence-brief.md).

---

## Principles

- **Scannable dossier**: The brief is designed for quick scanning. The user should
  grasp the person's key positions within 10 seconds of opening the panel.
- **Grounding-first**: Every quote is verbatim with a timestamp. Every Insight
  carries a grounding badge. The user always knows what is evidence vs
  interpretation.
- **Honest degradation**: When data is missing, the panel says so in plain language.
  No broken layouts, no spinners that never resolve.

---

## Scope

**In scope:**

- Guest Brief right rail panel (graph and search entry points)
- Guest Brief full-width view (browse entry point)
- Person header with grounding rate
- Topic summary stats bar
- Known Positions section (collapsible topic groups with strongest Insight)
- Best Quotes section (ranked list)
- Phase 6 Potential Challenges section (placeholder)
- All five degradation states (FR3.1 -- FR3.5)

**Non-goals:**

- Audio playback from timestamps (display only)
- Cross-corpus person views
- Phase 6 analysis UI implementation (only placeholder region)
- Position-change detection within a single person's arc (PRD-028 scope)

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

The Guest Brief panel has two presentation modes:

- **Right rail** (from graph or search entry point): same panel slot used for graph
  node detail and episode detail. Panel width follows the existing rail width
  (UXS-001).
- **Full-width** (from browse entry point): occupies the main content area. Layout
  is the same vertical stack but with wider cards and side-by-side topic groups
  when space permits.

The panel header shows "Guest Brief" with the `gi` domain token color, reflecting
that the primary data source is GIL Insights.

---

## Entry points

- Clicking a Person node in the Cytoscape graph opens the Person Landing in the
  right rail.
- Clicking a speaker name in a search result card (including lifted results with
  `lifted.speaker` or enriched sources from UXS-008) opens the Person Landing in
  the right rail.
- A dedicated person browse (accessible from the viewer navigation) opens the
  Person Landing in full-width mode.

---

## Person Landing component

This UXS **owns the shared Person Landing** -- the entry-point surface that hosts
both the Guest Brief and the Position Tracker ([UXS-009](UXS-009-position-tracker.md)).

### Structure

- The Person Landing is a container that shows the **person header** (shared between
  both views) and a **tab bar** to switch between "Brief" and "Positions".
- Default tab: "Brief" (Guest Brief).
- When the entry point is topic-specific (e.g. a "Track positions" link from
  PRD-026 with a pre-selected topic), the "Positions" tab is pre-selected with that
  topic active in the Position Tracker's topic selector.

### Tab bar

- Two tabs: "Brief" and "Positions".
- Tab style: `surface` background, `border` bottom border. Active tab uses `primary`
  bottom border (2px) and `primary` text. Inactive tab uses `muted` text.
- Tab bar sits below the person header and above the view-specific content.
- Keyboard-navigable: left/right arrow keys switch tabs, Enter/Space activates.

### Shared person header

The person header is rendered once in the Person Landing container (not duplicated
per view). Both the Guest Brief and Position Tracker inherit the same header:

- Person display name (`text-lg font-semibold`) and canonical `person:{slug}` ID
  (`muted`, `text-xs`, monospace).
- Appearance count badge (`surface` chip, `text-sm`): "N episodes".
- Date range (first and last appearance) in `muted` `text-xs`.
- Grounding rate badge (from Guest Brief data): visible on both tabs.

### Implementation

- New: `web/gi-kg-viewer/src/components/person/PersonLanding.vue`
- Hosts: `GuestBrief.vue` (this UXS) and `PositionTracker.vue` (UXS-009)
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
- Grounding rate badge: percentage pill. Intent tokens: `success` for >= 80%,
  `muted` for 50--79%, `warning` for < 50%. Hidden when no Insights exist (FR3.1).

### Action buttons

Below the header, a row of action buttons:

- **View in graph** -- navigates to the Graph tab and focuses on this person's node.
  Uses `primary` token.
- **Track positions** -- navigates to the Position Tracker (PRD-028) for this person
  (topic selector opens there). Uses `primary` token.

### Topic summary stats bar

A horizontal row of four metric chips (`surface` background, `border`, `text-sm`):

- **Topics**: total topic count.
- **Insights**: total Insight count.
- **Quotes**: total quote count.
- **Grounding**: grounding rate as percentage.

Each chip has a label (`muted`, `text-xs`) above the value (`text-sm font-medium`).
When all values are zero (FR3.1), the bar shows with zero values and `disabled` text.

### Known Positions section

Section header: "Known Positions" (`text-base font-semibold`).

- Topics are listed as collapsible groups on `surface` background, sorted by Insight
  count descending.
- Each topic group header (collapsed state):
  - Topic display name (`text-sm font-medium`).
  - Insight count badge (`surface` chip, `text-xs`): "N insights".
  - Strongest Insight text preview (`muted`, `text-sm`, truncated to ~100
    characters with ellipsis).
  - A chevron icon indicating expand/collapse state.
  - Two action links (`link` token, `text-xs`):
    - "Track positions" -- navigates to Position Tracker for this person + topic.
    - "View topic" -- navigates to Topic Entity View (PRD-026).
- Each topic group (expanded state):
  - All Insights for this topic, each as a compact card following the **shared
    InsightCard component** from [UXS-001](UXS-001-gi-kg-viewer.md). This view
    uses the following InsightCard slots: insight text, `insight_type` badge,
    `position_hint` bar, confidence score, grounding badge, and episode
    attribution:
    - Insight text (`text-sm`).
    - `insight_type` badge: small pill (`text-xs`). Token mapping matches
      UXS-009: "claim" -> `primary`, "recommendation" -> `success`,
      "observation" -> `muted`, "question" -> `link`, other -> `muted`.
    - `position_hint` indicator: small horizontal bar (same as UXS-009).
    - Confidence score (`muted`, `text-xs`). Hidden when unavailable.
    - Grounding badge: "grounded" (`success`, `text-xs`) or "ungrounded"
      (`warning`, `text-xs`).
    - Episode title and publish date (`muted`, `text-xs`).
  - The `strongest_insight` card is visually distinguished: `primary` left border
    (3px) and a small "Strongest" label (`primary`, `text-xs`).

### Best Quotes section

Section header: "Best Quotes" (`text-base font-semibold`).

- A vertical list of quote cards, ranked by the server (FR1.4 in PRD-029).
- Each quote card:
  - Blockquote style: `border` left border (3px), `surface` background, indent
    (padding-left 12px).
  - Quote text (`text-sm`, italic, `canvas-foreground`).
  - Topic tag: small chip (`surface` background, `kg` domain token border,
    `text-xs`) with topic display name. Clickable -- navigates to Topic Entity
    View.
  - Episode title and publish date (`muted`, `text-xs`).
  - Timestamp display (`muted`, `text-xs`): "MM:SS -- MM:SS". Hidden when
    timestamps are unavailable.
- Default: 10 quotes shown. "Show more quotes" control (`primary`, `text-sm`) loads
  additional quotes.

### Potential Challenges section (Phase 6)

- Visible only when `potential_challenges` is non-empty.
- Section header: "Potential Challenges" (`text-base font-semibold`) with a `warning`
  token icon.
- Each challenge card (`surface` background, `warning` left border 3px):
  - Topic display name (`text-sm font-medium`).
  - This person's position summary (`text-sm`, `canvas-foreground`).
  - "vs" divider (`muted`, `text-xs`).
  - Conflicting guest: display name (`text-sm font-medium`, clickable -- navigates
    to their Guest Brief) and their position summary (`text-sm`).
  - Episode references (`muted`, `text-xs`).
  - `derived: true` badge and provider attribution (`muted`, `text-xs`):
    "AI-generated -- derived from corpus data."
- Hidden entirely until Phase 6 ships (not shown as an empty placeholder).

---

## Graceful degradation

All degradation states use `muted` text and honest language. No broken layouts.

- **Person has 0 Insights** (FR3.1): Person header shows with KG metadata. Stats bar
  shows zeros with `disabled` text. Known Positions shows: "No grounded insights
  found for this person. Insights appear when the pipeline runs with GIL extraction
  enabled." (`muted`, `text-sm`, centered). Best Quotes section is hidden.
- **Person has Insights but 0 grounded Quotes** (FR3.2): Known Positions displays
  Insights with "ungrounded" badges. Best Quotes shows: "No grounded quotes
  available. Grounding improves with diarization and GIL extraction quality."
  (`muted`, `text-sm`). Grounding rate badge shows 0% with `warning` token.
- **Topic has 0 Insights** (FR3.3): Topic group appears with "0 insights" badge.
  Expanding shows: "No insights on this topic. The topic was mentioned but not
  deeply analysed." (`muted`, `text-sm`).
- **Lift fails** (FR3.4): Search result card shows raw chunk without speaker link.
  No Guest Brief navigation offered. (This state is in the search panel, not the
  Guest Brief panel itself.)
- **No bridge.json for all episodes** (FR3.5): Brief area shows: "No cross-layer
  data available. Run the pipeline with bridge generation enabled." (`muted`,
  `text-sm`, centered). Person header still shows KG-derived metadata if available.

---

## E2E contract

New visible labels and selectors require updates to the
[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
before or with implementation. Key surfaces:

- Right rail panel with `aria-label="Guest Brief"` (use `exact: true`)
- Person header (display name, appearance count, grounding rate badge)
- Topic summary stats bar (four metric chips)
- Known Positions section: topic group headers, expand/collapse, strongest Insight
  highlight, "Track positions" and "View topic" links
- Best Quotes section: quote cards, topic tags, timestamps
- "View in graph" and "Track positions" buttons
- All five degradation empty states
- Phase 6: Potential Challenges section, challenge cards, `derived` badge

---

## Accessibility

- **Focus:** Visible focus ring on all interactive elements (topic groups, buttons,
  quote cards, links). Uses `primary` token, 2px solid, 2px offset.
- **Contrast:** WCAG AA for all text on `surface` backgrounds. Badge text meets
  contrast requirements against badge backgrounds.
- **Keyboard:** Topic groups are expandable via Enter/Space. Quote cards and links
  are focusable. Stats bar chips are not interactive (informational only).
- **Screen reader:** Topic groups use `role="region"` with `aria-label` including
  topic name. Strongest Insight is announced as "Strongest position on [topic]."
  Challenge cards use `role="article"` with descriptive `aria-label`.

---

## Revision history

| Date       | Change                                                         |
| ---------- | -------------------------------------------------------------- |
| 2026-04-13 | Initial draft (PRD-029 companion)                              |
