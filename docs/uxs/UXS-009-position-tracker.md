# UXS-009: Position Tracker

- **Status**: Draft
- **Authors**: Marko
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states
- **Related PRDs**:
  - [PRD-028: Position Tracker](../prd/PRD-028-position-tracker.md) -- full
    requirements and API contract
- **Related RFCs**:
  - [RFC-072: Canonical Identity Layer](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) --
    bridge artifact, Flagship 1, query Pattern A
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Related UX specs**:
  - [UXS-004: Graph Exploration](UXS-004-graph-exploration.md) -- "View in graph"
    handoff
  - [UXS-005: Semantic Search](UXS-005-semantic-search.md) -- search entry point
    for speaker click
  - [UXS-007: Topic Entity View](UXS-007-topic-entity-view.md) -- cross-linked from
    topic selector
  - [UXS-008: Enriched Search](UXS-008-enriched-search.md) -- enriched search entry
    point; speaker names in enriched sources can open Person Landing
  - [UXS-010: Person Profile](UXS-010-person-profile.md) --
    "Open profile" handoff; **owns the shared Person Landing** that hosts both
    Person Profile and Position Tracker
- **Implementation paths**:
  - New: `web/gi-kg-viewer/src/components/person/PositionTracker.vue`
  - Existing: `web/gi-kg-viewer/src/components/graph/GraphNodeRailPanel.vue` (rail
    host)
  - Existing: `web/gi-kg-viewer/src/stores/` (new `person.ts` store or extend
    existing)

---

## Summary

The Position Tracker is a person + topic navigable surface that shows how a person's
stated positions on a topic evolve across episodes. This UXS defines the visual
contract for the panel layout, timeline cards, Insight and Quote card components,
topic selector, and graceful degradation states. All
tokens reference [UXS-001](UXS-001-gi-kg-viewer.md). Functional requirements are in
[PRD-028](../prd/PRD-028-position-tracker.md).

---

## Principles

- **Chronological narrative**: The primary axis is time. The user reads the arc top to
  bottom (most recent first) and sees how positions evolved.
- **Grounding-first**: Every Insight is accompanied by its verbatim quote. The quote
  is the evidence; the Insight is the interpretation.
- **Honest degradation**: When data is missing, the panel says so in plain language.
  No broken layouts, no spinners that never resolve.

---

## Scope

**In scope:**

- Position Tracker right rail panel (graph and search entry points)
- Position Tracker full-width view (browse entry point)
- Person header, topic selector, insight type filter
- Episode timeline with Insight and Quote cards
- All five degradation states (FR3.1 -- FR3.5)

**Non-goals:**

- Audio playback from timestamps (display only)
- Cross-corpus person views

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

The Position Tracker panel has two presentation modes:

- **Right rail** (from graph or search entry point): same panel slot used for graph
  node detail and episode detail. Panel width follows the existing rail width
  (UXS-001).
- **Full-width** (from browse entry point): occupies the main content area. Layout
  is the same vertical stack but with wider cards and more horizontal space for
  Insight text.

The panel header shows "Position Tracker" with the `gi` domain token color, reflecting
that the primary data source is GIL Insights.

---

## Entry points

The Position Tracker lives within the **shared Person Landing** component (owned by
[UXS-010](UXS-010-person-profile.md)). All entry points navigate to the
Person Landing, which defaults to the Person Profile tab. The user toggles to the
Position Tracker from there. When the entry point is topic-specific, the Position
Tracker tab is preselected with that topic active.

- Clicking a Person node in the Cytoscape graph opens the Person Landing in the
  right rail.
- Clicking a speaker name in a search result card (including lifted results with
  `lifted.speaker` or enriched sources from UXS-008) opens the Person Landing in
  the right rail.
- A dedicated person browse (accessible from the viewer navigation) opens the
  Person Landing in full-width mode.

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
- **Open profile** -- navigates to the Person Profile (PRD-029) for this person.
  Uses `primary` token.

### Topic selector

- A searchable dropdown (`surface` background, `border`, `text-sm`) populated from
  the person's known topics with display names.
- Placeholder text: "Select a topic to see position arc..."
- A "View topic" link (`link` token, `text-xs`) next to the selected topic navigates
  to the PRD-026 Topic Entity View.
- When the person has no topics: the dropdown is disabled with `disabled` text,
  "No topics found."

### Insight type filter

- A segmented control (`surface` background, `border`) with three options:
  - "Claims" (default, selected state uses `primary` fill)
  - "Recommendations"
  - "All"
- Compact: `text-xs`, fits on one line next to the topic selector in full-width mode,
  or below it in right rail mode.

### Episode timeline

- Episodes are displayed as cards in a vertical stack, ordered by `publish_date`
  (most recent first).
- A thin connecting line (`border` token, 2px) runs vertically between cards on the
  left edge, creating the timeline metaphor.
- Each episode card:
  - `surface` background, `border` border, rounded corners (UXS-001 radius).
  - Episode title (`text-sm font-medium`), podcast title (`muted`, `text-xs`),
    publish date (`muted`, `text-xs`).
  - Insight count badge: "N insights" (`surface` chip, `text-xs`).

### Insight cards

Insight cards follow the **shared InsightCard component** defined in
[UXS-001](UXS-001-gi-kg-viewer.md). This view uses the following InsightCard slots:
insight text, `insight_type` badge, `position_hint` bar, confidence score, grounding
badge. Episode attribution is provided by the parent episode card context.

Within each episode card, Insights are listed vertically, ordered by `position_hint`
ascending (early in episode first).

- Insight text (`text-sm`, `canvas-foreground`).
- `insight_type` badge: small pill (`text-xs`).
  - "claim" -> `primary` token.
  - "recommendation" -> `success` token.
  - "observation" -> `muted` token.
  - "question" -> `link` token.
  - Other/unknown -> `muted` token.
- `position_hint` indicator: a small horizontal bar (40px wide, 4px tall) filled
  proportionally. Uses `primary` token fill on `border` background. Tooltip shows
  the numeric value.
- Confidence score: `muted` `text-xs`, e.g. "0.88 confidence". Hidden when not
  available.
- Grounding badge: "grounded" (`success` token, `text-xs`) or "ungrounded"
  (`warning` token, `text-xs`).

### Quote cards

Within each Insight, supporting quotes are listed:

- Blockquote style: `border` left border (3px), `surface` background with slight
  indent (padding-left 12px).
- Quote text (`text-sm`, italic, `canvas-foreground`).
- Timestamp display (`muted`, `text-xs`): formatted as "MM:SS -- MM:SS" from
  `timestamp_start_ms` and `timestamp_end_ms`. When timestamps are unavailable,
  hidden entirely (no "unknown" placeholder).
- Episode attribution (`muted`, `text-xs`): shown only when the quote is displayed
  outside its episode context (e.g. in a future cross-reference view).

---

## Graceful degradation

All degradation states use `muted` text and honest language. No broken layouts.

- **Person has 0 Insights** (FR3.1): Person header shows with KG metadata. Topic
  selector is empty/disabled. Arc area shows: "No grounded insights found for this
  person. Insights appear when the pipeline runs with GIL extraction enabled."
  (`muted`, `text-sm`, centered in arc area).
- **Person has Insights but 0 grounded Quotes** (FR3.2): Insight cards display with
  "ungrounded" badge. Quote section within each Insight is hidden. A note below the
  Insights: "These insights are not grounded in verbatim quotes. Grounding improves
  with diarization and GIL extraction quality." (`muted`, `text-xs`).
- **Selected topic has 0 Insights** (FR3.3): Topic appears in selector. Arc area
  shows: "No insights on this topic for [person name]. Try selecting a different
  topic or viewing all types." (`muted`, `text-sm`, centered).
- **Lift fails** (FR3.4): Search result card shows raw chunk without speaker link.
  No Position Tracker navigation offered. (This state is in the search panel, not
  the Position Tracker panel itself.)
- **No bridge.json for all episodes** (FR3.5): Arc area shows: "No cross-layer data
  available. Run the pipeline with bridge generation enabled." (`muted`, `text-sm`,
  centered). Person header still shows KG-derived metadata if available.

---

## E2E contract

New visible labels and selectors require updates to the
[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
before or with implementation. Key surfaces:

- Right rail panel with `aria-label="Position Tracker"` (use `exact: true`)
- Person header (display name, appearance count)
- Topic selector dropdown
- Insight type filter segmented control
- Episode timeline cards
- Insight cards with type badges and grounding badges
- Quote blockquotes with timestamps
- "View in graph" and "Open profile" buttons
- All five degradation empty states

---

## Accessibility

- **Focus:** Visible focus ring on all interactive elements (topic selector, filter,
  buttons, episode cards). Uses `primary` token, 2px solid, 2px offset.
- **Contrast:** WCAG AA for all text on `surface` backgrounds. Badge text meets
  contrast requirements against badge backgrounds.
- **Keyboard:** Topic selector is keyboard-navigable (arrow keys, Enter to select).
  Episode cards are focusable for screen reader navigation.
- **Screen reader:** Episode cards use `role="article"` with `aria-label` including
  episode title and date. Insight type badges use `aria-label` for the full type name.

---

## Revision history

| Date       | Change                                                         |
| ---------- | -------------------------------------------------------------- |
| 2026-04-13 | Initial draft (PRD-028 companion)                              |
