# UXS-007: Topic Entity View

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states
- **Related PRDs**:
  - [PRD-026: Topic Entity View](../prd/PRD-026-topic-entity-view.md) -- full
    requirements and data sources
- **Related RFCs**:
  - [RFC-073: Enrichment Layer Architecture](../rfc/RFC-073-enrichment-layer-architecture.md) --
    `topic_cooccurrence` and `temporal_velocity` enrichers that provide the data
  - [RFC-072: Canonical Identity Layer](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) --
    bridge artifact powering cross-episode topic queries
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Related UX specs**:
  - [UXS-003: Corpus Library](UXS-003-corpus-library.md) -- Episode subject rail shared with
    Topic Entity View
  - [UXS-004: Graph Exploration](UXS-004-graph-exploration.md) -- "View in graph"
    handoff
  - [UXS-005: Semantic Search](UXS-005-semantic-search.md) -- "Search this topic"
    handoff
  - [UXS-008: Enriched Search](UXS-008-enriched-search.md) -- enriched search
    handoff; topic pills in enriched sources open this view
  - [UXS-009: Position Tracker](UXS-009-position-tracker.md) -- person chip click
    can open Person Landing with Position Tracker
  - [UXS-010: Person Profile](UXS-010-person-profile.md) --
    person chip click can open Person Landing with Person Profile
- **Implementation paths**:
  - New: `web/gi-kg-viewer/src/components/topic/TopicEntityView.vue`
  - Existing: `web/gi-kg-viewer/src/components/graph/GraphNodeRailPanel.vue` (rail
    host)
  - Existing: `web/gi-kg-viewer/src/stores/artifacts.ts` (enrichment data fetch)
  - MVP slice (GitHub **#548**): `web/gi-kg-viewer/src/components/shared/TopicTimelineDialog.vue`
    from graph **Topic** node detail — CIL **`/api/topics/.../timeline`** only; full
    Topic Entity View layout (charts, enrichers, shared **InsightCard**) remains future work.
- **Shell IA:** [VIEWER_IA.md](VIEWER_IA.md) — subject rail as single context layer; navigation axes; status bar

---

## Summary

For shell layout, the three navigation axes, subject rail persistence and clearing, status bar, and first-run empty corpus behavior, see **[VIEWER_IA.md](VIEWER_IA.md)**. This document specifies **Topic Entity View** in the subject rail only (sections, density, handoffs).

The Topic Entity View is a concept-first navigable surface in the right rail panel
where `topic:{slug}` is the subject. The user navigates *to a topic* and sees the
entire corpus through that lens: which episodes discuss it, how frequently over time,
what the most insightful things said about it are, and who said them.

This UXS defines the visual contract for the Topic Entity View panel layout, section
density, chart appearance, and graceful degradation states. All tokens reference
[UXS-001](UXS-001-gi-kg-viewer.md). Functional requirements are in
[PRD-026](../prd/PRD-026-topic-entity-view.md).

---

## Placement

The Topic Entity View is presented in the **right rail panel** of the viewer (same
panel used for graph node detail and episode detail in Library). It is not a new main
tab. The panel header shows "Topic" with the `kg` domain token color, consistent with
KG identity coloring in UXS-001.

---

## Entry points

- Clicking a Topic node in the Cytoscape graph opens the Topic Entity View in the
  right rail (same mechanism as existing node detail, expanded to full Topic Entity
  layout).
- Topic pills on **Semantic Search** episode result cards are clickable and open the
  Topic Entity View. **Corpus Library** episode **list** rows do **not** show topic
  pills ([UXS-003](UXS-003-corpus-library.md) keeps that catalog dense; Digest carries
  CIL topic pills that open **Graph**). From Library, reach this view via Graph topic
  nodes, Digest/Dashboard entry points, or Search — not from list-row chips.
- Topic rows in the Dashboard Content Intelligence section (from `temporal_velocity`
  enricher output) are clickable and open the Topic Entity View.

---

## Section layout

The panel is a single scrollable column on `surface` background. Sections stack
vertically with `border` dividers between them:

### Topic header

- Topic display name (`text-lg font-semibold`) and canonical `topic:{slug}` ID
  (`muted`, `text-xs`, monospace).
- Total episode count badge (`surface` chip, `text-sm`).
- Trend badge: `accelerating` / `stable` / `declining` / `insufficient data`.
  Uses intent tokens: `success` for accelerating, `muted` for stable, `warning` for
  declining, `disabled` for insufficient data. Badge is a small pill with icon +
  label.
- Date range (first and last appearance) in `muted` `text-xs`.

### Timeline section

- Monthly bar chart using Chart.js with `series-1` token for bars.
- Bars are clickable -- clicking a bar filters the Insights section to that month.
  Active bar uses `primary` fill; inactive bars use `series-1` at reduced opacity.
- When `sources.gi: true` appears in the bridge for a given episode, the
  corresponding bar segment is visually distinguished using `gi` domain token. This
  indicates months where structured Insights exist vs months where the topic was
  mentioned but not deeply analysed.
- Insight line below the chart (`muted`, `text-sm`, italic) when the data supports a
  clear takeaway (e.g. "This topic appeared in 14 episodes in Q1 2026, up from 2 in
  2024"). Computed by the viewer from enricher data, not a separate enricher output.
- Chart height: ~120px (compact, data-dense).

### Insights section

- Scrollable list of grounded Insight cards on `surface` background. Insight cards
  follow the **shared InsightCard** contract in
  [UXS-001 — InsightCard (shared component)](UXS-001-gi-kg-viewer.md#insightcard-shared-component).
  This view uses the following InsightCard slots: insight text, grounding badge,
  speaker chip, episode attribution, and supporting quote blockquote.
- Each card: insight text (`text-sm`), speaker name chip (`muted`), episode title +
  publish date (`muted`, `text-xs`), supporting verbatim quote in a `border`-left
  blockquote with a timestamp jump link (`link` token).
- Maximum 20 shown by default; "Show more" control (`primary`, `text-sm`) loads
  additional results via API pagination.
- When no grounded Insights exist, honest empty state: `muted` text, "No grounded
  insights yet for this topic. Insights appear when the pipeline runs with GIL
  extraction enabled."
- Filterable by person (clicking a person chip in the Persons section filters this
  list).

### Persons section

- Compact horizontal list of person chips (`surface` background, `border`, `text-sm`).
- Each chip: display name, Insight count badge, most recent episode date (`muted`,
  `text-xs`).
- When `grounding_rate` corpus enricher has run: small grounding quality badge on each
  chip -- percentage of grounded Insights. Intent tokens: `success` for >= 80%,
  `muted` for 50-79%, `warning` for < 50%. Hidden when enricher has not run.
- Clicking a person chip filters the Insights section. A secondary action (e.g.
  dedicated link icon or long-press) opens the Person Landing (UXS-010) for that
  person, giving access to their Person Profile and Position Tracker.
- Sorted by Insight count descending.
- When diarization is unavailable: `muted` note, "Speaker attribution requires
  diarization. Enable pyannote in pipeline config."

### Related topics section

- Compact list of topic chips (`surface` background, `kg` domain token border,
  `text-sm`).
- Each chip: display name and co-occurrence count badge.
- Clicking a related topic navigates to that topic's Topic Entity View (replaces
  current panel content).
- Maximum 8 related topics shown.
- Hidden entirely when `topic_cooccurrence` enricher has not run.

---

## Action buttons

Below the header, a row of two equal-width buttons:

- **View in graph** -- navigates to the Graph tab and focuses/centers on this topic's
  node in the Cytoscape canvas. Uses `primary` token.
- **Search this topic** -- prefills the semantic search panel with the topic label.
  Uses `primary` token. Consistent with existing "Search topic" affordance in Digest.

---

## Graceful degradation

All degradation states use `muted` text and honest language:

- Missing `temporal_velocity` -> Timeline section shows: "Run `podcast enrich` to
  generate topic analytics."
- Missing `topic_cooccurrence` -> Related Topics section is hidden entirely.
- Missing GIL Insights -> Insights section shows honest empty state (see above).
- Missing `grounding_rate` -> Person grounding badges hidden; chips still show.
- Core artifacts only -> Header shows episode count from KG MENTIONS scan; Timeline
  hidden; Insights hidden; Related Topics hidden. View is still useful for basic
  topic identity.

---

## Accessibility

- All interactive elements (topic chips, person chips, buttons, timeline bars) are
  keyboard-focusable and have visible focus indicators using UXS-001 `focus-ring`
  token.
- Screen reader: panel announces "Topic: [name]" on open. Trend badge, section
  headings, and empty states use `aria-label` or `aria-live` for dynamic content.
- Colour is not the sole differentiator for intent tokens -- trend badges include
  both icon and text label; grounding badges include percentage text.
- Minimum contrast: all text meets WCAG 2.1 AA (4.5:1 for `text-sm`, 3:1 for
  `text-lg`).

---

## E2E contract

New visible labels and selectors require updates to the
[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
before or with implementation. Key surfaces:

- Right rail panel with `aria-label="Topic"` (use `exact: true` in automation)
- Topic header, trend badge, timeline chart
- Insights list, person chips, related topic chips
- "View in graph" and "Search this topic" buttons
- All degradation empty states

---

## Revision history

| Date       | Change                                                         |
| ---------- | -------------------------------------------------------------- |
| 2026-04-13 | Initial draft (PRD-026 companion)                              |
| 2026-04-19 | Entry points: Library rows per UXS-003; InsightCard anchor     |
