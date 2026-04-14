# UXS-008: Enriched Search

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states
- **Related PRDs**:
  - [PRD-027: Enriched Search](../prd/PRD-027-enriched-search.md) -- full requirements,
    data flow, and dependencies
- **Related RFCs**:
  - [RFC-073: Enrichment Layer Architecture](../rfc/RFC-073-enrichment-layer-architecture.md) --
    enrichment trust contract; QueryEnricher protocol (Phase 4 extension)
  - [RFC-072: Canonical Identity Layer](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) --
    bridge artifact enabling chunk-to-Insight lift
  - [RFC-061: Semantic Corpus Search](../rfc/RFC-061-semantic-corpus-search.md) --
    FAISS search API this feature extends
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Related UX specs**:
  - [UXS-005: Semantic Search](UXS-005-semantic-search.md) -- baseline search panel
    this feature extends
  - [UXS-007: Topic Entity View](UXS-007-topic-entity-view.md) -- topic tags in
    enriched sources open Topic Entity View
  - [UXS-009: Position Tracker](UXS-009-position-tracker.md) -- speaker names in
    enriched sources open Person Landing with Position Tracker
  - [UXS-010: Person Profile](UXS-010-person-profile.md) --
    speaker names in enriched sources open Person Landing with Person Profile
- **Implementation paths**:
  - Existing: `web/gi-kg-viewer/src/components/search/SearchPanel.vue` (extended)
  - Existing: `web/gi-kg-viewer/src/components/search/ResultCard.vue`
  - Existing: `web/gi-kg-viewer/src/stores/search.ts` (extended)
  - New: enriched answer panel component (within search/)

---

## Summary

Enriched Search extends the existing semantic search panel (UXS-005) with a
query-time LLM enrichment layer that lifts raw FAISS transcript chunks into
structured, grounded, attributed answers. This UXS defines the visual contract for
the Enriched Answer panel, provider attribution, source citations, degraded states,
and the "Enhanced" availability indicator. All tokens reference
[UXS-001](UXS-001-gi-kg-viewer.md). Functional requirements are in
[PRD-027](../prd/PRD-027-enriched-search.md).

---

## Enriched Answer panel

### Placement

The Enriched Answer panel appears **above** the raw search results list, not
replacing or modifying existing search UI elements (UXS-005). It is visually distinct
from raw results.

### Visual treatment

- `gi` domain token border or background tint (consistent with GIL identity in
  UXS-001) to signal grounded provenance.
- Visible **"AI-generated / grounded"** badge (`muted` text, `text-xs`, with a small
  icon) in the panel header. This signals the derived nature of the content.
- Panel background: `surface` with `gi` left border (4px solid).
- Panel padding: standard card padding (same as search result cards).

### Answer text

- Synthesised paragraph or short structured response produced by the LLM enricher.
- `text-sm` body text, `surface-foreground` color.
- Speaker names in the answer text are visually emphasized (`font-semibold`).
- Specific positions from grounded Insights are referenced inline.

### Sources section

Below the answer text, a **Sources** section lists all Insights used to generate the
answer:

- Each source shows: speaker name (`font-semibold`), Insight text (`text-sm`),
  episode title and publish date (`muted`, `text-xs`), and a timestamp deep-link
  (`link` token -- opens episode at that moment if audio is available, otherwise
  highlights the transcript segment).
- **Speaker names are clickable** (`link` token, cursor pointer) and open the
  Person Landing (UXS-010) for that person, giving access to their Person Profile and
  Position Tracker.
- **Topic tags** on sources (when present) are clickable and open the Topic Entity
  View (UXS-007) for that topic.
- Sources are collapsible -- defaults to showing 3 sources with a "Show all N
  sources" control (`primary`, `text-sm`) if more exist.
- **Grounded source count** indicator: "Based on N grounded insights" (`muted`,
  `text-xs`) above the sources list.

### Provider attribution

- Muted caption below the answer: placeholder copy "Synthesised by …" including
  provider name and model id (e.g. "Synthesised by OpenAI gpt-4o-mini"). Uses `muted` token, `text-xs`.
- One-line explanation: "This answer was generated from grounded insights in your
  corpus. All sources are verbatim quotes with timestamps." Uses `muted` token,
  `text-xs`.

### Source-to-result linking

Sources that were used in the answer are visually linked to their corresponding raw
search result cards below. Implementation options (choose one):

- Highlight badge on the raw result card (small `gi` dot or "Used in answer" chip)
- Hover on a source highlights the corresponding result card with `overlay`
  background

---

## Enhanced availability indicator

When enrichment is available, a subtle **"Enhanced"** chip appears in the search
panel header:

- Small pill: `surface` background, `gi` border, `text-xs`.
- Visible only when `enriched_search_available: true` from `GET /api/health`.
- Hidden when enrichment is not configured -- no degraded state, no error.

---

## Advanced search integration

The Advanced search modal (UXS-005) gains a new toggle:

- **"Enriched answers"** -- on/off, defaults to on when enrichment is configured.
- Same toggle style as existing Advanced search controls.
- When off, the Enriched Answer panel is hidden for that query; raw results are
  unchanged.

---

## Degraded states

**Degradation philosophy:** Enriched Search uses a **silent degradation** strategy
that differs from the "honest empty state" approach used in UXS-007, UXS-009, and
UXS-010. When enrichment is unavailable, the panel is hidden entirely -- the user
sees the baseline search experience (UXS-005) with no indication that enrichment
exists. This is intentional: enrichment is additive, and advertising an unavailable
feature would confuse users who have not configured an LLM provider. The other views
(Topic Entity, Position Tracker, Person Profile) use honest empty states because the
user has already navigated *to* that feature and expects to see content.

All degradation is silent and honest:

- **No grounded Insights lifted:** Enriched Answer panel is hidden entirely. Raw
  results shown as normal. No error message.
- **LLM call fails (timeout, provider error):** Panel shows minimal error state:
  "Enrichment unavailable for this query." (`muted` text, `text-sm`). Raw results
  remain unaffected.
- **All lifted Insights are `grounded: false`:** Panel is hidden. The system does not
  synthesise from ungrounded Insights.
- **Latency > 5 seconds:** Loading indicator in the Enriched Answer panel area
  (skeleton using `surface` / `border` stripes, same pattern as UXS-001 loading
  state). Raw results are shown immediately while the enricher works.
- **Enrichment not configured:** Search panel is unchanged from UXS-005. No mention
  of missing configuration unless the user explicitly inspects health.

---

## Accessibility

- The Enriched Answer panel uses `role="region"` with
  `aria-label="Enriched answer"`.
- The "AI-generated / grounded" badge is announced by screen readers.
- Source collapse/expand uses `aria-expanded` and is keyboard-operable.
- Speaker name links and topic tag links are keyboard-focusable with visible focus
  indicators using UXS-001 `focus-ring` token.
- The "Enhanced" chip uses `aria-label="Enriched search available"`.
- Colour is not the sole differentiator -- the `gi` border is supplemented by the
  text badge.
- Minimum contrast: all text meets WCAG 2.1 AA (4.5:1 for `text-sm`, 3:1 for
  `text-lg`).

---

## E2E contract

New visible labels and selectors require updates to the
[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
before or with implementation. Key surfaces:

- Enriched Answer panel (with `aria-label="Enriched answer"`)
- "AI-generated / grounded" badge
- Sources section (collapsible)
- "Enhanced" chip in search panel header
- "Enriched answers" toggle in Advanced search
- All degradation states (hidden panel, error state, loading skeleton)

Playwright coverage: `e2e/enriched-search.spec.ts` with mocked enricher responses
for enriched state, disabled state, and degraded state (no grounded Insights).

---

## Revision history

| Date       | Change                                                         |
| ---------- | -------------------------------------------------------------- |
| 2026-04-13 | Initial draft (PRD-027 companion)                              |
