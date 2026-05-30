# PRD-033: Search-Powered Surface Enhancements

- **Status**: Draft
- **Author**: Marko
- **Created**: 2026-05-24
- **Target**: v2.7 (foundation), podcast product (full)
- **Related PRDs**:
  - `docs/prd/PRD-031-search.md` — Search product
  - `docs/prd/PRD-032-hybrid-corpus-search.md` — retrieval backend (dependency)
  - `docs/prd/PRD-026-topic-entity-view.md` — Topic Entity View panel
  - `docs/prd/PRD-029-person-profile.md` — Person Landing content
  - `docs/prd/PRD-028-position-tracker.md` — position evolution data
- **Related RFCs**:
  - `docs/rfc/RFC-090-hybrid-retrieval.md`, `docs/rfc/RFC-091-kg-proximity-signal.md`
  - `docs/rfc/RFC-080-graph-visualization-extensions.md` — graph viz extensions
- **Related issues**: #658 (Graph filter UX), #669 (Library), #670 (Digest), #671 (Search/Explore), #672 (right rail), #673 (nav), #674 (cross-surface flows) — all CLOSED

> **Stabilization note (2026-05-30):** Rebased to this repo. Original draft referenced
> "RFC-077 (graph visualization extensions)" and "RFC-080 (corpus impact surface)" — both wrong
> here. Corrected: graph viz extensions → **RFC-080**. A dedicated **corpus-impact / coverage
> surface** does not exist as its own doc and is marked **[TBD — not yet specified]**. This PRD is
> a cross-surface companion to PRD-031/032: it specifies how the retrieval backend propagates
> across viewer surfaces, not a standalone feature.

---

## Summary

PRD-032 delivers a two-tier hybrid retrieval backend with BM25, vector, and KG-proximity signals.
This PRD covers how that capability propagates across the viewer's surfaces — Search/Explore,
Library, Digest, Detail panels, Graph, and Dashboard — and what becomes possible on each that
wasn't before. It is not a cosmetic refresh: the retrieval upgrade changes what information can be
surfaced, how confidently, and in response to what triggers.

## Background & Context

- **What problem this solves.** Each viewer surface today is limited by single-signal,
  insight-only retrieval. Library filtering is presence/absence; Digest bands are static; node
  clicks dead-end; Dashboard cards aren't drillable to evidence.
- **Why now.** Once PRD-032's backend lands, every surface can be re-grounded in actual,
  attributable corpus evidence. The cross-surface flow gaps in #674 become fixable.
- **How it relates to existing features.** This PRD is the consumer-side companion to PRD-031
  (Search product) and PRD-032 (backend). It depends on the right-rail panels (#672), the filter
  chip bars (#658/#669/#670/#671), and the canonical identity layer (RFC-072).

### What the backend unlocks per surface

| Old capability | New capability |
| --- | --- |
| Vector similarity only | BM25 + vector + KG proximity, mixed-tier results |
| Insight nodes only | Transcript segments + insights, raw evidence accessible |
| No query intent | Query-type routing — entity, synthesis, temporal, evidence |
| Static ranked lists | Compound results (segment + insight linked) |
| No coverage signal | Corpus coverage counts (richer impact surface **[TBD]**) |
| Raw MCP dumps | LITM-aware briefing packs (requires MCP layer **[TBD]**) |

## Goals

- Propagate hybrid retrieval consistently across all viewer surfaces, not just Search.
- Turn dead-end node clicks into evidence-grounded panels (#672).
- Make filtering rank by relevance (retrieval-backed), not just presence/absence.
- Establish cross-show synthesis as a first-class, differentiated capability (Digest).

## Non-Goals

- Building the retrieval backend (PRD-032 / RFC-090 own that).
- Net-new surfaces — this enhances existing surfaces only.
- Surfaces whose content depends on capabilities not yet built (contradiction edges, MCP packs,
  corpus-impact surface) ship as graceful placeholders, not blockers.

## Personas

- **Beta researcher**: Navigates across surfaces (Search → Detail → Graph) and expects continuity
  and real content at each stop.
- **Operator (you)**: Uses Dashboard/Library coverage signals to guide corpus expansion.

## User Stories

- _As a beta researcher, I can filter Library by a topic and get episodes ranked by relevance, so
  that a 700+ episode corpus doesn't return an undifferentiated list._
- _As a beta researcher, I can click a speaker name in Digest and open their Person Landing, so
  that names are navigation anchors, not plain text._
- _As a beta researcher, I can see a cross-show synthesis band in Digest, so that I get a
  horizontal slice across shows on a topic._
- _As a beta researcher, I can click a Graph node and open an evidence-grounded panel, so that the
  graph is a navigation surface, not just a visualization._
- _As an operator, I can see a coverage-gaps card on the Dashboard, so that I know which topics are
  thinly covered._

## Functional Requirements

### FR1: Search and Explore

_Primary surface for PRD-031/032; specified here for cross-surface consistency._

- **FR1.1**: Two-tier results in the list with a `source_tier` indicator; compound results appear
  as a single card with both layers accessible.
- **FR1.2**: Named-entity search returns exact-match results (purely retrieval quality, no UI
  change).
- **FR1.3**: Raw-evidence toggle ("Insights" / "Transcript" / "Both") constraining to
  segment-tier results (uses the `raw_evidence` query type).
- **FR1.4**: Query-type indicator near the search bar (transparency / debuggability).
- **FR1.5 (Explore)**: Corpus-coverage card above results when a topic/person is selected — N
  shows, M episodes, date range, contradiction count. Placeholder in v2.7; full when the
  corpus-impact surface exists **[TBD]**. Affected: #671.

### FR2: Library (#669)

- **FR2.1**: "Why this episode" relevance snippet on rows when a search/filter context is active —
  top-scoring segment or insight for the current context. Triggered only with active context;
  rows stay clean by default.
- **FR2.2**: Topic/person filter chips rank episodes by hybrid relevance, not presence/absence.
- **FR2.3**: Coverage-gap indicator on show groups for a filtered view (requires corpus-impact
  surface **[TBD]**; placeholder in v2.7).

### FR3: Digest (#670)

- **FR3.1**: Topic bands ranked by retrieval signal (insight density, contradiction presence,
  cross-show coverage), not just recency/frequency.
- **FR3.2**: Cross-show synthesis band — for a topic, the top insight from each show that covers
  it. A genuine product differentiator (the corpus moat).
- **FR3.3**: Contradiction indicator in bands when typed contradiction KG edges exist (**not yet
  built**; placeholder until then).
- **FR3.4**: Speaker names become interactive — resolve to canonical person IDs (RFC-072) and open
  Person Landing (#672). _Depends on the entity resolver, which **does not exist yet**._

### FR4: Detail Panels (#672)

- **FR4.1 Person Landing**: Dynamically assembled via an `entity_lookup`, BM25-heavy query scoped
  to `speaker_id = canonical_id` — top insight, top segment, shows ranked by contribution, topics
  by insight density, position evolution (if PRD-028 data exists). Returns compound results.
- **FR4.2 Topic Entity View**: `cross_show_synthesis`, KG-proximity-weighted, unscoped — cross-show
  coverage summary, dominant positions, contradictions (when edges exist), key voices, raw
  evidence.
- **FR4.3 Episode Detail**: Navigable transcript with highlighted matched segments; "related
  insights" via a KG-proximity query scoped to the episode's topics.

### FR5: Graph (#658)

- **FR5.1**: Node size reflects a retrieval signal (insight density / cross-show breadth /
  search-context relevance), not just degree.
- **FR5.2**: Edge weight reflects retrieval confidence (insight count × confidence). Feeds the
  graph-viz edge-weight target in RFC-080.
- **FR5.3**: Contradiction edges shown as a distinct edge type when present (**not yet built**).
- **FR5.4**: Node click opens the populated Detail panel (FR4), not a dead-end generic panel.

### FR6: Dashboard

- **FR6.1**: Briefing cards grounded in retrieval (top insight + top segment per topic) via a
  briefing-pack call. _Requires MCP layer **[TBD]**; until then, cards use existing enrichment._
- **FR6.2**: "View evidence" affordance opening the briefing-pack view (RFC-093).
- **FR6.3**: Activity chart driven by retrieval-volume signal (query volume by topic over time).
  Tufte-compliant: single y-axis, no dual-axis, no decorative lines.
- **FR6.4**: Coverage-gaps card (corpus quality for the operator). Requires corpus-impact surface
  **[TBD]**.
- Design constraints unchanged: 5-second-answer cards; enrichment feeds retrieval, retrieval feeds
  cards; Tufte chart discipline.

## Success Metrics

- Cross-surface navigation rate (Search → Detail → Graph) increases vs current baseline.
- Library filter → episode click-through increases with relevance snippets active.
- Digest cross-show synthesis band engagement: any non-zero rate (validates the differentiator).
- Node-click → populated-panel rate: ~100% of clicks reach real content (no dead-ends).

## Dependencies

- **PRD-032 / RFC-090** hybrid retrieval backend — **Hard** — Draft.
- **RFC-091** KG proximity (Topic Entity View, related insights) — **Hard for those** — Draft;
  depends on graph-access prerequisites.
- **#672** right-rail panels — **Hard** — CLOSED.
- **RFC-072** canonical identity / entity resolver — **Hard for interactive names** — **Partial**
  (no resolver yet).
- **RFC-080** graph-visualization extensions — **Soft** — for edge-weight/node-size signals.
- **MCP layer** (Dashboard briefing packs) — **[TBD — not yet specified]**.
- **Corpus-impact / coverage surface** (coverage cards/gaps) — **[TBD — not yet specified]**.
- **Typed contradiction KG edges** — **not yet built** (placeholders degrade gracefully).

## Constraints & Assumptions

**Constraints:**

- Surfaces depending on unbuilt capabilities ship placeholders, never blockers.
- Dashboard cards remain 5-second answers; retrieval happens async/in background.

**Assumptions:**

- Right-rail panels (#672) and filter chip bars (#658/#669/#670/#671) are built or buildable.
- Canonical IDs become available via the entity resolver prerequisite (#849).

## Design Considerations

### OQ-1: PanelRetrievalStore

A `PanelRetrievalStore` (Pinia) caches retrieval results per entity ID with a short TTL; panel
opens trigger a retrieval call, subsequent opens within TTL are instant. Resolves the #672
`subjectStore` open question. Decide before podcast-product panel work.

### OQ-2: Search context propagation

A shared `activeSearchContext` store so Graph reflects an active search (retrieval-score-weighted
nodes). Useful but adds cross-surface state complexity — defer to podcast product, design
separately.

### OQ-3: Dashboard async card content

Briefing-pack-backed cards add load latency — render skeleton-first, populate async. Confirm the
dashboard architecture supports async card content before implementing.

### OQ-4: Contradiction-edge dependency

Multiple surfaces show placeholders for contradiction content; all depend on
contradiction-as-KG-edge, which has no RFC yet. Schedule that prerequisite before podcast product
to avoid shipping a cluster of "coming soon" placeholders.

## Cross-Surface Flows (Issue #674 Revisited)

Issue #674 identified five cross-surface flow gaps. Hybrid search addresses four:

| Gap | How hybrid search fixes it |
| --- | --- |
| Feed names in Digest non-interactive | Show names resolve via canonical identity → show-scoped Library view |
| Library rows require too many clicks to reach graph | Relevance snippets give enough context to decide before clicking; compound results reduce pogo-sticking |
| Digest topic bands switch tabs instead of opening Topic Entity View | Topic Entity View now has real content; band click opens right rail |
| Speaker names in Search plain text | Entity resolver maps names → canonical IDs → Person Landing |
| _(fifth gap: not directly addressed here)_ | — |

## Milestone Assignment

| Surface | Change | Milestone |
| --- | --- | --- |
| Search/Explore | Two-tier results, raw-evidence toggle, query-type indicator | v2.7 |
| Search/Explore | Corpus coverage card in Explore | podcast product |
| Library | Hybrid retrieval ranking on filter chips | v2.7 |
| Library | Relevance snippets on rows (active context) | v2.7 |
| Library | Coverage gap indicator | podcast product |
| Digest | Topic bands ranked by retrieval signal | v2.7 |
| Digest | Cross-show synthesis band | podcast product |
| Digest | Contradiction indicator | podcast product (when edges exist) |
| Digest | Speaker names interactive | v2.7 (needs entity resolver) |
| Detail — Person Landing | Retrieval-grounded content | podcast product |
| Detail — Topic Entity View | Retrieval-grounded content | podcast product |
| Detail — Episode Detail | Related insights, segment highlights | podcast product |
| Graph | Node size / search-context prominence | podcast product |
| Graph | Contradiction edges visible | podcast product (when edges exist) |
| Graph | Node click → populated panel | podcast product |
| Dashboard | Briefing cards + "view evidence" | podcast product (needs MCP) |
| Dashboard | Coverage gaps card | podcast product |
| Dashboard | Retrieval-volume activity chart | podcast product |

## Related Work

- `docs/prd/PRD-031-search.md`, `docs/prd/PRD-032-hybrid-corpus-search.md`
- `docs/rfc/RFC-090-hybrid-retrieval.md`, `docs/rfc/RFC-091-kg-proximity-signal.md`,
  `docs/rfc/RFC-093-litm-context-packs.md`
- `docs/rfc/RFC-080-graph-visualization-extensions.md`
- GitHub issues #658, #669, #670, #671, #672, #673, #674 (all CLOSED).

## Release Checklist

- [ ] PRD reviewed and approved
- [ ] PRD-032 / RFC-090 backend available
- [ ] v2.7 surface slices (Search two-tier, Library ranking + snippets, Digest band ranking) shipped
- [ ] Entity resolver prerequisite filed and tracked (interactive names)
- [ ] Placeholder behavior verified for unbuilt-capability surfaces (no broken UI)
- [ ] Corpus-impact surface and MCP layer prerequisites scheduled before podcast-product slices
