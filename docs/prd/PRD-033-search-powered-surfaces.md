# PRD-033: Search-Powered Surface Enhancements

- **Status**: Draft
- **Author**: Marko
- **Created**: 2026-05-24
- **Revised**: 2026-06-04 — re-grounded on the shipped backend; generic capability→surface framing
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
> surface** does not exist as its own doc and is marked **[TBD — not yet specified]**.
>
> **Revision note (2026-06-04) — re-grounded on the shipped foundation.** The retrieval backend
> and the relational edge layer have **shipped** (PRD-032 / RFC-090; the linker + `Person→Insight`,
> `Insight→Entity`, `Podcast→Episode` edges in #874). This PRD is now framed **generically**: given
> the capabilities we have actually built, how does each viewer surface consume them to deliver
> value. It is **not** gated on any one future capability. Capabilities that some surfaces could
> additionally use but that are **not yet built** — contradiction edges, MCP briefing packs, a
> coverage surface — are **orthogonal**: surfaces deliver the value they can today and add those
> affordances later. They are fenced off in [§Orthogonal capabilities](#orthogonal-capabilities-not-this-prds-spine)
> and never block this work. (Note: KG-proximity was evaluated and **rejected as a retrieval
> signal** — RFC-091 Decision Record; ranking is hybrid BM25+dense, relational structure comes from
> the edges.)

---

## Summary

The retrieval foundation has shipped. This PRD is the **consumer-side spec**: it inventories every
viewer surface and, for each, defines **which shipped capability it consumes and the value it
delivers**. The objective is simple — **bring everything we've built to the UI** — not to wait on
any single future capability. Each surface ships the value its consumed capabilities already
support; orthogonal additions land independently.

## What the foundation provides (shipped — the consumable inventory)

| Capability (shipped) | What a surface does with it |
| --- | --- |
| **Hybrid retrieval** (BM25 + dense + RRF) | Rank by relevance, not presence/absence; exact names + phrases *and* semantics |
| **Two-tier + compound results** | Show a synthesized insight together with its grounding transcript segment |
| **Query-intent routing** | Adapt the retrieval strategy to the query (entity / evidence / synthesis / semantic) |
| **Canonical entity resolver** (RFC-072 / #849) | Turn a name in text into a navigable canonical entity id |
| **`Person→Insight`** (who said what, #874) | Assemble a person's positions — "what does X think about Y" |
| **`Insight→Entity`** (insight about whom, #874) | Ground an insight in the people/orgs it concerns |
| **`Podcast→HAS_EPISODE→Episode`** (#874) | Resolve a show name to show-scoped navigation |
| **Cross-show shared topics** | Group a topic's coverage across multiple shows |

Every surface below consumes a subset of this inventory. Nothing here is "coming soon."

## Goals

- Propagate hybrid retrieval and the relational layer **consistently across all viewer surfaces**,
  not just Search.
- Turn dead-end node clicks into **evidence-grounded panels** (#672).
- Make filtering **rank by relevance** (retrieval-backed), not presence/absence.
- Make names and shows **navigable** (resolver + `Person→Insight` + `HAS_EPISODE`).
- Establish **cross-show synthesis** as a first-class capability (Digest), using hybrid retrieval +
  shared topics.

## Non-Goals

- Building the retrieval backend or the edge layer (PRD-032 / RFC-090 / #874 own those).
- Net-new surfaces — this enhances existing surfaces only.
- Any **orthogonal** capability (contradiction edges, MCP packs, coverage surface). Surfaces that
  could use them add the affordance when they exist; this PRD never blocks on them.

## Personas

- **Beta researcher**: Navigates across surfaces (Search → Detail → Graph) and expects continuity
  and real, attributable content at each stop.
- **Operator**: Uses coverage signals (where available) to guide corpus expansion.

## Functional Requirements

Each requirement names the **shipped capability** it consumes. Orthogonal additions are marked
`[orthogonal]` and are not prerequisites.

### FR1: Search and Explore (#671)

- **FR1.1** Two-tier results with a `source_tier` indicator; **compound results** render as one
  card exposing both the insight and its grounding segment. *(two-tier + compounds)*
- **FR1.2** Named-entity search returns exact-match results — retrieval quality, no UI change.
  *(hybrid)*
- **FR1.3** Raw-evidence toggle ("Insights" / "Transcript" / "Both") constraining to segment-tier.
  *(two-tier + intent routing)*
- **FR1.4** Query-type indicator near the search bar (transparency). *(intent routing)*
- **FR1.5** Speaker/entity names in results resolve to canonical ids and link to their Detail panel.
  *(resolver + `Person→Insight` / `Insight→Entity`)*

### FR2: Library (#669)

- **FR2.1** "Why this episode" relevance snippet on rows when a search/filter context is active —
  the top-scoring segment or insight for the context; rows stay clean by default. *(hybrid)*
- **FR2.2** Topic/person filter chips rank episodes by **hybrid relevance**, not presence/absence.
  *(hybrid)*
- **FR2.3** Show name → show-scoped Library view. *(`HAS_EPISODE`)*

### FR3: Digest (#670)

- **FR3.1** Topic bands ranked by retrieval signal (insight density, cross-show coverage), not just
  recency/frequency. *(hybrid + shared topics)*
- **FR3.2** **Cross-show synthesis band** — for a topic, the top insight from each show that covers
  it. The corpus differentiator. *(hybrid + shared topics)*
- **FR3.3** Speaker/show names are interactive — resolve to canonical ids → open Detail.
  *(resolver + `Person→Insight` + `HAS_EPISODE`)*

### FR4: Detail Panels (#672)

- **FR4.1 Person Landing** — assembled from `Person→Insight` (the person's positions), grounded
  segments (compounds), shows they appear on, and topics they discuss; an entity-scoped hybrid query
  fills supporting evidence. *(`Person→Insight` + compounds + resolver)*
- **FR4.2 Topic Entity View** — cross-show coverage of the topic: dominant insights ranked by
  hybrid, the entities involved (`Insight→Entity`), and key voices (`Person→Insight`), with raw
  evidence. *(hybrid + shared topics + `Insight→Entity` + `Person→Insight`)*
- **FR4.3 Episode Detail** — navigable transcript with highlighted matched segments; "related
  insights" via a topic-scoped hybrid query. *(hybrid + two-tier)*

### FR5: Graph (#658)

- **FR5.1** Node size reflects a retrieval signal (insight density / cross-show breadth /
  search-context relevance), not just degree. *(hybrid)*
- **FR5.2** Edge weight reflects retrieval confidence (insight count × confidence). Feeds RFC-080.
  *(two-tier + edges)*
- **FR5.3** Node click opens the **populated** Detail panel (FR4), not a dead-end. *(all of FR4)*

### FR6: Dashboard

- **FR6.1** Briefing cards grounded in retrieval (top insight + top segment per topic). Until the
  MCP layer exists `[orthogonal]`, cards use a direct hybrid query rather than a briefing pack.
  *(hybrid + compounds)*
- **FR6.2** Activity chart driven by retrieval-volume signal (query volume by topic over time).
  Tufte-compliant: single y-axis, no dual-axis, no decorative lines.

## Orthogonal capabilities (not this PRD's spine)

These are independent capabilities that some surfaces could *additionally* use. They are **not
built**, **not prerequisites**, and **do not block** anything above. Each surface ships its
shipped-capability value now and adds these affordances if/when they land.

| Orthogonal capability | Surfaces that would add an affordance | Status |
| --- | --- | --- |
| **Contradiction edges** | Digest contradiction indicator; Graph contradiction edge type; Topic Entity View "where they disagree" | not built; separate capability/RFC |
| **MCP / LITM briefing packs** (RFC-093) | Dashboard briefing cards + "view evidence" | not built; `[TBD]` |
| **Corpus-impact / coverage surface** | Explore coverage card; Library coverage-gap indicator; Dashboard coverage-gaps card | not built; `[TBD]` |

When one lands, the relevant surface gains the affordance — additive, never blocking.

## Success Metrics

- Cross-surface navigation rate (Search → Detail → Graph) increases vs current baseline.
- Library filter → episode click-through increases with relevance snippets active.
- Digest cross-show synthesis band engagement: any non-zero rate (validates the differentiator).
- Node-click → populated-panel rate: ~100% of clicks reach real content (no dead-ends).
- Name/show → Detail navigation works for resolved entities (no plain-text dead-ends).

## Design Considerations

### OQ-1: PanelRetrievalStore

A `PanelRetrievalStore` (Pinia) caches retrieval results per entity id with a short TTL; panel opens
trigger a retrieval call, subsequent opens within TTL are instant. Resolves the #672 `subjectStore`
open question. Decide before podcast-product panel work.

### OQ-2: Search-context propagation

A shared `activeSearchContext` store so Graph reflects an active search (retrieval-score-weighted
nodes). Useful but adds cross-surface state complexity — defer to podcast product, design
separately.

### OQ-3: Dashboard async card content

Retrieval-backed cards add load latency — render skeleton-first, populate async. Confirm the
dashboard architecture supports async card content before implementing.

## Cross-Surface Flows (Issue #674 Revisited)

Issue #674 identified five cross-surface flow gaps. The shipped foundation addresses four:

| Gap | How the shipped foundation fixes it |
| --- | --- |
| Feed names in Digest non-interactive | Show names resolve via canonical identity + `HAS_EPISODE` → show-scoped Library view |
| Library rows require too many clicks to reach graph | Relevance snippets give context before clicking; compound results reduce pogo-sticking |
| Digest topic bands switch tabs instead of opening Topic Entity View | Topic Entity View has real content (hybrid + shared topics) → band click opens right rail |
| Speaker names in Search plain text | Entity resolver + `Person→Insight` map names → canonical ids → Person Landing |
| *(fifth gap: not directly addressed here)* | — |

## Milestone Assignment

| Surface | Change | Consumes | Readiness |
| --- | --- | --- | --- |
| Search/Explore | Two-tier results, raw-evidence toggle, query-type indicator | two-tier, intent routing | ready |
| Search/Explore | Interactive entity names → Detail | resolver, edges | ready |
| Library | Hybrid ranking on filter chips; relevance snippets | hybrid | ready |
| Library | Show name → show-scoped view | `HAS_EPISODE` | ready |
| Digest | Topic bands by retrieval signal; cross-show synthesis band | hybrid, shared topics | ready |
| Digest | Interactive speaker/show names | resolver, edges | ready |
| Detail — Person Landing | Retrieval-grounded positions | `Person→Insight`, compounds | ready |
| Detail — Topic Entity View | Cross-show coverage + entities + voices | hybrid, edges | ready |
| Detail — Episode Detail | Related insights, segment highlights | hybrid, two-tier | ready |
| Graph | Node size / edge weight; node click → populated panel | hybrid, edges, FR4 | ready |
| Dashboard | Briefing cards (direct hybrid); retrieval-volume chart | hybrid, compounds | ready |
| *Orthogonal affordances* | contradiction / coverage / MCP cards | (orthogonal) | when built |

## Related Work

- `docs/prd/PRD-031-search.md`, `docs/prd/PRD-032-hybrid-corpus-search.md`
- `docs/rfc/RFC-090-hybrid-retrieval.md`, `docs/rfc/RFC-091-kg-proximity-signal.md`,
  `docs/rfc/RFC-093-litm-context-packs.md`
- `docs/rfc/RFC-080-graph-visualization-extensions.md`
- GitHub issues #658, #669, #670, #671, #672, #673, #674 (all CLOSED).

## Release Checklist

- [ ] PRD reviewed and approved
