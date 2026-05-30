# PRD-031: Search

- **Status**: Draft
- **Author**: Marko
- **Created**: 2026-05-24
- **Target**: v2.7 (retrieval foundation) + Media Alpha (full Search UI)
- **Related PRDs**:
  - `docs/prd/PRD-032-hybrid-corpus-search.md` — backend capability
  - `docs/prd/PRD-033-search-powered-surfaces.md` — surface propagation
  - `docs/prd/PRD-021-semantic-corpus-search.md` — predecessor (FAISS semantic search)
  - `docs/prd/PRD-028-position-tracker.md` — position evolution data (temporal intent)
- **Related RFCs**:
  - `docs/rfc/RFC-090-hybrid-retrieval.md` — hybrid retrieval pipeline
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` — canonical identity
- **Related issues**: #849 (Search retrieval prerequisites), #466 (GI/KG depth roadmap — CLOSED, superseded), #484 (Semantic Corpus Search — predecessor), #671 (Search/Explore filter UX, CLOSED), #672 (right rail entity views, CLOSED), #674 (cross-surface flows, CLOSED)

> **Stabilization note (2026-05-30):** This doc was rebased from an earlier draft (PRD-029)
> authored against a different doc-numbering universe. Numbers, titles, and module paths
> have been corrected to this repo. Forward-references to capabilities that do not yet
> exist here (an MCP tool layer; a corpus-impact / coverage surface; typed contradiction
> KG edges) are marked **[TBD — not yet specified]** rather than linked to a number.

---

## Summary

Search in the viewer today is a single-signal FAISS vector lookup over Grounded Insight Layer
(GIL) insight nodes (see `docs/prd/PRD-021-semantic-corpus-search.md`, shipped). It answers
"what is similar to this query" — but not "who said what, across which shows, and how does it
relate to everything else in the corpus." **Search** is the first version of search that
earns the name: multi-signal (BM25 + vector + KG proximity), intent-aware, corpus-oriented,
provenance-grounded, and structured for how humans actually ask questions about media. It is the
product layer over the retrieval backend specified in RFC-090 and PRD-032.

## Background & Context

- **What problem this solves.** Users don't experience current search as a search engine — they
  experience it as autocomplete with extra steps. It surfaces one signal (semantic similarity),
  ignores named entities, ignores show boundaries, ignores time, ignores contradiction, and dumps
  results with no structure. Speaker names are plain text; there is no provenance, no synthesis,
  and no way to orient to the corpus before diving in.
- **Why now.** PRD-021 / RFC-061 delivered semantic corpus search (FAISS over insights). The
  GI/KG depth roadmap (#466) explicitly calls for moving beyond shallow single-signal retrieval —
  named-entity recall, relational context, corpus-level behavior. Search is the product
  framing of that next step.
- **How it relates to existing features.** Search consumes the hybrid retrieval backend
  (RFC-090 plus the additive signals in RFC-091/092/093) and the canonical identity layer
  (RFC-072). It is complementary to the Graph surface, not a replacement.

## Goals

- Deliver meaningful search-quality improvement through hybrid retrieval (BM25 + vector + KG
  proximity).
- Make search results navigable into the full knowledge graph — not a dead end.
- Surface corpus context before and alongside results (scope, coverage, contradictions).
- Support four distinct query intents with appropriate result presentation for each.
- Lay the Search surface foundation that later enrichers can build on.

## Non-Goals

- Full AI synthesis over results (that is the autoresearch loop — Search is retrieval +
  structure, not generation).
- Real-time ingestion (corpus is batch-indexed; Search operates on committed corpus state).
- Public-facing search (this is the viewer for beta users, not an embeddable widget).
- Replacing the Graph surface — Search and Graph are complementary, not competing.

## Personas

- **Beta researcher**: Power user exploring the corpus for media-intelligence work.
  - Needs to find specific people, quotes, and cross-show positions quickly.
  - Search gives named-entity recall, provenance, and graph navigation from results.
- **Operator (you)**: Owns corpus quality and coverage.
  - Needs to understand what the corpus covers densely vs sparsely.
  - The corpus scope bar and coverage intent surface this at query time.
- **Agent (future)**: Autoresearch / MCP consumer issuing programmatic queries.
  - Needs shaped, intent-aware results rather than raw lists.
  - Served by the briefing-pack tool specified in RFC-093 **[TBD — MCP tool layer not yet built]**.

## User Stories

- _As a beta researcher, I can search "Sam Altman" and find every place his name appears across
  shows — not just things "about OpenAI" — so that exact named-entity queries work._
- _As a beta researcher, I can see "this query spans 6 shows, 34 episodes, 12 speakers" before
  reading results, so that I understand the size and shape of what's out there._
- _As a beta researcher, I can click a speaker name in a result to open that person's entity view,
  so that a result is a graph entry point, not a dead end._
- _As a beta researcher, I can see upfront that a topic has conflicting positions across shows, so
  that I know to look at the contradiction breakdown before reading individual quotes._
- _As a beta researcher, I can compare how two shows cover a topic side by side, so that I get
  cross-show synthesis instead of an interleaved ranked list._
- _As a beta researcher, I can ask how a topic's coverage changed over time and see an arc, so that
  temporal queries track position evolution, not just similarity._

## Functional Requirements

### FR1: Query Intent Routing

Search routes each query to one of four intents. Routing is automatic (no user
configuration) but the detected intent is surfaced lightly so users understand the result
structure they see.

- **FR1.1**: Detect intent across four types — `entity_lookup`, `cross_show_synthesis`,
  `temporal_tracking`, `corpus_coverage` — using the rules-based router in RFC-090 §3.6.
- **FR1.2**: Default to hybrid retrieval for unclassified queries.
- **FR1.3**: Surface the detected intent in the UI ("Searching as: entity lookup") with an
  optional manual override.
- **FR1.4**: Intent routing is rules-based in v2.7; ML-upgradable per RFC-092 (no UI change).

| Intent | Trigger signals | Result presentation |
|--------|----------------|---------------------|
| `entity_lookup` | Person name, show name, specific term | Entity card header + insight list below |
| `cross_show_synthesis` | Comparative language, multiple show refs, "vs", "across" | Grouped by show + synthesis header |
| `temporal_tracking` | Date references, "over time", "evolution", "changed" | Timeline view + date-ordered results |
| `corpus_coverage` | "how much", "which shows", "covered" | Coverage map, not result list |

### FR2: Query Bar

- **FR2.1**: Single full-width text input, prominent.
- **FR2.2**: Query-intent badge below the input (auto-detected, lightly displayed, overridable).

### FR3: Corpus Scope Bar

- **FR3.1**: Compact indicator immediately below the query bar, before results, showing coverage
  of the current result set: shows · episodes · speakers · date range.
- **FR3.2**: Updates on each search; active filters reduce the counts in real time.
- **FR3.3**: Links to a full coverage map for `corpus_coverage` intent queries. _(The richer
  coverage/"impact" visualization is **[TBD — corpus-impact surface not yet specified]**;
  v2.7 ships basic counts only.)_

```text
Searching across: 6 shows  ·  34 episodes  ·  12 speakers  ·  Date range: Jan 2023 – Mar 2026
```

### FR4: Filter Chip Bar

- **FR4.1**: Unified filter chip bar per issue #671 (CLOSED). Chips: Show, Person, Date Range,
  Content Type.
- **FR4.2**: Applied chips shown inline; active filters reduce the corpus scope bar numbers.

### FR5: Result Cards (Grounded Insight Cards)

Each result card is a grounded insight card, not a text snippet.

- **FR5.1**: Show name, episode title, date — all clickable, navigate to show/episode.
- **FR5.2**: Exact grounding quote (the evidence, not a summary).
- **FR5.3**: Speaker name clickable → Person Entity View in right rail (#672, CLOSED).
- **FR5.4**: Topic tag clickable → Topic Entity View in right rail (#672, CLOSED).
- **FR5.5**: Confidence indicator (enricher confidence score; subtle).
- **FR5.6**: Contradiction badge when the insight is in a detected contradiction cluster.
  _Depends on typed contradiction KG edges, which **do not exist yet** (see RFC-091 open
  questions); until then this badge is hidden / degrades gracefully._

```text
┌─────────────────────────────────────────────────────────┐
│  [Show name] · [Episode title] · [Date]                 │
│                                                          │
│  "[Exact quote supporting this insight]"                 │
│                                                          │
│  ── [Speaker name ↗] ── [Topic tag] ── [Confidence] ──   │
│                                          ⚡ Contradiction │
└─────────────────────────────────────────────────────────┘
```

### FR6: Result Presentation by Intent

- **FR6.1 `entity_lookup`**: Entity card header (Person Landing or Topic Entity View, #672) with
  canonical identity, key positions, coverage summary; insight list below, filtered to the entity.
- **FR6.2 `cross_show_synthesis`**: Results grouped by show; each group header shows the show's
  dominant coverage angle. Synthesis panel above: "N shows covered this topic — A: pro / B:
  skeptical / C: neutral." _(Synthesis derivation source is an open question — see below.)_
- **FR6.3 `temporal_tracking`**: Results ordered by episode date ascending with a lightweight
  timeline (dots sized by insight count per episode).
- **FR6.4 `corpus_coverage`**: No result list — a coverage table (shows × time periods, cells =
  episode count). This is the Explore surface bleeding into Search for coverage queries.

### FR7: Right Rail (Search Context)

- **FR7.1**: Clicking a speaker or topic from a result opens Person Landing / Topic Entity View
  (#672) in the right rail. Search is the primary entry point for these panels.
- **FR7.2**: Right rail persists across result scrolling (#673, CLOSED); clicking a different
  speaker updates it without closing.

## Success Metrics

**Retrieval quality (internal eval loop):**

- Named-entity recall: >85% (entity appears in top-5 when queried by exact name).
- Hybrid vs vector-only nDCG@10: >10% improvement on the test query set.
- Cross-show query diversity: ≥2 distinct shows in top-5 for cross-show queries.

**User behavior (beta):**

- Right-rail activation rate from Search results: >30% of sessions (navigability).
- Contradiction badge click-through: any non-zero rate (value of surfacing).
- Query refinement rate: <40% (high refinement ⇒ intent routing is wrong).

**Qualitative (beta feedback):**

- Can users find a specific person's statements without knowing the episode?
- Do users understand what they're searching across before reading results?
- Do cross-show queries feel meaningfully different from single-show results?

## Dependencies

- **RFC-090** Hybrid Retrieval Pipeline (LanceDB + RRF) — **Hard** — Draft.
- **RFC-072** Canonical Identity — **Hard** — **Partial**: slug-based IDs exist
  (`src/podcast_scraper/identity/slugify.py`); the freeform-text → canonical-ID **entity
  resolver this PRD assumes does not exist yet** (tracked in #849).
- **Issue #672** Person Landing + Topic Entity View — **Hard** — issue CLOSED; panels specified,
  build state to confirm.
- **Issue #671** Filter chip bar — **Soft** — CLOSED.
- **Issue #673** Right rail persistence — **Soft** — CLOSED.
- **RFC-088** Enrichment Layer Architecture (contradiction signals) — **Soft** — contradiction
  badge degrades gracefully; typed contradiction edges are **not yet built**.
- **MCP tool layer** — **[TBD — not yet specified]** — required only for the agent-facing
  briefing pack (RFC-093), not for the viewer.

## Constraints & Assumptions

**Constraints:**

- Operates on committed (batch-indexed) corpus state; no real-time indexing.
- Local-first: retrieval and embedding run on the operator's machine (Apple M-series), no cloud
  dependency required for core search.

**Assumptions:**

- GIL insight nodes carry grounding quotes with timestamps (`timestamp_start_ms` /
  `timestamp_end_ms` on `SupportingQuote`) — verified in `src/podcast_scraper/gi/contracts.py`,
  which makes segment↔insight linking feasible.
- Embedding model is `sentence-transformers/all-MiniLM-L6-v2` (384-dim), per the current FAISS
  index.

## Design Considerations

### Intent routing: rules-based now, ML later

- **Option A — Rules-based router (v2.7)**: Deterministic keyword/regex routing (RFC-090 §3.6).
  - **Pros**: Ships immediately; no training data needed; transparent.
  - **Cons**: Misclassifies ambiguous/novel phrasings.
- **Option B — ML classifier (later)**: Fine-tuned classifier (RFC-092).
  - **Pros**: Robust; improves with query data.
  - **Cons**: Needs ≥500 labeled queries from the eval loop first.
  - **Decision**: Ship A now; B is config-switchable once data exists. Misrouting degrades to
    sub-optimal weights, not wrong results — RRF is robust to weight perturbation.

## Open Questions

1. **Intent routing override.** Allow manual intent selection? Leaning yes, optional — auto-detect
   default, dropdown to override.
2. **Synthesis panel generation.** Is the cross-show synthesis header derived at index time
   (preferred — no per-query LLM calls) or query time? Needs enricher output structured enough for
   index-time derivation (RFC-088).
3. **Contradiction edge availability.** The contradiction badge depends on typed contradiction KG
   edges, which **do not exist yet**. Define the fallback (enrichment-time metadata scan) explicitly.
4. **Search vs Explore boundary.** `corpus_coverage` intent overlaps the Explore surface. Likely
   converging; for now treat as separate surfaces with a navigation link.

## Related Work

- PRD-032: `docs/prd/PRD-032-hybrid-corpus-search.md` — backend capability.
- PRD-033: `docs/prd/PRD-033-search-powered-surfaces.md` — cross-surface propagation.
- PRD-021: `docs/prd/PRD-021-semantic-corpus-search.md` — predecessor.
- PRD-026: `docs/prd/PRD-026-topic-entity-view.md` — Topic Entity View panel.
- PRD-029: `docs/prd/PRD-029-person-profile.md` — Person profile content.
- RFC-090/091/092/093 — the retrieval backend and additive signals.
- Issue #849: Search retrieval prerequisites (survivors of #466).
- Issue #466: GI + KG depth roadmap (CLOSED, superseded).
- Issues #671, #672, #673, #674 (all CLOSED).

## Release Checklist

**v2.7 (retrieval foundation):**

- [ ] RFC-090 LanceDB backend + RRF fusion layer
- [ ] Query intent router (rules-based)
- [ ] Corpus scope bar (basic counts)
- [ ] Hybrid result cards with full provenance
- [ ] Speaker/topic names as clickable links (#672 prerequisites confirmed)
- [ ] Filter chip bar (#671)

**Media Alpha (full Search):**

- [ ] Cross-show synthesis grouping
- [ ] Temporal timeline view
- [ ] Contradiction badge (requires typed KG contradiction edges — prerequisite RFC/issue)
- [ ] Right rail integration with persistence (#672 + #673)
- [ ] Corpus coverage intent → coverage map view
- [ ] Entity card header for `entity_lookup` intent

**V3:**

- [ ] ML query intent classifier (RFC-092)
- [ ] Saved searches
- [ ] Search result export (research bundle)
- [ ] Search as MCP tool endpoint (agents call Search directly — requires MCP layer)
