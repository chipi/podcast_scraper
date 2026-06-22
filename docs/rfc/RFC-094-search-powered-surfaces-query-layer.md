# RFC-094: Search-Powered Surfaces — Relational-Query Layer + Front-End Retrieval State

- **Status**: Draft
- **v2 closure (RFC-097, 2026-06-20)**: stable `person:` / `topic:` / `org:` IDs formalized via CIL (RFC-072 + RFC-097); v2-materialized ABOUT + MENTIONS_PERSON + MENTIONS_ORG edges enable `who_said_what_about(topic)`, `insights_about(person)`, `mentions_of_person(org)` queries against per-artifact data (no CorpusGraph composition required). MCP exposure (RFC-095), panel caching (OQ-1), and coverage-aware filtering remain open. See [RFC-097](RFC-097-unified-kg-gi-ontology-v2.md).
- **Authors**: Marko
- **Stakeholders**: Core team
- **Related PRDs**:
  - `docs/prd/PRD-033-search-powered-surfaces.md` — the consumer-side spec this RFC implements the shared layer for
  - `docs/prd/PRD-031-search.md`, `docs/prd/PRD-032-hybrid-corpus-search.md`
- **Related RFCs**:
  - `docs/rfc/RFC-090-hybrid-retrieval.md` — hybrid retrieval backend (ranking)
  - `docs/rfc/RFC-091-kg-proximity-signal.md` — KG proximity **rejected as a retrieval signal**; relational structure comes from edges, not proximity
  - `docs/rfc/RFC-093-litm-context-packs.md` — MCP briefing packs (orthogonal; Dashboard)
- **Related ADRs**:
  - _(none yet)_

---

## Abstract

PRD-033 maps how every viewer surface consumes the shipped retrieval + relational-edge foundation.
Most surfaces (Search, Library, Digest) already have what they need — the existing hybrid
`/api/search` path. But the **Detail panels and Graph** need *relational* queries that **do not
exist yet**: "what positions does this person hold," "who said what about this topic," "what entities
does this insight concern," "what episodes belong to this show." The edges to answer these shipped
in #874 (`Person→Insight`, `Insight→Entity`, `Podcast→HAS_EPISODE→Episode`) and live in the
cross-layer `CorpusGraph` — but nothing exposes them as queries.

This RFC defines the **one shared, cross-cutting piece** PRD-033 depends on: a **relational-query
layer** (a small read API over the corpus graph + hybrid index) and the **front-end retrieval-state
architecture** that surfaces use to consume it (resolving PRD-033 OQ-1/2/3). The surface UIs
themselves are execution against this contract and are tracked as per-surface issues, not here.

**Ranking comes from hybrid (BM25+dense); relational structure comes from the edges.** KG-proximity
is not used (RFC-091 Decision Record).

## Problem Statement

The relational edges are in the corpus but **unconsumed** — the same dormancy risk that retired
KG-proximity. Concretely:

- **No relational query API.** `positions_of(person)` etc. require traversing `CorpusGraph`
  (`Person ←SPOKEN_BY← Quote ←SUPPORTED_BY← Insight`, `Insight ─MENTIONS→ Entity`,
  `Podcast ─HAS_EPISODE→ Episode`) + ranking with hybrid. There is no module or route that does this.
- **No front-end retrieval-state architecture.** PRD-033 OQ-1/2/3 are unresolved: per-entity panel
  caching, cross-surface search context, async card rendering.

Without this shared layer, every Detail/Graph surface would re-implement graph traversal + caching
ad hoc.

## Goals

1. A **read-only relational-query layer** exposing a small, fixed set of queries over the shipped
   edges + hybrid index, with stable request/response contracts.
2. Resolve PRD-033 **OQ-1/2/3** (panel cache, search context, async cards) as concrete front-end
   patterns.
3. Be **additive and non-blocking**: no new extraction, no new edges; consumes only what shipped.

## Non-Goals

- The surface UIs (per-surface issues).
- New edges or extraction (contradiction, etc. are orthogonal — PRD-033 §Orthogonal capabilities).
- The MCP briefing-pack layer (RFC-093) — Dashboard's richest cards are orthogonal.

## Design & Implementation

### 1. Relational-query layer (`search/relational_queries.py`)

A read-only module over `get_corpus_graph(corpus_dir)` (#849 cache) + the hybrid retrieval path.
Each function takes a resolved canonical id (via the existing `EntityResolver`) and returns a typed
result. Ranking of supporting evidence uses hybrid; structure uses graph traversal.

| Query | Inputs | Returns | Backed by |
| --- | --- | --- | --- |
| `positions_of(person_id)` | `person:` id | the person's insights (their positions) + grounding segments | `Person→Insight` + compounds |
| `who_said(topic_id)` | `topic:` id | per-person insights on the topic | `Insight ─ABOUT→ Topic` + `Person→Insight` |
| `insights_about(entity_id)` | `person:`/`org:` id | insights that concern the entity | `Insight ─MENTIONS→ Entity` |
| `entities_in(insight_id)` | `insight:` id | the people/orgs an insight concerns | `Insight ─MENTIONS→ Entity` |
| `episodes_of(show_id)` | `podcast:` id | the show's episodes | `Podcast ─HAS_EPISODE→ Episode` |
| `cross_show_synthesis(topic_id)` | `topic:` id | top insight **per distinct show** covering the topic | hybrid scoped to topic, grouped by show |
| `related_insights(scope)` | topic/episode scope | insights ranked for the scope | hybrid (topic/episode-scoped) |

Contracts are plain dataclasses (id, text, score, source_tier, show_id, episode_id, grounding). All
queries degrade gracefully (empty result, never raise) and are bounded by `k`.

### 2. API surface (`server/routes/relational.py`)

Thin FastAPI routes wrapping the layer, one per query, e.g. `GET /api/relational/positions?person=…`.
Same auth/middleware as `/api/search`. (MCP-tool exposure of the same functions is a later,
orthogonal step — RFC-093.)

### 3. Front-end retrieval state (resolves PRD-033 OQ-1/2/3)

- **OQ-1 `PanelRetrievalStore` (Pinia).** Keyed by canonical id, short TTL. A panel open triggers one
  relational-query call; opens within TTL are instant. The single cache all Detail panels share.
- **OQ-2 `activeSearchContext` (Pinia).** The current search/filter context, so Library rows show
  "why this episode" snippets and Graph can weight nodes by the active query. Read-only consumers;
  one writer (the search bar).
- **OQ-3 async cards.** Retrieval-backed cards render skeleton-first and populate from the layer
  async; no blocking on retrieval.

### 4. How surfaces consume it (maps to PRD-033 FRs)

| Surface | Uses |
| --- | --- |
| Search / Library / Digest | existing `/api/search` (+ `activeSearchContext`); **no new layer needed** — can start immediately |
| Detail — Person Landing | `positions_of` + compounds (`PanelRetrievalStore`) |
| Detail — Topic Entity View | `cross_show_synthesis` + `who_said` + `insights_about` |
| Detail — Episode Detail | `related_insights(episode)` |
| Graph | node/edge signals from the layer; node click → populated Detail (above) |
| Dashboard | `related_insights(topic)` for cards (briefing-pack form is orthogonal, RFC-093) |

## Rollout & Sequencing

1. **Layer first** (`relational_queries.py` + routes) — unblocks Detail + Graph.
2. **Front-end stores** (OQ-1/2/3).
3. **Per-surface issues** consume the layer. Search/Library/Digest can proceed in parallel before the
   layer lands (they reuse `/api/search`).

## Open Questions

1. **OQ-A Caching scope.** Per-corpus process cache for `CorpusGraph` already exists (#849); the
   panel TTL is front-end. Confirm no server-side per-query cache is needed at current scale.
2. **OQ-B MCP exposure.** Whether the same functions are surfaced as MCP tools now or with RFC-093.
   Default: routes now, MCP with the briefing-pack work.

## Benefits

- The shipped edges stop being dormant — a real consumer, with a stable contract.
- One place defines the relational queries; surfaces don't re-implement traversal.
- Front-end retrieval state is decided once, not per panel.

## Migration Path

Additive — new module + routes + Pinia stores. No changes to existing search, indexing, or edges.
