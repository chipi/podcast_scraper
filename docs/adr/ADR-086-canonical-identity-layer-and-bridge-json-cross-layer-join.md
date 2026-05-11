# ADR-086: Canonical Identity Layer and Per-Episode bridge.json Cross-Layer Join

- **Status**: Accepted
- **Date**: 2026-05-08
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-072](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md)
- **Related ADRs**: [ADR-052](ADR-052-separate-gil-and-kg-artifact-layers.md),
  [ADR-053](ADR-053-grounding-contract-for-evidence-backed-insights.md),
  [ADR-061](ADR-061-faiss-phase-1-with-post-filter-metadata.md),
  [ADR-062](ADR-062-sentence-boundary-transcript-chunking.md)

## Context & Problem Statement

GIL (**`gi.json`**), KG (**`*.kg.json`**), and semantic search (FAISS metadata) evolved with
**different id prefixes and join semantics** for the same real-world entities. Query-time string
heuristics were fragile and blocked cross-layer features (person across episodes, transcript lift to
Insights, graph navigation by stable topic).

## Decision

1. **Canonical Identity Layer (CIL)** — Introduce shared **`person:`**, **`org:`**, and
   **`topic:`** identifiers (slug rules and builders in-repo) as the **corpus-wide vocabulary** for
   joins. GIL and KG **remain separate artifacts and schemas** (**[ADR-052](ADR-052-separate-gil-and-kg-artifact-layers.md)**);
   CIL does not merge layers into one file.

2. **Per-episode `bridge.json`** — Emit a **small join artifact** under episode metadata that maps
   layer-local ids (for example GIL **`speaker:`** / KG **`entity:person:`**) to canonical CIL ids
   and records edges needed for cross-layer queries. The bridge is **write-time explicit**, not a
   runtime guess.

3. **Additive GIL v1.1 fields** — **`insight_type`** and **`position_hint`** stay in **`gi.json`**
   as typed, query-friendly enrichments; they do not replace the grounding contract
   (**[ADR-053](ADR-053-grounding-contract-for-evidence-backed-insights.md)**).

4. **Filesystem-first** — No database prerequisite for CIL + bridge; optional future Postgres
   projection (**[ADR-054](ADR-054-relational-postgres-projection-for-gil-and-kg.md)**) consumes the
   same artifacts.

5. **HTTP surface** — Read-oriented **`/api/persons/*`** and **`/api/topics/*`** (and related
   routes) expose CIL-backed navigation; semantic search **lift** uses bridge + offset alignment
   where enabled.

## Rationale

- **Stable joins** — One place to resolve “same person, two id schemes” for viewer + API consumers.
- **Preserves layer independence** — Teams can still evolve GIL and KG schemas under ADR-052;
   bridge is the **seam**, not a wholesale merge.
- **Search alignment** — FAISS chunks can lift to attributed insights when offsets and bridge rows
   agree (**[ADR-061](ADR-061-faiss-phase-1-with-post-filter-metadata.md)**,
   **[ADR-062](ADR-062-sentence-boundary-transcript-chunking.md)**).

## Alternatives Considered

1. **Merge GIL + KG into one mega-schema** — Rejected; violates ADR-052 and raises migration risk.
2. **Query-time fuzzy entity resolution only** — Rejected; non-reproducible, expensive, and weak for
   API contracts.
3. **RDBMS as primary identity store** — Deferred; ADR-054 remains future; files stay canonical.

## Consequences

- **Positive**: Cross-episode graph expansion, person or topic APIs, and transcript lift share one
   identity story.
- **Negative**: Pipeline and index code must keep **`bridge.json`** coherent with **`gi.json`** /
   **`*.kg.json`**; offset regressions need CI guards.
- **Neutral**: **`RFC-072`** remains the normative field-level spec; this ADR records the **architectural**
  contract only.

## Implementation Notes

- **Code**: `src/podcast_scraper/builders/` (bridge paths, slug builders), server routes under
  **`server/routes`**, graph utilities in **`web/gi-kg-viewer`**
- **Normative detail**: [RFC-072](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md)

## References

- [RFC-072: Canonical identity layer and cross-layer bridge](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md)
