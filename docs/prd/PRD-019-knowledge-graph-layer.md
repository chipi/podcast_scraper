# PRD-019: Knowledge Graph Layer (KG)

- **Status**: 📋 Draft
- **Authors**: Podcast Scraper Team
- **Related RFCs**:
  - RFC-055 (Knowledge Graph — Core Concepts & Data Model)
  - RFC-056 (Knowledge Graph — Use Cases & End-to-End Consumption)
- **Related PRDs** (reference only; separate features):
  - [PRD-017: Grounded Insight Layer](PRD-017-grounded-insight-layer.md) (GI / GIL — evidence-first insights and quotes)
  - [PRD-018: Database Projection](PRD-018-database-projection-gil-kg.md) (Postgres projection for **GIL and KG** — RFC-051)
- **Related Documents**:
  - `docs/architecture/kg/ontology.md` — Human-readable ontology (**v1 frozen**, GitHub #464)
  - `docs/architecture/kg/kg.schema.json` — Machine-readable schema (published; validated in CI via `make validate-kg-schema`)
  - `docs/guides/KNOWLEDGE_GRAPH_GUIDE.md` — User-facing guide (updated with implementation)
- **Related UX specs**:
  - [UXS-001: GI / KG viewer](../uxs/UXS-001-gi-kg-viewer.md)

## Summary

The **Knowledge Graph Layer (KG)** is a **separate product feature** from the **Grounded Insight Layer (GIL / GI)** in [PRD-017](PRD-017-grounded-insight-layer.md). KG focuses on **structured linking and discovery**: entities, topics, relationships, and cross-episode connections suitable for graph-style queries and analytics.

**GIL** answers: *“What are the takeaways, and what verbatim evidence supports them?”*
**KG** answers: *“What entities and relationships can we extract or infer from this corpus?”*

Both features may consume the same upstream artifacts (transcripts, metadata, speaker signals) but **must remain independently toggleable** and **must not share the same primary artifact contract** (`gi.json` is reserved for GIL; KG defines its own storage and schema per RFC-055).

## Background & Context

Podcast libraries are easier to explore when content is not only summarized (PRD-005) or evidence-backed (PRD-017) but also **connected**: people, organizations, themes, and how they co-occur across episodes. Classic **knowledge graph** patterns (nodes, typed edges, optional global or logical union across episodes) support research, recommendations, and downstream tools that are **not** the primary goal of GIL’s grounding contract.

**Relationship to GIL (PRD-017):**

- **Complementary**: GIL emphasizes **trust** via quotes and grounding; KG emphasizes **structure** and **linking**. An episode may have **both** a `gi.json` (GIL) and a KG artifact (per RFC-055) when both features are enabled.
- **Not a duplicate**: KG is **not** a rename of GIL. User-facing CLI and config **must** distinguish **`gi`** (grounded insights) from **`kg`** (knowledge graph operations) as specified in RFC-055.
- **Shared inputs**: Transcripts (PRD-001), metadata (PRD-004), speaker detection (PRD-008) are natural inputs; neither feature should duplicate scrape/transcribe orchestration.

## Goals

1. **Structured graph export**: Represent episode-scoped and optionally cross-episode **nodes and edges** (entities, topics, relationships) in a **documented** format.
2. **Independent feature flag**: Enable or disable KG **without** requiring GIL or changing GIL semantics.
3. **Clear consumption path**: Document how consumers query or merge KG data (files, and **relational projection** per [PRD-018](PRD-018-database-projection-gil-kg.md) / [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md) when enabled).
4. **Stable contracts**: Versioned schema and ontology artifacts under `docs/architecture/kg/` aligned with RFC-055.
5. **Documented consumption**: End-to-end use cases, query patterns, and CLI expectations for KG are specified in RFC-056 and `KNOWLEDGE_GRAPH_GUIDE.md`, in **parity with** the GIL split (PRD-017 / RFC-049 + RFC-050 + `GROUNDED_INSIGHTS_GUIDE.md`).

## Documentation parity with GIL (PRD-017)

KG is intentionally documented to the **same artifact set** as GIL where it applies:

| Artifact | GIL | KG |
| --- | --- | --- |
| Product requirements | PRD-017 | PRD-019 (this document) |
| Core data model & artifact | RFC-049 | RFC-055 |
| Use cases & consumption | RFC-050 | RFC-056 |
| User guide | `GROUNDED_INSIGHTS_GUIDE.md` | `KNOWLEDGE_GRAPH_GUIDE.md` |
| Relational serving (optional) | PRD-018 / RFC-051 | Same PRD/RFC (separate tables) |

Shared projection work does **not** duplicate here: [PRD-018](PRD-018-database-projection-gil-kg.md) /
[RFC-051](../rfc/RFC-051-database-projection-gil-kg.md) cover Postgres export for **both**
features.

## Non-Goals (v1)

- Replacing GIL or merging KG into `gi.json`
- Perfect entity resolution across the open web (v1 may use episode-local or slug-level identities)
- Real-time streaming graph updates
- Mandatory global graph database (logical union of per-episode files remains valid per RFC-055)
- Truth verification or fact-checking (extraction quality and confidence may be modeled; “true in the world” is out of scope)

## Personas

- **Researchers & analysts**: Explore **who/what** appears across episodes and how topics connect.
- **Developers**: Integrate KG outputs into search, RAG, or visualization tools with a **stable JSON** contract.
- **Operators**: Run KG alongside GIL with predictable cost and config.

## User Stories

- _As a researcher, I can see which entities and topics recur across episodes from the same show so that I can navigate themes without listening to every minute._
- _As a developer, I can load per-episode KG artifacts into my tool or database so that I can build cross-episode queries._
- _As an operator, I can enable KG independently of grounded insights so that I can compare outputs or control cost._
- _As a developer, I can validate KG artifacts against a published JSON Schema in CI so that regressions are caught early._
- _As an analyst, I can aggregate entities or topics across episodes from a run without requiring a graph database for v1._

## Functional Requirements

### FR1: Feature Control

- **FR1.1**: Config flag (e.g. `generate_kg` or equivalent per RFC-055) enables KG extraction when true; default off.
- **FR1.2**: KG runs only when prerequisites in RFC-055 are satisfied (e.g. transcript available).

### FR2: Artifacts & Provenance

- **FR2.1**: Per-episode KG output path and filename pattern defined in RFC-055 (distinct from `*.gi.json`).
- **FR2.2**: Artifacts include `schema_version`, extraction provenance, and `episode_id` as specified in RFC-055.
- **FR2.3**: Machine-readable validation against `docs/architecture/kg/kg.schema.json` when schema is published.

### FR3: Ontology & Relationships

- **FR3.1**: Documented node and edge types in `docs/architecture/kg/ontology.md` (entities, topics, relationships as scoped in RFC-055).
- **FR3.2**: Cross-episode identity strategy (e.g. slugs, stable IDs) described in RFC-055; semantic merging deferred if marked out of scope.

### FR4: Integration

- **FR4.1**: Module boundaries respected (orchestration vs extraction vs I/O per existing architecture).
- **FR4.2**: Metadata document may reference KG artifact with provenance index only (mirrors GIL index pattern in PRD-017 where applicable).

### FR5: Consumption & operator experience

- **FR5.1**: RFC-056 defines v1 use cases, query patterns, and illustrative consumption contracts for KG.
- **FR5.2**: User-facing documentation in `KNOWLEDGE_GRAPH_GUIDE.md` stays aligned with implemented config, paths, and `kg` CLI behavior.
- **FR5.3**: Optional SQL access remains in scope of PRD-018 / RFC-051 when KG tables are projected (no duplicate projection spec in PRD-019).

## Non-functional requirements (NFR)

- **Testability**: KG code paths are unit- and integration-testable; schema validation is
  enforceable in CI when KG is generated in tests (see RFC-055 testing notes).

- **Observability**: Extraction should emit structured metrics or logging consistent with
  pipeline patterns (align with PRD-016 / existing metrics where applicable).

- **Versioning**: `schema_version` and ontology evolution follow the same discipline as
  GIL (`docs/architecture/kg/` as source of truth).

## Success Metrics

- **Schema compliance**: 100% of generated KG artifacts validate against published schema in CI when KG is exercised.
- **Independence**: With GIL off and KG on, pipeline completes and writes KG outputs only; with KG off and GIL on, behavior unchanged from PRD-017 baseline.
- **Documentation**: PRD-017 cross-links to this PRD where “graph vs grounded insights” confusion could arise.
- **Consumption spec**: RFC-056 and `KNOWLEDGE_GRAPH_GUIDE.md` reviewed and updated as part of any KG release that changes behavior or CLI.

## Acceptance criteria (v1)

- **Product**: FR1–FR5 satisfied for a representative run (transcript → KG artifact → validate
  against published schema).

- **Docs**: PRD-019, RFC-055, RFC-056, `KNOWLEDGE_GRAPH_GUIDE.md`, and `docs/architecture/kg/ontology.md`
  are mutually consistent for the shipped artifact shape.

- **Independence**: With only KG enabled, pipeline behavior does not depend on GIL
  (`generate_gi` may be false).

## Dependencies

- **PRD-001**, **PRD-004**: Transcripts and metadata as inputs
- **PRD-008** (optional): Richer speaker/entity signals when available
- **RFC-055**: Technical design, schema, and implementation plan
- **RFC-056**: Use cases, consumption patterns, and CLI expectations
- **PRD-017**: Reference for parallel feature pattern and artifact coexistence; not a runtime dependency for KG v1

## Constraints & Assumptions

- KG must remain **optional** and **backward compatible** when disabled.
- Coexistence with GIL must be **explicit** in docs and config to avoid conflating **`gi`** and **`kg`** user-facing commands.

## Design Considerations

- **Naming**: User-facing **grounded insights** remain under **`gi`** (RFC-049 / RFC-050);
  knowledge graph operations and artifacts use **`kg`** (RFC-055). Do not interchange **`gi`**

  and **`kg`** in user-facing strings.

- **Evolution**: Entity-rich KG may later link to GIL insight IDs in a future RFC; v1 focuses on a clean KG core.

## Open Questions

- **v1 node/edge freeze**: **Done (GitHub #464).** `docs/architecture/kg/kg.schema.json`, `docs/architecture/kg/ontology.md`, and the `build_artifact` pipeline are aligned for v1; strict validation enforces `extraction.model_version` and Entity `role` values shipped by the builder.
- Relative **implementation priority** of `kg export` vs `gi export` (both are in PRD-018 / RFC-051 scope; shipping order may differ).
- Which **`kg`** subcommands ship in v1 vs later (RFC-056 lists illustrative commands).

## Related Work

- [PRD-017: Grounded Insight Layer](PRD-017-grounded-insight-layer.md)
- [PRD-018: Database Projection](PRD-018-database-projection-gil-kg.md) (GIL + KG serving layer)
- [RFC-055: Knowledge Graph Layer — Core](../rfc/RFC-055-knowledge-graph-layer-core.md)
- [RFC-056: Knowledge Graph Layer — Use Cases & Consumption](../rfc/RFC-056-knowledge-graph-layer-use-cases.md)
- [Knowledge Graph Guide](../guides/KNOWLEDGE_GRAPH_GUIDE.md)
