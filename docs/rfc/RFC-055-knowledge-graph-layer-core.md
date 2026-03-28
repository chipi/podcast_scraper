# RFC-055: Knowledge Graph Layer — Core Concepts & Data Model

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, downstream consumers
- **Related PRDs**:
  - `docs/prd/PRD-019-knowledge-graph-layer.md` (Knowledge Graph Layer — **KG**)
  - `docs/prd/PRD-017-grounded-insight-layer.md` (Grounded Insight Layer — **GI / GIL** — separate feature)
- **Related RFCs** (reference — analogous patterns):
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md` (GIL core — artifact shape, co-location, schema discipline)
  - `docs/rfc/RFC-053-adaptive-summarization-routing.md` (adaptive routing — shared `EpisodeProfile`; optional `route_kg_extraction` per content shape)
  - `docs/rfc/RFC-004-filesystem-layout.md` (output layout)
- **Related Documents**:
  - `docs/kg/ontology.md` — Human-readable ontology (draft)
  - `docs/kg/kg.schema.json` — JSON Schema (**to be added** when implementation lands; see §Schema)
  - `docs/ARCHITECTURE.md` — Module boundaries

## Abstract

This RFC defines the **Knowledge Graph Layer (KG)** as a **separate** feature from the **Grounded Insight Layer (GIL)** defined in RFC-049. KG focuses on **entities, topics, and relationships** suitable for graph-style consumption across episodes, while **GIL** remains **evidence-first** (insights + verbatim quotes + grounding) in `gi.json`.

KG introduces its **own** per-episode artifact contract, config surface, and optional CLI namespace **`kg`**, distinct from **`gi`** (grounded insights). Implementation may reuse transcript and metadata inputs; it **must not** overload `gi.json` as the KG canonical store.

**Architecture alignment:**

- Follows the same **per-episode file** pattern as GIL where possible (co-located under episode output, logical union for global views).
- Respects **module boundaries** (`workflow` orchestration, dedicated `kg` package for extraction/serialization — exact layout in implementation PR).
- References **RFC-049** only for **patterns** (schema version, provenance), not for GI semantics.

## Problem Statement

Downstream users need **linking and structure** (who, what, how topics connect) that is **not** the primary contract of GIL. Without a dedicated KG design:

- **Confusion** arises between “graph” and “grounded insights” in naming and artifacts.
- **Scope creep** risks overloading GIL with entity extraction concerns deferred in PRD-017.
- **Consumers** lack a **stable KG** contract to build against.

**Use cases:**

1. **Cross-episode theme exploration**: Find recurring entities or topics across a feed.
2. **Structured export**: Load KG JSON into a database or visualization tool.
3. **Parallel operation with GIL**: Same run produces **`gi.json`** and KG artifact when both flags are on.

## Goals

1. **Define KG ontology** (node/edge types v1) in `docs/kg/ontology.md` and keep it in sync with implementation.
2. **Define storage**: Per-episode KG artifact filename, JSON shape, `schema_version`, provenance fields.
3. **Define configuration**: Feature flag and model/provider hooks (aligned with `Config` patterns used for GIL).
4. **Separate CLI naming**: **`kg`** subcommands for KG operations vs **`gi`** for GIL (per PRD-019 and user-facing consistency).
5. **Validation**: Publish `docs/kg/kg.schema.json` and validate in CI when KG is generated in tests.

## Constraints & Assumptions

**Constraints:**

- **Must not** store KG as the primary payload inside `gi.json`.
- **Must** be disable-by-default until implementation ships.
- **Must** remain compatible with existing output directory layout (ADR-003/004 family); exact paths specified at implementation time.
- Naming: **`gi`** = grounded insights; **`kg`** = knowledge graph — do not interchange in user-facing strings.

**Assumptions:**

- Transcripts are available for KG extraction v1.
- Global merge / DB projection: **KG relational serving** is covered by [PRD-018](../prd/PRD-018-database-projection-gil-kg.md) / [RFC-051](RFC-051-database-projection-gil-kg.md) (same RFC as GIL projection, **separate tables**) — optional for v1 until artifacts stabilize.

## Design & Implementation (High Level)

### 1. Artifact

- **Format**: JSON document per episode, distinct from `*.gi.json`.
- **Contents**: `schema_version`, `episode_id`, extraction metadata, `nodes`, `edges` (or equivalent graph serialization), with types enumerated in `docs/kg/ontology.md`.
- **Co-location**: Same episode directory family as transcripts/metadata; exact relative path TBD (e.g. `metadata/*.kg.json` — finalize during implementation to mirror GIL conventions).

### 2. Ontology v1 (initial categories)

Exact types are **draft** until implementation; initial buckets:

- **Episode-level anchor** (link to episode id).
- **Entity-like nodes** (e.g. person, organization — naming TBD in ontology).
- **Topic / theme** nodes (distinct from GIL “Topic” if needed to avoid collision — prefix or separate namespace in IDs).
- **Edges**: co-occurrence, mentioned_in, related_to (subset to be minimal for v1).

Update **`docs/kg/ontology.md`** as the source of truth; RFC references it by path.

### 3. Config (illustrative)

- Boolean **`generate_kg`** (or name aligned with config naming review) — default `false`.
- Optional model/provider keys for extraction tier (follow patterns from GIL and summarization providers; details in implementation).

### 4. CLI (illustrative)

- **`kg`** namespace for inspect/export/query **KG** artifacts.
- **`gi`** remains **only** for GIL (`gi.json` / grounded insights).

### 5. Relationship to GIL

| Aspect | GIL (RFC-049 / PRD-017) | KG (this RFC / PRD-019) |
| --- | --- | --- |
| Primary question | What is claimed, and what evidence supports it? | What is linked to what (entities, themes)? |
| Canonical artifact | `gi.json` | KG artifact per RFC implementation |
| User CLI | `gi` | `kg` |
| Grounding contract | Required (quotes + spans) | Not the KG v1 focus; confidence may apply to extractions |

Cross-links between artifacts (e.g. KG node referencing a GIL `insight_id`) are **optional** and **out of scope for v1** unless explicitly added in a follow-up RFC.

## Schema

- Add **`docs/kg/kg.schema.json`** before or with the first implementation PR.
- Provide **`make validate-kg-schema`** or reuse a generic JSON schema validation pattern (mirror `validate-gi-schema` tooling).

## Testing Strategy

- Unit tests: node/edge builders, ID stability, schema validation.
- Integration tests: transcript → KG artifact → validation.
- E2E (optional): enable `generate_kg` in a config path and assert artifact presence.

## Rollout

- Document flag in `docs/api/CONFIGURATION.md` when implemented.
- Link PRD-019 and this RFC from `GROUNDED_INSIGHTS_GUIDE.md` or a dedicated **`docs/guides/KNOWLEDGE_GRAPH_GUIDE.md`** when written.

## Alternatives Considered

1. **Extend `gi.json` with KG nodes** — Rejected: blurs contracts and complicates GIL consumers.
2. **Only GIL, no KG** — Rejected for users who need entity/graph workflows without grounding-first semantics.

## References

- [PRD-019: Knowledge Graph Layer](../prd/PRD-019-knowledge-graph-layer.md)
- [PRD-017: Grounded Insight Layer](../prd/PRD-017-grounded-insight-layer.md)
- [RFC-049: GIL Core](RFC-049-grounded-insight-layer-core.md) (pattern reference)
