# KG Ontology (Draft)

**Status:** Draft — aligns with [PRD-019](../prd/PRD-019-knowledge-graph-layer.md) and [RFC-055](../rfc/RFC-055-knowledge-graph-layer-core.md).

**Scope:** The **Knowledge Graph Layer (KG)** models **entities, themes, and relationships** for discovery and linking. It is **not** the same as the **Grounded Insight Layer (GIL)** in `docs/gi/ontology.md`, which centers on **insights**, **quotes**, and **grounding**.

## Design principles

1. **Episode-anchored**: Every KG graph is produced **per episode** in v1; global merge is a logical union of files or a future projection layer.
2. **Stable IDs**: Node IDs should be deterministic where possible (episode-scoped prefixes).
3. **Explicit separation from GIL**: Do not reuse `Insight` / `Quote` semantics from GIL; cross-linking GIL and KG is **optional** and **post-v1** unless specified in RFC-055 updates.

## Node types (v1 — illustrative)

Exact fields are finalized when `kg.schema.json` lands. Illustrative categories:

| Type | Description |
| --- | --- |
| `Episode` | Anchor for the episode (id aligns with pipeline episode identity). |
| `Entity` | Typed real-world items (e.g. person, organization) when extraction exists. |
| `Topic` | Theme or subject label (slug or normalized string — rules TBD). |

## Edge types (v1 — illustrative)

| Type | Description |
| --- | --- |
| `MENTIONS` | Entity or topic appears in a span of the episode (optional offsets in implementation). |
| `RELATED_TO` | Typed weak link between nodes (semantics TBD). |

## Versioning

- **`schema_version`** in each artifact (string, semver or project convention).
- Bump ontology and schema together when breaking changes occur.

## Related

- [GIL ontology](../gi/ontology.md) — grounded insights, quotes, SUPPORTED_BY (different feature).
