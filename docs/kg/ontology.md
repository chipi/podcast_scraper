# KG Ontology (v1)

**Status:** **v1 frozen** (GitHub #464) — matches the shipped `build_artifact` pipeline and **`docs/kg/kg.schema.json`**. For design history see [PRD-019](../prd/PRD-019-knowledge-graph-layer.md), [RFC-055](../rfc/RFC-055-knowledge-graph-layer-core.md), and [RFC-056](../rfc/RFC-056-knowledge-graph-layer-use-cases.md).

**Shipping note:** The pipeline emits **Episode**, **Topic**, and **Entity** nodes plus **MENTIONS** edges (Topic or Entity → Episode) for extraction modes `stub`, `summary_bullets`, and `provider`. **`RELATED_TO`** is defined in the schema for forward compatibility but is **not** emitted by the v1 builder. **`extraction.model_version`** is `stub`, `summary_bullets`, or `provider:<model>`; ML-only summarization may fall back when `extract_kg_graph` is unavailable (see [Knowledge Graph Guide](../guides/KNOWLEDGE_GRAPH_GUIDE.md)). Sibling pattern for GIL: Issue #460 in `docs/gi/ontology.md`.

**Scope:** The **Knowledge Graph Layer (KG)** models **entities, themes, and relationships** for discovery and linking. It is **not** the **Grounded Insight Layer (GIL)** (`docs/gi/ontology.md` — insights, quotes, grounding).

## Design principles

1. **Episode-anchored**: Every KG graph is produced **per episode** in v1; global merge is a logical union of files or a future projection layer.
2. **Stable IDs**: Node IDs are episode-scoped and deterministic for a given extraction inputs (see § Identity conventions).
3. **Explicit separation from GIL**: Do not reuse `Insight` / `Quote` semantics from GIL; cross-linking GIL and KG is **optional** and **post-v1** unless specified in RFC-055 updates.

## Identity conventions (v1 shipped)

| Node type | ID pattern (examples) |
| --- | --- |
| `Episode` | `kg:episode:{episode_id}` |
| `Topic` | `kg:topic:{episode_id}:{slug}` (summary bullets) or `kg:topic:{episode_id}:llm:{slug}` (provider partial) |
| `Entity` | `kg:entity:{episode_id}:{role}:{index}` (pipeline hosts/guests) or `kg:entity:{episode_id}:llm:{n}` (provider partial) |

**Slug:** Derived from the topic label via the pipeline slugifier (lowercase, hyphenated, max length capped in code) — must be non-empty in artifacts.

**Edges:** **`MENTIONS`** is directed **`from`** the Topic or Entity **`to`** the Episode anchor node.

## Node types (v1)

Fields and enums are normative in **`kg.schema.json`**.

| Type | Description |
| --- | --- |
| `Episode` | Anchor: `podcast_id`, `title`, `publish_date` (required). Optional `audio_url`, `duration_ms` in schema for consumers; v1 builder does not set them. |
| `Entity` | `name`, `entity_kind` (`person` \| `organization`). Optional `role`: `host`, `guest`, or `mentioned` (v1 builder always sets one when emitting an Entity). |
| `Topic` | `label`, `slug` (both required, non-empty). |

## Edge types (v1)

| Type | Description |
| --- | --- |
| `MENTIONS` | Topic or Entity **→** Episode (appears-in-episode). Optional `properties` object (often empty `{}`). |
| `RELATED_TO` | **Reserved** — not emitted by the v1 builder; allowed in schema for forward compatibility. |

## Provenance

- **`schema_version`:** `1.0` (string, matches JSON Schema pattern).
- **`extraction.model_version`:** `stub` \| `summary_bullets` \| `provider:<summarization_model_id>`.
- **`extraction.extracted_at`:** ISO-8601 timestamp (UTC `Z` in shipped output).
- **`extraction.transcript_ref`:** Relative transcript path or label for the text used in extraction.

## Versioning

- Bump **`schema_version`**, **`kg.schema.json`**, and this file together for breaking changes after v1.

## Related

- [GIL ontology](../gi/ontology.md) — grounded insights, quotes, SUPPORTED_BY (separate feature).
