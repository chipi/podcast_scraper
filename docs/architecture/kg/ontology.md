# KG Ontology (v1)

**Status:** **v1 frozen** (GitHub #464) — matches the shipped `build_artifact` pipeline and **`docs/architecture/kg/kg.schema.json`**. For design history see [PRD-019](../../prd/PRD-019-knowledge-graph-layer.md), [RFC-055](../../rfc/RFC-055-knowledge-graph-layer-core.md), and [RFC-056](../../rfc/RFC-056-knowledge-graph-layer-use-cases.md).

**Shipping note:** The pipeline emits **Episode**, **Topic**, and **Entity** nodes plus **MENTIONS** edges (Topic or Entity → Episode) for extraction modes `stub`, `summary_bullets`, and `provider`. **`RELATED_TO`** is defined in the schema for forward compatibility but is **not** emitted by the v1 builder. **`extraction.model_version`** is `stub`, verbatim `summary_bullets`, `provider:<model>` (transcript `extract_kg_graph`), or `provider:summary_bullets:<model>` (LLM topics derived from summary bullets); ML-only summarization falls back to verbatim bullets when no API provider is available (see [Knowledge Graph Guide](../../guides/KNOWLEDGE_GRAPH_GUIDE.md)). Sibling pattern for GIL: Issue #460 in `docs/architecture/gi/ontology.md`.

**Scope:** The **Knowledge Graph Layer (KG)** models **entities, themes, and relationships** for discovery and linking. It is **not** the **Grounded Insight Layer (GIL)** (`docs/architecture/gi/ontology.md` — insights, quotes, grounding).

## Design principles

1. **Episode-anchored**: Every KG graph is produced **per episode** in v1; global merge is a logical union of files or a future projection layer.
2. **Stable IDs**: Episode nodes use `episode:{episode_id}`; Topic/Entity ids are global (slug-based) so merged graphs connect across episodes (see § Identity conventions).
3. **Explicit separation from GIL**: Do not reuse `Insight` / `Quote` semantics from GIL; cross-linking GIL and KG is **optional** and **post-v1** unless specified in RFC-055 updates.

## Identity conventions (v1 shipped)

| Node type | ID pattern (examples) |
| --- | --- |
| `Episode` | `episode:{episode_id}` (same anchor string as artifact `episode_id`; shared with GI) |
| `Topic` | `topic:{slug}` — global by normalized label slug (bullets or provider) |
| `Entity` | **v1.2 (RFC-072):** `person:{slug}` or `org:{slug}`; properties use **`kind`**: `person` \| `org`. **Legacy:** `entity:person:{slug}` / `entity:organization:{slug}` with **`entity_kind`** — migrate with `scripts/migrate_kg_entity_ids.py`. Properties include **`name`** and optional **`label`**. |

**Slug:** Derived from the topic label via the pipeline slugifier (lowercase, hyphenated, max length capped in code) — must be non-empty in artifacts.

**Edges:** **`MENTIONS`** is directed **`from`** the Topic or Entity **`to`** the Episode anchor node.

## Node types (v1)

Fields and enums are normative in **`kg.schema.json`**.

| Type | Description |
| --- | --- |
| `Episode` | Anchor: `podcast_id`, `title`, `publish_date` (required). Optional `audio_url`, `duration_ms` in schema for consumers; v1 builder does not set them. |
| `Entity` | `name`, `entity_kind` (`person` \| `organization`). Optional `label` (graph/UI; pipeline copies `name`). Optional `role`: `host`, `guest`, or `mentioned` (v1 builder sets one when emitting an Entity). Optional **`description`** (v1.1 / GitHub #487): short episode-specific context from LLM extraction. |
| `Topic` | `label`, `slug` (both required, non-empty). Optional **`description`** (v1.1 / #487) for richer embedding context. |

## Edge types (v1)

| Type | Description |
| --- | --- |
| `MENTIONS` | Topic or Entity **→** Episode (appears-in-episode). Optional `properties` object (often empty `{}`). |
| `RELATED_TO` | **Reserved** — not emitted by the v1 builder; allowed in schema for forward compatibility. |

## Provenance

- **`schema_version`:** `1.0` (legacy), **`1.1`** (optional Topic/Entity `description` per #487), or **`1.2`** (RFC-072: `kind`, `person:` / `org:` ids).
- **`extraction.model_version`:** `stub` \| `summary_bullets` \| `provider:<summarization_model_id>` \| `provider:summary_bullets:<summarization_model_id>`.
- **`extraction.extracted_at`:** ISO-8601 timestamp (UTC `Z` in shipped output).
- **`extraction.transcript_ref`:** Relative transcript path or label for the text used in extraction.

## Versioning

- Bump **`schema_version`**, **`kg.schema.json`**, and this file together for breaking changes after v1.

## Related

- [GIL ontology](../gi/ontology.md) — grounded insights, quotes, SUPPORTED_BY (separate feature).
