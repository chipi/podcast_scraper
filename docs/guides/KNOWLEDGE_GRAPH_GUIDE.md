# Knowledge Graph Guide

This guide will document the **Knowledge Graph Layer (KG)**: structured **entities**,
**topics**, and **relationships** extracted from episode content for linking and discovery.
It complements the [Grounded Insights Guide](GROUNDED_INSIGHTS_GUIDE.md), which covers
evidence-backed insights (`gi.json`).

**Status:** KG is specified in [PRD-019](../prd/PRD-019-knowledge-graph-layer.md) and
[RFC-055](../rfc/RFC-055-knowledge-graph-layer-core.md) / [RFC-056](../rfc/RFC-056-knowledge-graph-layer-use-cases.md).
Operational details below will be filled in as implementation lands (config keys, file
paths, CLI subcommands).

---

## What Is the Knowledge Graph Layer?

KG answers: *“What entities and relationships can we extract or infer from this corpus?”*
It is **not** a rename of grounded insights. **GIL** (`gi`, `gi.json`) remains
**evidence-first** (insights and quotes). **KG** uses its **own** per-episode artifact and
**`kg`** CLI namespace.

| Aspect | GIL (PRD-017) | KG (PRD-019) |
| --- | --- | --- |
| Primary question | What is claimed, and what evidence supports it? | What is linked to what (entities, themes)? |
| Canonical artifact | `gi.json` | KG JSON per RFC-055 (filename TBD at implementation) |
| User-facing CLI | `gi` | `kg` |

---

## How KG fits with summaries and grounded insights

Episode **summaries**, **KG**, and **grounded insights (GIL)** are complementary:

| Layer | Role |
| --- | --- |
| **Summaries** ([PRD-005: Episode summarization](../prd/PRD-005-episode-summarization.md)) | **Consume** quickly: skim what an episode is about (low friction, broad coverage). |
| **KG (this guide)** | **Navigate** across many episodes: who and what show up, how themes and entities connect. |
| **Grounded insights** ([Grounded Insights Guide](GROUNDED_INSIGHTS_GUIDE.md)) | **Key value and trust**: takeaways linked to **verbatim quotes** when the grounding stack succeeds. |

Summaries are not a substitute for verification when claims matter; **GIL** is where you **stress-test** takeaways against the transcript. **KG** helps you **move around** your library; it does not replace reading summaries or checking grounded insights for defensible claims. The same mental model appears in [Grounded Insights Guide § Summaries, KG, and grounded insights](GROUNDED_INSIGHTS_GUIDE.md#summaries-kg-and-grounded-insights-how-they-fit-together).

---

## Enabling KG

**Planned (v1):** A config flag such as **`generate_kg`** (default `false`) and optional
provider/model settings aligned with RFC-055. Exact names will appear in
`docs/api/CONFIGURATION.md` when implemented.

Pipeline order for KG is **after** transcript (and typically after summarization/metadata)
when those are prerequisites; RFC-055 defines hard prerequisites (e.g. transcript
available).

---

## Output Artifacts

- **Per-episode KG JSON** co-located with episode outputs (path pattern finalized in
  RFC-055).

- **Ontology**: [docs/kg/ontology.md](../kg/ontology.md) (draft).
- **Schema**: `docs/kg/kg.schema.json` — published with or before first implementation.

Metadata may include a **lightweight index** to the KG artifact (mirroring the GIL index
pattern in metadata) per PRD-019 FR4.2.

---

## CLI (`kg` namespace)

RFC-055 reserves the **`kg`** namespace for inspect/export/validate-style commands. RFC-056
describes intended consumption patterns. **Concrete subcommands and flags** will be
documented here and in `docs/api/CLI.md` once available.

---

## Consumption and integration

- **File-based**: Scan per-episode KG JSON for corpus analytics (see RFC-056 use cases).
- **Database**: Optional relational projection per [PRD-018](../prd/PRD-018-database-projection-gil-kg.md) /
  [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md) — **separate** from GIL tables.

---

## Validation and troubleshooting

When `kg.schema.json` exists, validation should mirror the GIL pattern (e.g. a
**`validate`** command or CI check). Troubleshooting notes will be added after initial
implementation (common failure modes: missing transcript, extraction timeouts, schema
mismatches).

---

## Related documents

- [PRD-005: Episode summarization](../prd/PRD-005-episode-summarization.md) — summaries as the consumption layer alongside KG and GIL.
- [PRD-019: Knowledge Graph Layer](../prd/PRD-019-knowledge-graph-layer.md)
- [RFC-055: KG — Core Concepts & Data Model](../rfc/RFC-055-knowledge-graph-layer-core.md)
- [RFC-056: KG — Use Cases & Consumption](../rfc/RFC-056-knowledge-graph-layer-use-cases.md)
- [PRD-017: Grounded Insight Layer](../prd/PRD-017-grounded-insight-layer.md) (GIL)
- [Grounded Insights Guide](GROUNDED_INSIGHTS_GUIDE.md)
