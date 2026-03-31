# WIP: P4 — API / LLM KG for cleaner topics (ML follow-up)

**Status:** idea backlog — not implemented as a single task  
**See also:** [Knowledge Graph Guide § Choosing a mode](../guides/KNOWLEDGE_GRAPH_GUIDE.md#choosing-a-mode-operations), [kg-extraction-provider plan](kg-extraction-provider-plan.md)

---

## What “P4” refers to

Optional use of an **API summarization / KG-capable provider** so **Topic** labels (and optionally **entities**) are **not** limited to **verbatim ML summary bullets**.

- **Today (ML-heavy):** `kg_extraction_source: summary_bullets` + `summary_provider: transformers` / `hybrid_ml` → topics are **bullet text** (plus normalizations like prefix stripping). Cheap and predictable; quality tracks summarizer + ASR noise.
- **P4 direction:** Add **LLM cost/latency** to get **short thematic topic phrases** and/or **transcript-grounded** extraction via provider methods such as **`extract_kg_from_summary_bullets`** or **`extract_kg_graph`**.

## Two concrete patterns

| Pattern | Rough intent | Tradeoff |
| --- | --- | --- |
| **Bullets → LLM topics** | Keep ML summaries; one extra call to distill bullets into topic labels | Extra token spend; bullets-only context |
| **`kg_extraction_source: provider`** | LLM reads **transcript** for richer topics/entities | Highest cost/latency; needs a provider that implements `extract_kg_graph` |

## When to bother

- Operators want **browse/search-friendly** topic nodes without switching the **whole** summarization stack to an LLM.
- Verbatim bullets are **good enough** for linking episodes; P4 stays **out of scope**.

## Notes

- Local ML providers **do not** implement transcript `extract_kg_graph`; `provider` mode **falls back** when only ML is configured — see the KG guide.
- Wiring and `kg_extraction_provider` design choices are tracked in **`kg-extraction-provider-plan.md`** (broader than this P4 note).
