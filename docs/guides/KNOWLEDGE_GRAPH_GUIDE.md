# Knowledge Graph Guide

This guide documents the **Knowledge Graph Layer (KG)**: structured **entities**,
**topics**, and **relationships** extracted from episode content for linking and discovery.
It complements the [Grounded Insights Guide](GROUNDED_INSIGHTS_GUIDE.md), which covers
evidence-backed insights (`gi.json`).

**Status:** KG is specified in [PRD-019](../prd/PRD-019-knowledge-graph-layer.md) and
[RFC-055](../rfc/RFC-055-knowledge-graph-layer-core.md) / [RFC-056](../rfc/RFC-056-knowledge-graph-layer-use-cases.md).
**Implemented:** `generate_kg` + per-episode `*.kg.json`, configurable extraction
(`kg_extraction_source`: stub / summary_bullets / provider with LLM `extract_kg_graph`),
and the `kg` CLI (RFC-056).

---

## What Is the Knowledge Graph Layer?

KG answers: *“What entities and relationships can we extract or infer from this corpus?”*
It is **not** a rename of grounded insights. **GIL** (`gi`, `gi.json`) remains
**evidence-first** (insights and quotes). **KG** uses its **own** per-episode artifact and
**`kg`** CLI namespace.

| Aspect | GIL (PRD-017) | KG (PRD-019) |
| --- | --- | --- |
| Primary question | What is claimed, and what evidence supports it? | What is linked to what (entities, themes)? |
| Canonical artifact | `gi.json` | `*.kg.json` next to metadata (same basename as `.metadata.json`) |
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

- **Config:** `generate_kg: true` (default `false`). Requires `generate_metadata: true`
  (same rule as GIL).
- **CLI:** `--generate-kg` (with `--generate-metadata`).

### Extraction modes (GI-style)

| `kg_extraction_source` | Behavior |
| --- | --- |
| `summary_bullets` (default) | **Topic** nodes from the first `kg_max_topics` summary bullets (needs `generate_summaries` + bullets). **Entity** nodes from detected hosts/guests. `extraction.model_version` records `summary_bullets`. When the pipeline uses an LLM for KG-from-bullets, the effective backend follows **`kg_extraction_provider`** or **`summary_provider`** the same way as `provider` mode (see below). |
| `stub` | **Episode** + hosts/guests only; ignores summary bullets for topics. **`kg_extraction_provider`** is not used (no LLM KG client). |
| `provider` | Runs **`extract_kg_graph()`** on transcript text using **`kg_extraction_provider`** if set, otherwise **`summary_provider`** (same backend enum as summarization). **ML** providers (`transformers`, `hybrid_ml`) return no graph fragment — pipeline **falls back** to summary bullets when available. Optional **`kg_extraction_model`** overrides the chat model. **`kg_merge_pipeline_entities`** (default `true`) adds hosts/guests after LLM entities, deduped by **entity_kind + name** (same as LLM entity list). |

CLI flags: `--kg-extraction-source`, **`--kg-extraction-provider`**, `--kg-max-topics`,
`--kg-max-entities`, `--kg-extraction-model`, `--no-kg-merge-pipeline-entities`.

### KG LLM provider vs summary provider {#kg-llm-provider-vs-summary-provider}

- **`kg_extraction_provider`** (config) / **`--kg-extraction-provider`** (CLI): optional.
  **Unset** means reuse **`summary_provider`** for KG LLM calls — one client, no extra
  init/cleanup.
- **Set** to another registered summarization backend (e.g. OpenAI for summaries,
  Gemini for KG) when you want **`extract_kg_graph`** (and, when applicable, the
  bullets LLM path) to run on a **different** stack than episode summarization. The
  pipeline **`create_summarization_provider` → `initialize()`** for that episode and
  **`cleanup()`** afterward when the instance is not the summary provider (same pattern
  as GIL evidence providers).
- **Field reference:** [Configuration API — Knowledge Graph (KG)](../api/CONFIGURATION.md).

Pipeline: KG runs during **metadata generation**. Use a transcript on disk so provider
mode can read text; `extraction.transcript_ref` stays in the artifact for provenance.

---

## Output Artifacts

- **File:** `metadata/<episode_basename>.kg.json` (alongside `.metadata.json` / `.gi.json`).
- **Ontology:** [docs/architecture/kg/ontology.md](../architecture/kg/ontology.md) (**v1 frozen**, GitHub #464 — matches shipped pipeline).
- **Schema:** [docs/architecture/kg/kg.schema.json](../architecture/kg/kg.schema.json) — validate with
  `make validate-kg-schema [ARTIFACTS_DIR=path]`.

Episode **metadata** includes `knowledge_graph` when KG ran: `artifact_path`, `node_count`,
`edge_count`, `generated_at`, `schema_version` (provenance only; full graph is in `kg.json`).

---

## CLI (`kg` namespace)

Run as `python -m podcast_scraper.cli kg <subcommand> ...` (same entrypoint as `gi`).

| Subcommand | Purpose |
| --- | --- |
| **`kg validate`** | Validate one or more paths (files or directories) against `kg.schema.json`. Use **`--strict`** for full JSON Schema. **`-q`** / **`--quiet`**: only failures. |
| **`kg inspect`** | Summarize one episode artifact: **`--episode-path`** to `.kg.json`, or **`--output-dir`** + **`--episode-id`**. **`--format json`** for machine output. |
| **`kg export`** | Scan **`--output-dir`** for all `*.kg.json`. **`--format ndjson`** (default) or **`merged`**. **`--out PATH`** or stdout. **`--strict`** to require schema-valid artifacts. |
| **`kg entities`** | Cross-episode **entity roll-up** (counts, episodes, mentions). **`--min-episodes N`**, **`--format json`**. |
| **`kg topics`** | **Topic pair co-occurrence** within the same episode. **`--min-support N`**, **`--format json`**. |

See [CLI reference](../api/CLI.md#knowledge-graph-kg-subcommands) for examples.

---

## Consumption and integration

- **File-based**: Scan per-episode KG JSON for corpus analytics (see RFC-056 use cases).
- **Browser viewer (prototype)**: Load `*.kg.json` (and `*.gi.json`) in a static local UI —
  `make serve-gi-kg-viz`, then `http://127.0.0.1:8765/`. See [Development Guide — GI / KG
  browser viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) and
  [`web/gi-kg-viz/README.md`](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viz/README.md).
- **Database**: Optional relational projection per [PRD-018](../prd/PRD-018-database-projection-gil-kg.md) /
  [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md) — **separate** from GIL tables.

---

## Validation and troubleshooting

- **Strict JSON Schema:** `make validate-kg-schema` or
  `python scripts/tools/validate_kg_schema.py path/to/dir-or-file.kg.json`, or
  `python -m podcast_scraper.cli kg validate PATH [PATH...] --strict`
- **PRD-019 metrics (optional gates):** `make kg-quality-metrics DIR=path/to/run`
  or `python scripts/tools/kg_quality_metrics.py path … [--enforce --strict-schema]`.
  CI runs the same enforce pass as GIL on `tests/fixtures/gil_kg_ci_enforce` via
  `make quality-metrics-ci`.
- **Fixture:** `tests/fixtures/kg/minimal.kg.json` for smoke checks.
- **Acceptance (E2E configs):** `config/acceptance/kg/*.yaml` — mirrors
  `config/acceptance/gi/` (Planet Money + The Journal; ML, OpenAI, Ollama, Anthropic,
  Gemini, Mistral, DeepSeek, Grok). Stub-style configs use `kg_extraction_source: stub`
  (like GI default `gi_insight_source: stub`); bullet-driven configs use
  `acceptance_*_kg_ml_summary_bullets.yaml`. Run:
  `make test-acceptance CONFIGS="config/acceptance/kg/*.yaml"`.

Run metrics export (`metrics.json`) includes KG rollups: `kg_topic_nodes_total`,
`kg_entity_nodes_total`, `kg_extractions_stub` / `kg_extractions_summary_bullets` /
`kg_extractions_provider` / `kg_extractions_provider_summary_bullets` (LLM topics from
bullets only), `kg_avg_topics_per_artifact`, `kg_avg_entities_per_artifact`.

Failures during KG write are **non-fatal** (metadata is still written); check logs for
`KG artifact generation failed`. Common causes: disk permissions, or schema drift if
`kg.schema.json` was tightened without updating the builder.

---

## Choosing a mode (operations)

| When to use | Mode | Notes |
| --- | --- | --- |
| Fastest, no LLM cost, corpus smoke tests | `stub` | Episode + detected hosts/guests only; no LLM topics. |
| Good default when you already generate summary bullets | `summary_bullets` (default) | With an API `summary_provider`, one **extra** chat completion per episode derives short Topic labels (+ LLM entities) from bullets; `extraction.model_version` is `provider:summary_bullets:<model>`. ML-only summaries keep verbatim bullet labels (`summary_bullets`) with no KG LLM. |
| Richer topics/entities from transcript text | `provider` | **Extra** chat completion per episode on the **transcript** via `extract_kg_graph`. Adds latency and token cost on top of summarization. |

**ML / hybrid ML summarization:** `extract_kg_graph` is not implemented for local ML-only paths. With `kg_extraction_source: provider`, the pipeline **falls back** to `summary_bullets` when bullets exist; otherwise you may get a sparse graph (stub-like). Prefer **`summary_bullets`** or **`stub`** for ML-heavy runs unless you also use an API summarization provider. Topic labels copied from ML bullets may include **ASR/subword noise**; the pipeline strips a few known broken prefixes (e.g. hyphenated fragment starts) when normalizing bullet text for KG/GI consumers.

**Empty or tiny graphs:** Check `extraction.model_version` in `*.kg.json` (`stub`, verbatim `summary_bullets`, `provider:<model>` for transcript KG, or `provider:summary_bullets:<model>` for bullet-derived LLM KG). If provider calls fail, logs show a debug message and the builder may fall back. Validate artifacts with `kg validate --strict` and inspect counts via `kg inspect --format json`.

**JSONL metrics:** When `jsonl_metrics_enabled` is on, `episode_finished` lines include `kg_sec` (wall time for KG for that episode). `run_finished` lines include KG rollups (`kg_artifacts_generated`, `kg_failures`, `kg_provider_extractions`, extraction-mode counts, node totals) alongside the existing GI fields.

---

## Recorded product decisions (v1, KG shallow) {#recorded-product-decisions-v1-kg}

This table mirrors the **GIL v1 record** in [Grounded Insights Guide § Recorded product decisions (v1, issue 460)](GROUNDED_INSIGHTS_GUIDE.md#recorded-product-decisions-v1-issue-460) so operators who enable **both** flags see aligned expectations. It captures **what shallow v1 KG promises**, not the full [depth backlog](https://github.com/chipi/podcast_scraper/issues/466).

| Decision area | v1 choice |
| --- | --- |
| **Extraction + ML** | Default **`kg_extraction_source: summary_bullets`** (topics from bullets, entities from pipeline hosts/guests). **`stub`** = episode + hosts/guests only—good for smoke tests. **`provider`** calls LLM **`extract_kg_graph`** on the summarization provider; **`transformers` / `hybrid_ml`** do **not** implement it—the pipeline **falls back** to summary bullets when available, else a sparse graph. The CLI emits a **warning** when **`kg_extraction_source: provider`** and **`summary_provider`** is ML (outside pytest). Details: [Choosing a mode (operations)](#choosing-a-mode-operations). |
| **Entity / topic identity** | **Episode-local** labels and slugs; **no** web-scale entity resolution or global canonical IDs (per [PRD-019](../prd/PRD-019-knowledge-graph-layer.md) non-goals). **`kg entities`** roll-ups match strings/slugs as extracted—treat counts as **indicative**, not a curated knowledge base. |
| **Consumption CLI** | **`kg validate`**, **`inspect`**, **`export`**, **`entities`**, **`topics`** only—file scan and aggregations ([RFC-056](../rfc/RFC-056-knowledge-graph-layer-use-cases.md)). **No** `kg query` IR or NL layer in v1 ([GitHub #466](https://github.com/chipi/podcast_scraper/issues/466)). |
| **GIL ↔ KG in artifacts** | **No** required links from KG nodes to **`insight_id`** or quotes in v1 (optional future work; same epic as above). |
| **Scale / SQL** | Same as GIL: **file-based** consumption first. **Postgres** ([PRD-018](../prd/PRD-018-database-projection-gil-kg.md), [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md)) is **separate**—track with a dedicated issue if none exists. |

---

## Related documents

- [PRD-005: Episode summarization](../prd/PRD-005-episode-summarization.md) — summaries as the consumption layer alongside KG and GIL.
- [PRD-019: Knowledge Graph Layer](../prd/PRD-019-knowledge-graph-layer.md)
- [RFC-055: KG — Core Concepts & Data Model](../rfc/RFC-055-knowledge-graph-layer-core.md)
- [RFC-056: KG — Use Cases & Consumption](../rfc/RFC-056-knowledge-graph-layer-use-cases.md)
- [PRD-017: Grounded Insight Layer](../prd/PRD-017-grounded-insight-layer.md) (GIL)
- [Grounded Insights Guide](GROUNDED_INSIGHTS_GUIDE.md)
- [Development Guide — GI / KG browser viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) — optional local UI for `kg.json` / `gi.json`
- [Recorded product decisions (v1, KG shallow)](#recorded-product-decisions-v1-kg) — v1 scope table (this guide)
