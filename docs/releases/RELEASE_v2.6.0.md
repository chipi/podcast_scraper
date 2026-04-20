# Release v2.6.0 — Corpus Intelligence UI, Semantic Search, and Operator Tooling

**Release Date:** April 2026  
**Type:** Minor Release  
**Last Updated:** April 2026

## Summary

v2.6.0 is a **minor release** that ships a **full browser surface** for exploring processed corpora alongside the pipeline: a **Vue 3 + Vite** GI/KG viewer on a **FastAPI** server, **semantic corpus search** (FAISS) exposed in the UI and CLI, **Corpus Library**, **Digest**, and **Dashboard** experiences, and a **graph exploration toolkit** for Cytoscape-based GI/KG views. On the **provider and evaluation** side, the release tightens how you **compare quality and cost**: the **Run Comparison** Streamlit app gains a **Performance** tab tied to **frozen YAML profiles** (RFC-064), sitting on top of the **seven LLM providers** and **hybrid ML** stack delivered in v2.5.0.

The **Python library** surface (`Config`, `run_pipeline`, `service.run`) stays backward compatible. HTTP, the SPA, and server-only routes are **additive** behind `pip install -e '.[server]'`. **GIL** and **KG** reach **Completed** status for **single-layer** artifacts and consumption (RFC-049 / RFC-050, RFC-055 / RFC-056); **cross-layer identity** remains future work (Draft RFC-072). **Multi-feed** corpus layout, manifest, and unified indexing ship per RFC-063.

---

## Spotlight — GI/KG viewer and corpus UI

v2.5.0 expanded **who** summarizes and detects speakers (providers). v2.6.0 expands **where** you inspect the results: a dedicated **GI/KG viewer** aligned with ADR-064–ADR-066 and RFC-062.

**Stack and entrypoints**

- **FastAPI** — `podcast serve` (optional `[server]` extra), CORS and static SPA, OpenAPI at `/docs`.
- **Vue 3 + Vite + Cytoscape** — SPA under `web/gi-kg-viewer`, shell **`corpusPath`**, shared design rules in UXS-001 and feature UXSs (Digest, Library, Graph, Search, Dashboard, …).
- **Playwright** — UI E2E in `web/gi-kg-viewer/e2e/`; contract summarized in `e2e/E2E_SURFACE_MAP.md`.

**Tabs and flows (high level)**

| Area | What you get | PRD / RFC |
| --- | --- | --- |
| **Library** | Feeds and episodes from disk, pagination and filters, episode detail (summary bullets, GI/KG paths), **FAISS similar episodes** when indexed, handoffs to **Graph** and **Search** | [PRD-022](../prd/PRD-022-corpus-library-episode-browser.md), [RFC-067](../rfc/RFC-067-corpus-library-api-viewer.md) |
| **Digest** | Rolling digest of recent work across feeds, **24h glance** from Library, optional **semantic topic bands** when a vector index exists | [PRD-023](../prd/PRD-023-corpus-digest-recap.md), [RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md) |
| **Dashboard** | **Pipeline** vs **Content intelligence** Chart.js panels, corpus stats, manifest awareness, capped **`run.json`** discovery, timelines for index / digest / GI-KG | [PRD-025](../prd/PRD-025-corpus-intelligence-dashboard-viewer.md), [RFC-071](../rfc/RFC-071-corpus-intelligence-dashboard-viewer.md) |
| **Graph** | Zoom (100% / numeric), Shift+drag box zoom, minimap v1, degree-bucket filter, built-in layouts, edge filters | [PRD-024](../prd/PRD-024-graph-exploration-toolkit.md), [RFC-069](../rfc/RFC-069-graph-exploration-toolkit.md) |
| **Search** | **Semantic** search over the corpus in the UI, consistent with CLI `podcast search` / indexing | [PRD-021](../prd/PRD-021-semantic-corpus-search.md), [RFC-061](../rfc/RFC-061-semantic-corpus-search.md), [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md) |

**HTTP surface**

- **`/api/corpus/*`** — Library and Dashboard data (feeds, episodes, detail, similar, aggregates as documented).
- **`GET /api/corpus/digest`** — Digest JSON for the Digest tab and health discovery.
- **`GET /api/search`** — Semantic search API used by the viewer ([Server Guide](../guides/SERVER_GUIDE.md)).
- **`POST /api/index/rebuild`** / **`GET /api/index/stats`** — Background vector index rebuild (**202** / **409** semantics) and staleness-oriented stats for operators.
- **`GET /api/health`** — May include **`corpus_library_api`**, **`corpus_digest_api`**, and related flags for capability discovery ([Migration Guide](../api/MIGRATION_GUIDE.md#v260-viewer-and-http)).

**Design and UX references**

- [Server Guide](../guides/SERVER_GUIDE.md) — full route table and behavior notes.
- [E2E Testing Guide](../guides/E2E_TESTING_GUIDE.md) — Playwright workflow.
- [Development Guide](../guides/DEVELOPMENT_GUIDE.md) — local serve, `SERVE_OUTPUT_DIR`, viewer development.
- [docs/uxs/index.md](../uxs/index.md) — UX specifications for shared tokens and feature surfaces.

---

## Spotlight — Semantic search, Grounded Insights (GIL), and Knowledge Graph (KG)

v2.6.0 ships the **retrieval** and **structured artifact** layers that the viewer tabs sit on: **vector search** over transcript chunks, **GIL** (`gi.json`) for grounded quotes and insights, and **KG** (`kg.json`) for entities, topics, and relationships. Together they explain why **Library** can show GI/KG paths, **Digest** can show topic bands when indexed, **Graph** can render Cytoscape views, and **Search** can return semantic hits.

### Semantic corpus search

- **FAISS** — `FaissVectorStore` implements the vector-store contract ([ADR-060](../adr/ADR-060-vectorstore-protocol-with-backend-abstraction.md)); embed, index, and query paths are specified in [RFC-061](../rfc/RFC-061-semantic-corpus-search.md) and [PRD-021](../prd/PRD-021-semantic-corpus-search.md).
- **CLI** — `podcast index` and `podcast search` for building and querying the corpus index; semantic exploration hooks for **`gi explore`** where documented in the CLI and guides.
- **HTTP** — **`GET /api/search`** for the viewer Search panel (same corpus root as the shell); index rebuild and stats under **`/api/index/*`** (see [Server Guide](../guides/SERVER_GUIDE.md)).
- **Viewer** — Semantic search wired to the API (**left** query column in the current shell; **similar episodes** in Library depend on the same index when present). Post–v2.6 viewer work moved corpus artifacts + **Data** cards onto **Dashboard** — see **RFC-062** and **UXS-006**.
- **After v2.6.0** — Draft [RFC-070](../rfc/RFC-070-semantic-corpus-search-platform-future.md) tracks optional backends (Qdrant, pgvector, and so on); **not** part of the v2.6.0 FAISS ship.

### Grounded Insight Layer (GIL)

- **Completed (single layer)** — [RFC-049](../rfc/RFC-049-grounded-insight-layer-core.md) (core model, `gi.json`, grounding contract) and [RFC-050](../rfc/RFC-050-grounded-insight-layer-use-cases.md) (consumption patterns, CLI and product use cases without cross-layer bridge).
- **What you can do** — Inspect grounded insights and quotes per episode; drive **Insight**-oriented flows in the viewer and documentation under [PRD-017](../prd/PRD-017-grounded-insight-layer.md) for the **single-layer** slice.
- **Not in v2.6.0** — **Canonical identity** and **`bridge.json`** for cross-layer joins remain **Draft** ([RFC-072](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md)); [PRD-017](../prd/PRD-017-grounded-insight-layer.md) stays **Partial** until that work lands.

### Knowledge Graph (KG)

- **Completed (single layer)** — [RFC-055](../rfc/RFC-055-knowledge-graph-layer-core.md) (ontology, `kg.json`, separation from GIL) and [RFC-056](../rfc/RFC-056-knowledge-graph-layer-use-cases.md) (roll-ups, `kg` CLI patterns, export-oriented use cases).
- **What you can do** — **Graph** tab and related viewer flows consume **KG** artifacts alongside GIL for exploration ([PRD-019](../prd/PRD-019-knowledge-graph-layer.md) single-layer scope; [RFC-069](../rfc/RFC-069-graph-exploration-toolkit.md) for interaction toolkit).
- **Same boundary as GIL** — Cross-layer alignment is **RFC-072**, not shipped in v2.6.0.

**Guides**

- [Semantic Search Guide](../guides/SEMANTIC_SEARCH_GUIDE.md) — indexing, `GET /api/search`, chunking, and operator notes.
- [Grounded Insights Guide](../guides/GROUNDED_INSIGHTS_GUIDE.md) — `gi.json`, schema, CLI.
- [Knowledge Graph Guide](../guides/KNOWLEDGE_GRAPH_GUIDE.md) — `kg.json`, entities, relationships, bridge placeholder for future CIL.

---

## Spotlight — Comparing providers and runs

v2.5.0 added **five** cloud LLM families plus **Ollama** on top of OpenAI and Gemini, with unified config and CLI flags. v2.6.0 improves **how you compare** them in practice—not only by reading docs, but by **joining quality metrics with frozen resource profiles**.

**1. Provider landscape (carried forward from v2.5.0, still the reference in v2.6.0)**

- **Local ML** — Whisper, spaCy, Transformers (default path).
- **Hybrid ML (RFC-042)** — MAP–REDUCE summarization with configurable REDUCE backends (transformers, Ollama, llama.cpp).
- **Cloud LLM** — OpenAI, Gemini, Anthropic, Mistral, DeepSeek, Grok, plus **Ollama** as local LLM host.

**2. Documentation-first comparison**

- [AI Provider Comparison Guide](../guides/AI_PROVIDER_COMPARISON_GUIDE.md) — decision matrices, cost and quality framing, “which provider?” narrative.
- [Provider Deep Dives](../guides/PROVIDER_DEEP_DIVES.md) — per-provider cards and quadrant-style comparisons.
- [Evaluation reports](../guides/eval-reports/index.md) — methodology (ROUGE, BLEU, embeddings, and related metrics) and report index.
- [ML Model Comparison Guide](../guides/ML_MODEL_COMPARISON_GUIDE.md) — local and hybrid model tradeoffs.

**3. Tooling new in the v2.6.0 track**

- **Run Comparison — Performance tab (RFC-066)** — Streamlit **`?page=performance`** joins **run metrics** from experiments with **frozen RFC-064 YAML profiles** so you can relate **summary quality** (eval runs under `data/eval/`) to **resource shape** (RSS, CPU, wall time by stage) on comparable fixtures.
- **Performance profiling framework (RFC-064)** — `config/profiles/`, captured artifacts under `data/profiles/`, `make profile-freeze` / `make profile-diff`, scripts described in [Performance profile guide](../guides/PERFORMANCE_PROFILE_GUIDE.md).
- **AutoResearch closure (RFC-057 / ADR-073)** — optimization loop and eval matrix work brought to a documented closure; silver references and broad config sweeps support **evidence-backed** model and prompt choices.

**4. Live pipeline visibility (developers)**

- **Live Pipeline Monitor (RFC-065)** — `--monitor`, `.pipeline_status.json`, terminal or log-friendly status; optional **`[monitor]`** extras (memray, py-spy). See [Live Pipeline Monitor guide](../guides/LIVE_PIPELINE_MONITOR.md).

Together, the v2.5.0 **provider breadth** and v2.6.0 **Performance tab + frozen profiles + eval library** give a coherent story: pick a provider or model, run **smoke or benchmark** evals, capture **profiles** on the same corpus shape, and inspect **quality vs cost** in one place.

---

## Multi-feed corpus, manifest, and indexing

- **RFC-063** — Multiple feeds, append/resume semantics, **layout A**, unified index behavior, **`corpus_manifest.json`** and run-summary hooks. See [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md).

---

## Pipeline download resilience and run metrics

- **Configurable HTTP retries** for media, transcripts, and RSS (`http_*`, `rss_*` on `Config`), plus **application-level episode retries** (`episode_retry_max`, `episode_retry_delay_sec`) after urllib3 exhaustion.
- **CLI** — `--http-retry-total`, `--http-backoff-factor`, `--rss-retry-total`, `--rss-backoff-factor`, `--episode-retry-max`, `--episode-retry-delay-sec` ([CLI](../api/CLI.md#control-options)).
- **`metrics.json`** — `http_urllib3_retry_events`, `episode_download_retries`, `episode_download_retry_sleep_seconds` ([Experiment Guide](../guides/EXPERIMENT_GUIDE.md#pipeline-run-metrics-download-resilience)).
- **Optional Issue #522-class extensions** — per-host throttling, `Retry-After`, circuit breaker, RSS conditional GET; fields and flags documented under [CONFIGURATION — Download resilience](../api/CONFIGURATION.md#download-resilience).
- **`failure_summary`** in `run.json` when episodes fail (counts by error type, failed episode identifiers).
- Download resilience: documented canonically under [CONFIGURATION.md — Download resilience](../api/CONFIGURATION.md#download-resilience) (inline YAML presets; no separate example file required).

---

## Operational observability (partial PRD-016)

Shipped in this release train: **test metrics** and **GitHub Pages** dashboards (RFC-025 / RFC-026), **live monitor** (RFC-065), **frozen profiles** and **Run Compare Performance** (RFC-064 / RFC-066). **RFC-027** items (for example CSV export) remain open.

---

## Documentation

- [API overview — HTTP / viewer](../api/index.md#http-viewer-api-server-extra)
- [Server Guide](../guides/SERVER_GUIDE.md)
- [Migration Guide — v2.6.0](../api/MIGRATION_GUIDE.md#v260-viewer-and-http)
- [E2E Testing Guide](../guides/E2E_TESTING_GUIDE.md)
- [Experiment Guide](../guides/EXPERIMENT_GUIDE.md) — pipeline `metrics.json` and download resilience
- [RFC index — v2.6.0 rows](../rfc/index.md) — RFC-049, 050, 055, 056, 057, 061–071

---

## Upgrade notes

- **Library users** — No code changes required for `run_pipeline` / `service.run`.
- **Viewer or HTTP consumers** — Install **`[server]`**, run `podcast serve` with a valid output directory, and align viewer `corpusPath` with your corpus root. See [Migration Guide](../api/MIGRATION_GUIDE.md#v260-viewer-and-http).
- **Health JSON** — Prefer explicit **`corpus_digest_api`** once on a current server build; older servers may omit it (see migration notes for Digest behavior).

---

## Related release

- [v2.5.0](RELEASE_v2.5.0.md) — LLM provider expansion, MPS exclusive mode, entity reconciliation, run manifests, LLM metrics.
