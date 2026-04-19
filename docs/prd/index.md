# Product Requirements Documents (PRDs)

## Purpose

Product Requirements Documents (PRDs) define the **what** and **why** behind each major feature or capability in `podcast_scraper`. They capture:

- **User needs** and use cases
- **Functional requirements** and success criteria
- **Design considerations** and constraints
- **Integration points** with existing features

PRDs serve as the foundation for technical design (RFCs) and help ensure features align with user needs and project goals.

**Status vocabulary:** Shipped PRDs use **`Implemented (vX.Y.Z)`** (or **`Partial`** / **`Draft`** while in flight). Use **`Completed`** for **RFC** and **ADR** status lines, not for PRD headers — keeps product docs consistent with the PRD index.

Features with meaningful **UI** may also link **[UX specifications](../uxs/index.md)** (UXS) for tokens, layout, and accessibility; RFCs then reference that UX contract alongside the PRD.

## How PRDs Work

1. **Define Intent**: PRDs describe the problem to solve and desired outcomes
2. **Guide Design**: RFCs reference PRDs (and UXSs when UI is in scope) so technical solutions meet requirements and experience constraints
3. **Track Implementation**: Release notes reference PRDs to show what was delivered
4. **Document Evolution**: PRDs capture design decisions and rationale

## Open PRDs

| PRD | Title | Related RFCs | Description |
| --- | ----- | ------------ | ----------- |
| [PRD-016](PRD-016-operational-observability-pipeline-intelligence.md) | Operational Observability & Pipeline Intelligence | RFC-025, 026, 027, 064, 065, 066 | **Partial (v2.6.0):** test metrics + GitHub Pages dashboards (RFC-025/026) and **live monitor** (RFC-065), **frozen perf profiles** + **compare Performance tab** (RFC-064/066) shipped; RFC-027 pipeline-metrics gaps (e.g. CSV) remain |
| [PRD-017](PRD-017-grounded-insight-layer.md) | Grounded Insight Layer (GIL) | RFC-042, 044, 052, 049, 050, 051, 072 | **Partial (v2.6.0):** [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md)/[044](../rfc/RFC-044-model-registry.md)/[052](../rfc/RFC-052-locally-hosted-llm-models-with-prompts.md)/[049](../rfc/RFC-049-grounded-insight-layer-core.md)/[050](../rfc/RFC-050-grounded-insight-layer-use-cases.md) **Completed** (single-layer); cross-layer work in Draft RFC-072 and RFC-051 |
| [PRD-019](PRD-019-knowledge-graph-layer.md) | Knowledge Graph Layer (KG) | RFC-042, 044, 052, 055, 056, 051, 072 | **Partial (v2.6.0):** [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md)/[044](../rfc/RFC-044-model-registry.md)/[052](../rfc/RFC-052-locally-hosted-llm-models-with-prompts.md)/[055](../rfc/RFC-055-knowledge-graph-layer-core.md)/[056](../rfc/RFC-056-knowledge-graph-layer-use-cases.md) **Completed** (single-layer); cross-layer work in Draft RFC-072 and RFC-051 |
| [PRD-030](PRD-030-viewer-feed-sources-and-pipeline-jobs.md) | Viewer operator surface — feeds, config, jobs | [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) | **Draft:** feeds + **operator YAML** (no secrets in file; `--config-file` else corpus default); Phase 2 jobs + **stale/cancel/reconcile** ([#626](https://github.com/chipi/podcast_scraper/issues/626)) |

## Implemented PRDs

| PRD | Title | Version | Related RFCs | Description |
| --- | ----- | ------- | ------------ | ----------- |
| [PRD-001](PRD-001-transcript-pipeline.md) | Transcript Acquisition Pipeline | v2.0.0 | RFC-001, 002, 003, 004, 008, 009 | Core pipeline for downloading transcripts |
| [PRD-002](PRD-002-whisper-fallback.md) | Whisper Fallback Transcription | v2.0.0 | RFC-004, 005, 006, 008, 010 | Automatic transcription fallback |
| [PRD-003](PRD-003-user-interface-config.md) | User Interfaces & Configuration | v2.0.0 | RFC-007, 008, 009 | CLI interface and configuration |
| [PRD-004](PRD-004-metadata-generation.md) | Per-Episode Metadata Generation | v2.2.0 | RFC-011, 012 | Structured metadata documents |
| [PRD-005](PRD-005-episode-summarization.md) | Episode Summarization | v2.3.0 | RFC-012 | Automatic summary generation |
| [PRD-006](PRD-006-openai-provider-integration.md) | OpenAI Provider Integration | v2.4.0 | RFC-013, 017, 021, 022, 029 | OpenAI API as optional provider |
| [PRD-008](PRD-008-speaker-name-detection.md) | Automatic Speaker Name Detection | v2.1.0 | RFC-010 | Auto-detect host/guest names via NER |
| [PRD-009](PRD-009-anthropic-provider-integration.md) | Anthropic Provider Integration | v2.4.0 | RFC-032 | Anthropic Claude API as optional provider |
| [PRD-010](PRD-010-mistral-provider-integration.md) | Mistral Provider Integration | v2.5.0 | RFC-033 | Mistral AI as complete OpenAI alternative |
| [PRD-011](PRD-011-deepseek-provider-integration.md) | DeepSeek Provider Integration | v2.5.0 | RFC-034 | DeepSeek AI - ultra low-cost provider |
| [PRD-012](PRD-012-gemini-provider-integration.md) | Google Gemini Provider Integration | v2.5.0 | RFC-035 | Google Gemini - 2M context, native audio |
| [PRD-013](PRD-013-grok-provider-integration.md) | Grok Provider Integration (xAI) | v2.5.0 | RFC-036 | Grok - xAI's AI model with real-time information access |
| [PRD-014](PRD-014-ollama-provider-integration.md) | Ollama Provider Integration | v2.5.0 | RFC-037 | Ollama - fully local/offline, zero cost |
| [PRD-021](PRD-021-semantic-corpus-search.md) | Semantic Corpus Search | v2.6.0 | RFC-061, 062, 070 | Shipped: FAISS, `podcast search` / `podcast index`, semantic `gi explore`, viewer + `podcast serve` ([RFC-061](../rfc/RFC-061-semantic-corpus-search.md)); platform backends / Qdrant — Draft [RFC-070](../rfc/RFC-070-semantic-corpus-search-platform-future.md) |
| [PRD-022](PRD-022-corpus-library-episode-browser.md) | Corpus Library & Episode Browser | v2.6.0 | RFC-067, 062, 061, 063 | Filesystem-first catalog in viewer: feeds/episodes, summaries, similar episodes, handoff to graph and semantic search ([RFC-067](../rfc/RFC-067-corpus-library-api-viewer.md) Phases 1–3) |
| [PRD-023](PRD-023-corpus-digest-recap.md) | Corpus Digest & Library Glance | v2.6.0 | RFC-068, 067, 061, 062 | Digest tab + 24h Library glance: diverse recent episodes, global topic bands, GI/KG badges, search/graph handoffs ([RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md)) |
| [PRD-024](PRD-024-graph-exploration-toolkit.md) | GI/KG Graph Exploration Toolkit | v2.6.0 | RFC-069, 062 | Graph tab toolkit: zoom 100%/%, Shift+drag box zoom, minimap v1, degree bucket filter, built-in layouts, edge filters ([RFC-069](../rfc/RFC-069-graph-exploration-toolkit.md)) |
| [PRD-025](PRD-025-corpus-intelligence-dashboard-viewer.md) | Corpus Intelligence Dashboard (GI/KG Viewer) | v2.6.0 | RFC-071, 062, 063, 061, 067, 068 | **Dashboard** tab: **Pipeline** vs **Content intelligence** Chart.js panels; corpus stats, manifest, **`run.json`** scan, index/digest glance ([RFC-071](../rfc/RFC-071-corpus-intelligence-dashboard-viewer.md)) |

## Gap analysis {:#gaps}

**Counts (reconcile when adding PRDs):** **30** PRD documents -- **4** open (Partial/Draft) above,
**18** implemented, **8** Draft (not indexed until promoted).
Use **`Implemented (vX.Y.Z)`**, **`Partial`**, or **`Draft`** in PRD headers — not **`Completed`**
(that label is for RFCs and ADRs).

| Gap type | What to do |
| --- | --- |
| **Partial PRD** | Finish the **open** RFCs named in that row before promoting the PRD to **Implemented**. |
| **Implemented PRD + open RFC** | Expected when the RFC is a **future** slice (e.g. [PRD-021](PRD-021-semantic-corpus-search.md) and Draft [RFC-070](../rfc/RFC-070-semantic-corpus-search-platform-future.md)). |
| **Viewer / UI** | [UX specifications](../uxs/index.md) and the [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md). |
| **Architecture vs code** | [ADR index — Gap analysis](../adr/index.md#gaps) **Code** column. |

**Open-program themes:** Ops observability gaps
([PRD-016](PRD-016-operational-observability-pipeline-intelligence.md) -- distinct from viewer
[PRD-025](PRD-025-corpus-intelligence-dashboard-viewer.md)), GIL/KG cross-layer work (PRD-017/019
Partial; Draft RFC-072), and several Draft PRDs not indexed: experiments (PRD-007), governance
(PRD-015), Postgres projection (PRD-018), diarization (PRD-020), topic view (PRD-026), enriched
search (PRD-027), position tracker (PRD-028), person profile (PRD-029).

**Related:** [RFC gap analysis](../rfc/index.md#gaps) (technical backlog), [ADR gap analysis](../adr/index.md#gaps)
(decisions and implementation state).

## Quick Links

- **[Architecture](../architecture/ARCHITECTURE.md)** - System design and module responsibilities
- **[RFCs](../rfc/index.md)** - Technical design documents
- **[Releases](../releases/index.md)** - Release notes and version history

---

## Creating New PRDs

Use the **[PRD Template](PRD_TEMPLATE.md)** as a starting point for new product requirements documents.
