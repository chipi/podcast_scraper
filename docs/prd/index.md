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
| [PRD-016](PRD-016-operational-observability-pipeline-intelligence.md) | Operational Observability & Pipeline Intelligence | RFC-025, 026, 027, 064, 065, 066 | **Partial (v2.6.0–v2.7):** test metrics + GitHub Pages dashboards ([RFC-025](../rfc/RFC-025-test-metrics-and-health-tracking.md)/[026](../rfc/RFC-026-metrics-consumption-and-dashboards.md)) + **live monitor** ([RFC-065](../rfc/RFC-065-live-pipeline-monitor.md)) + **frozen perf profiles** + **compare Performance tab** ([RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md)/[066](../rfc/RFC-066-run-compare-performance-tab.md)) + pipeline metrics infra ([RFC-027](../rfc/RFC-027-pipeline-metrics-improvements.md)) all shipped; proactive alerting now operator-side via Sentry+Grafana+Langfuse per [RFC-043](../rfc/RFC-043-automated-metrics-alerts.md). Promotion gated on remaining product gaps surfaced in [PRD-016](PRD-016-operational-observability-pipeline-intelligence.md). |
| [PRD-030](PRD-030-viewer-feed-sources-and-pipeline-jobs.md) | Viewer operator surface — feeds, config, jobs | [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) | **Draft:** structured **`feeds.spec.yaml`** + **operator YAML** (no secrets in file; `serve --config-file` else corpus default); Phase 2 jobs + **stale/cancel/reconcile** ([#626](https://github.com/chipi/podcast_scraper/issues/626)) |
| [PRD-031](PRD-031-search.md) | Search product | [RFC-090](../rfc/RFC-090-hybrid-retrieval.md) | **Draft:** product surface for corpus search over the hybrid backend. |
| [PRD-032](PRD-032-hybrid-corpus-search.md) | Hybrid Corpus Search (backend) | [RFC-090](../rfc/RFC-090-hybrid-retrieval.md), [091](../rfc/RFC-091-kg-proximity-signal.md), [092](../rfc/RFC-092-ml-query-router.md) | **Partial (shipped):** two-tier + BM25+dense+RRF + compounds, hybrid default-on (RFC-090). KG-proximity rejected (RFC-091); ML router gated (RFC-092). |
| [PRD-033](PRD-033-search-powered-surfaces.md) | Search-Powered Surface Enhancements | [RFC-094](../rfc/RFC-094-search-powered-surfaces-query-layer.md), [090](../rfc/RFC-090-hybrid-retrieval.md) | **Draft:** how each viewer surface consumes the shipped foundation (re-grounded 2026-06-04). Shared query layer = RFC-094; per-surface issues [#882](https://github.com/chipi/podcast_scraper/issues/882)–888. |
| [PRD-034](PRD-034-generic-mcp-server.md) | Generic MCP Server — capabilities as agent tools | [RFC-095](../rfc/RFC-095-generic-mcp-server.md) | **Draft:** expose the platform's read capabilities (search, RFC-094 relational, CIL, catalog) as composable MCP tools for agentic clients. Generic substrate; RFC-093 briefing pack plugs in. Decoupled from #861. |
| [PRD-035](PRD-035-learning-platform.md) | Learning Platform (parent) — player + capture + personal knowledge corpus | [RFC-098](../rfc/RFC-098-learning-platform-foundation.md)–101 | **Draft (v2.7):** end-user learning platform on the pipeline. Principles, `segments.json` contract, phasing P0–P3. Parent of PRD-036–041. |
| [PRD-036](PRD-036-foundation-identity.md) | Foundation / Identity (minimal multi-user) | [RFC-098](../rfc/RFC-098-learning-platform-foundation.md) | **Draft (v2.7, P0):** OAuth identity, per-user files (no DB), consumer API surface, slug contract, scrape-on-demand, reference player. |
| [PRD-037](PRD-037-discovery.md) | Discovery — search, library, scrape-on-demand | [RFC-098](../rfc/RFC-098-learning-platform-foundation.md) | **Draft (v2.7, P1):** Podcast Index search via pluggable `DiscoverySource`; per-user library; deduped scrape requests. |
| [PRD-038](PRD-038-catalog.md) | Catalog — ready/pending episode browser | [RFC-098](../rfc/RFC-098-learning-platform-foundation.md) | **Draft (v2.7, P1):** global + per-podcast episode views with status and enriched previews; graceful degradation. |
| [PRD-039](PRD-039-player.md) | Player — queue, transcript-sync, knowledge panel | [RFC-099](../rfc/RFC-099-learning-platform-consumer-client.md) / [RFC-100](../rfc/RFC-100-audio-bridge-subsystem.md) | **Draft (v2.7, P1):** Spotify-grade playback bridged from origin host; transcript sync; queue; inline insights + grounded in-episode search. |
| [PRD-040](PRD-040-capture.md) | Capture — highlights + notes | [RFC-099](../rfc/RFC-099-learning-platform-consumer-client.md) | **Draft (v2.7, P2):** Kindle-style highlights and notes, grounded to timestamps/offsets, per user. |
| [PRD-041](PRD-041-consolidation.md) | Consolidation — personal knowledge corpus | [RFC-101](../rfc/RFC-101-personal-knowledge-corpus.md) | **Draft (v2.7, P3):** per-user projection over GIL/KG; grounded recall scoped to heard episodes; spaced resurfacing; interest profile. |
| [PRD-042](PRD-042-home.md) | Home (Learning Hub) | [RFC-099](../rfc/RFC-099-learning-platform-consumer-client.md) | **Draft (v2.7, P1):** the app's launch surface — orient (what's new) / resume (continue) / discover-within (recommended + corpus-wide grounded search) / route; catalog moves to /catalog. GH #1090. |
| [PRD-043](PRD-043-knowledge-layer.md) | Knowledge Layer — Topic Clusters, Entity Cards & Personalized Discovery | [RFC-102](../rfc/RFC-102-knowledge-clusters-entity-cards.md) | **Draft (v2.8, Epic 3):** make knowledge first-class — topic clusters (cluster-first UI), person/topic entity cards, entities in search, and digest × interest-cluster personalized discovery. Merges the proposed Epic 3+4. |

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
| [PRD-017](PRD-017-grounded-insight-layer.md) | Grounded Insight Layer (GIL) | v2.6.0–v2.7 | RFC-042, 044, 049, 050, 052, 062, 072, 097 | Single-layer GIL + cross-layer bridge shipped: `gi.json` schema v3.0, Insight Explorer, ABOUT/MENTIONS_PERSON/MENTIONS_ORG attribution per Insight, CIL resolver+registry, `insight_type` marking. Postgres projection ([RFC-051](../rfc/RFC-051-database-projection-gil-kg.md), Draft) remains as orthogonal persistence-layer scope. |
| [PRD-019](PRD-019-knowledge-graph-layer.md) | Knowledge Graph Layer (KG) | v2.6.0–v2.7 | RFC-042, 044, 052, 055, 056, 062, 072, 097 | Single-layer KG + cross-layer bridge shipped: `kg.json` schema v2.0, Person/Org/Podcast first-class node types, entity identity via CIL (`entity:person:` → `person:`), `kg` CLI + entity roll-up + export. Postgres projection ([RFC-051](../rfc/RFC-051-database-projection-gil-kg.md), Draft) and asymptotic entity resolution remain open as orthogonal scope. |
| [PRD-021](PRD-021-semantic-corpus-search.md) | Semantic Corpus Search | v2.6.0–v2.7 | RFC-061, 062, 075, 090 | Shipped end-to-end: FAISS Phase 1 ([RFC-061](../rfc/RFC-061-semantic-corpus-search.md)) → LanceDB-first hybrid (BM25 + dense + RRF, [RFC-090](../rfc/RFC-090-hybrid-retrieval.md), [ADR-099](../adr/ADR-099-lancedb-first-single-index-search.md), PR #1010); FAISS retired. Corpus topic clustering layer ([RFC-075](../rfc/RFC-075-corpus-topic-clustering.md)) shipped with `topic_clusters.json` + viewer compound parents. Platform backends ([RFC-070](../rfc/RFC-070-semantic-corpus-search-platform-future.md)) **Superseded** by the LanceDB path. |
| [PRD-022](PRD-022-corpus-library-episode-browser.md) | Corpus Library & Episode Browser | v2.6.0 | RFC-067, 062, 061, 063 | Filesystem-first catalog in viewer: feeds/episodes, summaries, similar episodes, handoff to graph and semantic search ([RFC-067](../rfc/RFC-067-corpus-library-api-viewer.md) Phases 1–3) |
| [PRD-023](PRD-023-corpus-digest-recap.md) | Corpus Digest & Library Glance | v2.6.0 | RFC-068, 067, 061, 062 | Digest tab + 24h Library glance: diverse recent episodes, global topic bands, GI/KG badges, search/graph handoffs ([RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md)) |
| [PRD-024](PRD-024-graph-exploration-toolkit.md) | GI/KG Graph Exploration Toolkit | v2.6.0 | RFC-069, 062 | Graph tab toolkit: zoom 100%/%, Shift+drag box zoom, minimap v1, degree bucket filter, built-in layouts, edge filters ([RFC-069](../rfc/RFC-069-graph-exploration-toolkit.md)) |
| [PRD-025](PRD-025-corpus-intelligence-dashboard-viewer.md) | Corpus Intelligence Dashboard (GI/KG Viewer) | v2.6.0 | RFC-071, 062, 063, 061, 067, 068 | **Dashboard** tab: **Pipeline** vs **Content intelligence** Chart.js panels; corpus stats, manifest, **`run.json`** scan, index/digest glance ([RFC-071](../rfc/RFC-071-corpus-intelligence-dashboard-viewer.md)) |

## Gap analysis {:#gaps}

**Counts (reconcile when adding PRDs):** **30** PRD documents -- **2** open (Partial/Draft) above,
**20** implemented, **8** Draft (not indexed until promoted).
Use **`Implemented (vX.Y.Z)`**, **`Partial`**, or **`Draft`** in PRD headers — not **`Completed`**
(that label is for RFCs and ADRs).

| Gap type | What to do |
| --- | --- |
| **Partial PRD** | Finish the **open** RFCs named in that row before promoting the PRD to **Implemented**. |
| **Implemented PRD + open RFC** | Expected when the RFC is a **future** slice (e.g. PRD-017/019 still reference [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md) Postgres projection as future scope). |
| **Viewer / UI** | [UX specifications](../uxs/index.md) and the [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md). |
| **Architecture vs code** | [ADR index — Gap analysis](../adr/index.md#gaps) **Code** column. |

**Open-program themes:** Ops observability product gaps
([PRD-016](PRD-016-operational-observability-pipeline-intelligence.md) -- distinct from viewer
[PRD-025](PRD-025-corpus-intelligence-dashboard-viewer.md)), viewer operator surface
([PRD-030](PRD-030-viewer-feed-sources-and-pipeline-jobs.md)), and the Consumer Learning Platform
roll-out (PRD-035–042 / RFC-098–101). Several Draft PRDs not indexed: experiments (PRD-007),
governance (PRD-015), Postgres projection (PRD-018), diarization (PRD-020), topic view (PRD-026),
enriched search (PRD-027), position tracker (PRD-028), person profile (PRD-029).

**Related:** [RFC gap analysis](../rfc/index.md#gaps) (technical backlog), [ADR gap analysis](../adr/index.md#gaps)
(decisions and implementation state).

## Quick Links

- **[Architecture](../architecture/ARCHITECTURE.md)** - System design and module responsibilities
- **[RFCs](../rfc/index.md)** - Technical design documents
- **[Releases](../releases/index.md)** - Release notes and version history

---

## Creating New PRDs

Use the **[PRD Template](PRD_TEMPLATE.md)** as a starting point for new product requirements documents.
