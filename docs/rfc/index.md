# Requests for Comment (RFCs)

## Purpose

Requests for Comment (RFCs) define the **how** behind each feature implementation in `podcast_scraper`. They capture:

- **Technical design** and architecture decisions
- **Implementation details** and module boundaries
- **API contracts** and data structures
- **Testing strategies** and validation approaches

RFCs translate PRD requirements into concrete technical solutions and serve as living documentation for developers.

## How RFCs Work

1. **Reference PRDs**: RFCs implement requirements defined in PRDs
2. **Define Architecture**: RFCs specify module design, interfaces, and data flow
3. **Guide Implementation**: Developers use RFCs as blueprints for code changes
4. **Document Decisions**: RFCs capture design rationale and alternatives considered

## Open RFCs

| RFC | Title | Related PRD | Description |
| --- | ----- | ----------- | ----------- |
| [RFC-015](RFC-015-ai-experiment-pipeline.md) | AI Experiment Pipeline | PRD-007 | Technical design for configuration-driven experiment pipeline (CI integration pending) |
| [RFC-027](RFC-027-pipeline-metrics-improvements.md) | Pipeline Metrics Improvements | - | Improvements to pipeline metrics collection and reporting |
| [RFC-038](RFC-038-continuous-review-tooling.md) | Continuous Review Tooling | #45 | Dependabot, pydeps, pre-release checklist |
| [RFC-041](RFC-041-podcast-ml-benchmarking-framework.md) | Podcast ML Benchmarking Framework | PRD-007 | Repeatable, objective ML benchmarking system (CI integration pending) |
| [RFC-043](RFC-043-automated-metrics-alerts.md) | Automated Metrics Alerts | - | Automated regression alerts and PR comments for pipeline metrics |
| [RFC-051](RFC-051-database-projection-gil-kg.md) | Database Projection (GIL & Knowledge Graph) | PRD-018 | Relational export for GIL (`gi.json`) and KG (RFC-055) artifacts |
| [RFC-053](RFC-053-adaptive-summarization-routing.md) | Adaptive Summarization Routing Based on Episode Profiling | PRD-005 | Episode profiling; routes summarization, GIL (RFC-049), and KG (RFC-055) strategies |
| [RFC-054](RFC-054-e2e-mock-response-strategy.md) | Flexible E2E Mock Response Strategy | #135, #399, #401 | Flexible strategy for E2E mock responses supporting normal and advanced error handling scenarios |
| [RFC-058](RFC-058-audio-speaker-diarization.md) | Audio-Based Speaker Diarization | PRD-020 | pyannote.audio design accepted ([ADR-058](../adr/ADR-058-additive-pyannote-diarization-with-separate-extra.md)); **implementation not landed** in `main` (no `[diarize]` extra yet) |
| [RFC-059](RFC-059-speaker-detection-refactor-test-audio.md) | Speaker Detection Refactor & Test Audio Improvements | PRD-020 | Modularize speaker detection, unique test voices, commercial segments |
| [RFC-060](RFC-060-diarization-aware-commercial-cleaning.md) | Multi-Signal Commercial Detection & Cleaning | PRD-020 | Expanded patterns + positional heuristics (Phase 1, all providers); diarization-enhanced (Phase 2, future) |
| [RFC-070](RFC-070-semantic-corpus-search-platform-future.md) | Semantic Corpus Search — Platform & Future Backends | PRD-021 | Draft — Qdrant **`VectorStore`**, native filtering, pgvector/RFC-051, re-ranking, digest fusion; split from completed [RFC-061](RFC-061-semantic-corpus-search.md) |
| [RFC-072](RFC-072-canonical-identity-layer-cross-layer-bridge.md) | Canonical Identity Layer & Cross-Layer Bridge | PRD-017, PRD-019, PRD-021 | Shared `person:`/`org:`/`topic:` IDs, `bridge.json` per episode, Position Tracker and Guest Brief flagship use cases; supersedes cross-layer aspects of RFC-050/056 |

## Completed RFCs

| RFC | Title | Related PRD | Version | Description |
| --- | ----- | ----------- | ------- | ----------- |
| [RFC-001](RFC-001-workflow-orchestration.md) | Workflow Orchestration | PRD-001 | v2.0.0 | Central orchestrator for transcript acquisition pipeline |
| [RFC-002](RFC-002-rss-parsing.md) | RSS Parsing & Episode Modeling | PRD-001 | v2.0.0 | RSS feed parsing and episode data model |
| [RFC-003](RFC-003-transcript-downloads.md) | Transcript Download Processing | PRD-001 | v2.0.0 | Resilient transcript download with retry logic |
| [RFC-004](RFC-004-filesystem-layout.md) | Filesystem Layout & Run Management | PRD-001 | v2.0.0 | Deterministic output directory structure and run scoping |
| [RFC-005](RFC-005-whisper-integration.md) | Whisper Integration Lifecycle | PRD-002 | v2.0.0 | Whisper model loading, transcription, and cleanup |
| [RFC-006](RFC-006-screenplay-formatting.md) | Whisper Screenplay Formatting | PRD-002 | v2.0.0 | Speaker-attributed transcript formatting |
| [RFC-007](RFC-007-cli-interface.md) | CLI Interface & Validation | PRD-003 | v2.0.0 | Command-line argument parsing and validation |
| [RFC-008](RFC-008-config-model.md) | Configuration Model & Validation | PRD-003 | v2.0.0 | Pydantic-based configuration with file loading |
| [RFC-009](RFC-009-progress-integration.md) | Progress Reporting Integration | PRD-001 | v2.0.0 | Pluggable progress reporting interface |
| [RFC-010](RFC-010-speaker-name-detection.md) | Automatic Speaker Name Detection | PRD-008 | v2.1.0 | NER-based host and guest identification |
| [RFC-011](RFC-011-metadata-generation.md) | Per-Episode Metadata Generation | PRD-004 | v2.2.0 | Structured metadata document generation |
| [RFC-012](RFC-012-episode-summarization.md) | Episode Summarization Using Local Transformers | PRD-005 | v2.3.0 | Local transformer-based summarization |
| [RFC-013](RFC-013-openai-provider-implementation.md) | OpenAI Provider Implementation | PRD-006 | v2.4.0 | OpenAI API providers for transcription, NER, and summarization |
| [RFC-016](RFC-016-modularization-for-ai-experiments.md) | Modularization for AI Experiments | PRD-007 | v2.4.0 | Provider system architecture to support AI experiment pipeline |
| [RFC-017](RFC-017-prompt-management.md) | Prompt Management | PRD-006 | v2.4.0 | Versioned, parameterized prompt management system (Jinja2) |
| [RFC-018](RFC-018-test-structure-reorganization.md) | Test Structure Reorganization | - | v2.4.0 | Reorganized test suite into unit/integration/e2e directories |
| [RFC-019](RFC-019-e2e-test-improvements.md) | E2E Test Infrastructure and Coverage Improvements | PRD-001+ | v2.4.0 | Comprehensive E2E test infrastructure and coverage |
| [RFC-020](RFC-020-integration-test-improvements.md) | Integration Test Infrastructure and Coverage Improvements | PRD-001+ | v2.4.0 | Integration test suite improvements (10 stages, 182 tests) |
| [RFC-021](RFC-021-modularization-refactoring-plan.md) | Modularization Refactoring Plan | PRD-006 | v2.4.0 | Detailed plan for modular provider architecture |
| [RFC-022](RFC-022-environment-variable-candidates-analysis.md) | Environment Variable Candidates Analysis | - | v2.4.0 | Environment variable support for deployment flexibility |
| [RFC-024](RFC-024-test-execution-optimization.md) | Test Execution Optimization | - | v2.4.0 | Optimized test execution with markers, tiers, parallel execution |
| [RFC-025](RFC-025-test-metrics-and-health-tracking.md) | Test Metrics and Health Tracking | - | v2.4.0 | Metrics collection, CI integration, flaky test detection |
| [RFC-026](RFC-026-metrics-consumption-and-dashboards.md) | Metrics Consumption and Dashboards | - | v2.4.0 | GitHub Pages metrics JSON API and job summaries |
| [RFC-028](RFC-028-ml-model-preloading-and-caching.md) | ML Model Preloading and Caching | - | v2.4.0 | Model preloading for local dev and GitHub Actions caching |
| [RFC-029](RFC-029-provider-refactoring-consolidation.md) | Provider Refactoring Consolidation | PRD-006 | v2.4.0 | Unified provider architecture documentation |
| [RFC-030](RFC-030-python-test-coverage-improvements.md) | Python Test Coverage Improvements | - | v2.4.0 | Coverage collection in CI, threshold enforcement |
| [RFC-031](RFC-031-code-complexity-analysis-tooling.md) | Code Complexity Analysis Tooling | - | v2.4.0 | Radon, Vulture, Interrogate, and codespell integration |
| [RFC-032](RFC-032-anthropic-provider-implementation.md) | Anthropic Provider Implementation | PRD-009 | v2.4.0 | Technical design for Anthropic Claude API providers |
| [RFC-033](RFC-033-mistral-provider-implementation.md) | Mistral Provider Implementation | PRD-010 | v2.5.0 | Technical design for Mistral AI providers (all 3 capabilities) |
| [RFC-034](RFC-034-deepseek-provider-implementation.md) | DeepSeek Provider Implementation | PRD-011 | v2.5.0 | Technical design for DeepSeek AI (ultra low-cost) |
| [RFC-035](RFC-035-gemini-provider-implementation.md) | Gemini Provider Implementation | PRD-012 | v2.5.0 | Technical design for Google Gemini (2M context) |
| [RFC-036](RFC-036-grok-provider-implementation.md) | Grok Provider Implementation (xAI) | PRD-013 | v2.5.0 | Technical design for Grok (xAI's AI model) |
| [RFC-037](RFC-037-ollama-provider-implementation.md) | Ollama Provider Implementation | PRD-014 | v2.5.0 | Technical design for Ollama (local/offline) |
| [RFC-039](RFC-039-development-workflow-worktrees-ci.md) | Development Workflow | - | v2.4.0 | Git worktrees, Cursor integration, CI evolution |
| [RFC-023](RFC-023-readme-acceptance-tests.md) | README Acceptance Tests | - | v2.5.0 | Script-based acceptance tests (`make test-acceptance`) with YAML configs |
| [RFC-040](RFC-040-audio-preprocessing-pipeline.md) | Audio Preprocessing Pipeline | - | v2.5.0 | FFmpeg preprocessing, opus codec, audio caching, factory pattern |
| [RFC-042](RFC-042-hybrid-summarization-pipeline.md) | Hybrid Podcast Summarization Pipeline | - | v2.5.0 | Hybrid MAP-REDUCE with instruction-tuned LLMs |
| [RFC-044](RFC-044-model-registry.md) | Model Registry for Architecture Limits | - | v2.5.0 | Centralized registry for model architecture limits |
| [RFC-045](RFC-045-ml-model-optimization-guide.md) | ML Model Optimization Guide | PRD-005, PRD-007 | v2.5.0 | cleaning_v4 profile, preprocessing optimization, parameter tuning guide |
| [RFC-046](RFC-046-materialization-architecture.md) | Materialization Architecture | PRD-007 | v2.5.0 | Dataset materialization for honest evaluation comparisons |
| [RFC-047](RFC-047-run-comparison-visual-tool.md) | Lightweight Run Comparison & Diagnostics Tool | PRD-007 | v2.5.0 | Streamlit-based visual tool for comparing runs |
| [RFC-048](RFC-048-evaluation-application-alignment.md) | Evaluation ↔ Application Alignment | PRD-007 | v2.5.0 | Fingerprinting and single-path eval-app alignment |
| [RFC-049](RFC-049-grounded-insight-layer-core.md) | Grounded Insight Layer – Core Concepts & Data Model | PRD-017 | v2.6.0 | Core ontology, grounding contract, storage format for GIL |
| [RFC-050](RFC-050-grounded-insight-layer-use-cases.md) | Grounded Insight Layer – Use Cases & End-to-End Consumption | PRD-017 | v2.6.0 | Single-layer GIL consumption (CLI inspect, Insight Explorer, query patterns); cross-layer use cases moved to [RFC-072](RFC-072-canonical-identity-layer-cross-layer-bridge.md) |
| [RFC-052](RFC-052-locally-hosted-llm-models-with-prompts.md) | Locally Hosted LLM Models with Prompts | PRD-014 | v2.5.0 | Ollama provider and optimized prompt templates |
| [RFC-055](RFC-055-knowledge-graph-layer-core.md) | Knowledge Graph Layer — Core Concepts & Data Model | PRD-019 | v2.6.0 | KG ontology, artifacts, and separation from GIL |
| [RFC-056](RFC-056-knowledge-graph-layer-use-cases.md) | Knowledge Graph Layer — Use Cases & End-to-End Consumption | PRD-019 | v2.6.0 | Single-layer KG consumption (`kg` CLI, entity roll-up, export); cross-layer use cases moved to [RFC-072](RFC-072-canonical-identity-layer-cross-layer-bridge.md) |
| [RFC-057](RFC-057-autoresearch-optimization-loop.md) | AutoResearch Optimization Loop (Prompts & ML Params) | PRD-007 | v2.6.0 | Closed per [ADR-073](../adr/ADR-073-rfc057-autoresearch-closure.md); Tracks A/B complete; silver refs + 72-config eval matrix |
| [RFC-061](RFC-061-semantic-corpus-search.md) | Semantic Corpus Search (FAISS) | PRD-021 | v2.6.0 | Shipped: `FaissVectorStore`, `podcast search` / `index`, embed-and-index, semantic `gi explore`, `/api/search` ([ADR-060](../adr/ADR-060-vectorstore-protocol-with-backend-abstraction.md)); platform backends — [RFC-070](RFC-070-semantic-corpus-search-platform-future.md) (Draft) |
| [RFC-062](RFC-062-gi-kg-viewer-v2.md) | GI/KG Viewer v2 — Semantic Search UI | PRD-017, PRD-019, PRD-021 | v2.6.0 | FastAPI `podcast serve`, Vue 3 + Vite + Cytoscape SPA, Playwright UI E2E ([ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md)–[ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md)); platform routes remain v2.7 per ADR-064 |
| [RFC-063](RFC-063-multi-feed-corpus-append-resume.md) | Multi-Feed Corpus, Append/Resume, and Unified Discovery | #440+ | v2.6.0 | N feeds, layout A, opt-in append; unified index (#505); `corpus_manifest.json` / run summary (#506); extends RFC-004; see [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md) |
| [RFC-064](RFC-064-performance-profiling-release-freeze.md) | Performance Profiling and Release Freeze Framework | - | v2.6.0 | Frozen profiles under `data/profiles/`, `scripts/eval/freeze_profile.py`, `diff_profiles.py`, `make profile-freeze` / `profile-diff`; [guide](../guides/PERFORMANCE_PROFILE_GUIDE.md) |
| [RFC-065](RFC-065-live-pipeline-monitor.md) | Live Pipeline Monitor (macOS Developer Tooling) | #512 | v2.6.0 | `--monitor`, `.pipeline_status.json`, `rich` or `.monitor.log`; optional **`[monitor]`** memray + py-spy; tmux split deferred; [guide](../guides/LIVE_PIPELINE_MONITOR.md) |
| [RFC-066](RFC-066-run-compare-performance-tab.md) | Run Comparison Tool — Performance Tab | - | v2.6.0 | Streamlit **Performance** page (`?page=performance`) joining run metrics with frozen RFC-064 profiles |
| [RFC-067](RFC-067-corpus-library-api-viewer.md) | Corpus Library — Catalog API & Viewer | PRD-022 | v2.6.0 | Filesystem-first `/api/corpus/*`, Library tab, episode detail, FAISS similar episodes, handoffs to graph and `/api/search` (Phases 1–3) |
| [RFC-068](RFC-068-corpus-digest-api-viewer.md) | Corpus Digest — API & Viewer | PRD-023 | v2.6.0 | `GET /api/corpus/digest`, Digest tab, Library 24h glance, feed diversity, semantic topic bands; `corpus_digest_api` on `/api/health` |
| [RFC-069](RFC-069-graph-exploration-toolkit.md) | GI/KG Viewer — Graph Exploration Toolkit | PRD-024 | v2.6.0 | Zoom controls, % readout, Shift+drag box zoom, minimap v1, degree-bucket filter, built-in layouts, edge filters; extends RFC-062 |
| [RFC-071](RFC-071-corpus-intelligence-dashboard-viewer.md) | Corpus Intelligence Dashboard (GI/KG Viewer) | PRD-025 | v2.6.0 | **Dashboard** tab: **`/api/corpus/*`** aggregates + Chart.js (**Pipeline** / **Content intelligence**); manifest + capped **`run.json`** discovery; index/digest/GI-KG timelines; [PRD-025](../prd/PRD-025-corpus-intelligence-dashboard-viewer.md) |

## Gap analysis {:#gaps}

**Counts (reconcile when moving RFCs):** **71** files under `docs/rfc/RFC-*.md` — IDs **RFC-001–RFC-072**
with **no RFC-014**. **13** open and **58** completed in the tables above.

**Open RFC clusters:** AI experiment pipeline + ML benchmark CI (**RFC-015**, **RFC-041**), pipeline
metrics (**RFC-027**), continuous review (**RFC-038**), metrics alerts (**RFC-043**), Postgres
projection (**RFC-051**), adaptive summarization routing (**RFC-053**), E2E mock composition
(**RFC-054**), diarization and cleaning (**RFC-058**–**RFC-060**; **RFC-058** is design-accepted,
**not** fully landed in `main`), semantic search **platform** draft (**RFC-070**; **RFC-061** FAISS
path is **Completed**), canonical identity layer and cross-layer bridge (**RFC-072**).

**Closed program:** [RFC-057](RFC-057-autoresearch-optimization-loop.md) — **Completed**; closure summary
in [ADR-073](../adr/ADR-073-rfc057-autoresearch-closure.md).

**Maintenance:** Edit each RFC **`Status`** line when you move its row between **Open** and **Completed**.
Product gaps: [PRD gap analysis](../prd/index.md#gaps). Decision records: [ADR gap analysis](../adr/index.md#gaps).

## Quick Links

- **[PRDs](../prd/index.md)** - Product requirements documents
- **[Architecture](../architecture/ARCHITECTURE.md)** - System design and module responsibilities
- **[Releases](../releases/index.md)** - Release notes and version history

---

## Creating New RFCs

Use the **[RFC Template](RFC_TEMPLATE.md)** as a starting point for new technical design documents.

**Status vocabulary:** Use **Draft** while in flight and **Completed** when shipped (optionally with
version or caveats in the same line). Do not use **Accepted** for RFCs — that label is for **ADRs**
only.
