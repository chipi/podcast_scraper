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
| [RFC-087](RFC-087-vps-public-edge-multi-compose.md) | VPS public edge and multi–Docker Compose hosting | — | **Draft:** optional **public TLS edge** (Caddy / Traefik / Cloudflare Tunnel) terminating HTTPS for multi-host VPS routing while operators stay on **Tailscale**; isolation + GitOps pattern for additional repos on the same VPS. Cross-links: [PROD_RUNBOOK](../guides/PROD_RUNBOOK.md), [VPS_MULTI_APP_ONBOARDING](../guides/VPS_MULTI_APP_ONBOARDING.md). |
| [RFC-091](RFC-091-kg-proximity-signal.md) | KG Proximity Signal | [PRD-032](../prd/PRD-032-hybrid-corpus-search.md) | **Rejected as a retrieval signal** (Decision Record): refuted on every corpus/axis. KG value is meaning-bearing relational edges (Person→Insight, #874), not proximity ranking. |
| [RFC-092](RFC-092-ml-query-router.md) | ML Query Router | [PRD-032](../prd/PRD-032-hybrid-corpus-search.md) | **Draft:** rules router shipped; ML classifier gated on eval data ([#860](https://github.com/chipi/podcast_scraper/issues/860)). |
| [RFC-096](RFC-096-audio-pipeline-separation-and-viewer-media.md) | Audio pipeline separation and viewer media | — | **Draft:** `pipeline_stage` split ([#414](https://github.com/chipi/podcast_scraper/issues/414)); persist corpus **`media/`** + local **`GET /api/corpus/media`** + viewer transcript audio player ([#547](https://github.com/chipi/podcast_scraper/issues/547)). |
| [RFC-098](RFC-098-learning-platform-foundation.md) | Learning Platform Foundation | [PRD-036](../prd/PRD-036-foundation-identity.md) | **Draft (v2.7, P0):** OAuth identity, **plain per-user files** (no DB/persistence-layer work), dedicated `/api/app/*` consumer API, stable episode slug contract, episode-scoped grounded search (no request-time LLM), scrape guardrails. Keystone for RFC-099–101. |
| [RFC-099](RFC-099-learning-platform-consumer-client.md) | Learning Platform Consumer Client | [PRD-039](../prd/PRD-039-player.md) / [PRD-038](../prd/PRD-038-catalog.md) / [PRD-040](../prd/PRD-040-capture.md) | **Draft (v2.7, P1–P2):** new top-level PWA (Vue 3 + TS); transcript-sync engine, queue, Knowledge Panel, capture; a11y (WCAG 2.1 AA) + i18n from line one; consumes RFC-098. |
| [RFC-100](RFC-100-audio-bridge-subsystem.md) | Audio Bridge Subsystem | [PRD-039](../prd/PRD-039-player.md) | **Draft (v2.7):** resolve a fresh, playable **origin enclosure URL** per episode (bridge, never rehost); freshness/redirect/expiry handling; optional no-store pass-through only when a host forces it. New subsystem. |
| [RFC-101](RFC-101-personal-knowledge-corpus.md) | Personal Knowledge Corpus | [PRD-041](../prd/PRD-041-consolidation.md) | **Draft (v2.7, P3):** per-user projection over GIL/KG scoped to heard/captured episodes; grounded recall via retrieval (no LLM); cross-episode connections; spaced resurfacing; interest profile. The moat. |
| [RFC-102](RFC-102-knowledge-clusters-entity-cards.md) | Knowledge Clusters, Entity Cards & Personalized Discovery | [PRD-043](../prd/PRD-043-knowledge-layer.md) | **Draft (Epic 3):** §1 topic clusters + cluster-first Insights panel **(Implemented, #1092)**; §2/§3 person/topic entity cards; §4 entities in search; §5 digest × interest-cluster personalization. |

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
| [RFC-023](RFC-023-readme-acceptance-tests.md) | README Acceptance Tests | - | v2.5.0 | Script-based acceptance (`make test-acceptance`, `MAIN_ACCEPTANCE_CONFIG.yaml` fast matrix, `scripts/acceptance/`) — not pytest `tests/acceptance/` |
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
| [RFC-064](RFC-064-performance-profiling-release-freeze.md) | Performance Profiling and Release Freeze Framework | - | v2.6.0 | Frozen profiles under `data/profiles/`, `scripts/eval/profile/freeze_profile.py`, `diff_profiles.py`, `make profile-freeze` / `profile-diff`; [guide](../guides/PERFORMANCE_PROFILE_GUIDE.md) |
| [RFC-065](RFC-065-live-pipeline-monitor.md) | Live Pipeline Monitor (macOS Developer Tooling) | #512 | v2.6.0 | `--monitor`, `.pipeline_status.json`, `rich` or `.monitor.log`; optional **`[monitor]`** memray + py-spy; tmux split deferred; [guide](../guides/LIVE_PIPELINE_MONITOR.md) |
| [RFC-066](RFC-066-run-compare-performance-tab.md) | Run Comparison Tool — Performance Tab | - | v2.6.0 | Streamlit **Performance** page (`?page=performance`) joining run metrics with frozen RFC-064 profiles |
| [RFC-067](RFC-067-corpus-library-api-viewer.md) | Corpus Library — Catalog API & Viewer | PRD-022 | v2.6.0 | Filesystem-first `/api/corpus/*`, Library tab, episode detail, FAISS similar episodes, handoffs to graph and `/api/search` (Phases 1–3) |
| [RFC-068](RFC-068-corpus-digest-api-viewer.md) | Corpus Digest — API & Viewer | PRD-023 | v2.6.0 | `GET /api/corpus/digest`, Digest tab, Library 24h glance, feed diversity, semantic topic bands; `corpus_digest_api` on `/api/health` |
| [RFC-069](RFC-069-graph-exploration-toolkit.md) | GI/KG Viewer — Graph Exploration Toolkit | PRD-024 | v2.6.0 | Zoom controls, % readout, Shift+drag box zoom, minimap v1, degree-bucket filter, built-in layouts, edge filters; extends RFC-062 |
| [RFC-071](RFC-071-corpus-intelligence-dashboard-viewer.md) | Corpus Intelligence Dashboard (GI/KG Viewer) | PRD-025 | v2.6.0 | **Dashboard** tab: **`/api/corpus/*`** aggregates + Chart.js (**Pipeline** / **Content intelligence**); manifest + capped **`run.json`** discovery; index/digest/GI-KG timelines; [PRD-025](../prd/PRD-025-corpus-intelligence-dashboard-viewer.md) |
| [RFC-076](RFC-076-progressive-graph-expansion.md) | Progressive graph expansion (cross-episode) | #581 | v2.6.0 | `POST /api/corpus/node-episodes`, `onetap` rail / `dbltap` expand-collapse, bridge-only scan; extends RFC-069 |
| [RFC-084](RFC-084-corpus-backup-manifest-and-version-aware-restore.md) | Corpus snapshot backup manifest and version-aware restore | — | v2.6.0 | `snapshot.manifest.json`, `scripts/ops/corpus_snapshot/`, backup/restore workflows + `make restore-corpus` / `restore-corpus-prod`; GitHub #763 |
| [RFC-085](RFC-085-graph-handoff-orchestrator-retrospective.md) | Graph handoff orchestrator stabilization (retrospective) | PRD-024 | v2.6.0 | 8-state FSM with envelope + generation tokens across 13 entry points; viewer-side stuck detection + error strip; matrix-driven E2E coverage. Decisions in [ADR-094](../adr/ADR-094-graph-handoff-orchestrator-fsm.md). |
| [RFC-086](RFC-086-viewer-test-pyramid-and-production-shaped-fixtures.md) | Viewer test pyramid — three tiers (mocks, production-shaped, real corpus) | PRD-024 | v2.6.0 | Tier-1 mocked, Tier-2 production-shaped fixtures, Tier-3 real-corpus matrix; `make ci-ui-validation`; real-bug-first matrix-row rule. Decisions in [ADR-095](../adr/ADR-095-viewer-test-pyramid.md). |
| [RFC-090](RFC-090-hybrid-retrieval.md) | Hybrid Retrieval Pipeline | PRD-032 | v2.6.0 | Two-tier (segment + insight) + BM25 + dense + RRF over LanceDB; hybrid default-on; FAISS retired via [ADR-099](../adr/ADR-099-lancedb-first-single-index-search.md) / PR #1010. |
| [RFC-094](RFC-094-search-powered-surfaces-query-layer.md) | Search-Powered Surfaces Query Layer | PRD-033 | v2.6.0–v2.7 | Relational-query layer (positions_of / who_said / cross_show_synthesis / related_topics) + front-end retrieval state stack: OQ-1 panel cache (#1075, PR #1089), OQ-2 `activeSearchContext` Pinia store, OQ-3 skeleton-first async cards (PersonLanding + TopicEntity views). MCP exposure landed via PRD-034 / RFC-095. |
| [RFC-015](RFC-015-ai-experiment-pipeline.md) | AI Experiment Pipeline | PRD-007 | v2.4–v2.7 | Phases 1–3 (runner, metrics, storage/comparison) shipped earlier; Phase 4 CI integration shipped via standalone `.github/workflows/ai-experiment-pipeline.yml` workflow (operator-triggered, no auto-gating). |
| [RFC-041](RFC-041-podcast-ml-benchmarking-framework.md) | Podcast ML Benchmarking Framework | PRD-007 | v2.5–v2.7 | Phases 0–1 (dataset materialization, baseline, metrics) shipped earlier; Phase 4 CI integration shipped via standalone `.github/workflows/ml-benchmark.yml` workflow (smoke / benchmark dataset choice, operator-triggered, no auto-gating). |
| [RFC-077](RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) | Viewer feeds + operator config + jobs & hygiene | PRD-030 | v2.6.0–v2.7 | Phase 1a (`/api/feeds` + `feeds.spec.yaml`), Phase 1b (`/api/operator-config` + `viewer_operator.yaml` + `available_profiles`), Phase 2 (jobs API + asyncio subprocess runner + JSONL registry + stale/PID-orphan reconcile + cancel) all shipped; viewer Operator tab consumes the APIs. Tracks [#626](https://github.com/chipi/podcast_scraper/issues/626). |
| [RFC-093](RFC-093-litm-context-packs.md) | LITM-Aware MCP Context Packs | PRD-033 | v2.7 | Pack-builder (`search/context_pack.py`) shipped earlier with LITM positioning + token budget + tests; MCP tool (`corpus_briefing_pack`) registered in the existing RFC-095 server (2026-06-25, this branch). Autoresearch adoption decoupled — pack-builder is plain Python the autoresearch loop can consume at its own pace. Tracks [#861](https://github.com/chipi/podcast_scraper/issues/861). |
| [RFC-081](RFC-081-pre-prod-environment-and-control-plane.md) | Pre-prod environment on GitHub Codespaces (Phase 1) | — | v2.7 | `.github/workflows/deploy-codespace.yml` auto-fires on stack-test green; `cloud_thin` profile via GHCR `:main` + `:sha-<short>` tags; `backup-corpus.yml` + `verify-backup-restore.yml` + `post-deploy-smoke.yml` live; observability through the existing Grafana + Sentry stack. Always-on host follow-on lands as RFC-082. |
| [RFC-082](RFC-082-always-on-pre-prod-and-prod-hosting.md) | Always-on pre-prod / production hosting | — | v2.7 | `.github/workflows/deploy-prod.yml` deploys to the VPS via GitOps; `backup-corpus-prod.yml` + `prod-restore-corpus.yml` cover the persistence + restore contract; `docs/guides/PROD_RUNBOOK.md` covers day-2 ops. Public-edge TLS termination is NOT part of this RFC — see RFC-087. |
| [RFC-089](RFC-089-dgx-spark-tailnet-integration.md) | DGX Spark tailnet integration | — | v2.7 | Phases P0/P1/P2 shipped (tailnet join; vLLM serving Qwen3-30B-A3B-Instruct on `:8003`; `TailnetDgxProvider` + `local_dgx_*` profiles; autoresearch matrix on DGX; AI comparison guide updated). ADR-096 + ADR-097 Accepted. `DGX_RUNBOOK.md` covers day-2. **P3 (self-hosted GHA runner + pre-prod-uses-DGX-by-default) tracked as open sub-item [#813](https://github.com/chipi/podcast_scraper/issues/813)** — heavier infra change held for operator commitment. |
| [RFC-072](RFC-072-canonical-identity-layer-cross-layer-bridge.md) | Canonical Identity Layer + Cross-Layer Bridge | PRD-017 / PRD-019 | v2.6.0–v2.7 | `src/podcast_scraper/identity/` (resolver + registry + slugify) + `src/podcast_scraper/builders/bridge_builder.py` shipped; v3 additive fields (`insight_type`, `position_hint`) + materialized cross-layer descriptive edges (ABOUT / MENTIONS_PERSON / MENTIONS_ORG) land per-artifact via RFC-097. KL1/KL2/KL3/KL5 remain as designed future work. |
| [RFC-078](RFC-078-ephemeral-acceptance-smoke-test.md) | Ephemeral acceptance smoke test (full-stack) | — | v2.6.0 | Phase 1 shipped: `compose/docker-compose.stack-test.yml`, `tests/stack-test/` Playwright suite, `make stack-test-*` targets, `.github/workflows/stack-test.yml`. Follow-ups (`workflow_run`, merge policy, BuildKit cache) tracked as separate issues per the materialization rule. |
| [RFC-079](RFC-079-full-stack-docker-compose.md) | Full-stack Docker Compose topology | — | v2.6.0–v2.7 | Phase 1 (#659): `compose/docker-compose.stack.yml`, viewer + API + pipeline images, Makefile `stack-*` targets, `DOCKER_SERVICE_GUIDE.md`. Phase 2 (#660): Docker job execution via the Jobs API factory (`PODCAST_PIPELINE_EXEC_MODE=docker`). |
| [RFC-095](RFC-095-generic-mcp-server.md) | Generic MCP Server | PRD-034 | v2.7 | Epic #891 + slices #892/#893/#894 closed 2026-06-06/07. 17 tools registered in `src/podcast_scraper/mcp/` (incl. `corpus_briefing_pack` from RFC-093 via PR #1094). Library-wrap architecture, stdio transport. HTTP/SSE transport (OQ-1) deferred until remote-agent demand surfaces. |
| [RFC-097](RFC-097-unified-kg-gi-ontology-v2.md) | Unified KG + GI ontology v2 | PRD-017 / PRD-019 | v2.6.0–v2.7 | Anchor #1036 closed; chunks 1–8 in PR #1039; chunk 9 (#1073) in PR #1089. ADR-101 + ADR-102 + ADR-103 Accepted. Person Profile + Position Tracker viewer surfaces live via #1048/#1049/#1050. NER post-pass + corpus-level Topic clustering deliver deterministic typed connectivity under airgapped profiles (PR #1094). |
| [RFC-058](RFC-058-audio-speaker-diarization.md) | Audio-Based Speaker Diarization | PRD-020 / PRD-002 / PRD-008 | v2.6.0–v2.7 | pyannote provider + segment↔speaker alignment shipped (`src/podcast_scraper/diarization/`); ADR-058 ratified; providers wired into pipeline + evaluation harness. WhisperX comparison eval is future scope, not a gating deliverable. |
| [RFC-059](RFC-059-speaker-detection-refactor-test-audio.md) | Speaker Detection Refactor & Test Audio Improvements | PRD-020 / PRD-008 / PRD-002 | v2.6.0–v2.7 | `speaker_detectors/` package refactor (NER, hosts, guests, normalization) + test-audio improvements shipped; all 3 tracked issues (#269, #111, #109) closed. |
| [RFC-060](RFC-060-diarization-aware-commercial-cleaning.md) | Diarization-Aware Commercial Detection & Cleaning | PRD-020 / PRD-005 | v2.6.0–v2.7 | Phase 1 (#486) + Phase 2 (#488) closed; diarization signals layer shipped (`src/podcast_scraper/cleaning/commercial/diarization_signals.py`), integrated with hybrid summarization triggers. |
| [RFC-083](RFC-083-prod-failover-orchestration-and-cutover.md) | Production Failover — Orchestration, Spare Stack, and Traffic Cutover | — | v2.7 | #762/#763/#764 closed; ADR-089/090/091/092 Accepted; `prod-failover-stand-up.yml` orchestrator + 8 `drill-*.yml` workflows shipped (drill-bootstrap / drill-restore / drill-promote / drill-cutover / drill-rollback / drill-teardown / drill-status / drill-end-to-end). |
| [RFC-027](RFC-027-pipeline-metrics-improvements.md) | Pipeline Metrics Improvements | PRD-001 | v2.4–v2.7 | #120 closed 2026-02-03. `src/podcast_scraper/workflow/metrics.py` (1668 LOC) covers stage timing, per-episode timing, LLM call tracking, GI/KG metrics, cost monitoring, JSONL streaming. Two-tier output (DEBUG detail + INFO summary) + `to_json()` export shipped. RFC's specific Phase-2 field names drifted during evolution — the actual surface is richer. Proactive alerting split out to RFC-043. |
| [RFC-038](RFC-038-continuous-review-tooling.md) | Continuous Review Tooling | — | v2.4–v2.7 | Dependabot (#169 closed, `.github/dependabot.yml` + ADR-029) and pydeps module-coupling analysis (#170 closed, Makefile `deps-graph` / `call-graph` + ADR-030) both shipped. Pre-release checklist (#255) is tracked as a future enhancement (ADR-031) and does not block this RFC's promotion. |
| [RFC-043](RFC-043-automated-metrics-alerts.md) | Automated Metrics Alerts | — | v2.7 | Completed via redirect: operator-side alerting on Sentry + Grafana + Langfuse (the deployed o11y surface). Codebase emits everything alerts need; thresholds live with the vendor. Operator recipes documented in [`OBSERVABILITY_EXTENSIONS.md` §Operator alerting](../guides/OBSERVABILITY_EXTENSIONS.md#operator-alerting--sentry--grafana). Nightly `alerts[]` detection in `scripts/dashboard/generate_metrics.py` still feeds Grafana panels. Original PR-comment + webhook scripts deliberately abandoned. |
| [RFC-054](RFC-054-e2e-mock-response-strategy.md) | Flexible E2E Mock Response Strategy | — | v2.4–v2.7 | #135/#399/#401 closed. Implementation took a per-provider mock-client shape (`tests/fixtures/mock_server/{gemini,mistral}_mock_client.py`) plus dedicated unit suites for non-functional concerns (`test_retryable_errors.py` / `test_retry_integration.py` / `test_llm_circuit_breaker.py` / `test_provider_metrics.py` / `test_download_resilience.py`). Functional + non-functional separation, error / 429 / 5xx / timeout coverage all in place — just without the centralized router this RFC drafted. |
| [RFC-070](RFC-070-semantic-corpus-search-platform-future.md) | Semantic Corpus Search — Platform & Future Backends | PRD-021 | v2.6.0 | **Superseded by RFC-090.** FAISS retired via [ADR-099](../adr/ADR-099-lancedb-first-single-index-search.md) / PR #1010 in favour of LanceDB-first hybrid retrieval (BM25 + dense + RRF). Native filtering, hybrid retrieval, online clustering — RFC-070's three motivating outcomes — all shipped on that path. Qdrant + pgvector backends no longer planned. Cross-show Topic clustering shipped via RFC-097 chunk 9. |
| [RFC-073](RFC-073-autoresearch-v2-framework.md) | Autoresearch v2 Framework | PRD-007 | v2.6.0 | dev/held-out split (`curated_5feeds_dev_v1` + `curated_5feeds_benchmark_v2`), 40% fraction contestation, Efficiency rubric, seed plumbing for OpenAI, prose extraction before judging (`autoresearch/JUDGING.md`), v2 reference card (`autoresearch/openai_v2_comparison_2026-04-14.md`) all shipped. Cross-provider replication + multi-run averaging tracked as Future Work. |
| [RFC-074](RFC-074-process-safety-ml-workloads-macos.md) | Process Safety for ML Workloads on macOS | — | v2.6.0–v2.7 | Makefile `cleanup-processes` / `check-zombie` / `check-spotlight` targets shipped + wired as prerequisites for test-unit/test-integration/test-e2e. `PYTEST_WORKERS ?= 2` simplification, pre-commit `MAX_HOOK_SECONDS=120` watchdog (`.github/hooks/pre-commit`), SIGALRM-based preload timeout in `scripts/cache/preload_ml_models.py` (1200s default / 7200s `--production` / `PRELOAD_TIMEOUT` override) all live. No system-crash incidents recurred since. |
| [RFC-075](RFC-075-corpus-topic-clustering.md) | Corpus Topic Clustering Layer | PRD-021 | v2.6.0–v2.7 | All 4 phases + post-impl review shipped end-to-end. #551/#552/#553/#554/#555/#556 closed + label-quality follow-ups #580/#587/#590. `src/podcast_scraper/search/topic_clusters.py` + `topic-clusters` CLI + `GET /api/corpus/topic-clusters` route + viewer ingestion (`corpusTopicClustersApi.ts`) + Cytoscape `TopicCluster` compound parent style. v2 schema with v1 read-compat. Default threshold lowered to 0.70 after 1178-topic production sweep. Distinct from RFC-097 chunk 9: RFC-075 is the viewer-overlay path; RFC-097 chunk 9 is the KG-typed-connectivity path — they coexist. |

## Gap analysis {:#gaps}

**Counts (reconcile when moving RFCs):** **100** files under `docs/rfc/RFC-*.md` -- IDs **RFC-001--RFC-101**
with **no RFC-014**. **8** open (in-flight, partial implementation), **88** completed, and **3** Draft
(not indexed until promoted) in the tables above.

**Open RFC clusters:** Consumer Learning Platform foundation (**RFC-098**–**RFC-101**), VPS public edge
(**RFC-087**), ML query router (**RFC-092**), audio pipeline separation + viewer media (**RFC-096**),
KG proximity signal — rejected (**RFC-091**).

**Active RFCs (in implementation):** Enrichment-layer architecture
(**RFC-088**) — Epic [#1101](https://github.com/chipi/podcast_scraper/issues/1101) carries the live
implementation across 9 chunks (foundation + resilience + metrics/o11y + MCP correlation + 6
deterministic enrichers + topic_similarity + nli_contradiction + QueryEnricher + viewer
integration + profile-preset wiring + promotion). Live plan:
`docs/wip/RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md`. Chunk 0 (ADR-104) shipped on the
current branch.

**Draft RFCs (not indexed):** Postgres projection (**RFC-051**), adaptive summarization routing
(**RFC-053**).
These are discoverable by filename under `docs/rfc/` but excluded from the index per the
[index inclusion rule](../guides/MARKDOWN_LINTING_GUIDE.md) (Draft docs are not indexed).

### Open RFCs (detail)

| RFC | Theme | Notes |
| --- | --- | --- |
| [RFC-087](RFC-087-vps-public-edge-multi-compose.md) | VPS public edge + multi-Compose hosting | **Draft:** optional public TLS edge (Caddy / Traefik / Cloudflare Tunnel) for multi-host VPS routing while operators stay on Tailscale. |
| [RFC-091](RFC-091-kg-proximity-signal.md) | KG Proximity Signal | **Decision Record:** rejected as retrieval signal on every corpus/axis; KG value is meaning-bearing relational edges (Person→Insight, #874), not proximity ranking. |
| [RFC-092](RFC-092-ml-query-router.md) | ML Query Router | **Draft:** rules router shipped; ML classifier gated on eval data ([#860](https://github.com/chipi/podcast_scraper/issues/860)). |
| [RFC-096](RFC-096-audio-pipeline-separation-and-viewer-media.md) | Audio pipeline separation + viewer media | **Draft:** `pipeline_stage` split ([#414](https://github.com/chipi/podcast_scraper/issues/414)); persist corpus `media/` + `GET /api/corpus/media` + viewer transcript audio player ([#547](https://github.com/chipi/podcast_scraper/issues/547)). |
| [RFC-098](RFC-098-learning-platform-foundation.md) | Learning Platform Foundation | **Draft (v2.7, P0):** OAuth identity, plain per-user files, dedicated `/api/app/*` consumer API, stable episode slug contract, episode-scoped grounded search. Keystone for RFC-099–101. |
| [RFC-099](RFC-099-learning-platform-consumer-client.md) | Learning Platform Consumer Client | **Draft (v2.7, P1–P2):** new top-level PWA (Vue 3 + TS); transcript-sync engine, queue, Knowledge Panel, capture; consumes RFC-098. |
| [RFC-100](RFC-100-audio-bridge-subsystem.md) | Audio Bridge Subsystem | **Draft (v2.7):** resolve playable origin enclosure URL per episode (bridge, never rehost); freshness/redirect/expiry handling. |
| [RFC-101](RFC-101-personal-knowledge-corpus.md) | Personal Knowledge Corpus | **Draft (v2.7, P3):** per-user projection over GIL/KG scoped to heard/captured episodes; grounded recall via retrieval (no LLM); cross-episode connections; spaced resurfacing. |

### Recently completed (v2.6.0+)

| RFC | Delivered (high level) |
| --- | --- |
| [RFC-050](RFC-050-grounded-insight-layer-use-cases.md) | Single-layer GIL consumption; cross-layer → **RFC-072** |
| [RFC-056](RFC-056-knowledge-graph-layer-use-cases.md) | Single-layer KG consumption; cross-layer → **RFC-072** |
| [RFC-057](RFC-057-autoresearch-optimization-loop.md) | Closed per [ADR-073](../adr/ADR-073-rfc057-autoresearch-closure.md) |
| [RFC-061](RFC-061-semantic-corpus-search.md) | FAISS path, CLI + API + semantic `gi explore` |
| [RFC-062](RFC-062-gi-kg-viewer-v2.md) | Server + Vue SPA + Playwright ([ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md)–[ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md)) |
| [RFC-063](RFC-063-multi-feed-corpus-append-resume.md) | Multi-feed layout, manifest ([ADR-074](../adr/ADR-074-multi-feed-corpus-parent-layout-and-manifest.md)) |
| [RFC-064](RFC-064-performance-profiling-release-freeze.md) | Frozen profiles, freeze/diff scripts ([ADR-075](../adr/ADR-075-frozen-yaml-performance-profiles-for-release-baselines.md)) |
| [RFC-065](RFC-065-live-pipeline-monitor.md) | `--monitor`, `.pipeline_status.json`, optional `[monitor]` |
| [RFC-066](RFC-066-run-compare-performance-tab.md) | Streamlit **Performance** vs frozen profiles ([ADR-076](../adr/ADR-076-streamlit-for-operator-run-comparison-and-performance-views.md)) |
| [RFC-067](RFC-067-corpus-library-api-viewer.md) | `/api/corpus/*`, Library tab, similar episodes |
| [RFC-068](RFC-068-corpus-digest-api-viewer.md) | Digest API + tab, Library glance |
| [RFC-069](RFC-069-graph-exploration-toolkit.md) | Graph exploration toolkit |
| [RFC-071](RFC-071-corpus-intelligence-dashboard-viewer.md) | Dashboard tab, corpus intelligence panels |
| [RFC-076](RFC-076-progressive-graph-expansion.md) | Progressive graph expansion (`/api/corpus/node-episodes`, graph `onetap`/`dbltap`) |

Older **draft RFC audit** tables (pre-2026-04) are **archeology** — trust this index and each RFC’s
**Status** block.

### Recommendations

1. **Status changes** — Edit RFC body + this index **together**.
2. **Large deliveries without new ADRs** — Often **RFC + guides + API docs**; see [ADR gap analysis](../adr/index.md#gaps) for when an ADR is still worth extracting.
3. **Decision vs code** — Use **Open** / **Completed** here plus [`docs/adr/index.md`](../adr/index.md) **Code** column.

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
