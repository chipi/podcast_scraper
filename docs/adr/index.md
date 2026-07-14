# Architecture Decision Records (ADRs)

## Purpose

Architecture Decision Records (ADRs) capture the **what** and **why** of **decisions** taken after
technical discussion (often in an **RFC** or issue). **RFCs** hold the **full design**, alternatives,
and implementation journey; **Accepted** ADRs are the **short, immutable** record of what we chose.

## How ADRs Work

1. **Immutable Records**: Once an ADR is accepted, it remains unchanged unless superseded by a new ADR.
2. **Context Driven**: They explain the trade-offs and rationale behind a decision (not the whole solution document).
3. **Reference for Developers**: They provide onboarding context for why certain patterns (like the Provider Protocol) were chosen.

## ADR Index

**Code** (last column): **Yes** = reflected in the codebase; **Partial** = incomplete or
still rolling out; **No** = not started (including accepted ADRs waiting on implementation).

| ADR | Title | Status | Related RFC | Description | Code |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [ADR-001](ADR-001-hybrid-concurrency-strategy.md) | Hybrid Concurrency Strategy | Accepted | [RFC-001](../rfc/RFC-001-workflow-orchestration.md) | IO-bound threading, sequential CPU/GPU tasks | Yes |
| [ADR-002](ADR-002-security-first-xml-processing.md) | Security-First XML Processing | Accepted | [RFC-002](../rfc/RFC-002-rss-parsing.md) | Mandated use of defusedxml for RSS parsing | Yes |
| [ADR-003](ADR-003-deterministic-feed-storage.md) | Deterministic Feed Storage | Accepted | [RFC-004](../rfc/RFC-004-filesystem-layout.md) | Hash-based output directory derivation | Yes |
| [ADR-004](ADR-004-flat-filesystem-archive-layout.md) | Flat Filesystem Archive Layout | Accepted | [RFC-004](../rfc/RFC-004-filesystem-layout.md) | Flat directory structure per feed run | Yes |
| [ADR-005](ADR-005-lazy-ml-dependency-loading.md) | Lazy ML Dependency Loading | Accepted | [RFC-005](../rfc/RFC-005-whisper-integration.md) | Function-level imports for heavy ML libraries | Yes |
| [ADR-006](ADR-006-context-aware-model-selection.md) | Context-Aware Model Selection | Accepted | [RFC-010](../rfc/RFC-010-speaker-name-detection.md) | Automatic English model promotion (.en) | Yes |
| [ADR-007](ADR-007-universal-episode-identity.md) | Universal Episode Identity | Accepted | [RFC-011](../rfc/RFC-011-metadata-generation.md) | GUID-first deterministic episode ID generation | Yes |
| [ADR-008](ADR-008-database-agnostic-metadata-schema.md) | Database-Agnostic Metadata Schema | Accepted | [RFC-011](../rfc/RFC-011-metadata-generation.md) | Unified JSON format for SQL/NoSQL | Yes |
| [ADR-009](ADR-009-privacy-first-local-summarization.md) | Privacy-First Local Summarization | Accepted | [RFC-012](../rfc/RFC-012-episode-summarization.md) | Local Transformers over Cloud APIs | Yes |
| [ADR-010](ADR-010-hierarchical-summarization-pattern.md) | Hierarchical Summarization Pattern | Accepted | [RFC-012](../rfc/RFC-012-episode-summarization.md) | Map-reduce chunking for long transcripts | Yes |
| [ADR-011](ADR-011-secure-credential-injection.md) | Secure Credential Injection | Accepted | [RFC-013](../rfc/RFC-013-openai-provider-implementation.md) | Environment-based secret management | Yes |
| [ADR-012](ADR-012-provider-agnostic-preprocessing.md) | Provider-Agnostic Preprocessing | Accepted | [RFC-013](../rfc/RFC-013-openai-provider-implementation.md) | Shared pre-inference cleaning pipeline | Yes |
| [ADR-013](ADR-013-standalone-experiment-configuration.md) | Standalone Experiment Configuration | Accepted | [RFC-015](../rfc/RFC-015-ai-experiment-pipeline.md) | Separation of research params from code | Yes |
| [ADR-014](ADR-014-codified-comparison-baselines.md) | Codified Comparison Baselines | Accepted | [RFC-015](../rfc/RFC-015-ai-experiment-pipeline.md), [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md) | Objective delta measurement vs baseline artifacts | Yes |
| [ADR-015](ADR-015-deep-provider-fingerprinting.md) | Deep Provider Fingerprinting | Accepted | [RFC-016](../rfc/RFC-016-modularization-for-ai-experiments.md) | Hardware and environment tracking for reproducibility | Yes |
| [ADR-016](ADR-016-typed-provider-parameter-models.md) | Typed Provider Parameter Models | Accepted | [RFC-016](../rfc/RFC-016-modularization-for-ai-experiments.md) | Pydantic validation for backend parameters | Yes |
| [ADR-017](ADR-017-registered-preprocessing-profiles.md) | Registered Preprocessing Profiles | Accepted | [RFC-016](../rfc/RFC-016-modularization-for-ai-experiments.md) | Versioned cleaning logic tracking | Yes |
| [ADR-018](ADR-018-externalized-prompt-management.md) | Externalized Prompt Management | Accepted | [RFC-017](../rfc/RFC-017-prompt-management.md) | Versioned Jinja2 templates in prompts/ | Yes |
| [ADR-019](ADR-019-standardized-test-pyramid.md) | Standardized Test Pyramid | Accepted | [RFC-018](../rfc/RFC-018-test-structure-reorganization.md), [RFC-024](../rfc/RFC-024-test-execution-optimization.md) | Strict unit/integration/e2e tiering | Yes |
| [ADR-020](ADR-020-protocol-based-provider-discovery.md) | Protocol-Based Provider Discovery | Accepted | [RFC-021](../rfc/RFC-021-modularization-refactoring-plan.md) | Decoupling via PEP 544 Protocols | Yes |
| [ADR-021](ADR-021-acceptance-test-tier-as-final-ci-gate.md) | Acceptance Test Tier as Final CI Gate | Accepted | [RFC-023](../rfc/RFC-023-readme-acceptance-tests.md) | Fourth test tier for README/documentation accuracy; runs last in CI | Yes |
| [ADR-022](ADR-022-flaky-test-defense.md) | Flaky Test Defense | Accepted | [RFC-025](../rfc/RFC-025-test-metrics-and-health-tracking.md) | Automated retries and health reporting | Yes |
| [ADR-023](ADR-023-public-operational-metrics.md) | Public Operational Metrics | Accepted | [RFC-026](../rfc/RFC-026-metrics-consumption-and-dashboards.md) | Transparency via GitHub Pages dashboards | Yes |
| [ADR-024](ADR-024-unified-provider-pattern.md) | Unified Provider Pattern | Accepted | [RFC-029](../rfc/RFC-029-provider-refactoring-consolidation.md) | Type-based unified provider classes | Yes |
| [ADR-025](ADR-025-technology-based-provider-naming.md) | Technology-Based Provider Naming | Accepted | [RFC-029](../rfc/RFC-029-provider-refactoring-consolidation.md) | Clear library-based option naming | Yes |
| [ADR-026](ADR-026-per-capability-provider-selection.md) | Per-Capability Provider Selection | Accepted | [RFC-032](../rfc/RFC-032-anthropic-provider-implementation.md), [RFC-033](../rfc/RFC-033-mistral-provider-implementation.md), [RFC-034](../rfc/RFC-034-deepseek-provider-implementation.md), [RFC-035](../rfc/RFC-035-gemini-provider-implementation.md), [RFC-036](../rfc/RFC-036-grok-provider-implementation.md), [RFC-037](../rfc/RFC-037-ollama-provider-implementation.md) | Independent provider choice per capability; partial-protocol providers allowed | Yes |
| [ADR-027](ADR-027-unified-provider-metrics-contract.md) | Unified Provider Metrics Contract | Accepted | - | Standardized `ProviderCallMetrics` pattern for all providers | Yes |
| [ADR-028](ADR-028-unified-retry-policy-with-metrics.md) | Unified Retry Policy with Metrics | Accepted | - | Centralized retry for **LLM/API providers** with backoff and metrics (not RSS/media HTTP; see [CONFIGURATION — Download resilience](../api/CONFIGURATION.md#download-resilience)) | Yes |
| [ADR-029](ADR-029-grouped-dependency-automation.md) | Grouped Dependency Automation | Accepted | [RFC-038](../rfc/RFC-038-continuous-review-tooling.md) | Balanced Dependabot updates via grouping | Yes |
| [ADR-030](ADR-030-periodic-module-coupling-analysis.md) | Periodic Module Coupling Analysis | Accepted | [RFC-038](../rfc/RFC-038-continuous-review-tooling.md) | Nightly visualization of architecture health | Yes |
| [ADR-031](ADR-031-mandatory-pre-release-validation.md) | Mandatory Pre-Release Validation | Accepted | [RFC-038](../rfc/RFC-038-continuous-review-tooling.md) | Standardized checklist script for releases | Partial |
| [ADR-032](ADR-032-git-worktree-based-development.md) | Git Worktree-Based Development | Accepted | [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md) | Parallel stable dev environments | Yes |
| [ADR-033](ADR-033-stratified-ci-execution.md) | Stratified CI Execution | Accepted | [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md) | Fast push checks vs. full PR validation | Yes |
| [ADR-034](ADR-034-isolated-runtime-environments.md) | Isolated Runtime Environments | Accepted | [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md) | Independent venv per worktree | Yes |
| [ADR-035](ADR-035-linear-history-via-squash-merge.md) | Linear History via Squash-Merge | Accepted | [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md) | Clean, revertible main branch history | Yes |
| [ADR-036](ADR-036-standardized-pre-provider-audio-stage.md) | Standardized Pre-Provider Audio Stage | Accepted | [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md) | Mandatory optimization before any transcription | Yes |
| [ADR-037](ADR-037-content-hash-based-audio-caching.md) | Content-Hash Based Audio Caching | Accepted | [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md) | Shared optimized artifacts in .cache/ | Yes |
| [ADR-038](ADR-038-ffmpeg-first-audio-manipulation.md) | FFmpeg-First Audio Manipulation | Accepted | [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md) | System-level performance for audio pipelines | Yes |
| [ADR-039](ADR-039-speech-optimized-codec-opus.md) | Speech-Optimized Codec (Opus) | Accepted | [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md) | MP3 (`libmp3lame` @ 64 kbps) for preprocessed audio; Opus rejected (see ADR) | Yes |
| [ADR-040](ADR-040-explicit-golden-dataset-versioning.md) | Explicit Golden Dataset Versioning | Accepted | [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md) | Approved, frozen ground truth data versions | Yes |
| [ADR-041](ADR-041-multi-tiered-benchmarking-strategy.md) | Multi-Tiered Benchmarking Strategy | Accepted | [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md) | Fast PR smoke tests vs nightly full benchmarks | Yes |
| [ADR-042](ADR-042-heuristic-based-quality-gates.md) | Heuristic-Based Quality Gates | Accepted | [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md) | Regex-based detection of common AI failure modes | Yes |
| [ADR-043](ADR-043-hybrid-map-reduce-summarization.md) | Hybrid MAP-REDUCE Summarization | Accepted | [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md) | Compression (Classic) + Abstraction (Instruct LLM) | Yes |
| [ADR-044](ADR-044-local-llm-backend-abstraction.md) | Local LLM Backend Abstraction | Accepted | [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md) | Support for llama.cpp, ollama, and transformers | Yes |
| [ADR-045](ADR-045-strict-reduce-prompt-contract.md) | Strict REDUCE Prompt Contract | Accepted | [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md) | Mandatory markdown structure for LLM outputs | Yes |
| [ADR-046](ADR-046-mps-exclusive-mode-apple-silicon.md) | MPS Exclusive Mode for Apple Silicon | Accepted | [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md) | Serialize GPU work on MPS to prevent memory contention; default on | Yes |
| [ADR-047](ADR-047-proactive-metric-regression-alerting.md) | Proactive Metric Regression Alerting | Accepted | [RFC-043](../rfc/RFC-043-automated-metrics-alerts.md) | Automated PR comments and webhook notifications | Partial |
| [ADR-048](ADR-048-centralized-model-registry.md) | Centralized Model Registry | Accepted | [RFC-044](../rfc/RFC-044-model-registry.md), [RFC-029](../rfc/RFC-029-provider-refactoring-consolidation.md) | Single source of truth for model architecture limits | Yes |
| [ADR-049](ADR-049-materialization-boundary-for-eval-inputs.md) | Materialization Boundary for Evaluation Inputs | Accepted | [RFC-046](../rfc/RFC-046-materialization-architecture.md) | Preprocessing becomes dataset definition via materialization_id; chunking stays in run config | Yes |
| [ADR-050](ADR-050-single-code-path-eval-app-alignment.md) | Single Code Path for Evaluation and Application | Accepted | [RFC-048](../rfc/RFC-048-evaluation-application-alignment.md) | Eval and app share identical execution path; scorers are read-only observers | Yes |
| [ADR-051](ADR-051-per-episode-json-artifacts-with-logical-union.md) | Per-Episode JSON Artifacts with Logical Union | Accepted | [RFC-049](../rfc/RFC-049-grounded-insight-layer-core.md), [RFC-055](../rfc/RFC-055-knowledge-graph-layer-core.md), [RFC-061](../rfc/RFC-061-semantic-corpus-search.md) | Shard by episode (gi.json, kg.json); union at query time; optional materialization | Yes |
| [ADR-052](ADR-052-separate-gil-and-kg-artifact-layers.md) | Separate GIL and KG Artifact Layers | Accepted | [RFC-049](../rfc/RFC-049-grounded-insight-layer-core.md), [RFC-055](../rfc/RFC-055-knowledge-graph-layer-core.md) | Independent schemas, feature flags, CLI namespaces, and evolution paths | Yes |
| [ADR-053](ADR-053-grounding-contract-for-evidence-backed-insights.md) | Grounding Contract for Evidence-Backed Insights | Accepted | [RFC-049](../rfc/RFC-049-grounded-insight-layer-core.md), [RFC-050](../rfc/RFC-050-grounded-insight-layer-use-cases.md) | Explicit grounded boolean, verbatim quotes with spans, evidence chain | Yes |
| [ADR-054](ADR-054-relational-postgres-projection-for-gil-and-kg.md) | Relational Postgres Projection for GIL and KG | Accepted | [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md) | Files canonical, Postgres is derived; separate GIL/KG tables; provenance on every row | No |
| [ADR-055](ADR-055-adaptive-summarization-routing.md) | Adaptive Summarization Routing | Proposed | [RFC-053](../rfc/RFC-053-adaptive-summarization-routing.md) | Rule-based routing with episode profiling for summarization strategies | No |
| [ADR-056](ADR-056-composable-e2e-mock-response-strategy.md) | Composable E2E Mock Response Strategy | Superseded | [RFC-054](../rfc/RFC-054-e2e-mock-response-strategy.md) | Centralized response-profile router not built; per-provider mock clients + per-concern unit suites shipped instead. RFC-054 closed via redirect. | Yes (different shape) |
| [ADR-057](ADR-057-autoresearch-thin-harness-with-credential-isolation.md) | AutoResearch Thin Harness with Credential Isolation | Accepted | [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) | Thin control layer reusing existing eval; immutable score.py; AUTORESEARCH\_\* credential vars | Yes |
| [ADR-058](ADR-058-additive-pyannote-diarization-with-separate-extra.md) | Additive pyannote Diarization | Accepted | [RFC-058](../rfc/RFC-058-audio-speaker-diarization.md) | pyannote additive second pass; `[ml]`/`[dev]` deps; default on for local Whisper | Partial |
| [ADR-059](ADR-059-confidence-scored-multi-signal-commercial-detection.md) | Confidence-Scored Multi-Signal Commercial Detection | Accepted | [RFC-060](../rfc/RFC-060-diarization-aware-commercial-cleaning.md) | Confidence-scored candidates; Phase 1 (#486) + Phase 2 diarization signals (#488) both shipped | Yes |
| [ADR-060](ADR-060-vectorstore-protocol-with-backend-abstraction.md) | VectorStore Protocol with Backend Abstraction | Accepted (superseded by [ADR-099](ADR-099-lancedb-first-single-index-search.md)) | [RFC-061](../rfc/RFC-061-semantic-corpus-search.md) | PEP 544 protocol decoupling FAISS (Phase 1) from Qdrant (Phase 2) | Yes |
| [ADR-061](ADR-061-faiss-phase-1-with-post-filter-metadata.md) | FAISS Phase 1 with Post-Filter Metadata Strategy | Accepted (superseded by [ADR-099](ADR-099-lancedb-first-single-index-search.md)) | [RFC-061](../rfc/RFC-061-semantic-corpus-search.md) | Over-fetch + post-filter for CLI-scale; auto index type selection | Yes |
| [ADR-062](ADR-062-sentence-boundary-transcript-chunking.md) | Sentence-Boundary Transcript Chunking | Accepted | [RFC-061](../rfc/RFC-061-semantic-corpus-search.md) | Regex sentence split, configurable target/overlap tokens, timestamp interpolation | Yes |
| [ADR-063](ADR-063-transparent-semantic-upgrade-for-gi-explore.md) | Transparent Semantic Upgrade for gi explore | Accepted | [RFC-061](../rfc/RFC-061-semantic-corpus-search.md), [RFC-050](../rfc/RFC-050-grounded-insight-layer-use-cases.md) | Auto-detect vector index; semantic if available, substring fallback if not | Yes |
| [ADR-064](ADR-064-canonical-server-layer-with-feature-flagged-routes.md) | Canonical Server Layer with Feature-Flagged Route Groups | Accepted | [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md) | `server/` module with `podcast serve` CLI; viewer routes v2.6, platform routes v2.7 | Yes |
| [ADR-065](ADR-065-vue3-vite-cytoscape-frontend-stack.md) | Vue 3 + Vite + Cytoscape.js Frontend Stack | Accepted | [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md) | Unified frontend stack for viewer and future platform UI | Yes |
| [ADR-066](ADR-066-playwright-for-ui-e2e-testing.md) | Playwright for UI End-to-End Testing | Accepted | [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md) | Browser regression testing; extends ADR-020 test pyramid with UI layer | Yes |
| [ADR-067](ADR-067-pegasus-led-retirement-podcast-content.md) | Pegasus/LED Retirement for Podcast Content | Accepted | [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) | GSG pretraining mismatch → near-duplicate chunks → LED ngram exhaustion; reserved for news content type | Yes |
| [ADR-068](ADR-068-bart-led-as-ml-production-baseline.md) | BART+LED as Local ML Production Baseline | Accepted | [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) | Autoresearch sweep: +4.26% ROUGE-L over dev baseline (18.82%); 2 params accepted (max_new_tokens=550, num_beams=6) | Yes |
| [ADR-069](ADR-069-hybrid-ml-pipeline-as-production-direction.md) | Hybrid ML Pipeline as Primary Production Direction | Accepted | [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md), [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md) | BART MAP + Llama 3.2:3b REDUCE at 23.1% ROUGE-L; closes 70% of cloud quality gap; temp=0.5, top_p=1.0 | Yes |
| [ADR-070](ADR-070-bart-base-as-hybrid-map-stage.md) | BART-base as Hybrid MAP Stage | Accepted | [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) | BART beats LongT5 as MAP (21.2% vs 20.8%); pretraining alignment > context window size | Yes |
| [ADR-071](ADR-071-four-tier-summarization-strategy.md) | Four-Tier Summarization Strategy | Accepted | [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) | ML Dev / ML Prod / LLM Local / LLM Cloud — direct Llama 3.2:3b beats hybrid (24.3% vs 23.7%, 2x faster) | Yes |
| [ADR-072](ADR-072-llama32-3b-as-tier3-local-llm.md) | Llama 3.2:3b as Tier 3 Local LLM | Accepted | [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) | 3B beats 7-12B models — instruction-following > size; temp=0.3 direct, temp=0.5 hybrid; 26.4% ROUGE-L @ 7.5s/ep | Yes |
| [ADR-073](ADR-073-rfc057-autoresearch-closure.md) | RFC-057 Autoresearch Loop — Closure and Final State | Accepted | [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) | Closes RFC-057; documents Track A/B outcomes, silver refs, 72-config matrix, production defaults | Yes |
| [ADR-074](ADR-074-multi-feed-corpus-parent-layout-and-manifest.md) | Multi-Feed Corpus Parent Layout and Machine-Readable Manifest | Accepted | [RFC-063](../rfc/RFC-063-multi-feed-corpus-append-resume.md) | Layout A corpus parent; unified discovery; `corpus_manifest.json` / optional summaries as operational artifacts | Yes |
| [ADR-075](ADR-075-frozen-yaml-performance-profiles-for-release-baselines.md) | Frozen YAML Performance Profiles for Release Resource Baselines | Accepted | [RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md) | `data/profiles/*.yaml` + freeze/diff scripts; resource cost sibling to quality baselines | Yes |
| [ADR-076](ADR-076-streamlit-for-operator-run-comparison-and-performance-views.md) | Streamlit for Operator Run Comparison and Performance Views | Accepted | [RFC-047](../rfc/RFC-047-run-comparison-visual-tool.md), [RFC-066](../rfc/RFC-066-run-compare-performance-tab.md) | Eval / Performance UI stays in `tools/run_compare/`; Vue viewer stays corpus-first | Yes |
| [ADR-077](ADR-077-local-ollama-model-selection.md) | Local Ollama Model Selection | Accepted | — | Default Ollama models per profile and tier | Yes |
| [ADR-078](ADR-078-gil-evidence-bundling-per-provider-champions.md) | GIL Evidence Stack Bundling — Per-Provider Champion Modes | Accepted | — | `bundled_ab` default; Mistral=`bundled_b_only`; Ollama bundled-only (staged unviable on local) | Yes |
| [ADR-079](ADR-079-opentofu-for-always-on-hosting-iac.md) | OpenTofu for Always-On Hosting IaC | Accepted | [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) | OpenTofu + `hcloud` + `tailscale` providers; `infra/tofu` entry | Yes |
| [ADR-080](ADR-080-opentofu-state-sops-age-in-git.md) | OpenTofu State Encrypted In-Repo (sops + age) | Accepted | [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) | Committed `.enc` state; `TFSTATE_AGE_KEY` in CI | Yes |
| [ADR-081](ADR-081-drill-opentofu-workspace-tailscale-acl-ownership.md) | Drill OpenTofu Workspace and Tailscale ACL Ownership | Accepted | [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) | Workspace `drill`, `HCLOUD_TOKEN_DRILL`, prod-only `tailscale_acl` | Yes |
| [ADR-082](ADR-082-gitops-app-deploy-via-stack-test-and-gha.md) | GitOps App Deploy via stack-test and GitHub Actions | Accepted | [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) | Stack-test gate + publish + `deploy-prod`; `workflow_run` target; infra apply manual | Yes |
| [ADR-083](ADR-083-tailscale-private-ingress-always-on-vps.md) | Tailscale as Private Ingress for Always-On VPS | Accepted | [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) | App on tailnet; `tag:gha-deployer` SSH path | Yes |
| [ADR-084](ADR-084-full-stack-docker-compose-topology.md) | Full-Stack Docker Compose Topology (API, Viewer, Pipeline) | Accepted | [RFC-079](../rfc/RFC-079-full-stack-docker-compose.md) | `compose/docker-compose.stack.yml`; shared volume; optional Docker job exec | Yes |
| [ADR-085](ADR-085-ephemeral-stack-test-integration-gate.md) | Ephemeral Stack-Test Integration Gate on Main | Accepted | [RFC-078](../rfc/RFC-078-ephemeral-acceptance-smoke-test.md), [RFC-079](../rfc/RFC-079-full-stack-docker-compose.md) | Compose overlay + Playwright `tests/stack-test/`; distinct from ADR-021 | Yes |
| [ADR-086](ADR-086-canonical-identity-layer-and-bridge-json-cross-layer-join.md) | Canonical Identity Layer and Per-Episode bridge.json Cross-Layer Join | Accepted | [RFC-072](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) | CIL ids + `bridge.json` seam; GIL or KG stay separate (ADR-052) | Yes |
| [ADR-087](ADR-087-autoresearch-track-a-v2-dev-held-out-and-judging.md) | Autoresearch Track A v2 — Dev or Held-Out Split and Judging | Accepted | [RFC-073](../rfc/RFC-073-autoresearch-v2-framework.md), [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) | Disjoint held-out; fraction contestation; Efficiency rubric; seed wiring | Yes |
| [ADR-088](ADR-088-macos-local-ci-process-safety-for-ml-workloads.md) | macOS Local CI Process Safety for ML Workloads | Accepted | [RFC-074](../rfc/RFC-074-process-safety-ml-workloads-macos.md) | No parse-time ML probes; cleanup or zombie checks; agent no-pileup rules | Yes |
| [ADR-089](ADR-089-prod-failover-orchestrator-separate-from-drill.md) | Prod Failover Orchestrator Separate from DR Drill | Accepted | [RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md) | Own workflow family; reuse drill workspace/secrets; no auto-destroy; GitHub #764 | Yes |
| [ADR-090](ADR-090-prod-failover-dns-first-cutover.md) | Prod Failover — DNS-First Cutover on Tailnet | Accepted | [RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md) | Canonical hostname DNS flip primary; floating IP optional | Yes |
| [ADR-091](ADR-091-prod-failover-gha-triggers-and-gates.md) | Prod Failover — GHA Triggers and Gates | Accepted | [RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md) | Manual cutover/failback/teardown; freeze prod schedules; spare schedules off after restore | Yes |
| [ADR-092](ADR-092-corpus-snapshot-backup-manifest-and-newest-compatible-restore.md) | Corpus Snapshot Backup Manifest and Newest-Compatible Restore Default | Accepted | [RFC-084](../rfc/RFC-084-corpus-backup-manifest-and-version-aware-restore.md) | `snapshot.manifest.json`; dual placement; newest-compatible default; fail closed; GitHub #763 | Yes |
| [ADR-093](ADR-093-canonical-stack-contract-and-environment-adapters.md) | Canonical Stack Contract Versus Environment Adapters | Accepted | [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) | One topology/health/`stack-test` discipline; adapters for transport/secrets only; steady vs restore playbooks separate; GitHub #762 | Yes |
| [ADR-094](ADR-094-graph-handoff-orchestrator-fsm.md) | Graph Handoff Orchestrator — 8-State FSM with Envelope and Generation Tokens | Accepted | [RFC-085](../rfc/RFC-085-graph-handoff-orchestrator-retrospective.md) | Single FSM authoritative across 13 entry points; envelope contract; generation supersession; viewer-side stuck detection + error strip | Yes |
| [ADR-095](ADR-095-viewer-test-pyramid.md) | Viewer Test Pyramid — Three Tiers (Mocks, Production-Shaped, Real Corpus) | Accepted | [RFC-086](../rfc/RFC-086-viewer-test-pyramid-and-production-shaped-fixtures.md) | Tier-1 mocked, Tier-2 production-shaped fixtures, Tier-3 real-corpus matrix; `make ci-ui-validation`; real-bug-first matrix-row rule | Yes |
| [ADR-096](ADR-096-dgx-spark-prod-primary-with-fallback.md) | DGX Spark in prod — primary-with-fallback contract | Accepted | [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md) | Prod may use DGX when each stage names a cloud fallback; v1 stage is Whisper | Partial |
| [ADR-097](ADR-097-self-hosted-gha-runner-policy.md) | GitHub Actions self-hosted runner policy on public repos | Accepted | [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md) | Ephemeral runner + fork approval + workflow allowlist before `dgx-spark` | Partial |
| [ADR-098](ADR-098-embedding-provider-profile-axis.md) | Embedding provider as a profile axis, supersede RFC-089 §D4 | Accepted | [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md) (§D4 superseded) | `vector_embedding_provider` literal (sentence_transformers \| ollama); shim deleted; default stays `sentence_transformers` everywhere — empirical A/B showed MiniLM beats nomic under fair chunking | Partial |
| [ADR-099](ADR-099-lancedb-first-single-index-search.md) | LanceDB-first single-index search; retire FAISS | Accepted | [RFC-090](../rfc/RFC-090-hybrid-retrieval.md) | Serving ran LanceDB like FAISS (open-per-query, over-fetch with vectors, runtime fallback); LanceDB already holds all content so FAISS is a redundant copy. Make LanceDB the single index — opened once, native hybrid, `select()` projection — and remove FAISS (build/serve/config); rebuild, no migration. Shipped #1010 (`faiss_store.py` deleted) | Yes |
| [ADR-100](ADR-100-response-shape-guardrails-for-cloud-llm-providers.md) | Response shape guardrails for cloud LLM providers | Accepted | — | Snapshot envelope diffs on every cloud LLM SDK bump; quarantine drift before it surfaces as silent extraction regressions | Yes |
| [ADR-101](ADR-101-drop-legacy-kg-gi-shape.md) | Drop legacy KG + GI shape support | Accepted | [RFC-097](../rfc/RFC-097-unified-kg-gi-ontology-v2.md) (chunk 9) | Skip the 2–4 week bake gate from RFC-097 §161; flip KG v2.0 / GI v3.0 validators to strict immediately. No external consumers, all corpora regenerable | Yes |
| [ADR-102](ADR-102-retro-audit-marker-for-in-place-artifact-mutation.md) | `_retro_audit` marker for in-place artifact mutation | Accepted | [RFC-097](../rfc/RFC-097-unified-kg-gi-ontology-v2.md) | Canonical shape for the audit trail when in-place mutation is the only viable path for published artifacts (eval runs, prod corpora). Required keys + summary file format + migration-script grandfathering | Yes |
| [ADR-103](ADR-103-deterministic-connectivity-under-llm-free-profiles.md) | Deterministic connectivity under LLM-free profiles | Accepted | [RFC-094](../rfc/RFC-094-search-powered-surfaces-query-layer.md) | Closes #1058 — three deterministic post-passes (KG ORG nodes via spaCy NER, GI MENTIONS_ORG via the existing NER pass, corpus-level Topic clustering via sentence-transformers) plus a multi-show pre-seeded fixture so airgapped CI carries real connectivity data without ever calling a cloud LLM | Yes |
| [ADR-104](ADR-104-enrichment-layer-boundary-vs-kg-direct.md) | Enrichment-layer boundary vs KG-direct connectivity | Accepted | [RFC-088](../rfc/RFC-088-enrichment-layer-architecture.md), [RFC-097](../rfc/RFC-097-unified-kg-gi-ontology-v2.md) | Amends RFC-088 §Key Decision #1: "Enrichers never modify core artifacts produced by core pipeline stages." RFC-097 chunk 9's KG-direct `RELATED_TO` is part of the core pipeline (conforms to KG v2.0; runs in `workflow/orchestration.py`); enrichments live under `enrichments/` and serve ranking / scoring / UI consumption. Rubric: traversal → core, ranking → enrichments. Same signal can have two outputs (one each), no redundancy. | Doc-only |
| [ADR-105](ADR-105-response-shape-guardrails-for-self-deployed-services.md) | Response-shape guardrails for self-deployed services | Accepted | — | The self-hosted sibling of ADR-100 (cloud): snapshot envelope diffs on self-hosted service bumps. Renumbered 2026-07-07 from a duplicate ADR-099 (the other ADR-099 is LanceDB-first search). | Yes |
| [ADR-106](ADR-106-transformers-v5-ml-backend-unification.md) | transformers v5 upgrade + ML backend unification | Accepted | [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md) | #382 (`5d504f3d`): pin `transformers>=5.0.0`; collapse the two duplicated HF-seq2seq loading idioms (`summarizer.SummaryModel` + `hybrid_ml_provider.TransformersReduceBackend`) into shared `HFSeq2SeqBackend` + `HFEvidenceBackend`. Retroactive capture. | Yes |
| [ADR-107](ADR-107-ingestion-is-the-pipeline-drop-ingest-primitive.md) | Ingestion is the pipeline — drop the standalone `ingest` primitive | Accepted | [PRD-037](../prd/PRD-037-discovery.md) | #1069: the single-feed pipeline *is* ingestion; drop the `ingest` verb + `IngestPolicy`. Durable outcome — enrichment made a consistent peer of the pipeline across CLI / docker / scheduler / UI / auto-chain. | Yes |
| [ADR-108](ADR-108-nli-disagreement-enrichers-gated-dark.md) | Reimagine the NLI enrichers → `topic_consensus` (activated) + retire stance scoring for a CIL timeline | Accepted | [RFC-088](../rfc/RFC-088-enrichment-layer-architecture.md), [RFC-103](../rfc/RFC-103-momentum-layer.md) | `nli_contradiction`/`stance_disagreement` hit 0% precision. `nli_contradiction`→**`topic_consensus`**, activated at precision 0.91 via a composite of embedding cosine + low NLI contradiction (symmetric entailment failed on real data). Stance-over-time was **retired** as an enricher — its signal is absent on factual insights — and is now a read-time CIL conversation/position timeline coloured by the deterministic `insight_sentiment` (VADER). | Activated + retired |
| [ADR-109](ADR-109-per-episode-quality-telemetry.md) | Per-episode GI quality telemetry — make silent failures loud | Proposed | [ADR-053](ADR-053-grounding-contract-for-evidence-backed-insights.md) | Nine defects in one session all produced plausible output and reported success. GI quality is aggregated per RUN, so 30 broken episodes hide behind a healthy average — exactly how prod-v2 accumulated 125 unusable transcripts unnoticed. Emit per-episode quality + failure flags (`stub_fallback`, `zero_quotes`, `gate_failed_open`, `truncation_salvaged`, `insights_at_ceiling`), a per-episode log line, Prometheus gauges, and a run-level quality summary. Must catch the known failures on 10-20 episodes before the 100-episode reprocess. | Pending |
| [ADR-110](ADR-110-resolve-speaker-identity-after-diarization.md) | Ask who speaks AFTER we can hear them | Proposed | [#876](https://github.com/chipi/podcast_scraper/issues/876), [#1169](https://github.com/chipi/podcast_scraper/issues/1169) | The pipeline asks "who speaks?" **before the audio is even downloaded** — `detect_speakers(title, description, known_hosts)` cannot take a transcript. So an LLM shown only show notes returns the people they *mention* (that is how Elon Musk, named only as the man *suing* OpenAI, became a speaker), and `corroborate_guests` then checks that guess **against the same show notes it was guessed from** — circular. Measured on the prod detector (gemini, 50 eps): **24.0% of talk unattributed**, **70 proposed names deleted, 69 of them whole** — including **Rob Armstrong, co-host of FT Unhedged**, and Planet Money's entire newsroom. Fix: keep metadata detection as a cheap *proposal*; **resolve identity after diarization** against each voice's own turns; **delete positional talk-time painting** (the actual invention mechanism); keep the regex as the no-LLM/airgapped path. | Pending |

## Gap analysis {:#gaps}

**Counts (reconcile when adding ADRs):** **110** files under `docs/adr/ADR-*.md` (ADR-001–ADR-110;
numbering has historical gaps). ADR-099 was a duplicate — the response-shape-guardrails file was
renumbered to ADR-105 (2026-07-07); ADR-106/107/108 added the same day. From the index table: **1** **Proposed** (**ADR-055**, tied to Draft
RFC-053; ADR-104 was promoted to Accepted with RFC-088 chunk 8), **3** **Superseded** (**ADR-056**,
RFC-054 shipped in different shape — per-provider mocks rather than the centralized router this ADR
drafted; **ADR-060** + **ADR-061**, FAISS retired for LanceDB — superseded by ADR-099), **1** **Accepted** with **Code = No**
(**ADR-054**, tied to Draft RFC-051 Postgres projection), **3** **Accepted** with **Code = Partial**
(**ADR-031**, **ADR-047**, **ADR-058**). **Accepted** means ratified, not necessarily shipped.

### When to extract a new ADR

Use an ADR when one or more of these hold; otherwise an **RFC + normative doc** (API guide,
`docs/api/*.md`, UXS) is usually enough.

| ADR type | When to extract | Recent examples |
| --- | --- | --- |
| **Closure / program outcome** | A **large RFC program** ends; you need an immutable summary. | [ADR-073](ADR-073-rfc057-autoresearch-closure.md) closes [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) |
| **Empirical production defaults** | Benchmarks change **default models/tiers** you must freeze for onboarding. | [ADR-067](ADR-067-pegasus-led-retirement-podcast-content.md)–[ADR-072](ADR-072-llama32-3b-as-tier3-local-llm.md) |
| **Stack & ownership boundary** | **Who owns HTTP**, which **frontend stack**, which **UI E2E runner**. | [ADR-064](ADR-064-canonical-server-layer-with-feature-flagged-routes.md)–[ADR-066](ADR-066-playwright-for-ui-e2e-testing.md) |
| **Heavy optional dependencies** | An extra **bloats install** or splits CUDA/CPU paths; defaults must not pay the cost. | [ADR-058](ADR-058-additive-pyannote-diarization-with-separate-extra.md) (amended: bundled in `[ml]`/`[dev]`) |
| **Cross-cutting protocol / contract** | Multiple subsystems share the **same interface**. | [ADR-060](ADR-060-vectorstore-protocol-with-backend-abstraction.md), [ADR-053](ADR-053-grounding-contract-for-evidence-backed-insights.md), [ADR-051](ADR-051-per-episode-json-artifacts-with-logical-union.md) |
| **Process / CI philosophy** | A **policy** decision that outlives one RFC. | [ADR-021](ADR-021-acceptance-test-tier-as-final-ci-gate.md) |

### When **not** to add an ADR

- **Viewer milestones** that do not change stack (e.g. [RFC-069](../rfc/RFC-069-graph-exploration-toolkit.md)) — **RFC + feature UXS (e.g. UXS-004) + UXS-001 hub + E2E map** suffice.
- **Single-route APIs** for the viewer with schema in code + tests (e.g. [RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md)) — Server Guide + tests suffice.
- **Operational tooling** without architectural boundary moves (e.g. [RFC-065](../rfc/RFC-065-live-pipeline-monitor.md)) — **RFC-first**.
- **Frozen artifact workflows** (e.g. [RFC-064](../rfc/RFC-064-performance-profiling-release-freeze.md)); **profile YAML** baselines are covered by **[ADR-075](ADR-075-frozen-yaml-performance-profiles-for-release-baselines.md)**.

### ADRs by implementation state

**Proposed**

| ADR | Primary RFC | Note |
| --- | --- | --- |
| [ADR-055](ADR-055-adaptive-summarization-routing.md) | [RFC-053](../rfc/RFC-053-adaptive-summarization-routing.md) | No episode profiling / routing in pipeline yet |
| [ADR-104](ADR-104-enrichment-layer-boundary-vs-kg-direct.md) | [RFC-088](../rfc/RFC-088-enrichment-layer-architecture.md), [RFC-097](../rfc/RFC-097-unified-kg-gi-ontology-v2.md) | Settles enrichment-layer ↔ KG-direct boundary (RFC-097 chunk 9 `RELATED_TO` is core, not enrichment violation). Doc-only; promoted to Accepted in RFC-088 implementation chunk 8 ([#1110](https://github.com/chipi/podcast_scraper/issues/1110)). |

**Superseded**

| ADR | Primary RFC | Note |
| --- | --- | --- |
| [ADR-056](ADR-056-composable-e2e-mock-response-strategy.md) | [RFC-054](../rfc/RFC-054-e2e-mock-response-strategy.md) | Composable ResponseProfile / Router not built; per-provider mock clients + per-concern unit suites shipped instead. RFC-054 closed via redirect. |

**Accepted, partial**

| ADR | Primary RFC / gap | Note |
| --- | --- | --- |
| [ADR-058](ADR-058-additive-pyannote-diarization-with-separate-extra.md) | [RFC-058](../rfc/RFC-058-audio-speaker-diarization.md) | Core pyannote provider + alignment shipped; diarization result caching deferred as future scope |
| [ADR-031](ADR-031-mandatory-pre-release-validation.md) | [RFC-038](../rfc/RFC-038-continuous-review-tooling.md) | Dependabot + pydeps shipped; pre-release checklist tracked as #255 future enhancement |
| [ADR-047](ADR-047-proactive-metric-regression-alerting.md) | [RFC-043](../rfc/RFC-043-automated-metrics-alerts.md) | Nightly `alerts[]` detection shipped; PR-comment + webhook scripts deliberately abandoned — operator-side Sentry/Grafana wiring is the destination (see [`OBSERVABILITY_EXTENSIONS.md`](../guides/OBSERVABILITY_EXTENSIONS.md#operator-alerting--sentry--grafana)) |

**Accepted, code not landed (expected)**

| ADR | Primary RFC | Note |
| --- | --- | --- |
| [ADR-054](ADR-054-relational-postgres-projection-for-gil-and-kg.md) | [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md) | Postgres projection deferred persistence-layer scope; PRD-017/019 still ship without it |

### Stale-audit corrections (reference)

Trust the **Code** column in the table above: **ADR-048** is implemented; **ADR-062** / **ADR-063**
are **Yes**; **ADR-064**–**ADR-066** are implemented; **ADR-021** is reflected in script-based
`make test-acceptance`.

### Situation cheat sheet

| Situation | Guidance |
| --- | --- |
| **Prefer a new ADR** | Irreversible stack boundary, cross-cutting protocol, frozen empirical default, heavy optional extra, or closure of a large program (e.g. **ADR-073**). |
| **Often RFC-only** | Bounded HTTP routes or viewer tabs where **ADR-064**–**ADR-066** + UXS already fix the stack (e.g. **RFC-067**, **RFC-068**, **RFC-069**, **RFC-071**). Corpus layout + manifest: **ADR-074**. Frozen resource baselines: **ADR-075**. Streamlit vs Vue for eval tools: **ADR-076**. Full-stack Compose + stack-test gate: **ADR-084**, **ADR-085**. CIL + `bridge.json`: **ADR-086**. Autoresearch Track A v2: **ADR-087**. macOS ML `make` safety: **ADR-088**. **Prod failover** design: **[RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md)**; decisions **ADR-089**–**ADR-091**. Corpus **snapshot** backup manifest + restore defaults: **[RFC-084](../rfc/RFC-084-corpus-backup-manifest-and-version-aware-restore.md)**; **ADR-092**. **Cross-surface stack contract vs adapters:** **ADR-093** ([#762](https://github.com/chipi/podcast_scraper/issues/762)). |
| **Proposed ADRs** | Promote **ADR-055** to **Accepted** (or supersede) when **RFC-053** ships end-to-end. **ADR-056** already Superseded (2026-06-26) — the centralized router this ADR drafted was replaced by per-provider mock clients + per-concern unit suites; no further action needed. |

### Future triggers

- **Multi-feed manifest** as an **immutable external contract** beyond [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md) — partially addressed by **ADR-074**.
- **`.pipeline_status.json` schema** if external monitors depend on it and breaking changes need versioning.
- **Profile YAML** for **non-Python** consumers beyond `tools/run_compare` / **`make profile-diff`** — partially addressed by **ADR-075**.
- **RFC-070** + **ADR-060** when platform vector backends land materially.
- **Full-stack Compose, stack-test, CIL or bridge, autoresearch v2, macOS ML process safety** — see **ADR-084**–**ADR-088** (normative detail remains in **RFC-072**, **RFC-073** v2 file, **RFC-074**, **RFC-078**, **RFC-079**).
- **Prod failover (stand up spare, validate, gated cutover)** — **[RFC-083](../rfc/RFC-083-prod-failover-orchestration-and-cutover.md)** (Draft); decisions **ADR-089**–**ADR-091**; GitHub #764.
- **Corpus snapshot tarball metadata + version-aware restore** — **[RFC-084](../rfc/RFC-084-corpus-backup-manifest-and-version-aware-restore.md)** (Completed); **ADR-092**; GitHub #763.

**Open decisions without ADRs:** see **Architecture Decision Candidates** below.

**Related:** [PRD gap analysis](../prd/index.md#gaps), [RFC gap analysis](../rfc/index.md#gaps).

---

## Architecture Decision Candidates

These items have been identified as potential architectural decisions but are currently under review.

| Candidate Decision | Origin | Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Informational-Only Metric Gates** | [RFC-043](../rfc/RFC-043-automated-metrics-alerts.md) | Open | Should regressions (runtime, coverage) block PRs or just notify? |
| **Excel-Based Result Aggregation** | [RFC-015](../rfc/RFC-015-ai-experiment-pipeline.md) | Open | Should we maintain `experiment_results.xlsx` or move fully to web? |
| **Manual vs. Automated Golden Creation** | [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md) | Open | Should golden data creation *always* require manual approval? |
| ~~Diarization-Free Dialogue Formatting~~ | [RFC-006](../rfc/RFC-006-screenplay-formatting.md) | Resolved → [ADR-058](ADR-058-additive-pyannote-diarization-with-separate-extra.md) | Additive pyannote diarization accepted; gap-based rotation preserved as default fallback |
| Minimalist Parser Dependency Strategy | [RFC-002](../rfc/RFC-002-rss-parsing.md) | Open | Raw ElementTree vs. external RSS libraries |
| Two-Phase Configuration Validation | [RFC-007](../rfc/RFC-007-cli-interface.md) | Open | argparse syntax + Pydantic semantic validation |

---

## Creating New ADRs

Use the **[ADR Template](ADR_TEMPLATE.md)** to document new architectural decisions. Decisions
typically originate from an **[RFC](../rfc/index.md)** that has been **reviewed** and often
**Completed** when implementation lands (RFCs use **Completed**, not **Accepted** — **Accepted**
is the ADR status).
