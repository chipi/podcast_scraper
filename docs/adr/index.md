# Architecture Decision Records (ADRs)

## Purpose

Architecture Decision Records (ADRs) capture the **what** and **why** of significant architectural decisions in `podcast_scraper`. While RFCs represent the proposal and journey, ADRs serve as the final, immutable record of truth for the project's architecture.

## How ADRs Work

1. **Immutable Records**: Once an ADR is accepted, it remains unchanged unless superseded by a new ADR.
2. **Context Driven**: They explain the trade-offs and rationale behind a decision.
3. **Reference for Developers**: They provide onboarding context for why certain patterns (like the Provider Protocol) were chosen.

## ADR Index

**Code** (last column): **Yes** = reflected in the codebase; **Partial** = incomplete or
still rolling out; **No** = not started (including accepted ADRs waiting on implementation,
and **Proposed** ADRs).

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
| [ADR-056](ADR-056-composable-e2e-mock-response-strategy.md) | Composable E2E Mock Response Strategy | Proposed | [RFC-054](../rfc/RFC-054-e2e-mock-response-strategy.md) | Separation of functional responses from non-functional behavior in tests | No |
| [ADR-057](ADR-057-autoresearch-thin-harness-with-credential-isolation.md) | AutoResearch Thin Harness with Credential Isolation | Accepted | [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) | Thin control layer reusing existing eval; immutable score.py; AUTORESEARCH\_\* credential vars | Yes |
| [ADR-058](ADR-058-additive-pyannote-diarization-with-separate-extra.md) | Additive pyannote Diarization with Separate `[diarize]` Extra | Accepted | [RFC-058](../rfc/RFC-058-audio-speaker-diarization.md) | pyannote as additive second pass; segment-level; separate \[diarize\] dependency group | No |
| [ADR-059](ADR-059-confidence-scored-multi-signal-commercial-detection.md) | Confidence-Scored Multi-Signal Commercial Detection | Accepted | [RFC-060](../rfc/RFC-060-diarization-aware-commercial-cleaning.md) | Confidence-scored candidates replace binary detection; pattern primary, diarization adjusts | No |
| [ADR-060](ADR-060-vectorstore-protocol-with-backend-abstraction.md) | VectorStore Protocol with Backend Abstraction | Accepted | [RFC-061](../rfc/RFC-061-semantic-corpus-search.md) | PEP 544 protocol decoupling FAISS (Phase 1) from Qdrant (Phase 2) | Yes |
| [ADR-061](ADR-061-faiss-phase-1-with-post-filter-metadata.md) | FAISS Phase 1 with Post-Filter Metadata Strategy | Accepted | [RFC-061](../rfc/RFC-061-semantic-corpus-search.md) | Over-fetch + post-filter for CLI-scale; auto index type selection | Yes |
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

## Gap analysis {:#gaps}

**Counts (reconcile when adding ADRs):** **78** files under `docs/adr/ADR-*.md` (ADR-001–ADR-078;
numbering has historical gaps). From the index table: **2** **Proposed** (**ADR-055**, **ADR-056**),
**3** **Accepted** with **Code = No** (**ADR-054**, **ADR-058**, **ADR-059**), **2** **Accepted** with
**Code = Partial** (**ADR-031**, **ADR-047**). **Accepted** means ratified, not necessarily shipped.

### When to extract a new ADR

Use an ADR when one or more of these hold; otherwise an **RFC + normative doc** (API guide,
`docs/api/*.md`, UXS) is usually enough.

| ADR type | When to extract | Recent examples |
| --- | --- | --- |
| **Closure / program outcome** | A **large RFC program** ends; you need an immutable summary. | [ADR-073](ADR-073-rfc057-autoresearch-closure.md) closes [RFC-057](../rfc/RFC-057-autoresearch-optimization-loop.md) |
| **Empirical production defaults** | Benchmarks change **default models/tiers** you must freeze for onboarding. | [ADR-067](ADR-067-pegasus-led-retirement-podcast-content.md)–[ADR-072](ADR-072-llama32-3b-as-tier3-local-llm.md) |
| **Stack & ownership boundary** | **Who owns HTTP**, which **frontend stack**, which **UI E2E runner**. | [ADR-064](ADR-064-canonical-server-layer-with-feature-flagged-routes.md)–[ADR-066](ADR-066-playwright-for-ui-e2e-testing.md) |
| **Heavy optional dependencies** | An extra **bloats install** or splits CUDA/CPU paths; defaults must not pay the cost. | [ADR-058](ADR-058-additive-pyannote-diarization-with-separate-extra.md) (**accepted; `[diarize]` not landed**) |
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
| [ADR-056](ADR-056-composable-e2e-mock-response-strategy.md) | [RFC-054](../rfc/RFC-054-e2e-mock-response-strategy.md) | Composable ResponseProfile / Router not implemented |

**Accepted, code not landed (expected)**

| ADR | Primary RFC | Note |
| --- | --- | --- |
| [ADR-054](ADR-054-relational-postgres-projection-for-gil-and-kg.md) | [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md) | Postgres projection future |
| [ADR-058](ADR-058-additive-pyannote-diarization-with-separate-extra.md) | [RFC-058](../rfc/RFC-058-audio-speaker-diarization.md) | No `[diarize]` extra in `pyproject.toml` yet |
| [ADR-059](ADR-059-confidence-scored-multi-signal-commercial-detection.md) | [RFC-060](../rfc/RFC-060-diarization-aware-commercial-cleaning.md) | Commercial detector as designed not landed |

**Accepted, partial**

| ADR | Gap |
| --- | --- |
| [ADR-031](ADR-031-mandatory-pre-release-validation.md) | `make pre-release` / checklist not fully aligned with [RFC-038](../rfc/RFC-038-continuous-review-tooling.md) |
| [ADR-047](ADR-047-proactive-metric-regression-alerting.md) | Alerts exist; **automated PR comments** not complete |

### Stale-audit corrections (reference)

Trust the **Code** column in the table above: **ADR-048** is implemented; **ADR-062** / **ADR-063**
are **Yes**; **ADR-064**–**ADR-066** are implemented; **ADR-021** is reflected in script-based
`make test-acceptance`.

### Situation cheat sheet

| Situation | Guidance |
| --- | --- |
| **Prefer a new ADR** | Irreversible stack boundary, cross-cutting protocol, frozen empirical default, heavy optional extra, or closure of a large program (e.g. **ADR-073**). |
| **Often RFC-only** | Bounded HTTP routes or viewer tabs where **ADR-064**–**ADR-066** + UXS already fix the stack (e.g. **RFC-067**, **RFC-068**, **RFC-069**, **RFC-071**). Corpus layout + manifest: **ADR-074**. Frozen resource baselines: **ADR-075**. Streamlit vs Vue for eval tools: **ADR-076**. |
| **Proposed ADRs** | Promote **ADR-055** / **ADR-056** to **Accepted** (or supersede) when **RFC-053** / **RFC-054** ship end-to-end. |

### Future triggers

- **Multi-feed manifest** as an **immutable external contract** beyond [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md) — partially addressed by **ADR-074**.
- **`.pipeline_status.json` schema** if external monitors depend on it and breaking changes need versioning.
- **Profile YAML** for **non-Python** consumers beyond `tools/run_compare` / **`make profile-diff`** — partially addressed by **ADR-075**.
- **RFC-070** + **ADR-060** when platform vector backends land materially.

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
