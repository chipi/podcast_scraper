# Architecture Decision Records (ADRs)

## Purpose

Architecture Decision Records (ADRs) capture the **what** and **why** of significant architectural decisions in `podcast_scraper`. While RFCs represent the proposal and journey, ADRs serve as the final, immutable record of truth for the project's architecture.

## How ADRs Work

1. **Immutable Records**: Once an ADR is accepted, it remains unchanged unless superseded by a new ADR.
2. **Context Driven**: They explain the trade-offs and rationale behind a decision.
3. **Reference for Developers**: They provide onboarding context for why certain patterns (like the Provider Protocol) were chosen.

## ADR Index

| ADR | Title | Status | Related RFC | Description |
| :--- | :--- | :--- | :--- | :--- |
| [ADR-001](ADR-001-hybrid-concurrency-strategy.md) | Hybrid Concurrency Strategy | Accepted | [RFC-001](../rfc/RFC-001-workflow-orchestration.md) | IO-bound threading, sequential CPU/GPU tasks |
| [ADR-002](ADR-002-security-first-xml-processing.md) | Security-First XML Processing | Accepted | [RFC-002](../rfc/RFC-002-rss-parsing.md) | Mandated use of defusedxml for RSS parsing |
| [ADR-003](ADR-003-deterministic-feed-storage.md) | Deterministic Feed Storage | Accepted | [RFC-004](../rfc/RFC-004-filesystem-layout.md) | Hash-based output directory derivation |
| [ADR-004](ADR-004-flat-filesystem-archive-layout.md) | Flat Filesystem Archive Layout | Accepted | [RFC-004](../rfc/RFC-004-filesystem-layout.md) | Flat directory structure per feed run |
| [ADR-005](ADR-005-lazy-ml-dependency-loading.md) | Lazy ML Dependency Loading | Accepted | [RFC-005](../rfc/RFC-005-whisper-integration.md) | Function-level imports for heavy ML libraries |
| [ADR-006](ADR-006-context-aware-model-selection.md) | Context-Aware Model Selection | Accepted | [RFC-010](../rfc/RFC-010-speaker-name-detection.md) | Automatic English model promotion (.en) |
| [ADR-007](ADR-007-universal-episode-identity.md) | Universal Episode Identity | Accepted | [RFC-011](../rfc/RFC-011-metadata-generation.md) | GUID-first deterministic episode ID generation |
| [ADR-008](ADR-008-database-agnostic-metadata-schema.md) | Database-Agnostic Metadata Schema | Accepted | [RFC-011](../rfc/RFC-011-metadata-generation.md) | Unified JSON format for SQL/NoSQL |
| [ADR-009](ADR-009-privacy-first-local-summarization.md) | Privacy-First Local Summarization | Accepted | [RFC-012](../rfc/RFC-012-episode-summarization.md) | Local Transformers over Cloud APIs |
| [ADR-010](ADR-010-hierarchical-summarization-pattern.md) | Hierarchical Summarization Pattern | Accepted | [RFC-012](../rfc/RFC-012-episode-summarization.md) | Map-reduce chunking for long transcripts |
| [ADR-011](ADR-011-unified-provider-pattern.md) | Unified Provider Pattern | Accepted | [RFC-029](../rfc/RFC-029-provider-refactoring-consolidation.md) | Type-based unified provider classes |
| [ADR-012](ADR-012-protocol-based-provider-discovery.md) | Protocol-Based Provider Discovery | Accepted | [RFC-021](../rfc/RFC-021-modularization-refactoring-plan.md) | Decoupling via PEP 544 Protocols |
| [ADR-013](ADR-013-technology-based-provider-naming.md) | Technology-Based Provider Naming | Accepted | [RFC-029](../rfc/RFC-029-provider-refactoring-consolidation.md) | Clear library-based option naming |
| [ADR-014](ADR-014-externalized-prompt-management.md) | Externalized Prompt Management | Accepted | [RFC-017](../rfc/RFC-017-prompt-management.md) | Versioned Jinja2 templates in prompts/ |
| [ADR-015](ADR-015-secure-credential-injection.md) | Secure Credential Injection | Accepted | [RFC-013](../rfc/RFC-013-openai-provider-implementation.md) | Environment-based secret management |
| [ADR-016](ADR-016-git-worktree-based-development.md) | Git Worktree-Based Development | Accepted | [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md) | Parallel stable dev environments |
| [ADR-017](ADR-017-stratified-ci-execution.md) | Stratified CI Execution | Accepted | [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md) | Fast push checks vs. full PR validation |
| [ADR-018](ADR-018-isolated-runtime-environments.md) | Isolated Runtime Environments | Accepted | [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md) | Independent venv per worktree |
| [ADR-019](ADR-019-provider-agnostic-preprocessing.md) | Provider-Agnostic Preprocessing | Accepted | [RFC-013](../rfc/RFC-013-openai-provider-implementation.md) | Shared pre-inference cleaning pipeline |
| [ADR-020](ADR-020-linear-history-via-squash-merge.md) | Linear History via Squash-Merge | Accepted | [RFC-039](../rfc/RFC-039-development-workflow-worktrees-ci.md) | Clean, revertible main branch history |
| [ADR-021](ADR-021-standardized-test-pyramid.md) | Standardized Test Pyramid | Accepted | [RFC-018](../rfc/RFC-018-test-structure-reorganization.md) | Strict unit/integration/e2e tiering |
| [ADR-022](ADR-022-flaky-test-defense.md) | Flaky Test Defense | Accepted | [RFC-025](../rfc/RFC-025-test-metrics-and-health-tracking.md) | Automated retries and health reporting |
| [ADR-023](ADR-023-public-operational-metrics.md) | Public Operational Metrics | Accepted | [RFC-026](../rfc/RFC-026-metrics-consumption-and-dashboards.md) | Transparency via GitHub Pages dashboards |
| [ADR-024](ADR-024-standalone-experiment-configuration.md) | Standalone Experiment Configuration | Accepted | [RFC-015](../rfc/RFC-015-ai-experiment-pipeline.md) | Separation of research params from code |
| [ADR-025](ADR-025-codified-comparison-baselines.md) | Codified Comparison Baselines | Accepted | [RFC-015](../rfc/RFC-015-ai-experiment-pipeline.md) | Objective delta measurement vs baseline artifacts |
| [ADR-026](ADR-026-explicit-golden-dataset-versioning.md) | Explicit Golden Dataset Versioning | Accepted | [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md) | Approved, frozen ground truth data versions |
| [ADR-027](ADR-027-deep-provider-fingerprinting.md) | Deep Provider Fingerprinting | Accepted | [RFC-016](../rfc/RFC-016-modularization-for-ai-experiments.md) | Hardware and environment tracking for reproducibility |
| [ADR-028](ADR-028-typed-provider-parameter-models.md) | Typed Provider Parameter Models | Accepted | [RFC-016](../rfc/RFC-016-modularization-for-ai-experiments.md) | Pydantic validation for backend parameters |
| [ADR-029](ADR-029-registered-preprocessing-profiles.md) | Registered Preprocessing Profiles | Accepted | [RFC-016](../rfc/RFC-016-modularization-for-ai-experiments.md) | Versioned cleaning logic tracking |
| [ADR-030](ADR-030-multi-tiered-benchmarking-strategy.md) | Multi-Tiered Benchmarking Strategy | Accepted | [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md) | Fast PR smoke tests vs nightly full benchmarks |
| [ADR-031](ADR-031-heuristic-based-quality-gates.md) | Heuristic-Based Quality Gates | Accepted | [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md) | Regex-based detection of common AI failure modes |
| [ADR-032](ADR-032-standardized-pre-provider-audio-stage.md) | Standardized Pre-Provider Audio Stage | Accepted | [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md) | Mandatory optimization before any transcription |
| [ADR-033](ADR-033-content-hash-based-audio-caching.md) | Content-Hash Based Audio Caching | Accepted | [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md) | Shared optimized artifacts in .cache/ |
| [ADR-034](ADR-034-ffmpeg-first-audio-manipulation.md) | FFmpeg-First Audio Manipulation | Accepted | [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md) | System-level performance for audio pipelines |
| [ADR-035](ADR-035-speech-optimized-codec-opus.md) | Speech-Optimized Codec (Opus) | Accepted | [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md) | Opus at 24kbps for intermediate artifacts |
| [ADR-036](ADR-036-hybrid-map-reduce-summarization.md) | Hybrid MAP-REDUCE Summarization | Accepted | [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md) | Compression (Classic) + Abstraction (Instruct LLM) |
| [ADR-037](ADR-037-local-llm-backend-abstraction.md) | Local LLM Backend Abstraction | Accepted | [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md) | Support for llama.cpp, ollama, and transformers |
| [ADR-038](ADR-038-strict-reduce-prompt-contract.md) | Strict REDUCE Prompt Contract | Accepted | [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md) | Mandatory markdown structure for LLM outputs |
| [ADR-039](ADR-039-grouped-dependency-automation.md) | Grouped Dependency Automation | Accepted | [RFC-038](../rfc/RFC-038-continuous-review-tooling.md) | Balanced Dependabot updates via grouping |
| [ADR-040](ADR-040-periodic-module-coupling-analysis.md) | Periodic Module Coupling Analysis | Accepted | [RFC-038](../rfc/RFC-038-continuous-review-tooling.md) | Nightly visualization of architecture health |
| [ADR-041](ADR-041-mandatory-pre-release-validation.md) | Mandatory Pre-Release Validation | Accepted | [RFC-038](../rfc/RFC-038-continuous-review-tooling.md) | Standardized checklist script for releases |
| [ADR-042](ADR-042-proactive-metric-regression-alerting.md) | Proactive Metric Regression Alerting | Accepted | [RFC-043](../rfc/RFC-043-automated-metrics-alerts.md) | Automated PR comments and webhook notifications |
| [ADR-043](ADR-043-unified-provider-metrics-contract.md) | Unified Provider Metrics Contract | Accepted | - | Standardized `ProviderCallMetrics` pattern for all providers |
| [ADR-044](ADR-044-unified-retry-policy-with-metrics.md) | Unified Retry Policy with Metrics | Accepted | - | Centralized retry logic with exponential backoff and metrics tracking |
| [ADR-045](ADR-045-composable-e2e-mock-response-strategy.md) | Composable E2E Mock Response Strategy | Proposed | [RFC-054](../rfc/RFC-054-e2e-mock-response-strategy.md) | Separation of functional responses from non-functional behavior in tests |
| [ADR-046](ADR-046-adaptive-summarization-routing.md) | Adaptive Summarization Routing | Proposed | [RFC-053](../rfc/RFC-053-adaptive-summarization-routing.md) | Rule-based routing with episode profiling for summarization strategies |
| [ADR-047](ADR-047-centralized-model-registry.md) | Centralized Model Registry | Proposed | [RFC-044](../rfc/RFC-044-model-registry.md) | Single source of truth for model architecture limits |

---

## Architecture Decision Candidates

These items have been identified as potential architectural decisions but are currently under review.

| Candidate Decision | Origin | Status | Description |
| :--- | :--- | :--- | :--- |
| **Informational-Only Metric Gates** | [RFC-043](../rfc/RFC-043-automated-metrics-alerts.md) | Open | Should regressions (runtime, coverage) block PRs or just notify? |
| **Excel-Based Result Aggregation** | [RFC-015](../rfc/RFC-015-ai-experiment-pipeline.md) | Open | Should we maintain `experiment_results.xlsx` or move fully to web? |
| **Manual vs. Automated Golden Creation** | [RFC-041](../rfc/RFC-041-podcast-ml-benchmarking-framework.md) | Open | Should golden data creation *always* require manual approval? |
| Diarization-Free Dialogue Formatting | [RFC-006](../rfc/RFC-006-screenplay-formatting.md) | Open | Gap-based speaker rotation vs. full diarization |
| Minimalist Parser Dependency Strategy | [RFC-002](../rfc/RFC-002-rss-parsing.md) | Open | Raw ElementTree vs. external RSS libraries |
| Two-Phase Configuration Validation | [RFC-007](../rfc/RFC-007-cli-interface.md) | Open | argparse syntax + Pydantic semantic validation |

---

## Creating New ADRs

Use the **[ADR Template](ADR_TEMPLATE.md)** to document new architectural decisions. Decisions typically originate from an **[RFC](../rfc/index.md)** that has been accepted and implemented.
