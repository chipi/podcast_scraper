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
| [RFC-015](RFC-015-ai-experiment-pipeline.md) | AI Experiment Pipeline | PRD-007 | Technical design for configuration-driven experiment pipeline |
| [RFC-016](RFC-016-modularization-for-ai-experiments.md) | Modularization for AI Experiments | PRD-007 | Provider system architecture to support AI experiment pipeline |
| [RFC-023](RFC-023-readme-acceptance-tests.md) | README Acceptance Tests | - | Acceptance tests validating README documentation accuracy |
| [RFC-027](RFC-027-pipeline-metrics-improvements.md) | Pipeline Metrics Improvements | - | Improvements to pipeline metrics collection and reporting |
| [RFC-032](RFC-032-anthropic-provider-implementation.md) | Anthropic Provider Implementation | PRD-009 | Technical design for Anthropic Claude API providers |
| [RFC-033](RFC-033-mistral-provider-implementation.md) | Mistral Provider Implementation | PRD-010 | Technical design for Mistral AI providers (all 3 capabilities) |
| [RFC-034](RFC-034-deepseek-provider-implementation.md) | DeepSeek Provider Implementation | PRD-011 | Technical design for DeepSeek AI (ultra low-cost) |
| [RFC-035](RFC-035-gemini-provider-implementation.md) | Gemini Provider Implementation | PRD-012 | Technical design for Google Gemini (2M context) |
| [RFC-036](RFC-036-groq-provider-implementation.md) | Groq Provider Implementation | PRD-013 | Technical design for Groq (ultra-fast) |
| [RFC-037](RFC-037-ollama-provider-implementation.md) | Ollama Provider Implementation | PRD-014 | Technical design for Ollama (local/offline) |
| [RFC-038](RFC-038-continuous-review-tooling.md) | Continuous Review Tooling | #45 | Dependabot, pydeps, pre-release checklist |
| [RFC-040](RFC-040-audio-preprocessing-pipeline.md) | Audio Preprocessing Pipeline | - | Optional audio preprocessing (VAD, normalization) before transcription |
| [RFC-041](RFC-041-podcast-ml-benchmarking-framework.md) | Podcast ML Benchmarking Framework | PRD-007 | Repeatable, objective ML benchmarking system |
| [RFC-042](RFC-042-hybrid-summarization-pipeline.md) | Hybrid Podcast Summarization Pipeline | - | Hybrid MAP-REDUCE with instruction-tuned LLMs (v2.5) |
| [RFC-043](RFC-043-automated-metrics-alerts.md) | Automated Metrics Alerts | - | Automated regression alerts and PR comments for pipeline metrics |
| [RFC-044](RFC-044-model-registry.md) | Model Registry for Architecture Limits | - | Centralized registry to eliminate hardcoded model limits throughout codebase |
| [RFC-045](RFC-045-ml-model-optimization-guide.md) | ML Model Optimization Guide | PRD-005, PRD-007 | Comprehensive guide for maximizing ML quality via preprocessing and parameter tuning |
| [RFC-046](RFC-046-materialization-architecture.md) | Materialization Architecture | PRD-007 | Shift preprocessing from run parameter to dataset materialization for honest comparisons |
| [RFC-047](RFC-047-run-comparison-visual-tool.md) | Lightweight Run Comparison & Diagnostics Tool | PRD-007 | Fast, one-page visual tool for comparing runs and diagnosing regressions |
| [RFC-048](RFC-048-evaluation-application-alignment.md) | Evaluation ↔ Application Tightening & Alignment | PRD-007 | Alignment rules ensuring evaluation results are representative of application behavior |

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
| [RFC-039](RFC-039-development-workflow-worktrees-ci.md) | Development Workflow | - | v2.4.0 | Git worktrees, Cursor integration, CI evolution |

## Quick Links

- **[PRDs](../prd/index.md)** - Product requirements documents
- **[Architecture](../ARCHITECTURE.md)** - System design and module responsibilities
- **[Releases](../releases/index.md)** - Release notes and version history

---

## Creating New RFCs

Use the **[RFC Template](RFC_TEMPLATE.md)** as a starting point for new technical design documents.
