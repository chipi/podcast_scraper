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
| [RFC-013](RFC-013-openai-provider-implementation.md) | OpenAI Provider Implementation | PRD-006 | - | Technical design for OpenAI API providers |
| [RFC-017](RFC-017-prompt-management.md) | Prompt Management | PRD-007 | - | Versioned, parameterized prompt management system |
| [RFC-018](RFC-018-test-structure-reorganization.md) | Test Structure Reorganization | - | - | Reorganize test suite into unit/integration/e2e directories |
| [RFC-019](RFC-019-e2e-test-improvements.md) | E2E Test Infrastructure and Coverage Improvements | PRD-001+ | - | Comprehensive E2E test infrastructure and coverage |
| [RFC-020](RFC-020-integration-test-improvements.md) | Integration Test Infrastructure and Coverage Improvements | PRD-001+ | - | Integration test suite improvements (10 stages, 182 tests) |
| [RFC-024](RFC-024-test-execution-optimization.md) | Test Execution Optimization | - | - | Optimize test execution with markers, tiers, parallel execution |
| [RFC-025](RFC-025-test-metrics-and-health-tracking.md) | Test Metrics and Health Tracking | - | - | Metrics collection, CI integration, flaky test detection |
| [RFC-028](RFC-028-ml-model-preloading-and-caching.md) | ML Model Preloading and Caching | - | - | Model preloading for local dev and GitHub Actions caching |
| [RFC-029](RFC-029-provider-refactoring-consolidation.md) | Provider Refactoring Consolidation | PRD-006 | - | Unified provider architecture documentation |
| [RFC-030](RFC-030-python-test-coverage-improvements.md) | Python Test Coverage Improvements | - | - | Coverage collection in CI, threshold enforcement |
| [RFC-021](RFC-021-modularization-refactoring-plan.md) | Modularization Refactoring Plan | PRD-006 | - | Detailed plan for modular provider architecture |
| [RFC-022](RFC-022-environment-variable-candidates-analysis.md) | Environment Variable Candidates Analysis | - | - | Environment variable support for deployment flexibility |
| [RFC-026](RFC-026-metrics-consumption-and-dashboards.md) | Metrics Consumption and Dashboards | - | - | GitHub Pages metrics JSON API and job summaries |
| [RFC-031](RFC-031-code-complexity-analysis-tooling.md) | Code Complexity Analysis Tooling | - | - | Code complexity, dead code, docstrings, spell checking |

## Quick Links

- **[PRDs](../prd/index.md)** - Product requirements documents
- **[Architecture](../ARCHITECTURE.md)** - System design and module responsibilities
- **[Releases](../releases/)** - Release notes and version history
