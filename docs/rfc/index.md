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

## RFC Index

| RFC | Title | Related PRD | Version | Status | Description |
| --- | ----- | ----------- | ------- | ------ | ----------- |
| [RFC-001](RFC-001-workflow-orchestration.md) | Workflow Orchestration | PRD-001 | v2.0.0 | ✅ Accepted | Central orchestrator for transcript acquisition pipeline |
| [RFC-002](RFC-002-rss-parsing.md) | RSS Parsing & Episode Modeling | PRD-001 | v2.0.0 | ✅ Accepted | RSS feed parsing and episode data model |
| [RFC-003](RFC-003-transcript-downloads.md) | Transcript Download Processing | PRD-001 | v2.0.0 | ✅ Accepted | Resilient transcript download with retry logic |
| [RFC-004](RFC-004-filesystem-layout.md) | Filesystem Layout & Run Management | PRD-001 | v2.0.0 | ✅ Accepted | Deterministic output directory structure and run scoping |
| [RFC-005](RFC-005-whisper-integration.md) | Whisper Integration Lifecycle | PRD-002 | v2.0.0 | ✅ Accepted | Whisper model loading, transcription, and cleanup |
| [RFC-006](RFC-006-screenplay-formatting.md) | Whisper Screenplay Formatting | PRD-002 | v2.0.0 | ✅ Accepted | Speaker-attributed transcript formatting |
| [RFC-007](RFC-007-cli-interface.md) | CLI Interface & Validation | PRD-003 | v2.0.0 | ✅ Accepted | Command-line argument parsing and validation |
| [RFC-008](RFC-008-config-model.md) | Configuration Model & Validation | PRD-003 | v2.0.0 | ✅ Accepted | Pydantic-based configuration with file loading |
| [RFC-009](RFC-009-progress-integration.md) | Progress Reporting Integration | PRD-001 | v2.0.0 | ✅ Accepted | Pluggable progress reporting interface |
| [RFC-010](RFC-010-speaker-name-detection.md) | Automatic Speaker Name Detection | - | v2.1.0 | ✅ Accepted | NER-based host and guest identification |
| [RFC-011](RFC-011-metadata-generation.md) | Per-Episode Metadata Generation | PRD-004 | v2.2.0 | ✅ Accepted | Structured metadata document generation |
| [RFC-012](RFC-012-episode-summarization.md) | Episode Summarization Using Local Transformers | PRD-005 | v2.3.0 | ✅ Accepted | Local transformer-based summarization |
| [RFC-013](RFC-013-openai-provider-implementation.md) | OpenAI Provider Implementation | PRD-006 | - | 📋 Draft | Technical design for OpenAI API providers (speaker detection, transcription, summarization) |
| [RFC-015](RFC-015-ai-experiment-pipeline.md) | AI Experiment Pipeline | PRD-007 | - | 📋 Draft | Technical design for configuration-driven experiment pipeline |
| [RFC-016](RFC-016-modularization-for-ai-experiments.md) | Modularization for AI Experiments | PRD-007 | - | 📋 Draft | Provider system architecture to support AI experiment pipeline |
| [RFC-017](RFC-017-prompt-management.md) | Prompt Management | PRD-007 | - | 📋 Draft | Versioned, parameterized prompt management system |

## Quick Links

- **[PRDs](../prd/index.md)** - Product requirements documents
- **[Architecture](../ARCHITECTURE.md)** - System design and module responsibilities
- **[Releases](../releases/)** - Release notes and version history
