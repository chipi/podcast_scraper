# Product Requirements Documents (PRDs)

## Purpose

Product Requirements Documents (PRDs) define the **what** and **why** behind each major feature or capability in `podcast_scraper`. They capture:

- **User needs** and use cases
- **Functional requirements** and success criteria
- **Design considerations** and constraints
- **Integration points** with existing features

PRDs serve as the foundation for technical design (RFCs) and help ensure features align with user needs and project goals.

## How PRDs Work

1. **Define Intent**: PRDs describe the problem to solve and desired outcomes
2. **Guide Design**: RFCs reference PRDs to ensure technical solutions meet requirements
3. **Track Implementation**: Release notes reference PRDs to show what was delivered
4. **Document Evolution**: PRDs capture design decisions and rationale

## Open PRDs

| PRD | Title | Related RFCs | Description |
| --- | ----- | ------------ | ----------- |
| [PRD-007](PRD-007-ai-experiment-pipeline.md) | AI Experiment Pipeline | RFC-015 | Configuration-driven experiment pipeline |

## Completed PRDs

| PRD | Title | Version | Related RFCs | Description |
| --- | ----- | ------- | ------------ | ----------- |
| [PRD-001](PRD-001-transcript-pipeline.md) | Transcript Acquisition Pipeline | v2.0.0 | RFC-001, 002, 003, 004, 008, 009 | Core pipeline for downloading transcripts |
| [PRD-002](PRD-002-whisper-fallback.md) | Whisper Fallback Transcription | v2.0.0 | RFC-004, 005, 006, 008, 010 | Automatic transcription fallback |
| [PRD-003](PRD-003-user-interface-config.md) | User Interfaces & Configuration | v2.0.0 | RFC-007, 008, 009 | CLI interface and configuration |
| [PRD-004](PRD-004-metadata-generation.md) | Per-Episode Metadata Generation | v2.2.0 | RFC-011, 012 | Structured metadata documents |
| [PRD-005](PRD-005-episode-summarization.md) | Episode Summarization | v2.3.0 | RFC-012 | Automatic summary generation |
| [PRD-006](PRD-006-openai-provider-integration.md) | OpenAI Provider Integration | v2.4.0 | RFC-013, 017, 021, 022, 029 | OpenAI API as optional provider |
| [PRD-008](PRD-008-speaker-name-detection.md) | Automatic Speaker Name Detection | v2.1.0 | RFC-010 | Auto-detect host/guest names via NER |

## Quick Links

- **[Architecture](../ARCHITECTURE.md)** - System design and module responsibilities
- **[RFCs](../rfc/index.md)** - Technical design documents
- **[Releases](../releases/)** - Release notes and version history
