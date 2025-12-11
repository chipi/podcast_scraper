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

## PRD Index

| PRD | Title | Version | Status | Description |
| --- | ----- | ------- | ------ | ----------- |
| [PRD-001](PRD-001-transcript-pipeline.md) | Transcript Acquisition Pipeline | v2.0.0 | ✅ Implemented | Core pipeline for downloading published transcripts from RSS feeds |
| [PRD-002](PRD-002-whisper-fallback.md) | Whisper Fallback Transcription | v2.0.0 | ✅ Implemented | Automatic transcription fallback when episodes lack published transcripts |
| [PRD-003](PRD-003-user-interface-config.md) | User Interfaces & Configuration | v2.0.0 | ✅ Implemented | CLI interface and configuration file support (JSON/YAML) |
| [PRD-004](PRD-004-metadata-generation.md) | Per-Episode Metadata Generation | v2.2.0 | ✅ Implemented | Structured metadata documents (JSON/YAML) for database ingestion and search |
| [PRD-005](PRD-005-episode-summarization.md) | Episode Summarization | v2.3.0 | ✅ Implemented | Automatic summary and key takeaways generation using local transformer models |
| [PRD-006](PRD-006-openai-provider-integration.md) | OpenAI Provider Integration | - | 📋 Draft | Add OpenAI API as optional provider for speaker detection, transcription, and summarization |
| [PRD-007](PRD-007-ai-experiment-pipeline.md) | AI Experiment Pipeline | - | 📋 Draft | Configuration-driven experiment pipeline for rapid iteration on models, prompts, and parameters |

## Quick Links

- **[Architecture](../ARCHITECTURE.md)** - System design and module responsibilities
- **[RFCs](../rfc/index.md)** - Technical design documents
- **[Releases](../releases/)** - Release notes and version history
