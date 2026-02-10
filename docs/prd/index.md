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
| [PRD-007](PRD-007-ai-quality-experiment-platform.md) | AI Quality & Experimentation Platform | RFC-015, 016, 041 | Integrated platform for experimentation and benchmarking |
| [PRD-015](PRD-015-engineering-governance-productivity.md) | Engineering Governance & Productivity Platform | RFC-018-024, 030, 031, 038, 039 | Integrated system for developer velocity and quality |
| [PRD-016](PRD-016-operational-observability-pipeline-intelligence.md) | Operational Observability & Pipeline Intelligence | RFC-025, 026, 027 | System for managing operational health and visibility |
| [PRD-017](PRD-017-grounded-insight-layer.md) | Grounded Insight Layer (GIL) | RFC-049, 050, 051 | Evidence-backed insights and quotes with grounding relationships |
| [PRD-018](PRD-018-database-projection-grounded-insight-layer.md) | Database Projection for GIL | RFC-049, 050, 051 | Fast, queryable database projection of GIL data |

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
| [PRD-009](PRD-009-anthropic-provider-integration.md) | Anthropic Provider Integration | v2.4.0 | RFC-032 | Anthropic Claude API as optional provider |
| [PRD-010](PRD-010-mistral-provider-integration.md) | Mistral Provider Integration | v2.5.0 | RFC-033 | Mistral AI as complete OpenAI alternative |
| [PRD-011](PRD-011-deepseek-provider-integration.md) | DeepSeek Provider Integration | v2.5.0 | RFC-034 | DeepSeek AI - ultra low-cost provider |
| [PRD-012](PRD-012-gemini-provider-integration.md) | Google Gemini Provider Integration | v2.5.0 | RFC-035 | Google Gemini - 2M context, native audio |
| [PRD-013](PRD-013-grok-provider-integration.md) | Grok Provider Integration (xAI) | v2.5.0 | RFC-036 | Grok - xAI's AI model with real-time information access |
| [PRD-014](PRD-014-ollama-provider-integration.md) | Ollama Provider Integration | v2.5.0 | RFC-037 | Ollama - fully local/offline, zero cost |

## Quick Links

- **[Architecture](../ARCHITECTURE.md)** - System design and module responsibilities
- **[RFCs](../rfc/index.md)** - Technical design documents
- **[Releases](../releases/index.md)** - Release notes and version history

---

## Creating New PRDs

Use the **[PRD Template](PRD_TEMPLATE.md)** as a starting point for new product requirements documents.
