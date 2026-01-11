# ADR-013: Technology-Based Provider Naming

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-029](../rfc/RFC-029-provider-refactoring-consolidation.md)

## Context & Problem Statement

Provider selection names were inconsistent (e.g., `ner` for technique, `openai` for company, `local` for location). This made the configuration API ambiguous for users.

## Decision

We standardize on **Technology-Based Naming** for all provider options:

- **`whisper`** (instead of `local_transcription`)
- **`spacy`** (instead of `ner`)
- **`transformers`** (instead of `local_summarization`)
- **`openai`** (remains the same as it identifies the technology stack)

## Rationale

- **Clarity**: Users immediately know which library or API is being used.
- **Consistency**: All options follow the same pattern (Library/Stack Name).
- **Extensibility**: It is clearer how to name new options (e.g., `ollama`, `anthropic`, `deepgram`).

## Alternatives Considered

1. **Location-Based (Local vs Cloud)**: Rejected as it's too vague; multiple local technologies might exist.

## Consequences

- **Positive**: Improved documentation clarity; more intuitive CLI flags.
- **Negative**: Requires a migration period with aliases for the old names (`ner`, `local`).

## References

- [RFC-029: Provider Refactoring Consolidation](../rfc/RFC-029-provider-refactoring-consolidation.md)
