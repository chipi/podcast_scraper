# ADR-011: Unified Provider Pattern

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-029](../rfc/RFC-029-provider-refactoring-consolidation.md)
- **Related PRDs**: [PRD-006](../prd/PRD-006-openai-provider-integration.md)

## Context & Problem Statement

Initially, each AI capability (transcription, speaker detection, summarization) had separate provider classes for local and OpenAI backends. This led to state duplication, inconsistent patterns for model loading, and complex factory logic that had to instantiate 6+ different classes.

## Decision

We adopt a **Unified Provider Pattern**. Instead of capability-based classes, we use type-based classes:

1. **`MLProvider`**: A single class that implements all three protocols (`TranscriptionProvider`, `SpeakerDetector`, `SummarizationProvider`) using local libraries (Whisper, spaCy, Transformers).
2. **`OpenAIProvider`**: A single class that implements all three protocols using the OpenAI API.

## Rationale

- **Efficiency**: Shares underlying resources (like the OpenAI client or local model loaders) across different tasks.
- **Consistency**: Standardizes the lifecycle (initialize, execute, cleanup) for all AI interactions.
- **Simplicity**: Dramatically reduces the number of classes and imports required in the core workflow.

## Alternatives Considered

1. **Capability-Specific Classes**: Rejected due to state duplication and maintenance overhead.
2. **Mixin Pattern**: Rejected as it obscured the lifecycle of shared resources like GPU memory.

## Consequences

- **Positive**: Reduced code volume; cleaner factories; simplified test mocking (mock one class instead of three).
- **Negative**: The provider classes are larger and have multiple responsibilities (though clearly partitioned by protocol).

## References

- [RFC-029: Provider Refactoring Consolidation](../rfc/RFC-029-provider-refactoring-consolidation.md)
