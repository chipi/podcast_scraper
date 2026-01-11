# ADR-010: Hierarchical Summarization Pattern

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-012](../rfc/RFC-012-episode-summarization.md)
- **Related PRDs**: [PRD-005](../prd/PRD-005-episode-summarization.md)

## Context & Problem Statement

Most local transformer models (like BART) have a strict context window of 1024 tokens. Podcast transcripts often exceed 10,000 tokens, making direct summarization impossible.

## Decision

We implement a **Hierarchical Summarization Pattern** (Map-Reduce):

1. **Map**: Split the transcript into semantic chunks that fit within the model's context window.
2. **Summarize**: Summarize each chunk individually.
3. **Reduce**: Combine the chunk summaries and summarize them again into a final episode abstract.

## Rationale

- **Feasibility**: Allows processing transcripts of any length on hardware with limited VRAM.
- **Context Preservation**: Ensures that details from the beginning, middle, and end of a long podcast are all reflected in the final summary.

## Alternatives Considered

1. **Truncation**: Summarizing only the first 1024 tokens; rejected as it misses the bulk of the content.
2. **Long-Context Models (LED)**: Supported as an alternative, but the hierarchical pattern remains the "safe" default for standard models.

## Consequences

- **Positive**: Handles 2+ hour episodes reliably.
- **Negative**: Slower than single-pass summarization; requires careful management of "summary of summaries" to avoid losing detail.

## References

- [RFC-012: Episode Summarization Using Local Transformers](../rfc/RFC-012-episode-summarization.md)
