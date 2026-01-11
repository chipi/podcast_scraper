# ADR-019: Provider-Agnostic Preprocessing

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-013](../rfc/RFC-013-openai-provider-implementation.md)

## Context & Problem Statement

Transcript cleaning (timestamp removal, sponsor block filtering) was originally implemented inside the summarization logic. Adding a second provider (OpenAI) would have meant duplicating this complex regex logic or risk inconsistent inputs.

## Decision

We move all cleaning and sanitization into a **Shared Preprocessing Module**.

- Cleaning happens *once* in the `metadata` or `workflow` layer.
- The "clean" text is then passed to the selected provider (`MLProvider` or `OpenAIProvider`).

## Rationale

- **Single Source of Truth**: Changes to how we detect "Sponsor Blocks" only need to be made in one place.
- **Efficiency**: Avoids redundant processing if multiple providers are used sequentially.
- **Comparability**: Ensures that when we compare a local model to OpenAI, they are both looking at the exact same cleaned text.

## Alternatives Considered

1. **Provider-Specific Cleaning**: Rejected as it makes benchmarking and cross-provider comparison invalid.

## Consequences

- **Positive**: Guarantees consistent input quality; simplifies provider implementation (they only handle inference).
- **Negative**: The preprocessing logic must remain generic enough to not over-clean text for specific models.

## References

- [RFC-013: OpenAI Provider Implementation](../rfc/RFC-013-openai-provider-implementation.md)
