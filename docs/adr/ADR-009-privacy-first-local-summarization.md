# ADR-009: Privacy-First Local Summarization

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-012](../rfc/RFC-012-episode-summarization.md)
- **Related PRDs**: [PRD-005](../prd/PRD-005-episode-summarization.md)

## Context & Problem Statement

Summarizing long podcast transcripts often involves sensitive or personal content. While Cloud LLMs (OpenAI, Anthropic) are easy to use, they introduce privacy risks, recurring costs, and internet dependencies.

## Decision

The project defaults to **Local Summarization** using Transformer models (BART, LED) running on the user's hardware. Cloud-based summarization is treated as an optional secondary provider.

## Rationale

- **Privacy**: Transcripts never leave the user's machine.
- **Cost**: Zero per-token cost allows for processing massive back-catalogs without financial overhead.
- **Reliability**: The pipeline works offline and is immune to API rate limits or service outages.

## Alternatives Considered

1. **Cloud-First**: Rejected to maintain the "toolbox" nature of the project and respect user privacy.
2. **Rule-based Summarization**: Rejected as it cannot capture the nuance and abstractive qualities of podcast conversations.

## Consequences

- **Positive**: High privacy; zero marginal cost.
- **Negative**: Requires significant local hardware (RAM/GPU); summaries may have lower "polish" than premium cloud models.

## References

- [RFC-012: Episode Summarization Using Local Transformers](../rfc/RFC-012-episode-summarization.md)
