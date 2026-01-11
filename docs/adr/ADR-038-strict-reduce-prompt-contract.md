# ADR-038: Strict REDUCE Prompt Contract

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md)

## Context & Problem Statement

LLM outputs can be unpredictable, often including conversational filler ("Sure, here is your summary...") or varying formatting. For a reliable pipeline, the output MUST be parsable and consistent.

## Decision

We enforce a **Strict REDUCE Prompt Contract**.

- The REDUCE phase prompt mandates a specific Markdown structure:
  - `Key takeaways:` (Bullet points)
  - `Topic outline:` (Bullet points)
  - `Action items (if any):` (Bullet points or "None")
- The system uses "zero-tolerance" instructions to prevent models from adding any introductory or concluding text.

## Rationale

- **Reliability**: Guarantees that the generated metadata always fits the expected schema.
- **User Experience**: Users get a predictable, actionable summary every time, regardless of the model being used.
- **Automation**: Enables simple post-processing and validation logic (e.g., checking for the presence of all three headers).

## Alternatives Considered

1. **JSON Output**: Considered but rejected for v1 as many smaller local models (7B) struggle with strict JSON syntax compared to Markdown.

## Consequences

- **Positive**: Consistent, high-quality structured summaries.
- **Negative**: Requires models with strong instruction-following capabilities (e.g., Qwen2.5, LLaMA 3).

## References

- [RFC-042: Hybrid Podcast Summarization Pipeline](../rfc/RFC-042-hybrid-summarization-pipeline.md)
