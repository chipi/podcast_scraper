# ADR-036: Hybrid MAP-REDUCE Summarization

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md)

## Context & Problem Statement

Classic summarization models (BART, LED) are efficient at compressing text but struggle with instruction-following, structure adherence, and filtering conversational noise. They often produce "extractive" summaries that leak scaffolding text or repeat duplicate ideas from different chunks.

## Decision

We adopt a **Hybrid MAP-REDUCE Summarization Strategy**:

1. **MAP Phase**: Uses **Classic Summarizers** (LED, LongT5) to compress transcript chunks into raw factual notes.
2. **REDUCE Phase**: Uses an **Instruction-Tuned LLM** (Qwen, LLaMA, Mistral) to synthesize those notes into a final, structured summary.

## Rationale

- **Separation of Concerns**: Classic models handle the "heavy lifting" of compression efficiently, while LLMs handle the "reasoning" of abstraction and structuring.
- **Quality**: Instruction-tuned models are far better at ignoring ads, deduplicating ideas, and following output schemas.
- **Efficiency**: Only the small, compressed notes are passed to the expensive LLM, keeping latency and memory usage manageable on local hardware.

## Alternatives Considered

1. **Pure Classic**: Rejected due to poor structure and extraction bias.
2. **Pure LLM**: Rejected for long podcasts due to massive context window requirements and high local compute cost.

## Consequences

- **Positive**: Dramatically higher summary quality; structured output guarantee; better ad/noise filtering.
- **Negative**: Requires loading two different classes of models.

## References

- [RFC-042: Hybrid Podcast Summarization Pipeline](../rfc/RFC-042-hybrid-summarization-pipeline.md)
