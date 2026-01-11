# ADR-006: Context-Aware Model Selection

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-010](../rfc/RFC-010-speaker-name-detection.md)
- **Related PRDs**: [PRD-008](../prd/PRD-008-speaker-name-detection.md)

## Context & Problem Statement

OpenAI Whisper provides generic multilingual models (`base`, `small`, etc.) and optimized English-only models (`base.en`, `small.en`). The English models are significantly more accurate and faster for English-only audio.

## Decision

The system automatically promotes requested models to their English variants if:

1. The detected or configured language is English (`en`).
2. The user requested a model size that has an `.en` variant (tiny, base, small, medium).
3. The user hasn't already explicitly specified a variant.

## Rationale

- **Quality by Default**: Most users want the best accuracy for their language without needing to know technical model suffixes.
- **Resource Efficiency**: English models are typically faster and smaller than their multilingual counterparts.

## Alternatives Considered

1. **Manual Selection Only**: Rejected as it places too much cognitive load on the user.
2. **Multilingual Default**: Safe, but leaves significant performance and accuracy on the table for the primary use case (English podcasts).

## Consequences

- **Positive**: Better transcription quality "out of the box."
- **Negative**: Can be surprising to users who explicitly wanted the multilingual model for some reason (e.g., occasional non-English segments).

## References

- [RFC-010: Automatic Speaker Name Detection](../rfc/RFC-010-speaker-name-detection.md)
