# ADR-028: Typed Provider Parameter Models

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-016](../rfc/RFC-016-modularization-for-ai-experiments.md)

## Context & Problem Statement

Passing AI parameters (temperature, chunk size, overlap) as raw dictionaries leads to "parameter drift," where different providers use different names for the same concept, or invalid values are only caught deep inside a model call.

## Decision

We use **Typed Provider Parameter Models** (Pydantic).

- Each AI task has a dedicated Pydantic model (e.g., `SummarizationParams`, `TranscriptionParams`).
- Providers MUST accept these models rather than raw dictionaries.
- The models include default values and strict validation rules (e.g., `temperature` between 0 and 1).

## Rationale

- **Type Safety**: Catches configuration errors at "experiment load time" rather than hours into a long processing run.
- **Consistency**: Ensures that `chunk_size` means the same thing across all local and cloud providers.
- **Self-Documentation**: The Pydantic models serve as the single source of truth for what parameters a provider supports.

## Alternatives Considered

1. **Raw Dicts**: Rejected due to lack of validation and high risk of typos.
2. **Config-Only Params**: Rejected as it prevents the experiment runner from varying parameters dynamically.

## Consequences

- **Positive**: Robust error handling; clear provider contracts; improved IDE autocompletion for researchers.
- **Negative**: Requires mapping core `Config` fields to these models.

## References

- [RFC-016: Modularization for AI Experiment Pipeline](../rfc/RFC-016-modularization-for-ai-experiments.md)
