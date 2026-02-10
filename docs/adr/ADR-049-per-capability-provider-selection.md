# ADR-049: Per-Capability Provider Selection

- **Status**: Accepted
- **Date**: 2026-02-10
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-032](../rfc/RFC-032-anthropic-provider-implementation.md), [RFC-033](../rfc/RFC-033-mistral-provider-implementation.md), [RFC-034](../rfc/RFC-034-deepseek-provider-implementation.md), [RFC-035](../rfc/RFC-035-gemini-provider-implementation.md), [RFC-036](../rfc/RFC-036-grok-provider-implementation.md), [RFC-037](../rfc/RFC-037-ollama-provider-implementation.md)
- **Related PRDs**: [PRD-006](../prd/PRD-006-openai-provider-integration.md), [PRD-009](../prd/PRD-009-anthropic-provider-integration.md), [PRD-010](../prd/PRD-010-mistral-provider-integration.md)–[PRD-014](../prd/PRD-014-ollama-provider-integration.md)

## Context & Problem Statement

The system has three AI capabilities: transcription, speaker detection, and summarization. Not all providers support all three. For example, Anthropic, DeepSeek, Grok, and Ollama do not support audio transcription; only Whisper, OpenAI, Mistral, and Gemini do. The pipeline must allow users to pick the best provider per capability without forcing a single provider for everything, and without requiring every provider to implement every protocol.

## Decision

We adopt **per-capability provider selection**:

1. **Independent config fields**: `transcription_provider`, `speaker_detector_provider`, and `summary_provider` are chosen independently. Each accepts only providers that implement that capability.
2. **Partial-protocol providers**: A provider may implement a subset of the three protocols (TranscriptionProvider, SpeakerDetector, SummarizationProvider). It is only offered in the config for capabilities it supports.
3. **No automatic fallback across providers**: If the user selects a provider that does not support a capability, that is a config error (e.g. selecting "anthropic" for transcription is invalid). Fallback (e.g. Whisper when no cloud transcription) is achieved by the user choosing a different provider for that capability, not by the pipeline auto-switching.

## Rationale

- **Clarity**: Users explicitly choose per capability; no hidden fallback behavior.
- **Consistency with ADR-011**: Each provider remains a unified class implementing one or more protocols; config simply restricts which providers appear per capability.
- **Extensibility**: New providers (e.g. "no transcription") are added by implementing only the protocols they support and registering for the corresponding config Literal.

## Alternatives Considered

1. **Single provider for all three**: Rejected; would force users to use the same vendor for transcription and summarization despite different capability matrices.
2. **Automatic fallback (e.g. always use Whisper for transcription if LLM provider doesn't support it)**: Rejected; implicit behavior would be surprising and would complicate config semantics and testing.

## Consequences

- **Positive**: Clear config model; provider capability matrix is documented; adding new providers only requires implementing the protocols they support and updating the config Literals for those capabilities.
- **Negative**: Config validation must keep per-capability allowlists in sync with provider implementations.

## Implementation Notes

- **Config**: `config.Config.transcription_provider`, `speaker_detector_provider`, `summary_provider` with Literal types that list only providers supporting that capability.
- **Pattern**: Factory functions (`create_transcription_provider`, etc.) only accept provider types that implement the relevant protocol; validation rejects invalid combinations.
- **Documentation**: Provider capability matrix in PRDs and configuration reference lists which providers support which capabilities.

## References

- [ADR-011: Unified Provider Pattern](ADR-011-unified-provider-pattern.md) – Type-based unified provider classes
- [ADR-012: Protocol-Based Provider Discovery](ADR-012-protocol-based-provider-discovery.md) – PEP 544 Protocols per capability
- [RFC-032: Anthropic Provider Implementation](../rfc/RFC-032-anthropic-provider-implementation.md) – Example: no transcription support
