# ADR-012: Protocol-Based Provider Discovery

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-021](../rfc/RFC-021-modularization-refactoring-plan.md)
- **Related PRDs**: [PRD-006](../prd/PRD-006-openai-provider-integration.md)

## Context & Problem Statement

The core workflow needs to interact with various AI backends without knowing their implementation details. Traditional inheritance creates rigid hierarchies that are hard to mock and extend.

## Decision

We use **PEP 544 Protocols** (Static Duck Typing) to define provider interfaces.

- Interfaces like `TranscriptionProvider` and `SummarizationProvider` are defined as `typing.Protocol`.
- Concrete implementations (ML, OpenAI, Custom) do not need to inherit from a base class; they only need to implement the required method signatures.

## Rationale

- **Decoupling**: The workflow depends on *behaviors*, not *class hierarchies*.
- **Flexibility**: Makes it trivial to add third-party providers (Deepgram, Anthropic) without modifying the internal inheritance tree.
- **Testability**: Simplifies mocking as any object matching the signature is valid.

## Alternatives Considered

1. **Abstract Base Classes (ABCs)**: Rejected because they force an inheritance relationship and can lead to "diamond" dependency issues.

## Consequences

- **Positive**: Clean module boundaries; strictly type-checked interfaces; easy extensibility.
- **Negative**: Requires careful documentation of method signatures since there is no single base class to inspect.

## References

- [RFC-021: Modularization Refactoring Plan](../rfc/RFC-021-modularization-refactoring-plan.md)
