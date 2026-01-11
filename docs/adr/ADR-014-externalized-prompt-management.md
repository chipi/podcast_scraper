# ADR-014: Externalized Prompt Management

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-017](../rfc/RFC-017-prompt-management.md)
- **Related PRDs**: [PRD-006](../prd/PRD-006-openai-provider-integration.md)

## Context & Problem Statement

AI prompts (for summarization and NER) were originally hardcoded strings. This made them difficult to version, impossible to test across different models without code changes, and hard to track for experimental reproducibility.

## Decision

We **Externalize Prompt Management**:

1. All prompts are stored as `.j2` (Jinja2) files in a dedicated `prompts/` directory.
2. Providers load and render prompts by logical name (e.g., `summarization/long_v1`).
3. The system records the SHA256 hash of the template used in the output metadata.

## Rationale

- **Iterative Speed**: Prompts can be updated without rebuilding or redeploying code.
- **Reproducibility**: The metadata hash ensures we always know exactly what prompt produced a specific result.
- **Dynamic Context**: Jinja2 allows passing runtime variables (like `paragraphs_max`) directly into the prompt logic.

## Alternatives Considered

1. **Prompt Engineering Frameworks (LangChain)**: Rejected as too heavy for our needs.
2. **Hardcoded Strings with Versioning**: Rejected as it clutters the codebase and requires git commits for every wording change.

## Consequences

- **Positive**: Clean code; better experimental rigor; easier collaboration with non-developers on prompt tuning.
- **Negative**: Adds a runtime dependency on `Jinja2` and requires managing an extra directory of assets.

## References

- [RFC-017: Prompt Management and Loading](../rfc/RFC-017-prompt-management.md)
