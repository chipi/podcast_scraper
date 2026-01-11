# ADR-005: Lazy ML Dependency Loading

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-005](../rfc/RFC-005-whisper-integration.md)
- **Related PRDs**: [PRD-002](../prd/PRD-002-whisper-fallback.md)

## Context & Problem Statement

ML libraries like `torch` and `openai-whisper` are heavy, adding hundreds of megabytes to the environment and increasing import times. Many users only need the basic transcript download functionality and may not have these libraries (or a GPU) installed.

## Decision

All heavy ML dependencies are **loaded lazily**. Imports for `torch`, `whisper`, and `transformers` are placed inside the functions that use them, rather than at the module top-level.

## Rationale

- **Portability**: The core `podcast_scraper` package can be installed and run for basic downloads in environments without ML support.
- **Startup Speed**: CLI commands that don't use ML (like `--help` or simple downloads) start instantly.
- **Graceful Degradation**: If a user tries to use an ML feature without the dependencies, we can catch the `ImportError` and provide a helpful installation guide instead of crashing.

## Alternatives Considered

1. **Optional Modules**: Splitting ML into separate subpackages; rejected as it complicates the internal API.
2. **Top-Level Try/Except**: Better than crashing, but still incurs import time overhead.

## Consequences

- **Positive**: Lightweight core; better user experience for non-ML users.
- **Negative**: Slightly violates PEP 8 (imports at top-level) and can hide missing dependency errors until runtime.

## References

- [RFC-005: Whisper Integration Lifecycle](../rfc/RFC-005-whisper-integration.md)
