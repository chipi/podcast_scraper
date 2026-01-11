# ADR-027: Deep Provider Fingerprinting

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-016](../rfc/RFC-016-modularization-for-ai-experiments.md)

## Context & Problem Statement

AI outputs are highly sensitive to environment variables (PyTorch version, GPU model, quantization level). Simply recording the model name (`whisper-large-v3`) is insufficient to reproduce a result exactly or to debug a sudden drop in quality.

## Decision

We implement **Deep Provider Fingerprinting**. Every AI-generated output (transcript, summary) must include a fingerprint containing:

- **Model Details**: Names and SHA256 hashes of local model weights.
- **Hardware**: Device name (`M1 Max`, `RTX 4090`), precision (`fp16`, `int8`).
- **Software**: Versions of `transformers`, `torch`, and the `podcast_scraper` package.
- **Git State**: Commit hash and "dirty" status of the source code.

## Rationale

- **Reproducibility**: Enables developers to perfectly recreate an experimental environment.
- **Debugging**: If metrics drop between two machines, the fingerprint quickly identifies if the cause is a library version mismatch or hardware difference.
- **Transparency**: Provides full "provenance" for every piece of generated content.

## Alternatives Considered

1. **Basic Metadata**: Rejected as it misses critical hardware/library variables that impact model non-determinism.

## Consequences

- **Positive**: Scientific-grade experimental tracking; easier cross-machine debugging.
- **Negative**: Adds a small amount of overhead to output metadata files.

## References

- [RFC-016: Modularization for AI Experiment Pipeline](../rfc/RFC-016-modularization-for-ai-experiments.md)
