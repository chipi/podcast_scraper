# ADR-027: Deep Provider Fingerprinting

- **Status**: Accepted ✅ Implemented
- **Date**: 2026-01-11
- **Updated**: 2026-01-16
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-016](../rfc/RFC-016-modularization-for-ai-experiments.md)

## Context & Problem Statement

AI outputs are highly sensitive to environment variables (PyTorch version, GPU model, quantization level). Simply recording the model name (`whisper-large-v3`) is insufficient to reproduce a result exactly or to debug a sudden drop in quality.

## Decision

We implement **Deep Provider Fingerprinting**. Every AI-generated output (transcript, summary) must include a fingerprint containing:

- **Run Context**: Run ID, baseline/reference ID, dataset ID, git commit/branch/dirty status
- **Provider**: Provider type, library, library version
- **Model**: Task, model name, model version/revision, endpoint, tokenizer details
- **Generation Params**: Temperature, top_p, max_new_tokens, min_new_tokens, repetition_penalty, seed
- **Preprocessing**: Profile ID, profile version, detailed steps (remove_timestamps, normalize_speakers, etc.)
- **Chunking**: Strategy, token/word chunk sizes, overlap, boundary heuristics
- **Prompts**: Template ID, template SHA256, parameters
- **Environment**: Python version, OS, dependencies
- **Runtime**: Device (MPS/CUDA/CPU), backend, torch version, dtype, inference backend, compile settings

**Implementation**: The fingerprint is stored as `fingerprint.json` in baseline/reference/run directories, with a reference from `predictions.jsonl` entries via `fingerprint_ref`.

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
