# ADR-033: Content-Hash Based Audio Caching

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md)

## Context & Problem Statement

Audio preprocessing (especially VAD) is computationally expensive. During development or when reprocessing episodes with different transcription models, running the full `ffmpeg` pipeline every time is wasteful.

## Decision

We implement **Content-Hash Based Audio Caching**.

- Preprocessed audio is stored in `.cache/preprocessing/`.
- The cache key is a hash of the **first 1MB of the raw audio** + the **preprocessing configuration string**.
- If a hit is found, the optimized artifact is reused immediately.

## Rationale

- **Developer Velocity**: Reprocessing an episode with a new model takes seconds instead of minutes.
- **Efficiency**: Avoids redundant disk I/O and CPU cycles.
- **Correctness**: Including the config string in the hash ensures the cache invalidates automatically if we change the VAD threshold or sample rate.

## Alternatives Considered

1. **Path-Based Caching**: Rejected as file paths often change or are temporary.
2. **Full File Hashing**: Rejected as hashing 100MB+ files is too slow for a "fast cache" check.

## Consequences

- **Positive**: Near-instant iteration during AI research; reduced system load.
- **Negative**: Requires periodic cleanup of the `.cache/` directory.

## References

- [RFC-040: Audio Preprocessing Pipeline for Podcast Ingestion](../rfc/RFC-040-audio-preprocessing-pipeline.md)
