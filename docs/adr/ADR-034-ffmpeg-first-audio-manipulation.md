# ADR-034: FFmpeg-First Audio Manipulation

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md)

## Context & Problem Statement

The pipeline needs to perform complex audio operations: format conversion, resampling, VAD (silence removal), and loudness normalization. Pure Python libraries are often slow or require complex chains of C-extensions.

## Decision

We standardize on **FFmpeg-First Audio Manipulation**.

- The pipeline calls the system `ffmpeg` binary directly via `subprocess`.
- We use complex filter chains (e.g., `silenceremove`, `loudnorm`) to perform multiple operations in a single pass.

## Rationale

- **Performance**: `ffmpeg` is highly optimized and far faster than equivalent Python libraries like `pydub` or `librosa`.
- **Completeness**: Supports every codec and filter we might ever need without adding dozens of Python dependencies.
- **Reliability**: Industry-standard tool with predictable behavior across platforms.

## Alternatives Considered

1. **Python Libraries (pydub/webrtcvad)**: Rejected due to performance bottlenecks and fragmented feature sets.

## Consequences

- **Positive**: Extremely fast audio pipelines; simplified Python dependency list.
- **Negative**: Requires the user to have `ffmpeg` installed on their system.

## References

- [RFC-040: Audio Preprocessing Pipeline for Podcast Ingestion](../rfc/RFC-040-audio-preprocessing-pipeline.md)
