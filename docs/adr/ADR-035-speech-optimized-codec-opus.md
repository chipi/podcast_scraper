# ADR-035: Speech-Optimized Codec (Opus)

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md)

## Context & Problem Statement

Intermediate audio artifacts (cached, preprocessed files) need to be small enough to stay under API limits (<25MB) while preserving enough quality for high-accuracy transcription. MP3 and WAV are either inefficient or too large.

## Decision

We standardize on the **MP3 Codec** at **64 kbps** for all preprocessed intermediate audio.

- Files are saved with the `.mp3` extension (MP3 codec for maximum compatibility).
- The bitrate is fixed at 64kbps, which provides good quality for speech while keeping file sizes manageable.

## Rationale

- **Compatibility**: MP3 is universally supported by all transcription APIs (OpenAI, local Whisper, etc.).
- **Reliability**: MP3 format is well-tested and doesn't have container format issues that can occur with OGG/Opus.
- **File Size**: A 60-minute podcast episode in MP3 64kbps is ~30MB, still fitting under API limits with preprocessing.
- **Quality**: 64kbps MP3 provides sufficient quality for accurate transcription while maintaining compatibility.

## Alternatives Considered

1. **Opus/OGG**: Initially chosen for better compression, but rejected due to format compatibility issues with OpenAI API.
2. **WAV**: Rejected as it is uncompressed and would immediately exceed upload limits.

## Consequences

- **Positive**: Maximum compatibility with all cloud APIs; reliable format detection; widely playable.
- **Negative**: Slightly larger file sizes compared to Opus (but still acceptable with preprocessing).

## References

- [RFC-040: Audio Preprocessing Pipeline for Podcast Ingestion](../rfc/RFC-040-audio-preprocessing-pipeline.md)
