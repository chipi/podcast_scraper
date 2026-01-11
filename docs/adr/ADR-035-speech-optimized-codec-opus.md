# ADR-035: Speech-Optimized Codec (Opus)

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md)

## Context & Problem Statement

Intermediate audio artifacts (cached, preprocessed files) need to be small enough to stay under API limits (<25MB) while preserving enough quality for high-accuracy transcription. MP3 and WAV are either inefficient or too large.

## Decision

We standardize on the **Opus Codec** at **24 kbps** for all preprocessed intermediate audio.

- Files are saved with the `.opus` extension.
- The bitrate is fixed at 24kbps, which is the "sweet spot" for speech clarity vs. size.

## Rationale

- **Quality**: Opus is the most advanced speech-optimized codec available, providing higher clarity at 24kbps than MP3 does at 64kbps.
- **Extreme Compression**: A 60-minute podcast episode in Opus 24kbps is only ~10MB, fitting comfortably under all known API limits.
- **Modern Support**: Native support in `ffmpeg`, OpenAI Whisper API, and most local Whisper implementations.

## Alternatives Considered

1. **MP3**: Rejected as it requires higher bitrates (and thus larger files) to achieve the same transcription accuracy.
2. **WAV**: Rejected as it is uncompressed and would immediately exceed upload limits.

## Consequences

- **Positive**: Maximum compatibility with cloud APIs; minimal disk usage for cache.
- **Negative**: Opus files are not as easily "playable" in legacy desktop players without special codecs.

## References

- [RFC-040: Audio Preprocessing Pipeline for Podcast Ingestion](../rfc/RFC-040-audio-preprocessing-pipeline.md)
