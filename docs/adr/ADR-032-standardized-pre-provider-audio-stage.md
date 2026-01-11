# ADR-032: Standardized Pre-Provider Audio Stage

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-040](../rfc/RFC-040-audio-preprocessing-pipeline.md)

## Context & Problem Statement

Transcription providers (OpenAI, Local Whisper) are sensitive to audio quality and file size. Raw podcast files are often high-fidelity stereo, contain long silences, and exceed API upload limits (e.g., OpenAI's 25MB limit). Handling this inside each provider caused logic duplication and inconsistent results.

## Decision

We introduce a **Standardized Pre-Provider Audio Stage**.

- Preprocessing happens in the core pipeline *before* any transcription provider is selected.
- All audio is converted to **Mono**, resampled to **16 kHz**, and processed with **Voice Activity Detection (VAD)** to remove silence.
- Loudness is normalized to a consistent target (e.g., -16 LUFS).

## Rationale

- **API Compatibility**: Guarantees that 100% of podcasts fit within provider upload limits (<25MB).
- **Cost/Performance**: Removing silence and music reduces transcription runtime and API costs by 30-60%.
- **Consistency**: All providers receive the same "optimized" speech-only signal, making benchmarking fair.

## Alternatives Considered

1. **Provider-Level Optimization**: Rejected as it duplicates logic and prevents cross-provider caching.
2. **No Preprocessing**: Rejected as most podcasts simply won't fit into the OpenAI API limit.

## Consequences

- **Positive**: Dramatically lower costs; faster processing; 100% API success rate.
- **Negative**: Adds a system dependency on `ffmpeg`.

## References

- [RFC-040: Audio Preprocessing Pipeline for Podcast Ingestion](../rfc/RFC-040-audio-preprocessing-pipeline.md)
