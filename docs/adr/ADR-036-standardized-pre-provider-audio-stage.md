# ADR-036: Standardized Pre-Provider Audio Stage

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

## Amendment (2026-07-12, GitHub #1173): silence removal is off

The VAD silence-removal clause in **Decision** is superseded; the rest of the ADR stands.

This ADR treated the stage as free — it optimizes the signal, nothing downstream can tell. That is
true of mono conversion, resampling, loudness normalization and MP3 encoding, all of which preserve
the audio's **duration**. It is not true of silence removal, which deletes interior pauses and so
hands the transcriber a **shorter timeline than the audio the rest of the system uses**. Transcript
timestamps are stored against the *original* file (the player seeks it, the KG cites it), so every
timestamp after a removed pause lands early and the error accumulates: measured on the prod corpus
at **-20 s** on a 25-minute episode (32 s of pauses cut) and **-162 s** on a 1 h 54 m one (401 s
cut).

The "Cost/Performance: 30-60%" rationale also over-credited this stage: the bitrate/sample-rate
work (#561) delivers the file-size win, while silence removal cut only ~3% of duration.

Preprocessing is therefore now **timeline-preserving by default**. Silence removal survives behind
`preprocessing_silence_removal` (default `false`) for deployments that never use transcript
timestamps. A unit test pins the invariant (duration in == duration out).

## References

- [RFC-040: Audio Preprocessing Pipeline for Podcast Ingestion](../rfc/RFC-040-audio-preprocessing-pipeline.md)
