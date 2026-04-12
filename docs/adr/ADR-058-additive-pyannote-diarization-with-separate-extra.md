# ADR-058: Additive pyannote Diarization with Separate `[diarize]` Extra

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-058](../rfc/RFC-058-audio-speaker-diarization.md)
- **Related PRDs**: [PRD-020](../prd/PRD-020-audio-speaker-diarization.md)

## Context & Problem Statement

The current screenplay formatting uses a time-gap heuristic to assign speakers: if a
silence gap exceeds `gap_s`, the next segment is attributed to a different speaker via
round-robin cycling. This produces systematically wrong attribution in rapid exchanges,
same-speaker pauses, and multi-guest panels.

Neural speaker diarization (voice-embedding-driven "who said what") solves this, but
there are multiple integration strategies: replace Whisper entirely with WhisperX, add
pyannote as a second pass after Whisper, or use cloud diarization APIs.

## Decision

We add pyannote.audio as an **additive second pass** after Whisper transcription, with
diarization as a **separate optional dependency group**.

1. **Additive, not replacement**: Whisper transcription is preserved as-is. pyannote
   runs after Whisper to produce speaker timelines, which are aligned to Whisper
   segments via maximum-overlap matching. The existing gap-based path is the default
   (`diarize=false`).
2. **Segment-level diarization**: One speaker is assigned per Whisper segment (not per
   word). This matches the current screenplay format and avoids the complexity of
   forced word-level alignment.
3. **Waveform loading via `torchaudio`**: Audio is loaded with `torchaudio.load()`
   and passed as a waveform to pyannote, avoiding a known 4x performance penalty when
   passing file paths (pyannote issue #1702).
4. **Separate `[diarize]` optional extra**: pyannote adds `speechbrain`, `asteroid`,
   and HuggingFace model downloads — significantly heavier than the base `[ml]` extra.
   Users who want Whisper without diarization should not pay this dependency cost.
   Install via `pip install -e ".[diarize]"`.
5. **Lazy import**: pyannote dependencies are imported at function level (matching
   ADR-005 pattern) so the package loads normally without `[diarize]` installed.

## Rationale

- **Additive**: Lower integration risk than replacing the entire Whisper pipeline with
  WhisperX. All existing Whisper code paths, tests, and behaviors are preserved. If
  pyannote has issues, the gap-based fallback works.
- **Segment-level**: Simpler alignment algorithm, matches current screenplay format.
  Word-level diarization requires forced alignment (WhisperX territory) and can be
  added later if needed.
- **Waveform loading**: Measured 12s vs 50s for 3-minute clips. This is a
  straightforward performance optimization that should always be used.
- **Separate extra**: pyannote's transitive dependency tree (`speechbrain`,
  `asteroid`, HuggingFace model downloads) is large. Bundling it into `[ml]` would
  penalize every ML user, not just those who want diarization.

## Alternatives Considered

1. **WhisperX as full pipeline replacement**: Rejected; replaces proven Whisper
   integration, slightly lower diarization accuracy (~5%), larger blast radius. Can be
   evaluated after diarization proves value.
2. **Voice-activity-based speaker change detection (Silero VAD)**: Rejected; still a
   heuristic with no voice identity. Only marginally better than gap-based rotation.
3. **Cloud diarization APIs (Google, AssemblyAI)**: Rejected; per-minute costs, vendor
   lock-in, no offline support. Conflicts with local-first philosophy.
4. **Merge `[diarize]` into `[ml]`**: Rejected; significantly increases dependency
   weight for all ML users.

## Consequences

- **Positive**: Accurate, voice-based speaker attribution. Auto speaker count detection
  (eliminates manual `screenplay_num_speakers`). Multi-speaker panels correctly handled.
  Downstream quality improves for GIL quotes and KG speaker nodes.
- **Negative**: HuggingFace token required (gated model). GPU strongly recommended
  (CPU ~8.5 min vs GPU ~1.5 min for 60 min audio). New dependency group to maintain.
- **Neutral**: `diarize=false` is the default. Zero impact on users who do not opt in.

## Implementation Notes

- **Module**: `src/podcast_scraper/providers/ml/diarization/`
- **Protocol**: `DiarizationProvider` (PEP 544) with `diarize()` method
- **Alignment**: Maximum-overlap matching between Whisper segments and diarization
  timeline, with carry-forward for gaps
- **Caching**: Diarization results cached by `sha256(audio_content) +
  diarization_config_hash` in `.cache/diarization/`
- **Config**: `diarize: bool = False`, `hf_token`, `num_speakers`, `min_speakers`,
  `max_speakers`, `diarization_device`, `diarization_model`
- **Relationship to ADR-005**: Follows lazy ML dependency loading pattern
- **Relationship to ADR-036**: Preprocessed audio (RFC-040) feeds into diarization

## References

- [RFC-058: Audio-Based Speaker Diarization](../rfc/RFC-058-audio-speaker-diarization.md)
- [RFC-059: Speaker Detection Refactor](../rfc/RFC-059-speaker-detection-refactor-test-audio.md)
- [ADR-005: Lazy ML Dependency Loading](ADR-005-lazy-ml-dependency-loading.md)
- [ADR-036: Standardized Pre-Provider Audio Stage](ADR-036-standardized-pre-provider-audio-stage.md)
- [PRD-020: Audio Speaker Diarization](../prd/PRD-020-audio-speaker-diarization.md)
