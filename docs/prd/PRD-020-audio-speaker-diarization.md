# PRD-020: Audio-Based Speaker Diarization & Commercial Content Cleaning

- **Status**: Draft
- **Related RFCs**:
  - [RFC-058](../rfc/RFC-058-audio-speaker-diarization.md) — audio-based speaker diarization (**open**)
  - [RFC-059](../rfc/RFC-059-speaker-detection-refactor-test-audio.md) — speaker detection refactor & test audio (**open**)
  - [RFC-060](../rfc/RFC-060-diarization-aware-commercial-cleaning.md) — multi-signal commercial detection & cleaning (**open**; Phase 2 diarization tie-in pending)
- **Supersedes non-goals in**: PRD-008

## Summary

Add true audio-based speaker diarization to the local transcription pipeline, replacing the current gap-based round-robin speaker rotation with neural voice-embedding-driven attribution. This enables accurate "who said what" labeling in screenplay-format transcripts, especially for interviews with rapid exchanges, multi-guest panels, and overlapping speech.

**Downstream value — commercial content cleaning:** Accurate diarization unlocks a critical downstream capability: identifying and removing host-read sponsor segments from transcripts before summarization. Commercial content leaking into summaries, quotes, and knowledge graphs fundamentally breaks the pipeline's core value. While expanded pattern matching and positional heuristics can improve commercial detection independently (RFC-060 Phase 1), diarization provides the "who + when" context needed to catch host-read ads that blend seamlessly into conversation (RFC-060 Phase 2). This makes diarization not just an accuracy improvement but an enabler of clean, trustworthy output across the entire pipeline.

## Background & Context

The current speaker attribution system has two separate layers that do fundamentally different jobs:

1. **Name detection** (PRD-008, spaCy NER): Extracts host/guest *names* from RSS metadata and episode titles. This identifies *who* the speakers are but not *when* they speak.

2. **Screenplay formatting** (`format_screenplay_from_segments()`): Assigns detected names to Whisper segments using a time-gap heuristic — when silence between segments exceeds `screenplay_gap_s`, it rotates to the next speaker index round-robin. This has no knowledge of actual voice characteristics.

This gap-based rotation is fundamentally unreliable:

- **Rapid exchanges** (< gap threshold): Consecutive turns from different speakers get merged into one speaker
- **Long pauses** by the same speaker: Trigger a false speaker switch
- **Multi-guest panels** (3+ speakers): Round-robin cycles through speakers in fixed order, producing nonsensical attribution
- **Cross-talk / overlap**: Merged into whichever speaker index is current

PRD-008 explicitly listed audio-based diarization as a non-goal. With the maturation of pyannote.audio (v4.0+) and WhisperX, production-quality diarization is now accessible as an optional ML dependency, making it practical to offer as an opt-in enhancement.

## Goals

- Replace gap-based speaker rotation with neural speaker diarization for local Whisper transcription
- Automatically detect the number of speakers from audio (eliminate manual `screenplay_num_speakers`)
- Produce screenplay transcripts with accurate, voice-consistent speaker labels
- Maintain the existing NER name-mapping pipeline (detect names from metadata, then map to diarized speaker IDs)
- Offer diarization as an optional, opt-in feature behind a CLI flag (`--diarize`)
- Keep the current gap-based fallback as default for users without GPU or HuggingFace token
- Enable diarization-enhanced commercial detection (RFC-060 Phase 2) — providing the "who + when" context that makes host-read sponsor identification reliable

## Non-Goals

- Real-time / streaming diarization during transcription
- Cross-episode speaker identification (recognizing the same speaker across different episodes via voiceprints) — future consideration
- Replacing cloud transcription providers (OpenAI, Gemini, Mistral APIs) — diarization applies only to local Whisper path
- Custom speaker model training or fine-tuning

## Personas

- **Podcast Archivist**: Maintains large transcript archives
  - Needs accurate speaker labels across hundreds of episodes
  - Gap-based rotation produces unreliable archives that require manual correction

- **Researcher / Analyst**: Studies conversation dynamics, discourse patterns
  - Needs to know exactly who said what for citation and analysis
  - Current round-robin attribution makes quantitative speaker analysis meaningless

- **Developer / Integrator**: Builds downstream tools on top of transcripts
  - Needs structured, reliable speaker-attributed segments for NLP pipelines
  - Unreliable attribution propagates errors into downstream systems (KG, GIL)

## User Stories

- _As a Podcast Archivist, I can enable `--diarize` so that my screenplay transcripts accurately attribute each line to the correct speaker based on their voice._

- _As a Researcher, I can process interview episodes and get correctly separated host vs. guest turns so that I can analyze speaking patterns and extract per-speaker quotes._

- _As a Developer, I can access diarized segments with speaker IDs and timestamps so that I can build speaker-aware downstream features (knowledge graphs, quote extraction)._

- _As any operator, I can run without `--diarize` and get the same gap-based behavior as before so that nothing breaks if I lack a GPU or HuggingFace token._

## Functional Requirements

### FR1: Diarization Pipeline

- **FR1.1**: When `--diarize` is enabled, run speaker diarization on the audio file after (or during) Whisper transcription to produce a speaker timeline: `[(start, end, speaker_id), ...]`
- **FR1.2**: Align Whisper transcription segments with the diarization timeline to assign each segment (or word) a speaker ID
- **FR1.3**: Auto-detect the number of speakers from audio (override with `--num-speakers N` if known)
- **FR1.4**: Support a minimum of 2 and maximum of 20 speakers

### FR2: Speaker Name Mapping

- **FR2.1**: Map diarized speaker IDs (e.g., `SPEAKER_00`, `SPEAKER_01`) to detected names from the existing NER pipeline (hosts from feed metadata, guests from episode titles)
- **FR2.2**: Use heuristic mapping: first speaker in intro → likely host; map to detected host name. Remaining speakers → map to detected guest names in order
- **FR2.3**: Fall back to `SPEAKER_00`, `SPEAKER_01`, etc. when NER name detection produces fewer names than diarized speakers

### FR3: Integration with Existing Pipeline

- **FR3.1**: Diarization must be optional — disabled by default, enabled via `--diarize` CLI flag or `diarize: true` in YAML config
- **FR3.2**: When disabled, current gap-based `format_screenplay_from_segments()` behavior is preserved exactly
- **FR3.3**: Diarized output uses the same screenplay text format (`SPEAKER: text\n`) for backward compatibility
- **FR3.4**: Diarization result is cached alongside transcript cache (keyed by audio content hash + diarization config)

### FR4: Configuration

- **FR4.1**: New CLI flags and config options:

  | Option | Default | Description |
  | --- | --- | --- |
  | `diarize` | `false` | Enable audio-based speaker diarization |
  | `hf_token` | `None` | HuggingFace access token for pyannote models |
  | `num_speakers` | `None` (auto) | Override auto-detected speaker count |
  | `min_speakers` | `2` | Minimum speakers for diarization |
  | `max_speakers` | `20` | Maximum speakers for diarization |
  | `diarization_device` | `auto` | Device for diarization model (`cpu`, `cuda`, `mps`) |

- **FR4.2**: HuggingFace token can be provided via `--hf-token`, `HF_TOKEN` env var, or `~/.huggingface/token`
- **FR4.3**: Clear error message when `--diarize` is used without a valid HuggingFace token

### FR5: Audio Preprocessing Compatibility

- **FR5.1**: Diarization must work with the existing FFmpeg preprocessing pipeline (mono, resample, loudnorm)
- **FR5.2**: If preprocessing is enabled, diarize the preprocessed audio (not the raw download)

## Success Metrics

- Diarization Error Rate (DER) < 15% on a benchmark set of 10+ podcast episodes with known speaker turns
- Screenplay attribution accuracy: >= 85% of lines attributed to the correct speaker (manual spot-check on 5 episodes)
- Processing overhead: < 2x total wall-clock time compared to Whisper-only (with GPU)
- Zero regression: all existing tests pass with `diarize=false` (default)
- Zero impact on cloud provider paths (OpenAI, Gemini, Mistral transcription)

## Dependencies

- PRD-008: Automatic Speaker Name Detection (NER pipeline provides names to map onto diarized speaker IDs)
- PRD-002: Whisper Fallback Transcription (provides Whisper segments to align with diarization)
- RFC-040: Audio Preprocessing (preprocessed audio feeds into diarization)
- External: HuggingFace account + accepted model terms for pyannote gated models

## Constraints & Assumptions

**Constraints:**

- GPU strongly recommended for practical diarization speed (CPU: ~8.5 min for 60 min audio; GPU: ~75 sec with WhisperX)
- HuggingFace token required for pyannote model access (gated models)
- Audio must be available locally (diarization cannot run on cloud-transcribed episodes without audio)
- Added dependency weight: pyannote.audio pulls in speechbrain, asteroid, torchaudio

**Assumptions:**

- Users opting into `--diarize` have GPU access or accept significantly longer processing times on CPU
- HuggingFace model terms are acceptable for the user's use case
- Podcast audio quality is generally sufficient for diarization (studio-recorded or high-quality remote)
- Most podcast episodes have 2-4 speakers

## Design Considerations

### DC1: WhisperX (Bundled) vs. Whisper + pyannote (Separate)

- **Option A: WhisperX** — Single library that wraps Whisper + pyannote + wav2vec2 alignment
  - **Pros**: Single pipeline, batched inference (70x realtime), word-level timestamps (~50ms), VAD pre-filtering, 228x speedup in speaker assignment, near 100% GPU utilization
  - **Cons**: Replaces current Whisper integration entirely, heavier dependency, slightly lower diarization accuracy (~5% more misattributions vs raw pyannote), less control over individual components

- **Option B: Whisper (current) + pyannote.audio (new)** — Keep existing Whisper, add pyannote as a second pass
  - **Pros**: Minimal changes to existing Whisper code, best diarization accuracy, modular (can swap diarization engine later)
  - **Cons**: Manual alignment code needed, pyannote has known GPU utilization issues (~10%), slower overall, two separate model loads

- **Recommendation**: Start with **Option B** (additive pyannote) for lower integration risk. Evaluate **Option A** (WhisperX) as a future optimization if diarization becomes a core feature and speed matters more. Option B preserves all existing Whisper code paths and tests.

### DC2: Diarization Scope

- **Segment-level diarization**: Assign one speaker per Whisper segment (~5-30 sec chunks). Simpler, matches current screenplay format.
- **Word-level diarization**: Assign speakers per word. Higher accuracy for rapid exchanges but requires WhisperX or custom forced alignment.
- **Recommendation**: Start with segment-level (aligns with Option B above). Word-level is a natural follow-up if WhisperX is adopted.

## Integration with Existing Pipeline

Audio-based diarization enhances the transcript pipeline by:

- **Transcription stage**: After Whisper produces segments, an optional diarization step assigns speaker IDs to each segment
- **Speaker detection stage**: NER-detected names (from PRD-008) are mapped to diarized speaker IDs instead of being assigned round-robin
- **Screenplay formatting**: `format_screenplay_from_segments()` uses diarized speaker assignments instead of gap-based rotation
- **Cache**: Diarization results cached alongside transcripts to avoid re-processing
- **GIL / KG downstream**: More accurate speaker attribution improves quote extraction and knowledge graph speaker nodes
- **Commercial cleaning**: Diarization data feeds into the `CommercialDetector` (RFC-060) to boost confidence when identifying host-read sponsor segments — host monologues in mid-episode, topic discontinuity, and single-speaker candidate regions are strong signals

## Example Output

**Current (gap-based, often wrong):**

```text
Lenny Rachitsky: Welcome back to the show. Today I'm joined by an amazing guest.
Sarah Chen: Thank you so much for having me, Lenny. I've been looking forward to this.
Lenny Rachitsky: So tell me about your journey.
Sarah Chen: Well, it started about five years ago when I was working at...
Lenny Rachitsky: ...a startup in San Francisco.
```

Note: The last line is misattributed — it's Sarah continuing her sentence after a brief pause, but the gap triggered a speaker rotation.

**With diarization (correct):**

```text
Lenny Rachitsky: Welcome back to the show. Today I'm joined by an amazing guest.
Sarah Chen: Thank you so much for having me, Lenny. I've been looking forward to this.
Lenny Rachitsky: So tell me about your journey.
Sarah Chen: Well, it started about five years ago when I was working at a startup in San Francisco.
```

## Open Questions

- Should diarization be part of the `ml` optional extra or a separate `diarize` extra in `pyproject.toml`?
- How to handle the HuggingFace token UX — should we guide users through model acceptance on first run?
- Should we expose diarization confidence scores in the output (e.g., per-segment confidence)?
- Priority: segment-level only, or invest in word-level from the start?

## Related Work

- PRD-008: Automatic Speaker Name Detection (current NER-based system, lists audio diarization as non-goal)
- RFC-010: Speaker Name Detection Technical Design
- RFC-005: Whisper Integration
- RFC-040: Audio Preprocessing Pipeline
- [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio/) — v4.0.4 (Feb 2026)
- [m-bain/whisperX](https://github.com/m-bain/whisperX) — v3.3.2, bundles Whisper + pyannote + alignment
- [pyannote.audio documentation](https://pyannote.github.io/pyannote-audio/)

## Release Checklist

- [ ] PRD reviewed and approved
- [ ] RFC-NNN created with technical design (diarization provider, alignment algorithm, config schema)
- [ ] pyannote.audio added to optional `[diarize]` extra in `pyproject.toml`
- [ ] `DiarizationProvider` implemented with segment-level speaker assignment
- [ ] Alignment logic: Whisper segments mapped to pyannote speaker IDs
- [ ] NER name mapping: diarized speaker IDs -> detected host/guest names
- [ ] CLI flags: `--diarize`, `--hf-token`, `--num-speakers`
- [ ] Cache integration: diarization results cached by audio hash
- [ ] Tests: unit tests for alignment, integration tests with sample audio
- [ ] Documentation: README section, config examples, HuggingFace setup guide
- [ ] Benchmark: DER measured on 10+ episodes, results documented
- [ ] Backward compatibility verified: all existing tests pass with `diarize=false`
