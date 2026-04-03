# RFC-058: Audio-Based Speaker Diarization

- **Status**: Draft
- **Authors**: Architecture Review
- **Stakeholders**: Core Pipeline, ML Providers, Transcription Consumers
- **Related PRDs**:
  - `docs/prd/PRD-020-audio-speaker-diarization.md`
  - `docs/prd/PRD-002-whisper-fallback.md`
  - `docs/prd/PRD-008-speaker-name-detection.md`
- **Related RFCs**:
  - `docs/rfc/RFC-005-whisper-integration.md` (Whisper lifecycle)
  - `docs/rfc/RFC-006-screenplay-formatting.md` (gap-based rotation — to be superseded)
  - `docs/rfc/RFC-010-speaker-name-detection.md` (NER name pipeline — preserved)
  - `docs/rfc/RFC-040-audio-preprocessing-pipeline.md` (FFmpeg preprocessing — feeds into diarization)
  - `docs/rfc/RFC-059-speaker-detection-refactor-test-audio.md` (companion: refactor + test improvements)
  - `docs/rfc/RFC-060-diarization-aware-commercial-cleaning.md` (downstream: diarization-aware sponsor detection)
- **Related Issues**:
  - Issue #414: Audio Pipeline Separation
- **Related Documents**:
  - `docs/architecture/ARCHITECTURE.md` (pipeline flow)

## Abstract

This RFC defines the technical design for adding neural speaker diarization to the local Whisper transcription pipeline via pyannote.audio. Diarization replaces the gap-based round-robin speaker rotation in `format_screenplay_from_segments()` with voice-embedding-driven speaker attribution, producing accurate "who said what" transcripts. The feature is opt-in (`--diarize`), preserves the existing pipeline as default, and reuses the NER name-mapping system from RFC-010 to assign real names to diarized speaker IDs.

**Architecture Alignment:** Diarization is introduced as an optional post-transcription stage in `episode_processor.py`, sitting between Whisper transcription and screenplay formatting. This follows the pipeline-stage pattern established by RFC-040 (audio preprocessing) and aligns with Issue #414's vision of separating audio processing into distinct stages.

## Problem Statement

The current screenplay formatting (`format_screenplay_from_segments()` in `ml_provider.py`) uses a time-gap heuristic to assign speakers:

```python
if prev_end is not None and start - prev_end > gap_s:
    current_speaker_idx = (current_speaker_idx + 1) % max(
        config.MIN_NUM_SPEAKERS, num_speakers
    )
```

This produces systematically wrong speaker attribution:

1. **Rapid exchanges** — When speakers alternate faster than `gap_s`, both turns are attributed to the same speaker
2. **Same-speaker pauses** — A long pause (thinking, drinking water) triggers a false speaker switch
3. **Multi-guest panels** — Round-robin cycling through 3+ speakers produces nonsensical attribution order
4. **Overlapping speech** — Cross-talk is always attributed to the current speaker index

These errors propagate into downstream features (GIL quote extraction, KG speaker nodes) and make screenplay transcripts unreliable for research or analysis.

**Use Cases:**

1. **Interview transcript accuracy**: Host and guest lines correctly attributed during rapid back-and-forth
2. **Panel discussion**: 3-4 speakers correctly identified and tracked throughout the episode
3. **Quote extraction**: GIL/KG downstream features receive reliably attributed speaker segments

## Goals

1. **Accurate diarization**: Replace gap-based rotation with neural speaker embeddings (target: < 15% DER)
2. **Transparent integration**: Diarization as an optional pipeline stage, not a replacement for Whisper
3. **Name preservation**: Map diarized speaker IDs to NER-detected names (RFC-010 pipeline intact)
4. **Backward compatibility**: `diarize=false` (default) preserves exact current behavior
5. **Cacheable**: Diarization results cached by audio content hash to avoid re-processing

## Constraints & Assumptions

**Constraints:**

- pyannote.audio models are gated on HuggingFace — requires user to accept model terms and provide access token
- GPU strongly recommended (CPU: ~8.5 min for 60 min audio; GPU with proper waveform loading: ~1.5 min)
- Only applies to local Whisper path — cloud providers (OpenAI, Gemini, Mistral APIs) transcribe without local audio access
- Must not break any existing tests when `diarize=false`
- pyannote adds significant dependency weight: `speechbrain`, `torchaudio`, HF model downloads

**Assumptions:**

- Users opting into `--diarize` accept the HuggingFace token requirement
- Podcast audio quality is generally sufficient for speaker embedding extraction
- Most podcast episodes have 2-4 speakers; maximum 20 is a safe upper bound
- Segment-level diarization (not word-level) is sufficient for the initial implementation

## Design & Implementation

### 1. New Module: `diarization/`

Create a new provider-style module under the ML providers directory:

```text
src/podcast_scraper/
├── providers/
│   └── ml/
│       ├── ml_provider.py          # existing — gains diarize_and_format() method
│       ├── speaker_detection.py    # existing — NER pipeline unchanged
│       └── diarization/            # NEW
│           ├── __init__.py
│           ├── base.py             # DiarizationProvider protocol
│           ├── pyannote_provider.py # pyannote.audio implementation
│           └── alignment.py        # Whisper segments ↔ diarization timeline alignment
```

### 2. DiarizationProvider Protocol

```python
from typing import Protocol, List, Tuple, Optional

class DiarizationSegment:
    """A diarized segment with speaker ID and time range."""
    start: float           # seconds
    end: float             # seconds
    speaker: str           # e.g. "SPEAKER_00"

class DiarizationResult:
    """Complete diarization output."""
    segments: List[DiarizationSegment]
    num_speakers: int
    model_name: str

class DiarizationProvider(Protocol):
    """Protocol for speaker diarization backends."""

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
    ) -> DiarizationResult:
        """Run speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            num_speakers: Known number of speakers (None = auto-detect)
            min_speakers: Minimum speakers for auto-detection
            max_speakers: Maximum speakers for auto-detection

        Returns:
            DiarizationResult with speaker-attributed segments
        """
        ...
```

### 3. pyannote.audio Implementation

```python
import torch
import torchaudio
from pyannote.audio import Pipeline

class PyAnnoteDiarizationProvider:
    """Speaker diarization using pyannote.audio community pipeline."""

    def __init__(
        self,
        hf_token: str,
        device: str = "auto",
        model_name: str = "pyannote/speaker-diarization-3.1",
    ):
        self._pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipeline.to(torch.device(device))

    def diarize(self, audio_path, *, num_speakers=None, min_speakers=2, max_speakers=20):
        # Load audio as waveform (avoids pyannote's slow file-path loading)
        waveform, sample_rate = torchaudio.load(audio_path)

        params = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        else:
            params["min_speakers"] = min_speakers
            params["max_speakers"] = max_speakers

        diarization = self._pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            **params,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizationSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
            ))

        unique_speakers = {s.speaker for s in segments}
        return DiarizationResult(
            segments=segments,
            num_speakers=len(unique_speakers),
            model_name=self._pipeline.model_name,
        )
```

Key implementation detail: loading audio via `torchaudio.load()` and passing the waveform directly to pyannote avoids a known 4x performance penalty when passing file paths (see pyannote issue #1702).

### 4. Alignment Algorithm

The alignment module maps Whisper transcription segments to diarization speaker IDs:

```python
def align_segments_to_speakers(
    whisper_segments: List[dict],
    diarization: DiarizationResult,
) -> List[Tuple[dict, str]]:
    """Assign a speaker ID to each Whisper segment based on diarization overlap.

    For each Whisper segment, find the diarization segment(s) that overlap it,
    then assign the speaker with the greatest overlap duration.

    Args:
        whisper_segments: Whisper segments with start/end/text
        diarization: DiarizationResult from diarization provider

    Returns:
        List of (whisper_segment, speaker_id) tuples
    """
```

**Algorithm:**

1. Build an interval index from diarization segments (for O(log n) lookups)
2. For each Whisper segment `[ws, we]`:
   a. Find all diarization segments that overlap `[ws, we]`
   b. Compute overlap duration with each: `overlap = min(we, de) - max(ws, ds)`
   c. Assign the speaker with the maximum total overlap
   d. If no overlap found (gap in diarization), carry forward the previous speaker
3. Return aligned `(segment, speaker_id)` pairs

**Edge cases:**

- **No diarization overlap**: Carry forward previous speaker (silence gaps between diarization segments)
- **Multiple speakers per Whisper segment**: Assign majority speaker (segment-level, not word-level)
- **Whisper segment extends beyond diarization**: Clip to diarization boundary

### 5. Integration into Pipeline

**In `episode_processor.py`:**

```python
def _diarize_and_format_screenplay(
    self,
    audio_path: str,
    whisper_segments: List[dict],
    speaker_names: List[str],
    cfg: Config,
) -> str:
    """Diarize audio and format screenplay with real speaker attribution."""

    # 1. Run diarization
    diarization_result = self._diarization_provider.diarize(
        audio_path,
        num_speakers=cfg.num_speakers,
        min_speakers=cfg.min_speakers,
        max_speakers=cfg.max_speakers,
    )

    # 2. Align Whisper segments to diarized speakers
    aligned = align_segments_to_speakers(whisper_segments, diarization_result)

    # 3. Map speaker IDs to detected names
    speaker_map = self._map_speakers_to_names(
        diarization_result, speaker_names
    )

    # 4. Format screenplay
    return self._format_diarized_screenplay(aligned, speaker_map)
```

**Pipeline flow with diarization enabled:**

```text
Download → [Preprocess (RFC-040)] → Whisper transcribe
                                        ↓
                                  segments + text
                                        ↓
                            ┌──── diarize=true? ────┐
                            │                       │
                       YES: pyannote               NO: gap-based
                       diarize audio               rotation (RFC-006)
                            │                       │
                       align segments               │
                       to speakers                  │
                            │                       │
                            └───────────────────────┘
                                        ↓
                              NER name mapping (RFC-010)
                                        ↓
                              Screenplay output
```

### 6. Speaker Name Mapping

Diarization produces anonymous IDs (`SPEAKER_00`, `SPEAKER_01`, ...). The NER pipeline (RFC-010) produces names. Mapping strategy:

1. **Intro heuristic**: The speaker with the most speaking time in the first 90 seconds → likely host → map to first detected host name
2. **Speaking time rank**: Remaining speakers ordered by total speaking time → map to remaining names in order
3. **Fallback**: If NER detected fewer names than diarized speakers, use `SPEAKER_N` for unmapped speakers

```python
def _map_speakers_to_names(
    self,
    diarization: DiarizationResult,
    detected_names: List[str],
) -> Dict[str, str]:
    """Map diarized speaker IDs to detected names.

    Strategy: speaker with most time in first 90s → host name;
    remaining by total speaking time → remaining names in order.
    """
```

### 7. Configuration

New config fields in `Config` (Pydantic model):

```python
# Diarization settings
diarize: bool = False
hf_token: Optional[str] = None  # also: HF_TOKEN env var, ~/.huggingface/token
num_speakers: Optional[int] = None  # None = auto-detect
min_speakers: int = 2
max_speakers: int = 20
diarization_device: str = "auto"  # "cpu", "cuda", "mps"
diarization_model: str = "pyannote/speaker-diarization-3.1"
```

CLI flags:

```text
--diarize / --no-diarize     Enable/disable speaker diarization
--hf-token TEXT              HuggingFace access token
--num-speakers INT           Override auto-detected speaker count
--min-speakers INT           Minimum speakers (default: 2)
--max-speakers INT           Maximum speakers (default: 20)
--diarization-device TEXT    Device for diarization (auto/cpu/cuda/mps)
```

### 8. Caching

Diarization results are expensive to compute. Cache alongside transcript cache:

- **Cache key**: `sha256(audio_content) + diarization_config_hash`
- **Cache value**: Serialized `DiarizationResult` (JSON)
- **Location**: `diarization_cache_dir` (defaults to `<output_dir>/.cache/diarization/`)
- **Invalidation**: Audio content change or diarization config change

### 9. Dependency Management

Add `pyannote.audio` as a new optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
diarize = [
    "pyannote.audio>=3.1",
    "torchaudio",
]
```

This keeps diarization dependencies separate from the existing `ml` extra. Users install with:

```bash
pip install -e ".[diarize]"
# or combined:
pip install -e ".[ml,diarize]"
```

Lazy import pattern (same as Whisper):

```python
def _import_pyannote():
    try:
        from pyannote.audio import Pipeline
        return Pipeline
    except ImportError:
        raise ProviderDependencyError(
            "pyannote.audio is required for speaker diarization. "
            "Install with: pip install -e '.[diarize]'"
        )
```

## Key Decisions

1. **Additive pyannote (not WhisperX replacement)**
   - **Decision**: Keep existing Whisper integration, add pyannote as a second pass
   - **Rationale**: Lower integration risk; preserves all existing Whisper code paths and tests; best diarization accuracy. WhisperX can be evaluated as a future optimization (see PRD-020 DC1)

2. **Segment-level (not word-level) diarization**
   - **Decision**: Assign one speaker per Whisper segment, not per word
   - **Rationale**: Simpler alignment; matches current screenplay format; word-level requires forced alignment (WhisperX territory). Can be added later

3. **Waveform loading via torchaudio**
   - **Decision**: Load audio with `torchaudio.load()` before passing to pyannote
   - **Rationale**: pyannote's file-path loading has a known 4x performance penalty (issue #1702). Waveform loading achieves 12s vs 50s for 3-min clips

4. **Separate `[diarize]` extra (not merged into `[ml]`)**
   - **Decision**: New optional dependency group
   - **Rationale**: pyannote adds `speechbrain`, `asteroid`, and HF model downloads — significantly heavier than spaCy. Users who want Whisper without diarization shouldn't pay this cost

5. **HuggingFace token from multiple sources**
   - **Decision**: Accept via `--hf-token`, `HF_TOKEN` env var, or `~/.huggingface/token`
   - **Rationale**: Follows HuggingFace ecosystem conventions; flexible for CI and local use

## Alternatives Considered

1. **WhisperX as full pipeline replacement**
   - **Description**: Replace current Whisper integration entirely with WhisperX
   - **Pros**: Single pipeline, batched inference (70x realtime), word-level timestamps, VAD pre-filtering
   - **Cons**: Replaces proven Whisper integration; slightly lower diarization accuracy (~5%); larger blast radius
   - **Why Deferred**: Too risky for initial implementation. Evaluate after diarization proves value

2. **Simple voice-activity-based speaker change detection**
   - **Description**: Use Silero VAD to detect speech segments and infer speaker changes from silence patterns
   - **Pros**: Lightweight, no HuggingFace token needed
   - **Cons**: Not actually diarization — still a heuristic; no voice embeddings; only marginally better than current gap-based approach
   - **Why Rejected**: Doesn't solve the core problem (no voice identity)

3. **Cloud diarization APIs (e.g., Google Speech-to-Text, AssemblyAI)**
   - **Description**: Use cloud APIs that include diarization
   - **Pros**: No local GPU needed; high accuracy
   - **Cons**: Per-minute costs; requires network; vendor lock-in; doesn't work offline
   - **Why Rejected**: Conflicts with project philosophy of local-first, optional cloud

## Testing Strategy

**Test Coverage:**

- **Unit tests**: Alignment algorithm, speaker name mapping, cache key generation, config validation
- **Integration tests**: pyannote pipeline on test audio fixtures (see RFC-059 for improved fixtures)
- **E2E tests**: Full pipeline with `--diarize` on sample podcast audio

**Test Organization:**

- `tests/unit/podcast_scraper/providers/ml/diarization/` — alignment, mapping, caching
- `tests/integration/providers/ml/test_diarization.py` — pyannote on fixture audio
- Marker: `@pytest.mark.diarization` (requires `[diarize]` extra + HuggingFace token)
- Marker: `@pytest.mark.slow` (diarization is inherently slow)

**Test Fixtures:**

- Depends on RFC-059 improvements: unique voices per speaker in test audio make diarization testable
- Without distinct voices, pyannote cannot distinguish speakers in TTS-generated audio
- May need a small set of real podcast audio snippets (CC-licensed) for integration tests

**Test Execution:**

- Unit tests: `make test` (fast, no ML dependencies)
- Integration tests: `make test-slow` or dedicated `make test-diarize` target
- CI: Skip diarization tests by default (no GPU, no HuggingFace token); run in dedicated slow CI job

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1**: Core diarization provider + alignment algorithm + unit tests
- **Phase 2**: Pipeline integration + caching + CLI flags + config
- **Phase 3**: Integration tests with real/improved test audio + benchmark on 10+ episodes
- **Phase 4**: Documentation, HuggingFace setup guide, examples

**Monitoring:**

- Diarization Error Rate (DER) measured on benchmark episodes
- Processing time logged per episode (diarization step isolated)
- Cache hit rate tracked in pipeline metrics

**Success Criteria:**

1. DER < 15% on benchmark episodes
2. Screenplay attribution >= 85% correct on manual spot-check
3. Processing overhead < 2x vs Whisper-only (with GPU)
4. Zero test regressions with `diarize=false`

## Relationship to Other RFCs

This RFC (RFC-058) is part of the audio pipeline evolution that includes:

1. **RFC-006: Screenplay Formatting** — Current gap-based system; RFC-058 provides an alternative path
2. **RFC-010: Speaker Name Detection** — NER pipeline preserved; names map to diarized speaker IDs
3. **RFC-040: Audio Preprocessing** — Preprocessed audio feeds into diarization
4. **RFC-059: Speaker Detection Refactor & Test Audio** — Companion RFC; refactors speaker detection module and improves test audio to make diarization testable

**Key Distinction:**

- **RFC-058** (this): Adds the diarization *capability* (pyannote, alignment, pipeline integration)
- **RFC-059**: Improves the *infrastructure* that diarization depends on (modular speaker detection, test audio with distinct voices)

Together, these RFCs deliver:

- Accurate, voice-based speaker attribution in transcripts
- Clean, modular speaker detection architecture ready for diarization integration
- Test fixtures capable of validating diarization quality

## Benefits

1. **Accurate speaker attribution**: Neural voice embeddings replace blind gap-based rotation
2. **Auto speaker count**: Eliminates manual `screenplay_num_speakers` configuration
3. **Multi-speaker support**: Correctly handles panels with 3+ speakers
4. **Downstream quality**: Improves GIL quote attribution and KG speaker nodes
5. **Opt-in safety**: Zero impact on existing users and workflows

## Migration Path

1. **Phase 1**: Ship with `diarize=false` default — zero user impact
2. **Phase 2**: Users opt in with `--diarize` — requires `[diarize]` install + HuggingFace token
3. **Phase 3**: Once validated, consider making diarization the default for `--screenplay` when `[diarize]` is installed
4. **Future**: Evaluate WhisperX for bundled pipeline if speed/word-level timestamps become priorities

## Open Questions

1. Should `[diarize]` be merged into `[ml]` extra or remain separate?
2. Should the alignment use an interval tree library (e.g., `intervaltree`) or a simple sorted-list scan?
3. How to handle the HuggingFace model acceptance UX — CLI wizard on first run?
4. Should diarization confidence scores be exposed in output metadata?
5. What real podcast audio (CC-licensed) to use for integration test benchmarks?

## References

- **Related PRD**: `docs/prd/PRD-020-audio-speaker-diarization.md`
- **Related Issues**: #414 (Audio Pipeline Separation)
- **Source Code**: `src/podcast_scraper/providers/ml/ml_provider.py` (screenplay formatting)
- **Source Code**: `src/podcast_scraper/providers/ml/speaker_detection.py` (NER pipeline)
- **External**: [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio/) (v4.0.4)
- **External**: [m-bain/whisperX](https://github.com/m-bain/whisperX) (v3.3.2, future option)
- **External**: [pyannote issue #1702](https://github.com/pyannote/pyannote-audio/issues/1702) (waveform loading performance)
