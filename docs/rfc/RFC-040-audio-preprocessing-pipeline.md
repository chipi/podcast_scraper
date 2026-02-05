# RFC-040: Audio Pre-Processing Pipeline for Podcast Ingestion

- **Status**: Draft
- **Authors**: Architecture Review
- **Stakeholders**: Core Pipeline, Transcription Providers
- **Related PRDs**:
  - N/A (new capability)
- **Related ADRs**:
  - [ADR-032: Standardized Pre-Provider Audio Stage](../adr/ADR-032-standardized-pre-provider-audio-stage.md)
  - [ADR-033: Content-Hash Based Audio Caching](../adr/ADR-033-content-hash-based-audio-caching.md)
  - [ADR-034: FFmpeg-First Audio Manipulation](../adr/ADR-034-ffmpeg-first-audio-manipulation.md)
  - [ADR-035: Speech-Optimized Codec (Opus)](../adr/ADR-035-speech-optimized-codec-opus.md)
- **Related RFCs**:
  - `docs/rfc/RFC-005-whisper-integration.md` (transcription pipeline)
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (provider pattern)
  - `docs/rfc/RFC-029-provider-refactoring-consolidation.md` (unified provider architecture)
- **Related Documents**:
  - `docs/ARCHITECTURE.md` (pipeline flow)
  - `docs/guides/PROVIDER_IMPLEMENTATION_GUIDE.md` (provider patterns)

## Abstract

This RFC proposes a dedicated **audio pre-processing stage** in the podcast ingestion pipeline to optimize raw audio files **before** they are sent to any transcription provider (local Whisper or OpenAI API). The pre-processor will convert audio to mono, resample to 16 kHz, remove silence using Voice Activity Detection (VAD), and normalize loudness. This is critical for API-based transcription where file size limits and per-minute costs make optimization essential. It also improves performance for local Whisper transcription.

**Architecture Alignment:** This RFC introduces preprocessing as a **pipeline stage that occurs before provider selection**, not as part of the provider itself. Audio is preprocessed in `episode_processor.py` after download but before passing to any transcription provider. This ensures all providers (Whisper, OpenAI, future providers) benefit from optimized audio, maintains separation of concerns, and addresses API upload size limits.

## Problem Statement

The current podcast ingestion pipeline downloads audio files and passes them directly to transcription providers (Whisper or OpenAI Whisper API). Podcast audio often contains:

- **Long silence periods** (intros, outros, pauses)
- **Music segments** (intro/outro themes, mid-roll music)
- **Inconsistent loudness levels** (varying recording quality across episodes)
- **Unnecessarily high fidelity** (48kHz, stereo, high bitrates)

This leads to:

1. **API upload size limits** — Many podcast episodes exceed provider file size limits
2. **Longer transcription runtimes** — More audio data to process (both local and API)
3. **Higher API costs** — Providers charge per audio minute, including non-speech segments
4. **Higher compute costs** — Local Whisper processes silence and music unnecessarily
5. **Larger transcripts** — Non-speech segments may produce "[MUSIC]", "[SILENCE]", or filler text
6. **Inconsistent quality** — Varying audio levels affect transcription accuracy

### API Provider File Size Limits (Official Documentation)

Different transcription providers impose various file size constraints:

| Provider | API Endpoint | Max File Size | Max Duration | Documentation | Status |
|----------|-------------|---------------|--------------|---------------|--------|
| **OpenAI Whisper** | Audio API | **25 MB** | N/A | [Official Docs](https://platform.openai.com/docs/guides/speech-to-text/whisper) | ✅ Supported |
| **Mistral Voxtral** | Audio API | **TBD** (likely similar to OpenAI) | TBD | [API Docs](https://docs.mistral.ai/) | 🔄 Planned (PRD-010) |
| **Google Gemini** | Multimodal API | **TBD** (native audio) | TBD | [AI Studio](https://ai.google.dev/) | 🔄 Planned (PRD-012) |
| **Grok** | N/A | ❌ No transcription | N/A | [API Docs](https://docs.x.ai) | ✅ Implemented (PRD-013, LLMs only) |
| **Google Cloud (sync)** | Speech-to-Text | **10 MB** | ~1 minute | [Quotas](https://cloud.google.com/speech-to-text/quotas) | ❌ Not planned |
| **Azure Speech** | Fast/Batch | **300 MB - 1 GB** | 120-240 min | [Service Limits](https://learn.microsoft.com/azure/ai-services/speech-service/speech-services-quotas-and-limits) | ❌ Not planned |

**Transcription Provider Support Matrix** (from PRDs):

| Provider | Transcription | Speaker Detection | Summarization | Notes |
|----------|---------------|-------------------|---------------|-------|
| **Local (Whisper)** | ✅ | ✅ (spaCy) | ✅ (Transformers) | No size limits |
| **OpenAI** | ✅ (25 MB limit) | ✅ | ✅ | Currently supported |
| **Mistral** | ✅ Voxtral (limit TBD) | ✅ | ✅ | Planned (PRD-010) |
| **Gemini** | ✅ Native audio (limit TBD) | ✅ | ✅ | Planned (PRD-012) |
| **Anthropic** | ❌ | ✅ | ✅ | No audio API (PRD-009) |
| **DeepSeek** | ❌ | ✅ | ✅ | No audio API (PRD-011) |
| **Grok** | ❌ | ✅ | ✅ | LLMs only (PRD-013) |
| **Ollama** | ❌ | ✅ | ✅ | Local LLMs (PRD-014) |

**Critical Constraints**:

- **OpenAI**: 25 MB hard limit (most restrictive known, currently supported)
- **Mistral & Gemini**: Limits unknown, assumed similar or more generous
- **Conservative Design**: Target <25 MB to ensure compatibility with all current and planned providers

**Common Denominator Approach**:

To support both current (OpenAI) and planned (Mistral, Gemini) API transcription providers, we adopt a **conservative <25 MB target**:

1. **Known constraint**: OpenAI = 25 MB (confirmed)
2. **Unknown constraints**: Mistral Voxtral and Gemini audio limits not yet documented
3. **Safe assumption**: Designing for 25 MB ensures compatibility even if future providers have similar or slightly more restrictive limits
4. **Benefit**: If Mistral/Gemini have more generous limits (>25 MB), preprocessed files will still be well-optimized

**Real-World Impact**:

- Raw podcast episodes (30-60 minutes) typically range from **30-100 MB** in MP3 format
- **70-80% of podcast episodes exceed the 25 MB limit** and cannot be uploaded to OpenAI without preprocessing
- Preprocessing is **mandatory** (not optional) for API-based transcription of typical podcast content
- **Target output**: 2-5 MB for typical episodes (10-25× reduction, well below any likely provider limit)

**Real-World Impact**:

- Raw podcast episodes (30-60 minutes) typically range from **30-100 MB** in MP3 format
- **70-80% of podcast episodes exceed the 25 MB limit** and cannot be uploaded without preprocessing
- Preprocessing is **not optional** for API-based transcription of typical podcast content

We need a **pipeline stage that preprocesses audio before it reaches any provider**, ensuring files fit within the 25 MB constraint while optimizing for both local and API-based transcription.

**Use Cases:**

1. **API Upload Size Compliance**: Reduce audio files to fit within 25 MB limit (OpenAI, Grok)
2. **Cost Optimization**: Reduce API costs by removing silence and music before API calls
3. **Performance Optimization**: Speed up local Whisper transcription by processing smaller audio files
4. **Quality Standardization**: Normalize audio levels across different podcast sources for consistent transcription quality

## Goals

1. **Ensure API compatibility** — All preprocessed audio must be **<25 MB** to support OpenAI & Grok
2. **Reduce audio size for API upload limits** — Target 10-25× reduction (typical 50 MB → 2-5 MB)
3. **Lower transcription costs** — Reduce API usage by processing less audio (30-60% cost savings)
4. **Improve transcription performance** — Faster processing for both local and API providers
5. **Improve transcription accuracy** — Consistent audio levels and speech-only content
6. **Standardize audio inputs** — All podcasts processed with consistent quality
7. **Maintain backward compatibility** — Preprocessing is optional and configurable
8. **Cache preprocessed audio** — Avoid reprocessing with content-based hashing
9. **Provider-agnostic** — All transcription providers receive optimized audio

## Constraints & Assumptions

**Constraints:**

- Must not break existing transcription workflows (backward compatibility)
- Must work **before** any provider is invoked (not provider-specific)
- Must ensure preprocessed files are **<25 MB** (OpenAI & Grok constraint - most restrictive)
- Must handle various input formats (MP3, M4A, WAV, etc.)
- Must be performant enough not to negate transcription time savings
- Must preserve audio segment boundaries for accurate transcription timing
- Must be optional and configurable (disabled by default for initial rollout)
- Must not require provider-specific implementations

**Assumptions:**

- `ffmpeg` is available on the system (already common dependency for audio processing)
- VAD preprocessing adds 10-30% overhead but saves 30-60% on transcription time (net win)
- Content-based hashing is sufficient for cache keys (no security sensitivity)
- Preprocessed audio can be cached in `.cache/` directory structure
- Aggressive VAD with conservative thresholds (preserve meaningful pauses) is acceptable

## Design & Implementation

### 1. Module Structure

Create a new `preprocessing` module following the provider pattern:

```text
src/podcast_scraper/
├── preprocessing/
│   ├── __init__.py
│   ├── base.py              # AudioPreprocessor protocol
│   ├── factory.py           # Factory function
│   ├── ffmpeg_processor.py  # FFmpeg-based implementation
│   └── cache.py             # Preprocessed audio caching
```

### 2. AudioPreprocessor Protocol

Define a protocol for audio preprocessing:

```python
# preprocessing/base.py
from typing import Protocol, Tuple
from pathlib import Path

class AudioPreprocessor(Protocol):
    """Protocol for audio preprocessing providers."""

    def preprocess(
        self,
        input_path: str,
        output_path: str,
    ) -> Tuple[bool, float]:
        """Preprocess audio file for transcription.

        Args:
            input_path: Path to raw audio file
            output_path: Path to save preprocessed audio

        Returns:
            Tuple of (success: bool, elapsed_time: float)
        """
        ...

    def get_cache_key(self, input_path: str) -> str:
        """Generate content-based cache key for input audio.

        Args:
            input_path: Path to audio file

        Returns:
            Cache key (hash of audio content + preprocessing settings)
        """
        ...
```

### 3. FFmpeg Implementation

Implement audio preprocessing using `ffmpeg`:

```python
# preprocessing/ffmpeg_processor.py
import hashlib
import logging
import subprocess
import time
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

class FFmpegAudioPreprocessor:
    """Audio preprocessor using ffmpeg for format conversion and silenceremove filter."""

    def __init__(self, cfg):
        """Initialize preprocessor with configuration.

        Args:
            cfg: Configuration object with preprocessing settings
        """
        self.cfg = cfg
        self.sample_rate = cfg.preprocessing_sample_rate  # Default: 16000
        self.silence_threshold = cfg.preprocessing_silence_threshold  # Default: -50dB
        self.silence_duration = cfg.preprocessing_silence_duration  # Default: 2.0s
        self.target_loudness = cfg.preprocessing_target_loudness  # Default: -16 LUFS

    def preprocess(self, input_path: str, output_path: str) -> Tuple[bool, float]:
        """Preprocess audio using ffmpeg pipeline.

        Pipeline stages:
        1. Convert to mono
        2. Resample to 16 kHz
        3. Remove silence (VAD)
        4. Normalize loudness to -16 LUFS
        5. Compress using speech-optimized codec

        Args:
            input_path: Path to raw audio file
            output_path: Path to save preprocessed audio

        Returns:
            Tuple of (success: bool, elapsed_time: float)
        """
        start_time = time.time()

        # Build ffmpeg command
        # -ac 1: convert to mono
        # -ar 16000: resample to 16 kHz
        # -af silenceremove: remove silence (conservative thresholds)
        # -af loudnorm: normalize loudness to -16 LUFS
        # -c:a libopus: speech-optimized codec
        # -b:a 24k: 24 kbps bitrate
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-ac", "1",  # Mono
            "-ar", str(self.sample_rate),  # 16 kHz
            "-af", (
                f"silenceremove="
                f"start_periods=1:"
                f"start_threshold={self.silence_threshold}:"
                f"start_duration={self.silence_duration}:"
                f"stop_periods=-1:"
                f"stop_threshold={self.silence_threshold}:"
                f"stop_duration={self.silence_duration},"
                f"loudnorm=I={self.target_loudness}:LRA=11:TP=-1.5"
            ),
            "-c:a", "libopus",  # Speech-optimized codec
            "-b:a", "24k",  # 24 kbps
            "-y",  # Overwrite output
            output_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=True,
            )
            elapsed = time.time() - start_time
            logger.debug("Audio preprocessing completed in %.1fs", elapsed)
            return True, elapsed

        except subprocess.CalledProcessError as exc:
            logger.error("FFmpeg preprocessing failed: %s", exc.stderr)
            return False, time.time() - start_time
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg preprocessing timed out after 300s")
            return False, time.time() - start_time
        except FileNotFoundError:
            logger.error("FFmpeg not found. Install ffmpeg to use audio preprocessing.")
            return False, 0.0

    def get_cache_key(self, input_path: str) -> str:
        """Generate cache key from file content hash + preprocessing config.

        Args:
            input_path: Path to audio file

        Returns:
            Cache key (SHA256 hash of content + config)
        """
        hasher = hashlib.sha256()

        # Hash file content (first 1MB for performance)
        try:
            with open(input_path, "rb") as f:
                hasher.update(f.read(1024 * 1024))
        except OSError as exc:
            logger.warning("Failed to hash audio file: %s", exc)
            # Use file path as fallback
            hasher.update(input_path.encode("utf-8"))

        # Hash preprocessing config to invalidate cache when settings change
        config_str = (
            f"{self.sample_rate}|"
            f"{self.silence_threshold}|"
            f"{self.silence_duration}|"
            f"{self.target_loudness}"
        )
        hasher.update(config_str.encode("utf-8"))

        return hasher.hexdigest()[:16]  # 16 hex chars (64 bits)
```

### 4. Caching Strategy

Implement caching to avoid reprocessing:

```python
# preprocessing/cache.py
import os
from pathlib import Path
from typing import Optional

PREPROCESSING_CACHE_DIR = ".cache/preprocessing"

def get_cached_audio_path(
    cache_key: str,
    cache_dir: str = PREPROCESSING_CACHE_DIR,
) -> Optional[str]:
    """Check if preprocessed audio exists in cache.

    Args:
        cache_key: Content-based cache key
        cache_dir: Cache directory path

    Returns:
        Path to cached audio if exists, None otherwise
    """
    cache_path = os.path.join(cache_dir, f"{cache_key}.mp3")
    if os.path.exists(cache_path):
        return cache_path
    return None

def save_to_cache(
    source_path: str,
    cache_key: str,
    cache_dir: str = PREPROCESSING_CACHE_DIR,
) -> str:
    """Save preprocessed audio to cache.

    Args:
        source_path: Path to preprocessed audio
        cache_key: Content-based cache key
        cache_dir: Cache directory path

    Returns:
        Path to cached audio
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.mp3")

    # Copy to cache (or move if source is temp)
    import shutil
    shutil.copy2(source_path, cache_path)

    return cache_path
```

### 5. Integration with Transcription Pipeline

Modify `episode_processor.py` to preprocess audio **before** passing to any provider:

```python
# episode_processor.py (modified)
def transcribe_media_to_text(
    job: models.TranscriptionJob,
    cfg: config.Config,
    whisper_model,
    run_suffix: Optional[str],
    effective_output_dir: str,
    transcription_provider=None,
    audio_preprocessor=None,  # NEW: Optional AudioPreprocessor
    pipeline_metrics=None,
) -> tuple[bool, Optional[str], int]:
    """Transcribe media file using transcription provider and save result.

    Audio preprocessing happens BEFORE provider selection - all providers
    receive optimized audio. This is critical for API providers with upload
    size limits (e.g., OpenAI 25 MB limit) and reduces costs for metered APIs.
    """
    # ... existing validation code ...

    temp_media = job.temp_media
    media_for_transcription = temp_media

    # NEW: Preprocess audio BEFORE passing to any provider
    # This happens at the pipeline level, not within providers
    if audio_preprocessor and cfg.preprocessing_enabled:
        cache_key = audio_preprocessor.get_cache_key(temp_media)

        # Check cache first
        cached_path = preprocessing.cache.get_cached_audio_path(cache_key)
        if cached_path:
            logger.debug("Using cached preprocessed audio: %s", cache_key)
            media_for_transcription = cached_path
        else:
            # Preprocess audio
            preprocessed_path = f"{temp_media}.preprocessed.mp3"
            success, preprocess_elapsed = audio_preprocessor.preprocess(
                temp_media, preprocessed_path
            )

            if success:
                # Save to cache
                cached_path = preprocessing.cache.save_to_cache(
                    preprocessed_path, cache_key
                )
                media_for_transcription = cached_path

                # Log size reduction
                original_size = os.path.getsize(temp_media)
                preprocessed_size = os.path.getsize(cached_path)
                reduction = (1 - preprocessed_size / original_size) * 100
                logger.info(
                    "    preprocessed audio: %.1f%% smaller (%.1fMB -> %.1fMB) in %.1fs",
                    reduction,
                    original_size / (1024 * 1024),
                    preprocessed_size / (1024 * 1024),
                    preprocess_elapsed,
                )

                # Record preprocessing time
                if pipeline_metrics:
                    pipeline_metrics.record_preprocessing_time(preprocess_elapsed)
            else:
                # Preprocessing failed, use original audio
                logger.warning("Audio preprocessing failed, using original audio")
                media_for_transcription = temp_media

    # Pass preprocessed (or original) audio to provider
    # Provider is agnostic to whether audio was preprocessed
    try:
        result, tc_elapsed = transcription_provider.transcribe_with_segments(
            media_for_transcription,  # Preprocessed or original - provider doesn't know/care
            language=cfg.language
        )
        # ... rest of transcription code ...
```

**Key Architecture Points**:

1. **Preprocessing happens at pipeline level** — Not inside providers
2. **All providers receive optimized audio** — Whisper, OpenAI, future providers all benefit
3. **Providers are agnostic** — They don't know if audio was preprocessed
4. **Cache is shared** — Same preprocessed audio reused across provider switches
5. **Separation of concerns** — Preprocessing is orthogonal to transcription

```

### 6. Configuration Fields

Add preprocessing configuration to `Config`:

```python
# config.py (additions)
class Config(BaseModel):
    # ... existing fields ...

    # Audio Preprocessing
    preprocessing_enabled: bool = Field(
        default=False,
        description="Enable audio preprocessing before transcription (default: False)"
    )
    preprocessing_sample_rate: int = Field(
        default=16000,
        description="Target sample rate for preprocessing (default: 16000 Hz)"
    )
    preprocessing_silence_threshold: str = Field(
        default="-50dB",
        description="Silence detection threshold (default: -50dB)"
    )
    preprocessing_silence_duration: float = Field(
        default=2.0,
        description="Minimum silence duration to remove in seconds (default: 2.0)"
    )
    preprocessing_target_loudness: int = Field(
        default=-16,
        description="Target loudness in LUFS for normalization (default: -16)"
    )
    preprocessing_cache_dir: Optional[str] = Field(
        default=None,
        description="Custom cache directory for preprocessed audio (default: .cache/preprocessing)"
    )
```

### 7. CLI Arguments

Add CLI flags for preprocessing:

```python
# cli.py (additions)
def _add_preprocessing_arguments(parser: argparse.ArgumentParser) -> None:
    """Add audio preprocessing arguments to parser."""
    preprocessing_group = parser.add_argument_group("Audio Preprocessing")
    preprocessing_group.add_argument(
        "--enable-preprocessing",
        action="store_true",
        dest="preprocessing_enabled",
        help="Enable audio preprocessing before transcription (experimental)"
    )
    preprocessing_group.add_argument(
        "--preprocessing-silence-threshold",
        type=str,
        default="-50dB",
        help="Silence detection threshold (default: -50dB)"
    )
    # ... other preprocessing flags ...
```

## Key Decisions

1. **Pipeline-Level Preprocessing, Not Provider-Level**
   - **Decision**: Preprocessing happens in `episode_processor.py` before provider invocation, not within providers
   - **Rationale**: API providers have upload size limits (OpenAI 25 MB). Preprocessing must happen before upload, not after. All providers benefit equally from optimized audio. Separation of concerns - preprocessing is orthogonal to transcription method.

2. **FFmpeg as Implementation**
   - **Decision**: Use `ffmpeg` for audio preprocessing
   - **Rationale**: Industry-standard tool, widely available, supports all needed operations (format conversion, VAD, loudness normalization). Alternative libraries (pydub, librosa) would add Python dependencies and may be slower.

3. **Optional and Disabled by Default**
   - **Decision**: Preprocessing is opt-in via `preprocessing_enabled=True`
   - **Rationale**: Maintains backward compatibility. Users can test preprocessing on their workloads before enabling permanently. Allows gradual rollout.

4. **Content-Based Caching**
   - **Decision**: Cache preprocessed audio using content hash + config hash
   - **Rationale**: Avoids reprocessing the same audio multiple times. Common in development/testing when reprocessing episodes. Cache invalidates automatically when preprocessing settings change.

5. **Conservative VAD Thresholds**
   - **Decision**: Use conservative silence detection thresholds by default
   - **Rationale**: Preserves meaningful pauses and prevents removing legitimate speech. Users can tune thresholds if needed.

6. **Opus Codec for Preprocessed Audio**
   - **Decision**: Use Opus codec at 24 kbps for preprocessed audio
   - **Rationale**: Opus is optimized for speech, provides excellent quality at low bitrates, and is widely supported. 24 kbps provides good speech quality while reducing file size significantly.

## Alternatives Considered

1. **No Preprocessing**
   - **Description**: Continue processing raw audio directly
   - **Pros**: Simpler pipeline, no additional dependencies
   - **Cons**: Higher costs, longer processing times, inconsistent quality
   - **Why Rejected**: Benefits of preprocessing outweigh the added complexity

2. **Text-Only Cleanup After Transcription**
   - **Description**: Remove silence markers ("[MUSIC]", "[SILENCE]") from transcripts post-transcription
   - **Pros**: Easier to implement, no audio processing needed
   - **Cons**: Does not reduce transcription time or costs, does not improve transcription quality
   - **Why Rejected**: Does not address the root cause (processing unnecessary audio)

3. **Python-Based VAD (webrtcvad, silero-vad)**
   - **Description**: Use Python VAD libraries instead of ffmpeg
   - **Pros**: No external dependencies, potentially more control
   - **Cons**: Adds Python dependencies, may be slower, requires manual audio format handling
   - **Why Rejected**: FFmpeg provides a complete solution (VAD + format conversion + normalization) in one tool

4. **Mandatory Preprocessing**
   - **Description**: Make preprocessing mandatory for all transcription
   - **Pros**: Ensures consistent quality across all workflows
   - **Cons**: Breaking change, may not be desirable for all use cases
   - **Why Rejected**: Too disruptive, opt-in approach allows users to test and validate benefits first

## Testing Strategy

**Test Coverage:**

- **Unit Tests**: FFmpeg command generation, cache key generation, error handling
- **Integration Tests**: Full preprocessing pipeline with real audio files, cache hit/miss scenarios
- **E2E Tests**: Preprocessing + transcription workflow with both Whisper and OpenAI providers

**Test Organization:**

- `tests/unit/test_preprocessing.py` — Unit tests for preprocessor logic
- `tests/integration/test_preprocessing_integration.py` — Integration tests with real audio
- `tests/e2e/test_preprocessing_e2e.py` — E2E tests with full pipeline
- Use `tests/fixtures/audio/` for test audio files (small samples)

**Test Execution:**

- Unit tests: Run in CI (fast, no external dependencies except ffmpeg availability check)
- Integration tests: Run in CI with `@pytest.mark.integration` and `@pytest.mark.skipif(not has_ffmpeg())`
- E2E tests: Run in nightly CI or on-demand

**Test Data:**

- Create small test audio files (~10s) with known characteristics:
  - `silence_test.mp3` — Audio with long silence periods
  - `music_test.mp3` — Audio with music intro/outro
  - `quiet_test.mp3` — Low-volume audio
  - `loud_test.mp3` — High-volume audio

**Performance Validation:**

- Measure preprocessing overhead vs. transcription time savings
- Target: Preprocessing adds <30% overhead but saves >30% transcription time (net positive)
- Track metrics: preprocessing time, audio size reduction, transcription time, total pipeline time

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1: Experimental (v2.5.0)** — Release as opt-in feature, document as experimental, gather user feedback
- **Phase 2: Refinement (v2.6.0)** — Refine thresholds based on feedback, improve caching, add more configuration options
- **Phase 3: Stable (v2.7.0)** — Mark as stable, consider enabling by default for new users

**Monitoring:**

- Track preprocessing metrics:
  - `preprocessing_time_seconds` — Time spent preprocessing
  - `preprocessing_audio_size_reduction` — Size reduction ratio
  - `preprocessing_cache_hit_rate` — Cache effectiveness
- Track transcription metrics impact:
  - `transcription_time_seconds` (before/after preprocessing)
  - `transcription_cost_usd` (for OpenAI provider)
  - `transcript_word_count` (may decrease with silence removal)
- Log warnings for:
  - FFmpeg not available
  - Preprocessing timeouts
  - Preprocessing failures (fallback to original audio)

**Success Criteria:**

1. ✅ Preprocessing reduces transcription time by ≥30% on average
2. ✅ Preprocessing reduces OpenAI API costs by ≥30% on average
3. ✅ Preprocessing maintains or improves transcription quality (manual review)
4. ✅ Cache hit rate ≥80% in development/testing scenarios
5. ✅ Zero breaking changes to existing workflows

## Expected Impact

| Area | Expected Improvement | Measurement |
|------|---------------------|-------------|
| Audio size | 10–25× smaller | File size comparison (MB) |
| API compatibility | 100% of podcasts fit <25 MB | Upload success rate |
| Preprocessing overhead | +10–30% time | Time measurement (seconds) |
| Transcription runtime | -30–60% faster | Time measurement (seconds) |
| OpenAI/Grok API cost | -30–60% reduction | API usage tracking ($) |
| Transcript size | -15–40% fewer tokens | Word count comparison |
| Transcription quality | Improved consistency | Manual QA review |
| Total pipeline time | -20–40% faster | End-to-end time measurement |

**Net Impact**: Despite preprocessing overhead, total pipeline time should decrease due to larger transcription time savings.

**File Size Target**: Preprocessed audio should average **2-5 MB** for typical 30-60 minute podcast episodes, well below the 25 MB limit with safety margin.

## Benefits

1. **Cost Reduction**: Significant reduction in OpenAI Whisper API costs by processing less audio
2. **Performance Improvement**: Faster transcription for both local Whisper and OpenAI API
3. **Quality Standardization**: Consistent audio levels and speech-only content improve transcription accuracy
4. **Flexibility**: Optional and configurable, users can tune for their specific needs
5. **Caching**: Preprocessed audio is cached, avoiding reprocessing in development/testing workflows
6. **Backward Compatibility**: Existing workflows continue to work unchanged

## Migration Path

**No migration needed** — preprocessing is opt-in:

1. **Existing Users**: No changes required, workflows continue as-is
2. **New Users**: Can optionally enable preprocessing in config or via CLI flag
3. **Gradual Adoption**: Users can test preprocessing on a subset of episodes before enabling globally

**Enabling Preprocessing:**

```yaml
# config.yaml
preprocessing_enabled: true  # Enable preprocessing
```

Or via CLI:

```bash
python3 -m podcast_scraper.cli https://feed.xml --enable-preprocessing
```

## Open Questions

1. **FFmpeg Availability Check**: Should we fail fast if ffmpeg is not available, or silently disable preprocessing?
   - **Proposed**: Warn and disable preprocessing if ffmpeg not found, log clear error message

2. **Codec Selection**: Should we support multiple output codecs (Opus, AAC, MP3) or standardize on Opus?
   - **Proposed**: Start with Opus only, add codec selection in future if users request it

3. **VAD Threshold Tuning**: Should we provide presets (conservative, balanced, aggressive) or expect users to tune manually?
   - **Proposed**: Start with single conservative default, add presets in Phase 2 based on feedback

4. **Preprocessing for OpenAI Only**: Should preprocessing be automatic for OpenAI provider (to save costs) but optional for Whisper?
   - **Proposed**: Keep consistent (manual opt-in for all providers), document cost savings for OpenAI users

5. **Cache Cleanup**: Should we provide a tool to clean preprocessing cache (similar to ML model cache cleanup)?
   - **Proposed**: Yes, add `cache --clean preprocessing` subcommand and include in `clean-all`

## References

- **Related RFC**: `docs/rfc/RFC-005-whisper-integration.md` (transcription pipeline)
- **Related RFC**: `docs/rfc/RFC-029-provider-refactoring-consolidation.md` (unified providers)
- **Source Code**: `podcast_scraper/episode_processor.py` (transcription orchestration)
- **External Tools**: FFmpeg documentation ([https://ffmpeg.org/](https://ffmpeg.org/))
- **VAD Reference**: FFmpeg silenceremove filter ([https://ffmpeg.org/ffmpeg-filters.html#silenceremove](https://ffmpeg.org/ffmpeg-filters.html#silenceremove))
