# RFC-005: Whisper Integration Lifecycle

- **Status**: Completed
- **Authors**: GPT-5 Codex (initial documentation)
- **Stakeholders**: Maintainers, operators enabling transcription fallback
- **Related PRD**: `docs/prd/PRD-002-whisper-fallback.md`

## Abstract

Document how the application loads, configures, and invokes OpenAI Whisper to transcribe podcast audio when transcripts are missing, including dependency handling and resource management.

## Problem Statement

Whisper brings heavyweight dependencies and variable runtime characteristics (CPU vs. GPU). We need a robust integration that loads models lazily, handles missing installations gracefully, and exposes consistent logging/metrics.

## Constraints & Assumptions

- Whisper dependency (`openai-whisper`) and `ffmpeg` may be absent; the system must emit actionable warnings and skip transcription rather than crash.
- Transcription runs sequentially to avoid resource contention and unpredictable GPU memory usage.
- English transcription is the default task (`language="en"`).

## Design & Implementation

1. **Model loading**
   - `whisper_integration.load_whisper_model(cfg)` imports the third-party `whisper` library from the `openai-whisper` package.
   - Validates requested model against `Config.VALID_WHISPER_MODELS`.
   - Logs model/device details; annotates model with `_is_cpu_device` to suppress FP16 warnings when necessary.
2. **Transcription execution**
   - `whisper.transcribe_with_whisper` wraps model invocation with progress reporting and timing.
   - Applies warning suppression (`warnings.catch_warnings`) when running on CPU to silence FP16 noise.
   - Returns `(result_dict, elapsed_seconds)` for logging by caller.
3. **Failure handling**
   - `ImportError` or runtime errors log warnings and result in `None` model so pipeline can continue without transcription.
   - Downstream callers remove temporary media whether transcription succeeds or fails.
4. **Configuration hooks**
   - CLI flags map to `Config` fields: `transcribe_missing`, `whisper_model`, `screenplay`, etc.
   - Whisper run suffix appended to output paths (see RFC-004).

## Key Decisions

- **Lazy import** prevents namespace collisions and lets transcript-only runs avoid Whisper dependency overhead.
- **Sequential processing** maximizes predictability and simplifies integration with progress reporter.
- **Environment detection** (CPU vs. GPU) drives logging and warning suppression for better UX.

## Alternatives Considered

- **Global import at module load time**: Rejected; would break environments without Whisper installed.
- **Parallel transcription**: Rejected to avoid complicated resource scheduling; revisit if demand emerges.

## Testing Strategy

- Unit tests mock Whisper library to verify loading paths and error handling.
- Integration tests use fixtures with fake Whisper objects to validate pipeline interactions without heavy dependencies.

## Rollout & Monitoring

- Logs make Whisper activation explicit: model name, device type, transcript saved path.
- Operators can monitor elapsed times per episode to tune model selection.

## References

- Source: `podcast_scraper/whisper_integration.py`
- Episode-level integration: `docs/rfc/RFC-003-transcript-downloads.md`
- Orchestrator: `docs/rfc/RFC-001-workflow-orchestration.md`
