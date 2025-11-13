# PRD-002: Whisper Fallback Transcription

## Summary
Provide an optional fallback path that transcribes podcast audio with OpenAI Whisper when no published transcript exists. This complements PRD-001 by guaranteeing text output coverage for every episode.

## Background & Context
- Many feeds omit transcripts or only publish them for recent episodes.
- Users want a single pipeline that can produce text for *all* items, defaulting to official transcripts but transcribing otherwise.
- Whisper support depends on local environment capabilities (GPU/CPU, `openai-whisper`, `ffmpeg`).

## Goals
- Detect episodes lacking transcripts and automatically produce a Whisper-based transcript when the feature is enabled.
- Keep Whisper integration optional so environments without the dependency can still run baseline downloads.
- Offer operators visibility into Whisper-specific work (media downloads, transcription progress, runtime).

## Non-Goals
- Training or fine-tuning custom ASR models.
- Multi-language transcription (initial release assumes English task mode).
- Parallel transcription (sequential processing keeps resource usage predictable).

## Personas
- **Archivist Ava**: Needs a complete text archive for compliance; missing transcripts must be filled automatically.
- **Creator Casey**: Runs Whisper in dry-run mode to understand impact before enabling it in production.

## User Stories
- *As Archivist Ava, I can enable `--transcribe-missing` and trust that every episode ends up with a transcript file.*
- *As Creator Casey, I can specify which Whisper model to use (e.g., `base`, `small`) based on latency/quality requirements.*
- *As any operator, I can see when media is downloaded, how large it is, and how long transcription takes.*
- *As any operator, I can format transcriptions as screenplay-style dialog with consistent speaker labeling.*
- *As any operator, I can have speaker names automatically detected from episode metadata without manual configuration (RFC-010).*
- *As any operator, I can specify the podcast language to optimize Whisper model selection (English vs multilingual) and improve transcription accuracy.*

## Functional Requirements
- **FR1**: Flag-controlled activation (`--transcribe-missing`) with model selection via `--whisper-model` (validated against supported list).
- **FR2**: Only queue Whisper jobs for episodes lacking a usable transcript after PRD-001 pipeline evaluation.
- **FR3**: Download enclosure media to a per-run temp folder with robust retry/backoff and cleanup after use.
- **FR4**: Reuse `--skip-existing` semantics so preexisting Whisper outputs short-circuit repeat work.
- **FR5**: Run Whisper sequentially, reporting progress through the shared progress API and logging per-episode timing.
- **FR6**: Support screenplay formatting via `--screenplay`, `--num-speakers`, `--speaker-names`, and `--screenplay-gap`.
- **FR7**: Respect `--dry-run` behavior (log planned downloads/transcriptions without touching disk).
- **FR8**: Append run suffixes indicating Whisper usage (e.g., `_whisper_base`) to distinguish outputs.
- **FR9**: Automatically detect speaker names from episode metadata using Named Entity Recognition (RFC-010) when `--auto-speakers` is enabled (default).
- **FR10**: Use detected speaker names in screenplay formatting unless manually overridden via `--speaker-names`.
- **FR11**: Support language configuration (`--language`) that drives both Whisper model selection (preferring `.en` variants for English) and NER model selection.
- **FR12**: Pass language parameter to Whisper transcription API to improve accuracy for non-English content.

## Success Metrics
- When enabled, 100% of episodes produce a transcript file (download or Whisper).
- Median transcription time stays within expectations for chosen model (documented in README).
- Temp media directory is always cleaned up (best-effort) even on failures.

## Dependencies
- Media download reliability and naming conventions from `docs/rfc/RFC-004-filesystem-layout.md`.
- Whisper loading/transcription mechanics in `docs/rfc/RFC-005-whisper-integration.md` and screenplay formatting in `docs/rfc/RFC-006-screenplay-formatting.md`.
- Automatic speaker name detection via Named Entity Recognition in `docs/rfc/RFC-010-speaker-name-detection.md`.

## Release Checklist
- [ ] README documents environment setup for Whisper (`openai-whisper`, `ffmpeg`).
- [ ] Tests cover dry-run, successful transcription, screenplay formatting, error handling (missing Whisper install).
- [ ] Logging reviewed for clear differentiation between download and transcription stages.

## Open Questions
- Should we allow per-episode inclusion/exclusion lists for Whisper (e.g., metadata filters)? Deferred.
- Do we need GPU auto-detection to warn users about potential performance impacts? Nice-to-have, not in scope.

## RFC-010 Integration

This PRD integrates with RFC-010 (Automatic Speaker Name Detection) to enhance Whisper transcription:

- **Automatic Speaker Detection**: When enabled, speaker names are extracted from episode titles, descriptions, and feed metadata using spaCy NER, eliminating the need for manual `--speaker-names` configuration.
- **Language Configuration**: The `--language` flag controls both Whisper model selection (preferring English-only `.en` variants when language is "en") and NER model selection (e.g., `en_core_web_sm` for English).
- **Model Selection**: For English podcasts, automatically prefer `.en` Whisper models (`base.en`, `small.en`, etc.) which perform better than multilingual variants.
- **Fallback Behavior**: If NER fails or spaCy is unavailable, fall back to default `["Host", "Guest"]` labels. Manual `--speaker-names` override always takes precedence.
- **Metadata Integration**: Detected speaker names are stored in episode metadata documents (per PRD-004) for downstream use cases.
