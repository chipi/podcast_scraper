# RFC-006: Whisper Screenplay Formatting

- **Status**: Accepted
- **Authors**: GPT-5 Codex (initial documentation)
- **Stakeholders**: Maintainers, operators formatting transcripts for dialog review
- **Related PRD**: `docs/prd/PRD-002-whisper-fallback.md`

## Abstract
Describe the algorithm that converts Whisper transcription segments into screenplay-style dialog with speaker attribution, including configuration hooks and formatting guarantees.

## Problem Statement
When Whisper is used to fill missing transcripts, some users prefer dialog-formatted output with alternating speaker labels. We must provide deterministic formatting driven by user-configurable speaker settings and silence gaps.

## Constraints & Assumptions
- Whisper segment data includes `start`, `end`, and `text` fields; we operate on the sorted list of segments.
- Users may supply explicit speaker names or fall back to enumerated labels (`SPEAKER 1`, etc.).
- Formatting should remain optional and degrade gracefully if segments are missing metadata.

## Design & Implementation
1. **Segment normalization**
   - Sort segments by `start` time, defaulting missing values to `0.0`.
   - Skip empty or whitespace-only segments.
2. **Speaker alternation**
   - Maintain `current_speaker_idx`, advancing when the gap between segment end and next start exceeds `cfg.screenplay_gap_s`.
   - Wrap speaker index modulo `max(config.MIN_NUM_SPEAKERS, cfg.screenplay_num_speakers)`.
3. **Line aggregation**
   - Consecutive segments assigned to same speaker are concatenated with spaces.
   - Preserve order in a list of `(speaker_idx, text)` tuples.
4. **Label resolution**
   - Map indices to user-provided `cfg.screenplay_speaker_names` when available; fallback to `SPEAKER <n>`.
5. **Output**
   - Join lines with newline separators and append trailing newline for POSIX-friendly files.
   - When formatting fails (invalid input), fall back to plain Whisper text to avoid data loss.

## Key Decisions
- **Gap-based alternation** rather than lexical cues keeps implementation simple and deterministic.
- **Speaker count minimum** ensures at least two speakers when screenplay mode is enabled, aligning with CLI validation.
- **Graceful fallback** prioritizes delivering usable transcripts even when formatting inputs are malformed.

## Alternatives Considered
- **Speaker diarization**: Not implemented due to complexity and external dependencies.
- **Timestamps in output**: Deferred; could be added as optional metadata in future iterations.

## Testing Strategy
- Unit tests feed synthetic segment lists to validate gap handling, speaker rotation, and aggregation.
- Integration tests toggle screenplay flags to confirm CLI wiring and fallback behavior.

## Rollout & Monitoring
- Logging warns when formatting fails and plain text is used instead.
- Future enhancements (e.g., custom templates) can extend this RFC while maintaining backward compatibility.

## References
- Source: `podcast_scraper/whisper.py` (`format_screenplay_from_segments`)
- CLI configuration: `docs/rfc/RFC-007-cli-interface.md`
- Configuration schema: `docs/rfc/RFC-008-config-model.md`
