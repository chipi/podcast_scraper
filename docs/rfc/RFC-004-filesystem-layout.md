# RFC-004: Filesystem Layout & Run Management

- **Status**: Completed
- **Authors**: GPT-5 Codex (initial documentation)
- **Stakeholders**: Maintainers, operators concerned with storage hygiene
- **Related PRDs**: `docs/prd/PRD-001-transcript-pipeline.md`, `docs/prd/PRD-002-whisper-fallback.md`

## Abstract

Define the rules governing output directories, filename sanitization, temporary storage, and run suffix semantics to guarantee deterministic, safe, and resumable filesystem interactions.

## Problem Statement

Without consistent naming and directory policies, transcript archives become hard to diff, risk clobbering existing data, and may interact poorly with user environments. The system requires cross-platform-safe filenames, predictable run folders, and cleanup of temporary artifacts.

## Constraints & Assumptions

- Target environments include macOS, Linux, and Windows; filenames must avoid reserved characters.
- Users may override output directories, but we encourage safe locations (home directory, platform data/cache roots).
- Whisper fallback stores intermediate media files that must be cleaned up to conserve disk.

## Design & Implementation

1. **Base output directory**
   - Default: `output/rss_<sanitized_host>_<hash>` where hash is first 8 chars of SHA-1 of the RSS URL.
   - `filesystem.derive_output_dir` handles overrides, invoking validation.
2. **Validation**
   - `filesystem.validate_and_normalize_output_dir` resolves paths, verifies they fall under safe roots (cwd, home, platform dirs), and warns otherwise.
3. **Run suffixes**
   - `filesystem.setup_output_directory` derives optional `run_suffix` based on `cfg.run_id` or Whisper usage.
   - Effective output path is `<output_dir>/run_<run_suffix>` when suffix present.
4. **Filename sanitization**
   - `filesystem.sanitize_filename` strips control characters, collapses whitespace, and replaces unsafe characters with `_`.
   - Episode filenames follow `<idx:04d> - <title_safe>[ _<run_suffix>].<ext>`.
5. **Whisper outputs**
   - `filesystem.build_whisper_output_name` truncates titles (32 chars) and appends run suffix when available.
   - Temporary media stored in `<effective_output_dir>/.tmp_media/` and removed post-transcription.
6. **Cleanup semantics**
   - `--clean-output` triggers deletion of existing output directory if not in dry-run mode.
   - Best-effort removal of temp directories even on errors (warnings if removal fails).

## Key Decisions

- **Hash-based directory suffix** avoids collisions between feeds hosted on same domain but different paths.
- **Suffix semantics** provide provenance (run ID, Whisper model) within output directories without complicating base naming.
- **Sanitization policy** prioritizes readability while remaining filesystem-safe across OSes.

## Alternatives Considered

- **Timestamped root directories**: Rejected in favor of deterministic names; use `--run-id auto` when unique runs are desired.
- **Per-episode subdirectories**: Rejected to keep archives flat and easy to diff.

## Testing Strategy

- Unit tests cover sanitization edge cases and output derivation logic.
- Integration tests ensure `--clean-output`, `--skip-existing`, and Whisper workflows interact correctly with directory management.

## Rollout & Monitoring

- Warnings emitted when users choose "unsafe" directories (outside recommended roots) for observability.
- Future enhancements (checksums, subdirectory partitioning) can extend this RFC with backward compatibility.

## References

- Source: `podcast_scraper/filesystem.py`
- Orchestrator usage: `docs/rfc/RFC-001-workflow-orchestration.md`
- Whisper specifics: `docs/rfc/RFC-005-whisper-integration.md`
