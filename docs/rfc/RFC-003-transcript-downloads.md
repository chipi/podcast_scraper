# RFC-003: Transcript Download Processing

- **Status**: Accepted
- **Authors**: GPT-5 Codex (initial documentation)
- **Stakeholders**: Maintainers, networking contributors
- **Related PRD**: `docs/prd/PRD-001-transcript-pipeline.md`

## Abstract

Explain how the system selects, downloads, and stores transcript assets derived from RSS metadata, including error handling, content-type reconciliation, and delay controls.

## Problem Statement

Transcript URLs vary in format, may expose inaccurate MIME types, and can include duplicates. The pipeline must consistently choose the best candidate, download it reliably, and save it under deterministic filenames while respecting user controls (`dry-run`, `skip-existing`, `delay-ms`).

## Constraints & Assumptions

- HTTP stack uses `requests` with retry-enabled adapters (RFC-004 covers filesystem interactions).
- Downloads should not halt the overall pipeline when individual episodes fail.
- User-specified preferences (`prefer_type`) are honored when selecting candidates.

## Design & Implementation

1. **Candidate selection**
   - `rss_parser.find_transcript_urls` provides candidates; `choose_transcript_url` applies preference ordering.
   - Preference list compares case-insensitively against MIME types and URL suffixes.
2. **Download execution**
   - `episode_processor.process_transcript_download` fetches bytes via `downloader.http_get` (streaming with progress updates).
   - Content-Type headers inform extension inference alongside declared types and URL heuristics.
3. **Filename derivation**
   - Base pattern: `<idx:04d> - <title_safe>[ _<run_suffix>]`.
   - Extension is re-evaluated post-download to capture actual media type.
   - Idempotency: `--skip-existing` checks for any file matching base pattern before download (supports historical runs with different extensions).
4. **Operational flags**
   - `--dry-run`: logs planned URL and destination path without network calls.
   - `--delay-ms`: optional sleep between episodes to respect rate-limits.
5. **Error handling**
   - Network exceptions log warnings and return `False`; pipeline continues.
   - Filesystem errors are logged and treated as failures for that episode.

## Key Decisions

- **Extension inference** after download ensures actual content type is reflected even when RSS metadata is wrong.
- **Glob-based skip check** allows for transcripts saved with different extensions or run suffixes while still stopping duplicate work.
- **Progress integration** uses shared factory to keep download UI consistent with other operations.

## Alternatives Considered

- **Strict MIME enforcement**: Requiring matching MIME types would drop transcripts due to inconsistent feeds; rejected in favor of heuristics.
- **Always overwriting files**: Rejected to maintain auditability and support manual curation between runs.

## Testing Strategy

- Unit tests cover `derive_transcript_extension` edge cases.
- Integration tests simulate HTTP responses with various headers to ensure extension selection behaves as expected.

## Rollout & Monitoring

- Logging includes success, failure, and skip events for traceability.
- Future enhancements (e.g., checksum validation) can extend this RFC without breaking existing behavior.

## References

- Source: `podcast_scraper/episode_processor.py`
- HTTP stack: `docs/rfc/RFC-004-filesystem-layout.md` and `podcast_scraper/downloader.py`
- Orchestration: `docs/rfc/RFC-001-workflow-orchestration.md`
