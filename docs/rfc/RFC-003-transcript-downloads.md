# RFC-003: Transcript Download Processing

- **Status**: Completed
- **Authors**: GPT-5 Codex (initial documentation)
- **Stakeholders**: Maintainers, networking contributors
- **Related PRD**: `docs/prd/PRD-001-transcript-pipeline.md`
- **Related guide**: [RSS and feed ingestion](../guides/RSS_GUIDE.md) (feed fetch, parsing, and `Episode` construction **before** transcript download)
- **Related ADRs**:
  - [ADR-001: Hybrid Concurrency Strategy](../adr/ADR-001-hybrid-concurrency-strategy.md)
  - [ADR-003: Deterministic Feed Storage](../adr/ADR-003-deterministic-feed-storage.md)
  - [ADR-004: Flat Filesystem Archive Layout](../adr/ADR-004-flat-filesystem-archive-layout.md)

## Abstract

Explain how the system selects, downloads, and stores transcript assets derived from RSS metadata, including error handling, content-type reconciliation, and delay controls.

## Problem Statement

Transcript URLs vary in format, may expose inaccurate MIME types, and can include duplicates. The pipeline must consistently choose the best candidate, download it reliably, and save it under deterministic filenames while respecting user controls (`dry-run`, `skip-existing`, `delay-ms`).

## Constraints & Assumptions

- HTTP stack uses `requests` with retry-enabled adapters (RFC-004 covers filesystem interactions). Retry counts and backoff factors are configurable via `Config` fields (`http_retry_total`, `http_backoff_factor`, `rss_retry_total`, `rss_backoff_factor`) with resilient defaults. An additional application-level episode retry (`episode_retry_max`, default 1) re-runs the full episode download on transient network errors after urllib3 retries are exhausted. Optional per-host pacing, circuit breaker, and RSS conditional GET (Issue #522) are documented in the same section. See [CONFIGURATION.md -- Download Resilience](../api/CONFIGURATION.md#download-resilience).
- Downloads should not halt the overall pipeline when individual episodes fail. End-of-run `failure_summary` in `run.json` aggregates failures by error type for triage.
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
- `run.json` includes a `failure_summary` when episodes fail, grouping failures by error type with counts and episode IDs.
- Future enhancements (e.g., checksum validation) can extend this RFC without breaking existing behavior.

## References

- Source: `podcast_scraper/episode_processor.py`
- HTTP stack: `docs/rfc/RFC-004-filesystem-layout.md` and `podcast_scraper/downloader.py`
- Orchestration: `docs/rfc/RFC-001-workflow-orchestration.md`
