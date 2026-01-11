# RFC-001: Workflow Orchestration

- **Status**: Completed
- **Authors**: GPT-5 Codex (initial documentation)
- **Stakeholders**: Project maintainers, pipeline integrators
- **Related PRD**: `docs/prd/PRD-001-transcript-pipeline.md`
- **Related ADRs**:
  - [ADR-001: Hybrid Concurrency Strategy](../adr/ADR-001-hybrid-concurrency-strategy.md)

## Abstract

Define how `workflow.run_pipeline` coordinates the transcript acquisition pipeline, including concurrency, run lifecycle, and interaction with episode processors.

## Problem Statement

We need a central orchestrator that translates a validated `Config` into concrete actions: preparing output directories, pulling RSS content, delegating episode work, and finalizing runs. Without a well-defined workflow contract, modules would duplicate logic and be difficult to reason about.

## Constraints & Assumptions

- Runs may target feeds with hundreds or thousands of episodes; orchestration must be memory-conscious and support concurrency for IO-bound operations.
- The workflow must respect dry-run semantics: no filesystem mutations when `cfg.dry_run` is true.
- Output directories must be deterministic and safe as defined in RFC-004.
- Whisper fallback is optional and only invoked when enabled and supported (see RFC-005).

## Design & Implementation

1. **Output directory setup**
   - Calls `filesystem.setup_output_directory(cfg)` to derive `effective_output_dir` and optional `run_suffix`.
   - Handles `--clean-output` and `--dry-run` permutations before creating directories.
2. **RSS fetch & materialization**
   - Uses `rss_parser.fetch_and_parse_rss(cfg)` to obtain a `models.RssFeed` (see RFC-002).
   - Converts items into `models.Episode` via `rss_parser.create_episode_from_item`.
3. **Concurrency model**
   - Builds argument tuples and executes `episode_processor.process_episode_download` either sequentially or using a `ThreadPoolExecutor` depending on `cfg.workers` and episode count.
   - Maintains thread-safe counters via locks only when concurrency is used.
4. **Whisper queue**
   - `episode_processor.process_episode_download` may enqueue `TranscriptionJob` instances in a shared list (guarded by a lock when needed).
   - After download phase, sequentially processes queued jobs via `episode_processor.transcribe_media_to_text` to keep resource usage predictable.
5. **Cleanup & Reporting**
   - Removes temporary media directory when present.
   - Returns `(count, summary)` indicating total transcripts saved or planned (dry-run).
   - Logging throughout uses module-level logger with configurable level (`workflow.apply_log_level`).

## Key Decisions

- **Thread pool for downloads**: IO-bound transcript downloads benefit from concurrency; CPU-bound Whisper work remains sequential.
- **Post-processing of Whisper jobs**: Running Whisper after downloads avoids mixing log contexts and simplifies progress reporting.
- **Dry-run handling**: Workflow is responsible for honoring dry-run semantics globally rather than leaving it to downstream modules.

## Alternatives Considered

- **AsyncIO-based orchestration**: Rejected due to higher complexity and limited benefit over thread pools for current workload.
- **Per-episode Whisper execution in worker threads**: Rejected to avoid GPU/CPU oversubscription and unpredictable performance.

## Testing Strategy

- Integration tests in `tests/test_podcast_scraper.py` cover success, dry-run, concurrency edge cases, and error handling.
- Use dependency injection hooks in `cli.main` to substitute mock `run_pipeline` for CLI-focused tests.

## Rollout & Monitoring

- Logging provides counts and summary statements for operations teams.
- Future telemetry hooks can be injected around `run_pipeline` without changing its signature.

## References

- Source: `podcast_scraper/workflow.py`
- Config contract: `docs/rfc/RFC-008-config-model.md`
- Episode processing: `docs/rfc/RFC-003-transcript-downloads.md`
