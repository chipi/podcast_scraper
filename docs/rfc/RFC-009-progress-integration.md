# RFC-009: Progress Reporting Integration

- **Status**: Accepted
- **Authors**: GPT-5 Codex (initial documentation)
- **Stakeholders**: Maintainers, UX contributors, embedding applications
- **Related PRDs**: `docs/prd/PRD-001-transcript-pipeline.md`, `docs/prd/PRD-003-user-interface-config.md`

## Abstract

Describe the pluggable progress reporting abstraction that decouples pipeline logic from specific UI implementations (e.g., `tqdm`), enabling both CLI and embedded usage scenarios.

## Problem Statement

Long-running downloads and transcriptions need user feedback. However, tying business logic to `tqdm` directly complicates embedding the library in other applications or headless environments. A narrow abstraction allows swapping progress reporters without invasive changes.

## Constraints & Assumptions

- Minimal API: only `update(advance: int)` is required for reporters.
- When no progress factory is registered, operations should still succeed without emitting progress (no-op behavior).
- The abstraction must work in both threaded contexts (download workers) and sequential loops (Whisper jobs).

## Design & Implementation

1. **Progress factory**
   - `progress.ProgressFactory` type alias defines a callable returning a context manager that yields a `ProgressReporter`.
   - `_progress_factory` module-level variable stores the active factory; defaults to `_noop_progress`.
2. **Registration**
   - `progress.set_progress_factory(factory)` installs a new factory, falling back to noop when `None` passed.
   - CLI registers `_tqdm_progress` at startup to provide terminal-friendly feedback.
3. **Usage pattern**
   - Callers use `with progress.progress_context(total, description) as reporter:` and invoke `reporter.update(count)`.
   - Works identically for streaming downloads (bytes) and discrete job counts (Whisper queue).
4. **Thread safety**
   - Factories are responsible for providing thread-safe reporters if needed. Default `_tqdm_progress` handles concurrent updates through `tqdm`’s internal locking.

## Key Decisions

- **Context manager API** ensures reporters can manage lifecycle (start/stop) gracefully without leaking handles.
- **No global `tqdm` import** keeps the base package lightweight for library consumers.
- **Backwards compatibility alias** `progress = progress_context` retained for older integrations.

## Alternatives Considered

- **Callbacks without context manager**: Rejected to avoid duplicated setup/teardown logic across call sites.
- **Observer/event bus**: Overkill for current use cases; may revisit if richer telemetry is required.

## Testing Strategy

- Unit tests verify the noop factory and `set_progress_factory` behavior.
- Integration tests ensure CLI installs `_tqdm_progress` and that progress updates do not raise errors in threaded downloads.

## Rollout & Monitoring

- Embedding applications can override the factory early in their startup to integrate with custom UIs (e.g., GUIs, logs-only environments).
- Documentation (README + PRDs) highlights the extensibility point for developers.

## References

- Source: `podcast_scraper/progress.py`
- CLI integration: `docs/rfc/RFC-007-cli-interface.md`
- Download usage: `docs/rfc/RFC-003-transcript-downloads.md`
