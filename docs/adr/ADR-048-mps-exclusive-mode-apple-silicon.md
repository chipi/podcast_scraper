# ADR-048: MPS Exclusive Mode for Apple Silicon

- **Status**: Accepted
- **Date**: 2026-02-10
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-042](../rfc/RFC-042-hybrid-summarization-pipeline.md) (hybrid pipeline context)
- **Related PRDs**: [PRD-002](../prd/PRD-002-whisper-fallback.md), [PRD-005](../prd/PRD-005-episode-summarization.md)

## Context & Problem Statement

On Apple Silicon, both Whisper (transcription) and local summarization models can use MPS (Metal Performance Shaders). When both run on the GPU concurrently, they compete for the same VRAM, leading to out-of-memory errors, instability, or segfaults. The pipeline already uses sequential ML execution (ADR-001), but episode-level parallelism can still allow transcription and summarization to overlap across episodes unless explicitly serialized.

## Decision

We adopt **MPS exclusive mode**: when enabled (default), GPU work is serialized so that transcription completes before summarization starts for a given episode. I/O (downloads, parsing) remains parallel. This is controlled by a config flag `mps_exclusive` (default `true`) and environment variable `MPS_EXCLUSIVE`.

- **Enabled (default)**: Serialize GPU work to prevent memory contention; recommended for most Apple Silicon systems.
- **Disabled**: Allow concurrent GPU operations for maximum throughput when the system has sufficient GPU memory (e.g. 48GB+ unified memory).

## Rationale

- **Reliability**: Prevents OOM and crashes on typical Mac laptops with limited unified memory.
- **Predictability**: Same behavior as ADR-001 (sequential ML) but applied explicitly to MPS to avoid cross-stage contention.
- **User choice**: Power users with large VRAM can disable for throughput; default is safe for everyone.

## Alternatives Considered

1. **Always serialize on MPS**: Rejected because it would remove the option for high-memory systems to maximize throughput.
2. **Detect VRAM and auto-toggle**: Rejected due to complexity and platform-specific detection; explicit config is simpler and more predictable.

## Consequences

- **Positive**: Stable default on Apple Silicon; fewer support issues; documented in ARCHITECTURE, TROUBLESHOOTING, and release notes.
- **Negative**: Default serialization can reduce throughput on high-memory systems unless users explicitly set `mps_exclusive: false`.

## Implementation Notes

- **Config**: `config.Config.mps_exclusive` (default `True`), CLI `--mps-exclusive` / `--no-mps-exclusive`, env `MPS_EXCLUSIVE`.
- **Orchestration**: When `mps_exclusive` is true, transcription is awaited before summarization is started for the same episode; a shared lock enforces serialization of GPU work.
- **Module**: `podcast_scraper/workflow/orchestration.py`, `podcast_scraper/workflow/stages/processing.py`, `podcast_scraper/config.py`.

## References

- [ADR-001: Hybrid Concurrency Strategy](ADR-001-hybrid-concurrency-strategy.md) – Sequential ML execution
- [RELEASE v2.5.0](../releases/RELEASE_v2.5.0.md) – MPS exclusive mode introduction (Issue #386)
- [TROUBLESHOOTING](../guides/TROUBLESHOOTING.md) – MPS and memory contention guidance
