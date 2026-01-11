# ADR-001: Hybrid Concurrency Strategy

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-001](../rfc/RFC-001-workflow-orchestration.md)
- **Related PRDs**: [PRD-001](../prd/PRD-001-transcript-pipeline.md)

## Context & Problem Statement

The podcast scraper handles multiple episodes per feed. Downloading transcripts is IO-bound and benefits from concurrency. However, ML-based tasks (Whisper transcription, summarization) are CPU/GPU-intensive and can cause resource oversubscription, out-of-memory errors, and unpredictable performance if run in parallel.

## Decision

We adopt a **Hybrid Concurrency Strategy**:

1. **IO-Bound Tasks (Downloads)**: Use a `ThreadPoolExecutor` with a configurable number of workers (defaulting to CPU count, capped at 8).
2. **Compute-Bound Tasks (ML)**: Execute strictly sequentially. Even when triggered by multiple parallel download workers, ML jobs are enqueued and processed one-by-one.

## Rationale

- **Predictability**: Sequential ML prevents GPU memory spikes and ensures stable latency per episode.
- **Simplicity**: Avoids complex resource scheduling and semaphore management for GPU access.
- **Efficiency**: Parallel downloads maximize network throughput without bottlenecking the local machine's processing power.

## Alternatives Considered

1. **AsyncIO**: Rejected due to increased complexity and limited benefit for our specific mixed IO/CPU workload.
2. **Parallel ML**: Rejected to avoid OOM issues on machines with limited VRAM (especially when running multiple large models).

## Consequences

- **Positive**: Stable resource usage; easy-to-track progress bars; no "GPU out of memory" errors in standard configurations.
- **Negative**: Total processing time is limited by the sequential nature of transcription/summarization.

## References

- [RFC-001: Workflow Orchestration](../rfc/RFC-001-workflow-orchestration.md)
