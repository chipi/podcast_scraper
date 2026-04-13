# WIP: Concurrent pipelines and `http_urllib3_retry_events`

**Status:** Open (documentation only; no implementation plan committed)
**Date:** 2026-04-12

## Problem

`http_urllib3_retry_events` in `metrics.json` is backed by a **process-wide** counter in
`podcast_scraper.rss.downloader`. It is **reset** when `configure_downloader()` runs at the start
of a normal `run_pipeline` / CLI / `service.run` invocation.

If two **`run_pipeline()`** calls run **concurrently** in the **same** Python process, urllib3
retry events from both runs contribute to one counter, so the value written to either run’s
`metrics.json` is **not** attributable to a single logical run.

## Current guidance

- Run **at most one** pipeline at a time **per process**, or use **separate processes** if you
  need isolated counts.
- See also [CONFIGURATION.md — Download resilience (threading and metrics)](../api/CONFIGURATION.md#download-resilience).

## Possible future directions (not decided)

- Per-run counter wired through a run-scoped object (e.g. `Metrics`) and thread-safe routing from
  `LoggingRetry` to the active run (non-trivial with thread-pool workers and shared sessions).
- **ContextVar** or similar for “current run metrics” (must propagate correctly into download
  worker threads).
- Document-only: keep global counter; recommend multiprocessing for parallel corpora.

No RFC/ADR opened yet; promote this WIP when a concrete product requirement appears.
