# ADR-175: Several API workers per server, off by default

- **Status**: Accepted (2026-11-08); implemented in `podcast_scraper/server/cli_handlers.py`
  (`serve --workers`), `podcast_scraper/server/leader.py` and
  `podcast_scraper/server/metrics_multiprocess.py`
- **Date**: 2026-11-08
- **Authors**: Marko Dragoljevic
- **Related**: [ADR-173](ADR-173-how-the-app-uses-database-connections.md) (database connections,
  PgBouncer readiness)

## Context & Problem Statement

`serve` runs one uvicorn process. A load test at 5, 20 and 50 concurrent listeners (2026-11-07) saturated
that one process at 50 while the database stayed idle: throughput stopped growing and a fast
route's p95 rose from 395 ms to 1.9 s. Python serves one CPU-bound request at a time per process,
so the way past it is more processes.

Three things in the server assume it is the only process:

1. **Work that must run once.** The cron scheduler fires scheduled jobs from the app's lifespan; N
   workers would fire each job N times.
2. **Metrics.** Each process counts its own requests, and a scrape reaches one worker at a time.
3. **Logging.** `serve` sets up app logging in its own process, which the workers do not inherit.

## Decision

1. **`serve --workers N`**, default 1 (or `PODCAST_SERVE_WORKERS`). One worker runs exactly as
   before. With more, uvicorn starts N processes, each building the app from the environment
   (`create_app_for_uvicorn`).
2. **One leader per host for the scheduler.** The workers take a file lock (`fcntl.flock`) on one
   file; the holder starts the scheduler, and the others retry every 30 s.
   - The kernel releases the lock when its process dies, however it dies.
   - A leader whose start fails gives the lock back.
   - The job-queue sweeper runs in every worker: it takes the job registry's cross-process lock
     already.
   - A Postgres advisory lock was not used: a session advisory lock breaks behind PgBouncer in
     transaction mode (ADR-173).
3. **Metrics across workers** use Prometheus multiprocess mode.
   - `PROMETHEUS_MULTIPROC_DIR` names a directory per container, emptied by `serve` before any
     worker starts, so a restart adds no old samples.
   - `/metrics` adds up every worker's files, and keeps the collectors that compute their samples
     at scrape time: build and schema info, job age, naming coverage, an extension's own.
   - Per-process collectors (process, platform, GC) are left out, because they describe one worker.
   - `serve` refuses `--workers N` with metrics on and no such directory, rather than serve one
     worker's counts.
   - The ledger-state gauge reports the most recent worker's value.
4. **Logging is set up in every worker**, as `serve` does in a single process.
5. **Each worker warms its own caches.** The read caches live in each process's memory, so a
   worker that skipped warming would answer cold. So cache memory grows with the number of
   workers.

## Consequences

- **Memory.** On prod one player-api worker holds 1.85–2.46 GB RSS (2026-11-08). N workers need
  about N times that, before any of it is cut. So N stays 1 until memory is cut or the box grows.
- **Connections.** Pools grow with workers: (pool size + overflow) × workers × services must stay
  under Postgres' `max_connections` (100 on prod), or PgBouncer goes in front (ADR-173).
- **Dashboards.** Counters and histograms keep their names and labels across workers. A gauge set
  in more than one worker needs a `multiprocess_mode`, or it carries a `pid` label.
- **The memory watchdog** acts per process, as before.

## Alternatives considered

- **gunicorn with uvicorn workers.** This was rejected: it is another process manager to run, and
  uvicorn's own `--workers` supervises and restarts workers.
- **A Postgres lease row for the leader.** This was rejected: it needs a table, hence a migration,
  for what a file lock on the one host already gives.
- **Warming once per host.** This was rejected: caches are per process, so the other workers would
  stay cold.
