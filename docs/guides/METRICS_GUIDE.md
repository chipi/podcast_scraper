# Metrics Guide

This guide describes **pipeline run metrics** written by the main podcast scraper (CLI,
`run_pipeline`, `service.run`). For **experiment / eval** metrics (ROUGE, baselines, promotion),
see [Experiment Guide](EXPERIMENT_GUIDE.md). For **CI test health** dashboards on GitHub Pages,
see [Test dashboard (GitHub Pages)](../ci/METRICS.md).

## Where metrics live

Each run directory can include `metrics.json` and related artifacts. Layout and filenames are
described in [Pipeline and Workflow — Run tracking](PIPELINE_AND_WORKFLOW.md#run-tracking-files-issue-379-429)
(Issues #379, #429).

## Pipeline run metrics (download resilience)

The pipeline records urllib3-level retry activity and optional application-level episode download
retries for operational triage. Configuration fields, CLI flags, and preset guidance:
[CONFIGURATION.md — Download resilience](../api/CONFIGURATION.md#download-resilience)
(including [recommended presets](../api/CONFIGURATION.md#recommended-presets-download-resilience) and [CLI parity](../api/CONFIGURATION.md#cli-vs-configuration-parity-download-resilience)).
Examples: `config/examples/config.example.download-resilience.yaml`,
`config/examples/config.example.download-resilience.polite.yaml`.

| Field | Meaning |
| :--- | :--- |
| `http_urllib3_retry_events` | Count of urllib3 retry scheduling events since the downloader was last configured (pipeline start); process-wide |
| `episode_download_retries` | Application-level episode download retries after urllib3 exhaustion, when `episode_retry_max` > 0 |
| `episode_download_retry_sleep_seconds` | Sum of configured backoff sleeps before those episode-level retries |
| `host_throttle_wait_seconds` | Wall time spent waiting on per-host throttle or `Retry-After` alignment (Issue #522) |
| `host_throttle_events` | Number of throttle / `Retry-After` wait episodes recorded for that metric |
| `retry_after_events` | Count of `Retry-After`-driven sleeps recorded in policy metrics |
| `retry_after_total_sleep_seconds` | Sum of those sleeps |
| `circuit_breaker_trips` | Times a circuit transitioned to open |
| `circuit_breaker_open_feeds` | Scope keys that opened (feed URL or host label) |
| `rss_conditional_hit` | RSS responses served from cache after HTTP 304 |
| `rss_conditional_miss` | RSS full-body 200 responses counted toward conditional GET |

**Semantics:** `http_urllib3_retry_events` is shared across all threads in the process. A single
`run_pipeline` per process is the supported model for a faithful count. For concurrent pipelines
or stronger isolation, see [WIP: concurrent pipelines and HTTP retry metrics](../wip/wip-concurrent-pipeline-http-retry-metrics.md).

**ADR-028** ([Unified retry policy with metrics](../adr/ADR-028-unified-retry-policy-with-metrics.md))
documents **LLM/API provider** retries and related metrics, not these HTTP download counters.

## Related references

- [EXPERIMENT_GUIDE — Pipeline run metrics (download resilience)](EXPERIMENT_GUIDE.md#pipeline-run-metrics-download-resilience) — same fields in the eval workflow context
- [METRICS_DOCS_AND_DASHBOARD_V2 (design notes)](../wip/METRICS_DOCS_AND_DASHBOARD_V2.md)
