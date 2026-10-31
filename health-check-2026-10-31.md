# Health check, 2026-10-31 (~06:05 UTC)

## Verdict

All systems nominal for prod. Source: Close-Listening-Obs, target `prod`; all 13 sources live, none failed or unconfigured.

## Coverage

Covered:
- Close-Listening-Obs, prod: health, Grafana alerts, Sentry/GlitchTip errors (24h), VictoriaLogs error logs (1h), VictoriaTraces, deploys, runs, enrichment, cost, cache, LLM/RSS breakers.
- GitHub `chipi/podcast_scraper`: the 50 newest open issues, and a closed `triage-fleet/filed` search (302 hits; I read only the newest 30).

NOT covered:
- Datadog: needs OAuth, not authorized.
- PagerDuty, Cloudflare, Slack, Linear and ~20 other plugin MCPs: proxy 403 on connect.
- `local-dev-observability` and `local-dev-content`: connection closed.
- Dev environment: only `prod` was queried.
- Open issues beyond the newest 50.
- Closed triage issues older than the newest 30.
- Play/iOS native crash reporting beyond what Sentry exposes.

"Since the last check" is not measurable: I have no prior-run state. The windows used were 24h for errors, 1h for logs and traces, and the last 10 deploys and runs.

## 1. Active incidents

None.
- `/api/health` is `ok`, code `2.7.0.dev0`, corpus built by `git_sha 48e3945` at 2026-10-31T03:36Z, no version warning.
- Grafana alerts: 0 total, 0 firing.
- LLM circuit breakers: all 8 providers closed, 0 trips. RSS breakers: 0 trips, 0 open feeds.

## 2. Errors and warnings

- Sentry/GlitchTip prod, 24h: 0 unresolved issues across podcast-api, podcast-pipeline, player and gi-kg-viewer.
- VictoriaLogs, error level, 1h: 5 lines, all from one source. The `alloy` container logs `error inspecting Docker container ... No such container b390c964…` for `component=loki.source.docker.orrery`, repeating every 5s.
  - Cause: Alloy is tailing a container that no longer exists.
  - The component name says "orrery", which CLAUDE.md names as another worktree on a shared machine. I have not confirmed the container's owner.
  - It is observability-agent noise, not an app error. It is not in the issue tracker (see section 5).

## 3. Performance anomalies

None serious.
- Last 10 deploys (all `chipi/infra`): 10 of 10 succeeded, failure rate 0.0.
- Nightly `full_incremental_pipeline` (10-28 through 10-31): succeeded, exit 0, each followed by a succeeded `corpus_enrichment`. The latest enrichment ran 03:49–03:56Z on 10-31.
- Enrichment (run `47a58bc2`): 390s, every enricher `ok` with 0 failed, timed out, quarantined or retried, circuit closed, `stale_days` 0. `topic_similarity` did not appear in the totals I read. I did not check whether it is enabled.
- LLM cost: 5 `llm_cost` events in 24h, all vLLM (Qwen3-30B), `estimated_cost_usd` 0.0, no guardrail triggered.
- Cache, minor: `kg.entity_id_map` has 0% hits, 2 misses, avg build 42.7s (85s total). `app_kg_entity_index` is also 0% hits, 2 misses, avg build 3.1s. That is a cold-start-sized cost, not an anomaly. I have no baseline to say whether it is normal.

## 4. Resolved items

I cannot list resolved Grafana or Sentry items: the tools return current state only. Indirect evidence:
- The prod nightly pipeline has run for 4 consecutive days (10-28 to 10-31), which matches the fix for the "ingestion not run in 36h" signal (#2302).

## 5. Triage-fleet cross-correlation

Read: the 50 newest open issues. 17 carry `triage-fleet/filed`. 13 of those carry `triage-fleet/stale-candidate`.

### Found by both (live signal and an issue)

- **Pipeline/ingestion down (#2302, 10-09)**: live evidence contradicts it. Nightly runs succeed and the API is up.
- **OOM kill on prod-podcast (#2299, 10-08)**: no firing alert now.
- **Enrichment FAILED/PARTIAL (#2382, 10-17)**: the latest enrichment shows no failures.
- **`topic_similarity` timeout (#2385)**: the latest enrichment shows no failures.
- **Invalid OpenAI key (#2371, #2354)**: the OpenAI breaker is closed with 0 failures. #2354 is dev-only (a MacBook LiteLLM proxy) and not testable from prod.
- **Transcription overran deadline (#2335, #2336, 10-12)**: no current evidence either way. No transcription-timeout signal in this window.

Each of those is a candidate to close from the prod side. The `stale-candidate` label already marks 13 of the 17.

### Found by the fleet, not visible live (cannot confirm either way)

- **#2396**, `Failed to fetch dynamically imported module` (player-1K, 10-27). The escalation text says only 2 events in 5 days. Sentry shows 0 in 24h.
- **#2391**, `Audio would not play (media error 4)` (10-19). Sentry shows 0 in 24h.
- **#2404**, low-signal aggregate for `player-obs`: `Failed to export spans batch due to timeout` (glitchtip PLAYER-OBS-4, 10-30). A span-export timeout, no user impact. 0 in the 24h Sentry window.
- **#2297 `ClientDisconnect`, #2348 MCP stream exception, #2376 language refusal, #2361 context limit 65536**: 0 in Sentry now. I did not open their bodies.
- **#2383**, "ops health check stale: daily pusher not running" (10-18). I have no live probe for the pusher, so it is neither confirmed nor refuted.

### Found by me, not in the issue tracker

1. **Alloy `No such container` error loop** (5s cadence, `loki.source.docker.orrery`). I searched only the 50 newest open issues and the closed triage list, so I cannot say it was never filed. I saw no matching title in either.
2. **`kg.entity_id_map` cold-build cost** (42.7s per build, 0% hit). I saw no matching title; same search limit as above.
3. **Connector gaps**: Datadog unauthorized, PagerDuty/Cloudflare/Slack/Linear down at the proxy, and two local-dev MCP servers closed. These are session-level gaps and belong in no issue.

### Open issues that are not signal-driven

About 33 of the 50 newest open issues are human-authored (speaker naming, ASR, DGX, discover arc, player native metrics). They are feature or investigation work and do not correlate with any live signal. Of these, #2370, #2353 and #2284 cover transcript-quality defects; the monitoring tools do not measure transcript quality.

## Not verified

- Whether the Alloy orrery tailer belongs to this deploy or to another project.
- Whether any triage issue's original condition is still reproducible; I only checked that no live alert or Sentry issue matches today.
- The remaining roughly 30 open triage-labeled issues beyond the 50 I read, if any.
