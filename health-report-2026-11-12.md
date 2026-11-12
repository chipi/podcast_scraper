# Health check — 2026-11-12

Sources reached: Close-Listening-Obs (prod: health, alerts, Sentry/GlitchTip errors, deploys, resilience, cost, runs, enrichment) and GitHub issues (chipi/podcast_scraper).
Not reached: Datadog (needs OAuth), PagerDuty/Slack/Linear etc. (proxy 403), local-dev-observability (connection closed). Not checked: VictoriaMetrics latency/RED (obs_surface not called), Umami analytics, Langfuse, per-episode events.

## 1. Active incidents
No incident declared. /api/health = ok, code 2.7.0.dev0, all API flags true. All LLM and RSS circuit breakers closed.

## 2. Errors and warnings (prod, 24h) — all podcast-pipeline; api, player, gi-kg-viewer have 0
| Sev | Issue | Count | Last seen (UTC) | Link |
|---|---|---|---|---|
| error | dgx-whisper did not recover in 900s pause-and-probe (3 related events: fuse open / endpoint / ResilienceFuseOpenError) | 2 each | 03:17 | https://glitchtip.tail6d0ed4.ts.net/podcast-pipeline/issues/769 , /770 , /771 |
| error | Whisper guardrail violation `length_floor_violated` | 10 + 8 | 03:17 / 01:35 | /issues/736 , /issues/737 |
| warning | dgx-whisper resilience fuse tripped (hard timeout) | 7 | 03:17 | /issues/767 |

Alert (Grafana, warning, firing since 2026-11-08 01:28Z): **OpenRouter credit low — USD 7.88 left (threshold 10).** At 0, every OpenRouter call fails.

## 3. Performance anomalies
- DGX Whisper endpoint unresponsive for 900s at ~03:17Z (the errors above).
- Cost: 681 LLM calls in 24h, all local vLLM (Qwen3-30B), USD 0.00.
- Not assessed: API latency/p95 (the /api/app/notifications p95 issue #2440 is not re-verified).

## 4. Resolved since last check
- Last 10 deploys all succeeded (failure rate 0.0); latest is deploy-litellm #7, 2026-11-09.
- Nightly full_incremental_pipeline and corpus_enrichment jobs on 11-10, 11-11 and 11-12 all `succeeded`, exit 0. Enrichers: 0 consecutive failures, none auto-disabled.
- Last closed triage issue: #2433 (Postgres login failures), closed 2026-11-06. I found no issue closed after that.

## Cross-correlation with triage-fleet GitHub issues
I listed the 40 newest open issues (411 open total) plus 15 recently closed. Older open issues were not read; a "not filed" finding below is based on the newest 40 and one keyword search.

**Found by both**
- DGX Whisper 900s no-recovery → #2475 (filed 03:24Z, 7 minutes after the error); earlier fuse-timeout → #2458.
- Whisper `length_floor` guardrail → #2428 (bug) and #2429 (adjust guardrail), both open.

**Found only by triage (not visible in my signals)**
- 11-11 batch (#2460–#2474): ML model preload failure, run-budget refusals ($185.86 estimate vs $5 cap; $5.51 vs $5 cap), spaCy speaker detection unavailable, `PODCAST_DATABASE_URL` missing in pipeline-dev, non-JSON KG replies, cleaning stage emptying a transcript (75000 → 0 chars), chunking audio-duration failure, summarisation prompt fabricating summaries. These look dev/pipeline-dev scoped; I did not verify them against prod.
- Backup stale for prod-podcast (#2455), memory.swap.max reset loop (#2447), /api/app/notifications p95 >10s (#2440), Turn extraction end_ms < start_ms (#2456).

**Found only by me (no matching issue found)**
- OpenRouter credit low, USD 7.88, active 4 days. A search for "openrouter credit" returned no issues. Issue #1634 (OpenRouter 403) is closed and a different problem.
- Gap: this is the only item that could stop all OpenRouter calls and nothing is tracking it. Whether anything else uses OpenRouter is not verified; prod cost shows only local vLLM, so the exposure may be limited.

**Caveat**
- Open issues are triage-fleet-heavy and several carry `stale-candidate` (#2415, #2385, #2383, #2382, #2396, #2391, #2376). I did not check whether those are still live.

No action requested.
