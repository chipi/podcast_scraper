# Observability architecture — emit open formats, ship pluggably

How the podcast app (and orrery) are observed on the **self-hosted** stack, and —
more importantly — the abstraction that keeps the app **vendor-neutral** so any
fork can point observability at a different backend without touching app code.

- **Decision record:** [ADR-119](../adr/ADR-119-vendor-neutral-event-emission.md)
  (this guide is the how-to; ADR-119 is the why) under
  [ADR-117](../adr/ADR-117-multi-tenant-observability-gitops.md) (multi-tenant split).
- **Full plan / phases:** `docs/wip/observability-app-surface-plan.md`.
- **Sibling guides:** [OBSERVABILITY_CONTROL_PLANE](OBSERVABILITY_CONTROL_PLANE.md)
  (the `podcast_obs` cross-source probe CLI/MCP — "what is a deploy doing now?"),
  [OBSERVABILITY_EXTENSIONS](OBSERVABILITY_EXTENSIONS.md) (alerting / Sentry / Grafana).

## The one idea

The app emits observability in **open, vendor-neutral formats** and depends on
nothing else. **Shipping is a separate, pluggable layer.** The reference backend is
self-hosted Grafana + VictoriaMetrics/VictoriaLogs/VictoriaTraces on the DGX
(tailnet-only), but that's a *sink*, not a coupling.

```text
APP (producer)                     SHIPPING (pluggable)          BACKEND (swappable)
  /metrics (Prometheus)      ─┐      Alloy scrape ───────────▶    VictoriaMetrics
  emit_event(...) JSONL/stdout┼──▶   Alloy tail (stdout+files) ─▶ VictoriaLogs
  OTLP spans (auto-instrument)┤      OTEL SDK → OTLP HTTP ──────▶ VictoriaTraces
  Sentry SDK                  ┤      DSN ──────────────────────▶  GlitchTip
  Langfuse span               ┘      SDK ──────────────────────▶  Langfuse
  depends on: open formats only      config / DSN, not app code
```

## Signal taxonomy — which signal goes where

| Signal | Open standard the app emits | Reference backend | How it ships |
| --- | --- | --- | --- |
| Metrics | Prometheus text (`/metrics`) | VictoriaMetrics | Alloy scrape (`api:8000/metrics`, host, cAdvisor) |
| Logs / events | canonical JSONL (`emit_event`) | VictoriaLogs | Alloy tails pipeline stdout + corpus JSONL files |
| Traces (spans) | OTLP (OpenTelemetry) | VictoriaTraces | OTEL SDK / `opentelemetry-instrument` → OTLP HTTP |
| Errors | Sentry protocol | GlitchTip (self-hosted) | Sentry SDK (DSN; Sentry Cloud swappable) |
| LLM prompt/cost | Langfuse span | Langfuse | `emit_langfuse_span` (in-code choke point) |

Rule of thumb: **metrics** = numbers over time; **logs/events** = discrete
structured facts (a cost record, a search, a job); **traces** = one request's span
waterfall; **errors** = exceptions with stacktraces; **LLM** = prompt/response/cost
detail. Keep them straight — don't cram a trace into a log or a metric into an event.

## Events / logs — the `emit_event` SDK

`src/podcast_scraper/obs/events.py`. One function, one envelope, two channels.

```python
from podcast_scraper.obs.events import emit_event

# sink="log" (default): one JSON line to stdout — for pipeline / ephemeral contexts.
emit_event("llm_cost", provider="openai", model="gpt-4o", cost_usd=0.012, run_id=rid)

# sink="file": append to a persistent corpus JSONL — for serve-side events that must
# survive with no agent attached (search, listen, job).
emit_event("search_query", sink="file", corpus_dir=corpus, query_type="semantic")
```

Envelope on every event: `{"ts", "schema", "event_type", ...fields}`. `None`
fields are dropped (lean). Telemetry **never raises** — a broken emit can't break
the caller. `logger=` preserves a caller's logger name; `ts=` backdates.

**Event catalog** (the contract — grep `emit_event(` for the live set):

| event_type | sink | emitter | key fields |
| --- | --- | --- | --- |
| `llm_cost` | log | `workflow/cost_monitoring.py` | provider, model, tokens, estimated_cost_usd, run_id, stage |
| `search_query` | file | `search/query_log.py` | query_type (no raw text) |
| `listen` | file | `server/app_user_state.py` | slug, feed_id, ts (epoch — *not yet on `emit_event`*, see ADR-119) |
| `job` | file | `.viewer/jobs.jsonl` | job lifecycle |

Adding an event: call `emit_event("<name>", ...)` at the emission site and add a
row here. That's it — no shipping/backend change.

## Collection (the reference sink — infra)

The prod-podcast **Alloy** collector (homelab repo
`infra/observability/hosts/prod-podcast/`, deployed at `/opt/vps-observability/`):

- **Metrics:** scrapes host node-exporter + cAdvisor + `api:8000/metrics` (60s).
- **Logs/events:** `loki.source.docker` scoped to the ephemeral `pipeline`/
  `pipeline-llm` runner containers (captures `emit_event` stdout) + `local.file_match`
  → `loki.source.file` for the corpus JSONL (`search/query_log.jsonl`,
  `users/*/listen.jsonl`, `.viewer/jobs.jsonl`). → VictoriaLogs.
- **Security logs:** sshd/fail2ban journal + Caddy access → VictoriaLogs.
- **Traces:** *not* via Alloy by default — the app exports OTLP directly (env-var
  driven). See `agentic-ai-homelab/docs/wip/podcast-otel-traces-handover.md`.

Dashboards are owned in-repo (`config/grafana/dashboards/vps/`) and pushed to the
shared Grafana `VPS — Podcast` folder with `scripts/ops/push-grafana-dashboards.sh`
(token in gitignored `.env`).

## How to point observability somewhere else (forkability)

Because the app emits only open formats, a fork changes **config, not code**:

- **Logs → a different store:** repoint Alloy's `loki.write` (or swap Alloy for
  Vector/Fluent Bit; add an `otelcol`/Kafka/S3 output). The app still just writes JSONL.
- **Metrics → a different TSDB:** point any Prometheus-compatible scraper at `/metrics`.
- **Traces → Jaeger/Tempo/Honeycomb:** change `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.
- **Errors → Sentry Cloud (or any Sentry-compatible):** change the DSN.
- **No backend at all:** events still land in stdout + the corpus JSONL files; the
  `emit_event` file sink works with no agent attached.

## Operating procedures (runbook)

The **minimal surface** — 4 dashboards in the `VPS — Podcast` Grafana folder, each
answering one question; raw logs/traces via **Explore** (no storage cost):

| Dashboard | Question | When |
| --- | --- | --- |
| **VPS Overview** | Is the box healthy? | daily glance; first stop in any incident |
| **Podcast App** (`$surface`) | Is it serving / what's it doing + costing? | daily; after deploy; weekly cost check |
| **Edge Security** | Is anyone attacking? | weekly; on a fail2ban alert |
| **Containers** | Per-container resource? | deep-dive (needs the cAdvisor fix) |

Procedures:

- **Daily 30-sec glance:** VPS Overview green (CPU/mem/disk < thresholds)? Podcast
  App → API 5xx rate flat?
- **After a deploy:** Podcast App → watch API 5xx + p95 for a spike → any failure,
  open **Explore → Traces** (service `podcast-api`) for the slow/failing span →
  **GlitchTip** for the exception + stacktrace.
- **Weekly:** Podcast App → `$surface=pipeline` → LLM cost trend; Edge Security →
  fail2ban bans / ssh failures.
- **Incident flow (down/slow):** VPS Overview (resource pressure?) → Podcast App
  (which `$surface`/route?) → Explore Traces (which span?) → Explore Logs
  (`instance:prod-podcast AND job:~"<surface>"`) → GlitchTip (stacktrace). The
  `run_id` tag joins a cost log, its trace, and its error across all three.
- **Differentiating the 3 podcast surfaces:** `$surface` dropdown on Podcast App
  (logs); OTEL `service.name` (`podcast-api` vs `podcast-pipeline`) + `http.route`
  on traces; the api route/handler label on metrics.

Deep-dive dashboards (Node Exporter Full, cAdvisor) stay in the Homelab folder —
not the daily surface. Keep the set at ~4; add a dashboard only when a recurring
question has no home.

## Verify (tailnet, DGX `100.69.49.126`)

```sh
# metrics: api RED + host + cadvisor for prod-podcast
curl -s "http://100.69.49.126:8428/api/v1/query?query=up{instance='prod-podcast'}"
# events/logs: what podcast-* jobs are landing
curl -sG "http://100.69.49.126:9428/select/logsql/query" \
  --data-urlencode "query=instance:prod-podcast AND job:~'podcast-' AND _time:1h | stats by (job) count()"
# traces: services reporting
curl -s "http://100.69.49.126:10428/select/jaeger/api/services"
```

Then Grafana → the `VPS — Podcast` folder (dashboards), Explore (logs/metrics),
Explore Traces (spans).
