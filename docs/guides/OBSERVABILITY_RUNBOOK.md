# Observability runbook — current state, production debugging, gaps

**The operational entry point for the next agent/operator.** Where the four signals
actually live *today*, how to debug prod when something breaks, and what's missing.

- **Design/why:** [OBSERVABILITY_ARCHITECTURE.md](OBSERVABILITY_ARCHITECTURE.md)
  ([ADR-119](../adr/ADR-119-vendor-neutral-event-emission.md) vendor-neutral emission,
  [ADR-117](../adr/ADR-117-multi-tenant-observability-gitops.md) multi-tenant split,
  [ADR-121] node-Alloy, [ADR-120] telemetry-never-breaks-the-app).
- **This doc = the how-to-operate + the honest current-state.** Last verified
  **2026-07-24** against the live systems.
- ⚠️ **The architecture guide has drifted** (backend host, dashboards, GlitchTip player
  project). Where they disagree, **this doc is the verified one** — see
  [Gaps / issues / drift](#gaps--issues--drift-the-assessment).

## 30-second orientation

Four signals, one join key. Everything lands on the **homelab** box (Mac mini, tailnet
name `homelab` = `100.87.33.61`), reached over Tailscale.

| Signal | App emits | Ships via | Backend (on `homelab`) | Grafana datasource |
| --- | --- | --- | --- | --- |
| **Metrics** | Prometheus `/metrics` (`prometheus_fastapi_instrumentator`) | VPS Alloy scrape → remote_write | **VictoriaMetrics** `:8428` | `VictoriaMetrics` |
| **Logs** | container stdout | VPS Alloy `loki.source.docker` | **VictoriaLogs** `:9428` | `VictoriaLogs` (uid `victorialogs`) |
| **Traces** | OTLP (`opentelemetry-instrument` auto) | OTEL SDK → OTLP HTTP | **VictoriaTraces** `:10428` | `VictoriaTraces (Tempo/Jaeger)` |
| **Errors** | Sentry SDK / browser SDK | DSN | **GlitchTip** `:8090` | (GlitchTip UI) |

**Join key = `trace_id`** (one request) and **`run_id`** (one pipeline run). Pivot between
signals on these. Grafana `http://homelab:3000`.

## Live topology (verified 2026-07-24)

```text
VPS (prod-podcast, tailnet)                      homelab (Mac mini, 100.87.33.61)
  operator api  ─ /metrics ───┐                    ┌── VictoriaMetrics :8428
  (compose-api-1)             │  VPS node Alloy     │
  player  api   ─ /metrics ───┼─ (/opt/vps-        ─┼── VictoriaLogs    :9428
  pipeline      ─ stdout  ────┤   observability)    │
  viewer        ─ stdout  ────┘  scrape + docker    └── VictoriaTraces  :10428
  operator/player api ─ OTLP ──────────────────────────▶  (traces, direct, not via Alloy)
  api (server) ─ Sentry DSN ───────────────────────────▶  GlitchTip :8090
  browser (player/operator) ─ Sentry ─▶ public ingest edge (telemetry.<domain>) ─▶ GlitchTip
                                                     Grafana :3000  (dashboards + Explore)
```

- **Collectors:** the **VPS node Alloy** (ADR-121, `/opt/vps-observability/`, config
  `base.alloy` + per-app drop-ins) ships VPS container logs + scrapes `/metrics`. A
  **separate homelab Alloy** (`agentic-ai-homelab/infra/observability/config.alloy`) scrapes
  the homelab host/GPU/vLLM — don't confuse the two.
- **Backends migrated DGX → homelab** (Mac mini). Any doc/command using `dgx-llm-1:8428/…`
  is stale — use `homelab:…`.

## Coverage matrix — what's actually ON

| Surface | Metrics | Logs (in VictoriaLogs) | Traces (in VictoriaTraces) | Errors (GlitchTip) |
| --- | --- | --- | --- | --- |
| **operator api** (`compose-api-1`) | ✅ `job=api,instance=prod-podcast` | ✅ `app=podcast,surface=api` (uvicorn access, **no trace_id**) | ✅ `service.name=podcast-api` | ✅ project `podcast` (1) |
| **operator viewer** (`compose-viewer-1`) | — | ✅ `app=podcast,surface=web` | — | browser → project 1 (client) |
| **player api** | ✅ (same instrumentator) | ✅ `app=player` | ❌ **OFF** (would be `player-api`; G0 held) | ✅ project **player (5)** |
| **player frontend** | — | ✅ `app=player,surface=web` | — | ✅ browser SDK → `telemetry.closelistening.app` → project 5 |
| **pipeline** (`pipeline`/`pipeline-llm`) | ✅ | ✅ carries `[run=… trace=…]` | ✅ `service.name=podcast-pipeline` (when run w/ OTEL) | ✅ project 1 |

Legend: ✅ live+verified · ❌ not flowing · — n/a.

## Production debugging runbook

**Grafana** `http://homelab:3000` → folders **Podcast Operator**, **Podcast Player**,
**Production Infra**; raw signals via **Explore**.

**Incident: API is slow or 5xx-ing**

1. **Production Infra → Host Overview** — box under resource pressure (CPU/mem/disk)?
2. **Podcast Operator/Player → Overview** — which route? 5xx rate + p95 latency
   (`http_request_duration_seconds` by `handler`,`method`).
3. **Explore → VictoriaTraces (Tempo)**, service `podcast-api` — open the slow/failing
   span; from the span **→ logs** (tracesToLogsV2, G3a: the "Logs for this span" button runs
   `${__trace.traceId}` against VictoriaLogs).
4. **GlitchTip** `:8090` (project `podcast` / `player`) — the exception + stacktrace.

**Incident: a specific error (from GlitchTip)**

- The GlitchTip event's `trace_id` tag → open that trace in VictoriaTraces → its logs.

**Investigate a whole pipeline run**

- Explore → VictoriaLogs: `app:podcast run_id:<id>` (pipeline logs carry `[run=…]`). Cost
  events (`llm_cost`), the run's trace (`podcast-pipeline`), and any error share `run_id`.

**The pivots (bidirectional where wired):**

- **log → trace:** a log line with `trace=<hex>` / `"trace_id":"<hex>"` shows a **"View
  trace"** link (VictoriaLogs `derivedFields`). *Live for pipeline logs; API access logs
  don't carry a trace id yet — see gaps.*
- **trace → logs:** the Tempo datasource `tracesToLogsV2` (G3a, added 2026-07-24). ✅
- **error → trace:** Sentry `before_send` stamps `trace_id`.
- **metric → trace:** ❌ no exemplars (gap).

**Quick CLI checks (tailnet, run against `homelab`):**

```sh
# traces: which services are reporting
curl -s http://homelab:10428/select/jaeger/api/services
# metrics: api RED for prod-podcast
curl -s "http://homelab:8428/api/v1/query?query=up{instance='prod-podcast'}"
# logs: recent api log lines
curl -sG http://homelab:9428/select/logsql/query \
  --data-urlencode 'query={app="podcast",surface="api"}' --data-urlencode limit=10
```

## Gaps / issues / drift (the assessment)

**Correlation gaps** (these limit prod debugging today):

1. **API request logs carry NO `trace_id`.** They're uvicorn access lines
   (`INFO: … "GET /… HTTP/1.1" 200`). The log→trace pivot is therefore **dark for API
   logs** (works only for pipeline logs). Fix built but **held**: the G1 pure-ASGI
   access-log middleware (`server/app.py::_AccessLogMiddleware`) stamps `trace=<hex>`. It's
   **API code → needs a main-branch image build** to reach prod (see the correlation plan).
2. **Player API is not traced.** `OTEL_TRACES_EXPORTER` is unset on the player (G0 config
   held on `production`). A user-triggered player error has **no trace** in VictoriaTraces
   yet. Fix is config (deploy-time env + `homelab` extra_hosts) — held.
3. **No metric exemplars.** A latency spike on a dashboard can't jump to an example trace.
   Requires app-side OTEL metrics emission — not started (phase 2).
4. **Health/metrics log noise.** Every `/health` + `/metrics` poll is logged (uvicorn),
   flooding VictoriaLogs. G4 (`OTEL_PYTHON_FASTAPI_EXCLUDED_URLS` + the middleware skip)
   quiets it — held with the correlation batch.

**Infra issues:**
5. **cAdvisor can't name containers** (#1272) — Docker 29 containerd image store leaves the
   RW-layer unresolvable, so per-container dashboards show cgroup ids, not names. Fleet-wide;
   not fixable by flag. Tracked separately.
6. **GlitchTip client-side reachability** — browsers can't reach tailnet-only GlitchTip, so
   the **player** frontend ships errors via a **public ingest edge** (`telemetry.closelistening.app`).
   The **operator** frontend has no such edge → its client-side errors are **not captured**.

**Doc drift (fix or distrust):**
7. `OBSERVABILITY_ARCHITECTURE.md` still says the backend is on the **DGX**
   (`dgx-llm-1:8428/9428/10428`) — **stale; it's `homelab`.** Verify commands there point at
   the wrong host.
8. Same guide's dashboard section (`config/grafana/dashboards/vps/` + push-script, "VPS —
   Podcast" folder) predates the **file-provisioned** homelab Grafana with **Podcast
   Operator** / **Podcast Player** folders.
9. Same guide says player backend errors "land under `api`" — **stale**: the player has its
   **own GlitchTip project (5)** + backend DSN (`PROD_SENTRY_DSN_PLAYER`).

## Where things live (file map)

**This repo (`podcast_scraper-infra`):**

- Traces (auto-instrument): `docker/api/Dockerfile` (CMD `opentelemetry-instrument …`),
  `docker/pipeline/entrypoint.sh` (wraps when `OTEL_TRACES_EXPORTER!=none`).
- OTEL env: `compose/docker-compose.stack.yml` (base, exporter=none), `…vps-prod.yml`
  (operator: otlp + `homelab` extra_hosts), `…player-public.yml` (player: G0, held).
- Correlation IDs: `src/podcast_scraper/utils/correlation.py` (`run_id`/`episode_id`/
  `current_trace_id`/`CorrelationFormatter`). API access log: `server/app.py::_AccessLogMiddleware` (G1, held).
- Events/logs SDK: `src/podcast_scraper/obs/events.py` (`emit_event`).
- Metrics: `prometheus_fastapi_instrumentator` wired in `server/app.py` (gated
  `PODCAST_METRICS_ENABLED`).
- Errors: `src/podcast_scraper/utils/sentry_init.py` (`before_send` stamps `trace_id`).
- Log shipping drop-ins: `infra/observability/{podcast,player,orrery}.alloy` (dropped into
  the VPS Alloy `config.d/` by `infra/deploy/deploy.sh` + `deploy-player.sh`).
- Correlation plan: `docs/wip/observability-correlation-id-enhancement.md`.

**Homelab repo (`agentic-ai-homelab`):**

- Backends: `infra/observability/docker-compose.yml` (VictoriaMetrics/Logs/Traces, Grafana,
  GlitchTip, cadvisor, alloy).
- Grafana provisioning: `infra/observability/backend/grafana/provisioning/datasources/`
  (`victorialogs.yml` = derivedFields log→trace; `victoriatraces.yml` = tracesToLogsV2
  trace→logs [G3a]) + `…/dashboards/` (folders: Homelab, Production Infra, Podcast Operator,
  Podcast Player, Orrery).
- Homelab-node Alloy: `infra/observability/config.alloy` (host/GPU/vLLM metrics).

**Access:** homelab over Tailscale, SSH key `~/.ssh/homelab_mini` (`ssh homelab`). The VPS
Alloy internals live on the VPS at `/opt/vps-observability/`.

## How to extend

- **Trace a new API surface:** it already runs `opentelemetry-instrument`; set
  `OTEL_TRACES_EXPORTER=otlp` + `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` +
  `OTEL_SERVICE_NAME=<name>` + `extra_hosts homelab:<tailnet-ip>` (mirror the player G0 or
  operator vps-prod config). Nothing else.
- **Ship a new container's logs:** add an `infra/observability/<app>.alloy` drop-in
  (scoped `loki.source.docker`, label `app=<x>`) + wire its deploy to drop it into the VPS
  Alloy `config.d/` and `docker kill -s HUP alloy`.
- **Add a metric:** it's already scraped if it appears on `/metrics` — nothing to ship.
- **Add an event:** `emit_event("<name>", …)` — see the architecture guide's event catalog.
