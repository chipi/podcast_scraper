# Full app-surface observability — plan (self-hosted)

**Goal:** full o11y (metrics **and** logs) over the podcast operator surface —
pipeline, LLM usage, ML usage, enrichers, podcast player, search — plus orrery,
on the self-hosted stack (VictoriaMetrics + VictoriaLogs + Grafana on the DGX,
tailnet-only). Moving off Grafana Cloud.

Status date: 2026-07-20. Grounded in a telemetry-emission recon of
`podcast_scraper-FUTURE` and `orrery` (see "Current emission inventory").

## Backend (have)

- VictoriaMetrics (metrics), VictoriaLogs (logs), Grafana 11.6 — DGX, tailnet.
- prod-podcast Alloy collector (60s): host + cAdvisor(host-cgroups) + **api
  `/metrics`** + security logs (sshd/fail2ban/caddy). Cloud agents being retired.
- Datasource uids: `victoriametrics`, `victorialogs`. Push to Grafana folder
  `vps-podcast` via the `podcast-deploy` SA token (dashboards-as-code).

## Current emission inventory (what each surface emits TODAY)

| Surface | Metrics | Logs (structured JSONL unless noted) | Gap |
|---|---|---|---|
| **Pipeline** | Prometheus histograms (post-job, gated `PODCAST_METRICS_ENABLED`): transcribe/summarize/gi/kg seconds; `podcast_pipeline_jobs_finished_total{status}`; scheduler counters. Rich `run.json` Metrics. | `pipeline.log` + stdout, `[run=<id>]` correlation stamp. | No real-time per-stage metric; histograms only appear after a job. |
| **LLM usage/cost** | none | `event_type=llm_cost` JSONL: provider, model, prompt/completion/cached tokens, `estimated_cost_usd`, stage, episode_id, run_id (`emit_llm_cost_event()`, cost_monitoring.py). | Log-only; no Prometheus. Needs the log stream shipped. |
| **ML inference** (whisper/pyannote/embeddings) | none | timings in `run.json` (`transcribe_time_by_episode`, `gi_times`, `kg_times`, `vector_index_seconds`); device + failures on error logs. | No real-time inference latency/RTF metric. |
| **Enrichers** (GI/KG) | scheduler counters only | health endpoints (`/api/enrichment/{status,health,metrics}`), enrichment event JSONL, rich `run.json` counts (gi/kg artifacts, failures). | No real-time per-enricher call-rate/latency metric. |
| **Player/playback** | HTTP RED only (instrumentator) | `listen.jsonl` per-user (open events, position). | No server-side business counters; user-private logs. |
| **Search** | HTTP RED only | `query_log.jsonl` (`query_type`, no raw text; daily rollups). | No latency-by-type / vector-vs-hybrid split. |
| **api HTTP** | `http_request_duration_seconds`, `http_requests_total` (RED) ✓ collected | — | — |
| **orrery** | none (nginx-exporter is "future") | nginx access/error + pipeline-runner stdout → (was) Cloud Loki. | No metrics; logs need repointing to VL. |

Central logging: `CorrelationFormatter` (text, run-id) or `JSONFormatter`
(`json_logs=True`) → stdout + `pipeline.log`. `/metrics` gated on
`PODCAST_METRICS_ENABLED=1` (on in prod — scheduler counters present).

**Key insight:** the operator-surface telemetry (LLM cost, search, pipeline
events, listen) already exists — as **structured JSONL logs** that we dropped as
the container "firehose." Recovering it = ship the **scoped, structured** app-log
stream to VictoriaLogs (api + pipeline containers, JSON-parsed), NOT the firehose.
No app changes for the log-derived surfaces. Real-time *metrics* for ML/search/
enrichers need app instrumentation (Phase 2).

## Architecture — emit open formats, ship pluggably (vendor-neutral)

**Principle:** the app emits observability in two **open, vendor-neutral formats**
and depends on nothing else. Shipping is a **separate, swappable** layer.
Grafana/Loki is today's *sink*, not a coupling — a forker swaps it by config.

```
APP (producer)                 SHIPPING (pluggable)        BACKEND (swappable)
  emit_event("llm_cost", …)      agent tails stdout +        VictoriaLogs /
    → canonical JSONL/stdout ──▶  /metrics; adapter out  ──▶ VictoriaMetrics
  /metrics (Prometheus)          (Alloy = reference impl)    (or Loki/Datadog/…)
  depends on: open formats only  config, not app code
```

Two emission standards:
1. **Metrics** → Prometheus text exposition (`/metrics`). Already pluggable — any
   scraper consumes it. No app coupling.
2. **Events/logs** → **canonical JSONL** on stdout (12-factor) + optional file.
   One envelope: `{ts, schema, event, run_id, …fields}`.

**Emit side (in-app SDK):** a single `emit_event(event_type, **fields)` producing
the canonical envelope. The existing `emit_llm_cost_event` / `append_query_event`
/ `append_listen_event` become thin wrappers → one code path, one schema. A small
**event catalog** (`llm_cost`, `pipeline_stage`, `ml_inference`, `enrichment`,
`search_query`, `listen`, `job`) is the contract. Vendor-neutral; the app never
imports a Loki/Grafana SDK. → **ADR in `podcast_scraper-FUTURE`.**

**Ship side (pluggable adapters, this repo / infra config):** contract = "canonical
JSONL on stdout/file" + "Prometheus `/metrics`". Reference adapter = Alloy →
VictoriaLogs/VictoriaMetrics. Swap by config: repoint `loki.write`, swap Alloy for
Vector/Fluent Bit, add `otelcol`/Kafka/S3/Datadog output. Zero app change.

**Why:** app = producer of open formats, infra = consumer/router (12-factor "logs
as event streams" + a typed envelope so events query in ANY backend). Metrics
already prove the model; events get the same discipline. Forkability is a feature.

### Concrete collection (prod, api-triggered pipeline containers)

- **Metrics:** Alloy scrapes `api:8000/metrics` (done) → VM. Phase 2 adds app
  instruments for real-time ML/search/enricher latency.
- **Events/logs:** Alloy captures (a) the **ephemeral pipeline-runner container**
  stdout (`docker compose run --rm` on the VPS — llm_cost/ML/pipeline events) via
  scoped `loki.source.docker`, and (b) the **corpus event files**
  (`corpus/search/query_log.jsonl`, `users/*/listen.jsonl`, `.viewer/jobs.jsonl`,
  pipeline.log) via `loki.source.file` over the `/rootfs` mount. Labeled
  `job=podcast-app`; VL `unpack_json` exposes the envelope fields. Bounded +
  structured (not the ungoverned firehose we dropped).
- **Dashboards:** `config/grafana/dashboards/vps/*.json`, pushed to `vps-podcast`.
- **Alerts:** re-home T-11 security to self-hosted; add cost-spike /
  pipeline-failure / error-rate alerts.

## Phased plan

**Phase 0 — infra (DONE):** collector, api /metrics scrape, node + edge-security
dashboards, Cloud podcast-agent removed (api metrics preserved).

**Phase 1 — log-derived app o11y (no app changes; HIGH VALUE):**
1. Alloy: scoped structured-log shipping (api + pipeline containers) → VL.
2. Dashboards (log-derived): **LLM Usage & Cost** (cost by stage/model/day,
   tokens, cache-hit, guardrails), **Pipeline & Jobs** (jobs_finished by status,
   stage histograms, scheduler), **Search** (volume by query_type + HTTP latency),
   **Enrichers** (health/circuit + gi/kg counts), **Player** (listen events + RED).

**Phase 2 — real-time metrics (app instrumentation, `podcast_scraper-FUTURE`):**
- ML inference: whisper RTF + duration, diarization duration, embedding time
  (prometheus_client histograms at the provider call sites).
- Search: latency histogram by `query_type`, vector-vs-hybrid success/fail.
- Enrichers: per-enricher call-rate + latency counters.
- Delivered as app-repo instructions (cross-repo); collected by the existing scrape.

**Phase 3 — orrery:**
- Repoint orrery's grafana-agent Loki → `victorialogs` (or fold into prod Alloy
  via docker_sd scoped to `orrery-*`).
- Repoint + push the existing `orrery-fixes` dashboards (`orrery-web-access`,
  `orrery-pipelines`) — swap datasource uid → `victorialogs`. Retire the
  `orrery-edge.json` placeholder.
- Later: nginx-prometheus-exporter sidecar → VM for orrery web metrics.

**Phase 4 — alerts + retention:**
- Re-home T-11 security alerts (`config/grafana/alerts/common/security.yaml`) to
  self-hosted Grafana alerting / vmalert.
- App alerts: LLM cost spike, pipeline failure rate, api 5xx rate, enricher
  circuit-open.
- Right-size VM (`VM_RETENTION`) / VL (`VLOGS_RETENTION`) on the DGX backend.

## Container metrics — known limitation + plan

The **Containers** dashboard is empty on prod. Root cause: Docker on the VPS uses
the **containerd image store** (`Storage Driver: overlayfs`), which has no classic
`overlay2` layerdb. cAdvisor's docker factory hard-fails per container ("failed to
identify the read-write layer ID … image/overlayfs/layerdb/mounts/<id>/mount-id:
no such file") and never creates the handler — so only host cgroups + `machine_*`
are exported, no per-container series. Tried + rejected: `cgroup: host` (necessary,
insufficient), cAdvisor `v0.52.1` (same failure; `v0.53+` not on gcr.io),
`--containerd` (registers a 2nd factory but the docker factory still owns the
`docker-*.scope` cgroups and still fails). This is a cAdvisor↔containerd-image-store
incompatibility, not a config error.

Plan (in priority order):
1. **Defer (recommended).** Per-container resource isn't load-bearing for a small
   VPS — host pressure shows in **VPS Overview**, and app health in **Podcast App**
   (api RED from `/metrics`). Leave the Containers dashboard as a deep-dive stub.
2. **Box → overlay2**, if per-container resource becomes needed: a maintenance
   window — set `/etc/docker/daemon.json` `{"features":{"containerd-snapshotter":false}}`,
   `systemctl restart docker` (restarts ALL containers, re-pulls images), redeploy.
   One-time; then the existing cAdvisor works unchanged.
3. **Watch for a cAdvisor** release that supports the containerd image store
   (check ghcr/quay for `v0.53+`); drop-in image bump if it lands.

## Open decisions

- **Log format in prod:** is `json_logs` on? If not, either enable it (clean JSON
  for VL `unpack_json`) or parse the embedded JSON events. Confirm before Phase 1.
- **Scoped log volume:** api + pipeline stdout at INFO — acceptable? Or filter to
  `event_type=*` structured lines only to stay lean.
- **orrery ownership:** fold orrery logs into the podcast box Alloy, or keep
  orrery's own agent (repointed)? Cross-repo (orrery-fixes has the dashboards).
- **Revoke** the exposed orrery Cloud `glc_` token after Phase 3.
