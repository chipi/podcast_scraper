# Observability correlation: traces ⇄ logs ⇄ metrics

**Status:** In progress (reconstructed 2026-07-23). G0 + G1 + G4 implemented this-repo
(held, pending a player deploy / main-image for G1); **G3a DONE** — `tracesToLogsV2`
trace→logs applied on the homelab Grafana 2026-07-24 (pending the homelab-repo commit; the
log→trace derivedField already existed). G5 + metric exemplars open. Live-state assessment:
`docs/guides/OBSERVABILITY_RUNBOOK.md`.
**Owner:** Marko
**Scope:** self-hosted o11y (VictoriaLogs + VictoriaTraces + VictoriaMetrics behind Grafana)

## Goal

One click to pivot between the three signals for a single request/run: from a log
line → its distributed trace → the metrics around it (and back). The join key is the
OTEL **`trace_id`** (plus the app-level `run_id`/`episode_id` for pipeline work).

## Current state (ADR-119 — traces signal; more built than it looks)

- **The API image is already OTEL-ready.** `docker/api/Dockerfile` runs
  `opentelemetry-bootstrap -a install` and its **CMD wraps the process in
  `opentelemetry-instrument`** — auto-instruments FastAPI request spans + outbound HTTP.
  Gated by `OTEL_TRACES_EXPORTER` (base `docker-compose.stack.yml` defaults it to `none`).
- **Operator surface is traced.** `docker-compose.vps-prod.yml` sets `OTEL_TRACES_EXPORTER:
  otlp` for `api` + `pipeline` + `pipeline-llm`, with `extra_hosts homelab:<tailnet-ip>`;
  the endpoint comes from the prod `.env` (deploy-prod renders it) →
  `http://homelab:10428/insert/opentelemetry/v1/traces` (**VictoriaTraces**).
- **Pipeline** correlation is done: `correlation.py` join-key layer (`run_id`/`episode_id`)
  + `CorrelationFormatter` stamps the OTEL `trace_id` on pipeline log lines (`4d368eb0`).
- Alloy drop-ins ship container logs → **VictoriaLogs** with `{app, surface, container}`.

## Gaps (the actual work — config, not new machinery; no new deps)

### G0 — Player surface is NOT traced [this repo] ← the main gap for player errors
`docker-compose.player-public.yml` sets no OTEL vars, so `OTEL_TRACES_EXPORTER=none` →
the player API (same image, already `opentelemetry-instrument`-wrapped) exports nothing.
**Fix:** mirror the operator:
- player-public.yml api service: `OTEL_TRACES_EXPORTER: otlp`, `OTEL_SERVICE_NAME:
  player-api` (distinct from operator `podcast-api`), `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL:
  http/protobuf`, `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: ${…:-}`, `extra_hosts
  homelab:${HOMELAB_TAILNET_IP}`.
- deploy-player.sh: resolve `HOMELAB_TAILNET_IP` (`tailscale ip -4 homelab`, like
  deploy-prod) + the workflow stages `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` into .env.player.

### G1 — `trace_id` on API request logs, both surfaces [this repo]
`opentelemetry-instrument` ships `LoggingInstrumentor`; enable
`OTEL_PYTHON_LOG_CORRELATION=true` so request logs carry `otelTraceID`/`otelSpanID`, and
ensure the api log format renders it. Set in `stack.yml` so both surfaces inherit.

### G4 — drop `/api/health` span noise, both surfaces [this repo]
Health probes create a span per poll. Set `OTEL_PYTHON_FASTAPI_EXCLUDED_URLS` (e.g.
`health,/api/health,/api/app/health`) in `stack.yml`. No code.

### G5 — errors ⇄ trace [both surfaces]
GlitchTip already receives errors (Sentry SDK). Confirm/enable that a captured error
carries the active `trace_id` (Sentry OTEL context / at minimum the error log line carries
it via G1), so an error pivots to its trace.

### G3a — Grafana "View trace" pivot [homelab repo: agentic-ai-homelab]
VictoriaLogs datasource `derivedFields` regex `trace_id` → "View trace" → VictoriaTraces;
back-link traces → logs; GlitchTip error → trace. Hand-off note like
`HOMELAB-HANDOVER-observability-reorg.md`. (Metric exemplars = later phase.)

## Proposed order

1. **G0** — player OTEL wiring (env + `homelab` host + deploy plumbing). The biggest gap;
   directly delivers "trace player errors." One player deploy (shared-edge; needs approval).
2. **G1 + G4** — log correlation + health-span exclusion in stack.yml (both surfaces).
   Rides the same player deploy; operator picks it up on its next deploy.
3. **G5** — verify error→trace tagging on both surfaces.
4. **G3a** — homelab Grafana pivot (hand-off note).
5. Metric exemplars — later phase, only if metric→trace pivot is wanted.

## Non-goals
- Not migrating off the Victoria stack. Not adding Tempo. Not per-request sampling tuning
  beyond dropping health noise. Not changing the operator's already-working trace export.

## Open questions
- Operator traces: confirmed *configured*, not confirmed *landing* in VictoriaTraces
  (needs a homelab-side query). Verify during G5.
- Player→homelab reachability: player runs on the same VPS/tailnet as the operator, so
  `extra_hosts homelab:<tailnet-ip>` + `:10428` should work — verify at deploy.
