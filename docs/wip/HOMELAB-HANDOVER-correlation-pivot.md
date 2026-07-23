# Homelab hand-off — Grafana traces⇄logs⇄errors pivot (correlation G3a)

**For:** the agentic-ai-homelab agent (owns Grafana + VictoriaLogs/Traces/Metrics).
**From:** podcast_scraper-infra. **Date:** 2026-07-23.
**Companion plan:** `docs/wip/observability-correlation-id-enhancement.md` (this repo).

## What changed on the app side (already deployed / deploying)

Both prod API surfaces now export **OTEL traces** to the homelab VictoriaTraces OTLP
ingest (`http://homelab:10428/insert/opentelemetry/v1/traces`), via the api image's
`opentelemetry-instrument` wrapper (ADR-119):

| Surface | `service.name` | How |
|---|---|---|
| Operator | `podcast-api` | `docker-compose.vps-prod.yml` (already on) |
| **Player (new)** | `player-api` | `docker-compose.player-public.yml` + deploy-player |

Also set on both (base `stack.yml` + player compose):
- `OTEL_PYTHON_LOG_CORRELATION=true` — injects `otelTraceID`/`otelSpanID` onto log records.
- `OTEL_PYTHON_FASTAPI_EXCLUDED_URLS=health` — `/health` probes no longer create spans.

Errors already flow to **GlitchTip** (Sentry SDK) from both surfaces.

## Your task (Grafana datasource config — lives in the homelab repo)

1. **Logs → trace ("View trace" button).** On the **VictoriaLogs** datasource add a
   `derivedField` that extracts the trace id from the log line and links to the
   **VictoriaTraces** datasource.
   - ⚠ **Verify the actual log format first** (see caveat below) — the regex depends on how
     the trace id renders. Likely one of: `otelTraceID=(\w+)` (LoggingInstrumentor default)
     or the app's `trace=(\w+)` token (pipeline `CorrelationFormatter`).
   - `internalLink` → VictoriaTraces datasource, `url` = the trace id.
2. **Trace → logs (back-link).** On the **VictoriaTraces** datasource configure
   `tracesToLogsV2` → VictoriaLogs, filtering `{service=...} | trace_id=<id>` (VictoriaLogs
   LogsQL), tags `service.name`.
3. **Error → trace (optional).** If GlitchTip events carry `trace_id` (Sentry OTEL context),
   add a link from the GlitchTip issue view / a Grafana panel to the trace. If they don't,
   note it — enabling the Sentry OTEL integration app-side is a follow-up (G5).

## Caveat — verify G1 before wiring the regex

`OTEL_PYTHON_LOG_CORRELATION=true` puts the ids on the LogRecord, but they only appear in
the **output** if the active formatter's format string references them. The api may use
uvicorn's own formatter or a custom one that doesn't. **Before writing the derivedField
regex, pull a real player-api log line from VictoriaLogs and confirm the trace id is
present and in what shape.** If it's absent, the app-side fix is to set
`OTEL_PYTHON_LOG_FORMAT` or add the field to the api log formatter (tracked in the plan doc
as an open item) — tell infra and we'll patch it.

## Verify end-to-end

Trigger a user-facing error on the player (e.g. a 500 on an app route), then:
1. Find the error in GlitchTip (player project) and/or the log line in VictoriaLogs
   (`{service_name="player-api"}` or `{app="player"}`).
2. Click **View trace** → the span shows in VictoriaTraces with `service.name=player-api`.
3. From the trace, back-link to its logs.

## Out of scope here (later phases)
- Metric exemplars (VictoriaMetrics histogram → exemplar trace) — needs app-side OTEL
  metrics emission; phase 2.
