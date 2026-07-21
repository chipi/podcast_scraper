# ADR-119: Vendor-neutral observability — canonical event emission + pluggable shipping

- **Status**: Proposed
- **Date**: 2026-07-20
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related ADRs**: [ADR-117](ADR-117-multi-tenant-observability-gitops.md) (the
  multi-tenant o11y ownership split this refines — each tenant owns its app
  telemetry; this ADR fixes *how* the app emits it)
- **Tracking**: app-surface o11y (`docs/wip/observability-app-surface-plan.md`)

## Context

ADR-117 says each tenant owns its app metrics/logs. It does not say *how* the app
should emit them. Today emission is ad-hoc: `emit_llm_cost_event()` writes JSON via
a logger to stdout; `append_query_event()` / `append_listen_event()` write JSON to
per-corpus files — three shapes, three code paths, no shared envelope, and an
implicit assumption that "logs go to Loki." When we moved off Grafana Cloud to a
self-hosted VictoriaMetrics/VictoriaLogs stack, nothing in the *app* had to change —
which is the property we want to make explicit and permanent, so a fork can ship to
any backend without touching app code.

## Decision

The app emits observability in **two open, vendor-neutral formats** and depends on
nothing else. **Shipping is a separate, pluggable layer.**

1. **Metrics** — Prometheus text exposition (`/metrics`). Already pluggable; any
   scraper consumes it.
2. **Events/logs** — **canonical JSONL** via a single
   `podcast_scraper.obs.events.emit_event(event_type, **fields)`.
   - Envelope: `{"ts", "schema", "event_type", ...fields}`. `schema` versions the
     shape; consumers key on `event_type`.
   - Two channels via `sink=`: `"log"` → one JSON line to the
     `podcast_scraper.events` logger → **process stdout** (for ephemeral pipeline
     runs — llm_cost, ml_inference, pipeline_stage); `"file"` → append to a
     **per-corpus JSONL** (for serve-side events that must persist without an agent
     attached — search_query, listen, job).
   - Telemetry **never breaks the caller** (every path swallows errors).
   - The existing emitters become **thin wrappers** over `emit_event` — one code
     path, one envelope, one place to evolve. An **event catalog** (the known
     `event_type`s + fields) is the contract.

3. **Shipping** is out-of-app config, not code. Contract = "canonical JSONL on
   stdout/file" + "Prometheus `/metrics`". The **reference adapter** is Grafana
   Alloy → VictoriaLogs/VictoriaMetrics (infra repo). Swapping is a shipping-config
   change (repoint `loki.write`; swap Alloy for Vector/Fluent Bit; add
   `otelcol`/Kafka/S3/Datadog output). The app never imports a vendor SDK.

### Signal taxonomy — same principle, every signal

The abstraction generalizes beyond metrics + logs. Every observability signal uses
an OPEN STANDARD the app emits, shipped to a SWAPPABLE self-hosted backend:

| Signal | Open standard (app emits) | Reference backend | Ship |
| --- | --- | --- | --- |
| Metrics | Prometheus `/metrics` | VictoriaMetrics | Alloy scrape |
| Logs / events | canonical JSONL (`emit_event`) | VictoriaLogs | Alloy tail |
| Traces (spans) | **OTLP** (OpenTelemetry) | VictoriaTraces | OTEL SDK / auto-instrument → OTLP HTTP |
| Errors | Sentry protocol | GlitchTip (self-hosted; Sentry Cloud swappable by DSN) | Sentry SDK |
| LLM prompt/cost | Langfuse span | Langfuse | `emit_langfuse_span` (in-code choke point) |

The app depends on the STANDARD, never the vendor — swap VictoriaTraces→Jaeger,
GlitchTip→Sentry, VictoriaLogs→Loki by config/DSN, no app change. Traces are
env-var driven (`opentelemetry-instrument`, zero app code); the OTLP traces
endpoint uses the traces-specific var with VictoriaTraces' non-standard full path.
This ADR governs the LOGS/EVENTS emit side (`emit_event`); the same discipline
applies across the whole table. Backend/handover pointers:
`agentic-ai-homelab/docs/wip/{podcast-otel-traces,glitchtip-*}-handover.md`.

## Consequences

**Positive:** app is vendor-neutral (Grafana/Loki is a swappable sink, per ADR-117
per-tenant ownership); one envelope makes events queryable in *any* backend;
forkability becomes a first-class feature; metrics and events follow the same
producer/consumer discipline.

**Negative:** a migration step — the three existing emitters must be refactored onto
`emit_event`, and existing Loki queries that assumed the old ad-hoc shapes may need
the added `schema`/`event_type` fields (additive, so low-risk).

**Neutral:** stdout is the canonical channel for pipeline events (12-factor); the
persistent corpus JSONL files remain for serve-side events that are naturally
file-shaped and must survive with no agent attached.

## Alternatives considered

- **Emit straight to an OTLP/OpenTelemetry SDK.** Rejected for now: heavier
  dependency + couples the app to a collector protocol; the JSONL-on-stdout contract
  is simpler, dependency-free, and a forker can bridge to OTLP in the shipping layer.
- **Keep the three ad-hoc emitters.** Rejected: no shared envelope, three places to
  evolve, and the "logs go to Loki" assumption stays implicit.
- **A vendor client in-app (Loki push SDK).** Rejected outright — that is exactly the
  coupling this ADR removes.
