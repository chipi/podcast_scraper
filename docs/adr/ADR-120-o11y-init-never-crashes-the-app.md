# ADR-120 — Observability init must never crash the app

**Status:** Accepted
**Date:** 2026-07-22

## Context

On 2026-07-22, while diagnosing the ADR-115 secrets cutover, a single bad
environment value took the whole api process down: `init_sentry` called
`sentry_sdk.init(dsn=...)` unguarded, so a **non-empty but malformed DSN** raised
`sentry_sdk.utils.BadDsn: Unsupported scheme ''` and crash-looped the app at
startup — while `/api/health` had been green the entire time it was actually
mis-delivering secrets.

Telemetry is *support* infrastructure. A misconfigured or unreachable telemetry
backend, a bad DSN, a missing optional package — none of these are worth taking
the product down for. The same principle already governs the deploy path
(`deploy.sh`: "telemetry must never break the app" — the `HOMELAB_TAILNET_IP`
fallback); this ADR makes it a code-level invariant too.

## Decision

**Every observability/telemetry init that loads config from the environment must
disable that feature on failure and let the app continue — never crash.**

Concretely, each o11y init point:

1. **Skips cleanly on an absent/empty value** (feature off, `debug` log).
2. **Wraps the actual init in `try/except`** so a *malformed* value (bad DSN, bad
   endpoint) or any init error logs **loudly** (`logger.exception` — an ERROR,
   which now surfaces in Sentry) and returns/continues **without** that feature.
3. Never `raise`s out of the init on a config/dependency problem — including a
   missing *optional* package (an app-up-without-metrics beats an app-down).

Failure is **loud in logs/Sentry, invisible to users.** Degrade, don't die.

## Consequences

**Positive**

- A bad secret/DSN/endpoint can no longer down the app (the BadDsn class).
- A mis-built image (missing optional o11y package) degrades to "no metrics"
  instead of a crash-loop.
- The `/api/health` probe stops being a false "all good" while telemetry is
  silently broken — the failure is an explicit ERROR log/Sentry event.

**Negative / neutral**

- A telemetry misconfig is now *quieter at the process level* — you must watch
  logs/Sentry (or a "telemetry disabled" signal) to notice it, rather than a hard
  crash. Mitigated by logging at ERROR (Sentry-visible) + the deploy-time checks.
- "Fail loud to catch a mis-built image at boot" is deliberately traded away for
  availability; deploy-time image/dependency checks are the right place for that.

## Applies to

- `init_sentry` (api + pipeline) — fixed 2026-07-22 (`c92854b8`).
- Metrics instrumentator (`server/app.py`) — softened from `raise RuntimeError` to
  log + skip.
- OTEL traces — already resilient (OTLP export failures are async + non-fatal).
- In-app OTEL span reads (`correlation.py`, `obs/events.py`) — already guarded.
- **Any new telemetry** (a new exporter, a Langfuse client in the app path, a log
  shipper) must follow this pattern.

## Alternatives considered

- **Fail loud on any o11y misconfig** (status quo for metrics). Rejected as the
  default: it makes non-critical support infra able to down the product. Kept only
  as a *deploy-time* check (image/dependency validation), not a runtime crash.
- **Silently swallow** o11y failures. Rejected: silence hid the original bug
  behind a green `/api/health`. Failures must be loud in logs/Sentry.
