# MCP o11y control-plane re-alignment — gap analysis & plan

**Date:** 2026-07-27
**Author:** Marko Dragoljevic
**Scope:** Re-align the `podcast_obs` MCP/CLI control plane with (a) the production observability stack
it actually runs against today and (b) the new signals just landed (per-episode manifest /
`pipeline_stage` events / `episode_id` correlation / diarization metrics / `capture_stage_exception`).
**Goal:** an agent (or operator) can say *"observe the API"* or *"observe the pipeline,"* get every
signal class for that surface, and **investigate** — pivot to the run / episode / trace / incident —
end-to-end, from a headless agentic environment.
**Inputs:** two read-only maps (the o11y emission+backend review, and a full `podcast_obs` surface
map) + `docs/guides/OBSERVABILITY_CONTROL_PLANE.md` / `OBSERVABILITY_RUNBOOK.md`.

---

## 0. TL;DR — the control plane points at the OLD stack

`podcast_obs` exists and is more capable than its doc says (**21 MCP tools**, a working `run_id`
correlate, a deterministic Langfuse trace join). But **every backend source is wired to the previous
generation stack** — Grafana **Cloud** Loki, **sentry.io**, Langfuse **Cloud** default — while
production moved to the self-hosted **homelab** box: **VictoriaLogs** (LogsQL), **VictoriaMetrics**
(PromQL), **VictoriaTraces** (OTLP/Jaeger), **GlitchTip** (`:8090`). `config/observability.example.yaml`
itself flags this drift. Net effect against the goal:

- **metrics** — no source at all (can't ask "API error-rate / CPU right now").
- **logs** — `loki.py` speaks Cloud-Loki LogQL; prod is VictoriaLogs/LogsQL (different endpoint +
  query language). The whole `emit_event` stream (`llm_cost` / **`pipeline_stage`** / `search_query`)
  is unreachable except the one hard-coded `llm_cost` slice.
- **errors** — `sentry.py` hardcodes `sentry.io`; no config field to point at GlitchTip.
- **traces** — only Langfuse (LLM-call-scoped); no VictoriaTraces, so a slow/failing **API request**
  has no trace path.
- **surface-scoping** — no `surface`/`component` (api vs pipeline) filter anywhere.
- the new signals from this week (`pipeline_stage`, per-episode manifest, `episode_id` pivot) have
  **no tool**.

So the re-alignment is not a tweak — it's re-pointing the sources at the current stack, adding the two
missing backends, and adding the surface/investigate ergonomics the goal needs.

---

## 1. Current state (what works, keep it)

- **21 MCP tools** (`mcp_server.py`), uniform `{ok, source, data|error, configured}` envelope,
  `stdio` + `streamable-http` transports (headless-reachable over tailnet, port 8848).
- **Good bones for the goal:** `prod_summary` (one-call glance across all sources),
  `prod_correlate(run_id)` (fans a run across Langfuse/Loki/Sentry/enrichment), `prod_usage`
  (cost sliced by provider/model/**episode_id**/run), the deterministic `trace = sha256(run_id)`
  join, credential-free `prod_api`/`enrichment` sources.
- **Config model is sound:** single-target env (`PODCAST_OBS_*`) or multi-target YAML with
  `<field>_env:` secret indirection. Re-pointing is mostly new source modules + config fields, not a
  rewrite.

---

## 2. Gap analysis (ranked; mapped to the "observe api/pipeline + investigate" goal)

### A. Backend drift — the control plane can't see prod (blocks the whole goal)
1. **No metrics source.** VictoriaMetrics (`:8428`, PromQL) has no module. → can't observe a surface's
   RED (rate/errors/duration), CPU/mem, or `/metrics` counters.
2. **Logs point at the wrong backend.** `loki.py` → Cloud-Loki LogQL; prod is VictoriaLogs
   (`:9428`, `/select/logsql/query`, LogsQL). Structurally incompatible. The `emit_event` stream
   (incl. the new **`pipeline_stage`**) is unreachable.
3. **Errors point at the wrong backend.** `sentry.py` hardcodes `sentry.io`; prod errors land in
   **GlitchTip** (`:8090`). No `sentry_url` config field exists → not even configurable.
4. **No general-trace source.** Only Langfuse. VictoriaTraces (`:10428`, `service.name=podcast-api`/
   `podcast-pipeline`) is invisible → can't pull the trace for a failing API request.

### B. Ergonomics the goal specifically needs
5. **No surface/component scoping.** The stack tags every signal `component`/`surface` =
   `api`/`pipeline`/`player`/`operator`/`moss`/`pyannote`, but no tool takes it. → *"observe the API"*
   vs *"observe the pipeline"* is not expressible as a first-class filter.
6. **`trace_id` is not a pivot key.** `prod_correlate` takes only `run_id`; the live stack's
   request-granularity join key is `trace_id` (log→trace→error). No tool for it.
7. **No `episode_id` pivot** despite `episode_id` now being stamped on logs / Langfuse / Sentry (our
   P0 correlation work). Only reachable indirectly via `prod_usage(group_by=episode_id)`.

### C. New signals unexposed
8. **`pipeline_stage` events** (per-stage cost/quality/versions, just shipped) — no query path
   (needs the VictoriaLogs source from gap #2).
9. **Run/episode artifacts unreachable.** No tool wraps the app's own
   `/api/corpus/documents/manifest`, `/api/corpus/runs/summary`, `/api/corpus/stats`, or per-episode
   `<base>.manifest.json`. `enrichment.run_summary()` is implemented but never registered — dead code.

### D. Hygiene / trust
10. **"Read-only" framing is false.** `enrichment_re_enable` / `enrichment_cancel` POST-mutate deploy
    state, contradicting the server's own instructions + the doc. An agent reasoning about safety is
    misled.
11. **CLI lags MCP** — no `resilience` / `usage` / `enrichment-*` subcommands.
12. **Doc + version drift.** `OBSERVABILITY_CONTROL_PLANE.md` says "11 tools" (actual: 21);
    `__version__=0.1.0` isn't surfaced in the tool instructions, so an agent can't negotiate schema.

---

## 3. Proposal — phased plan

Design principle: **keep the fast-glance shape, re-point the sources, and add two verb-level tools
that match the goal — `obs_surface(...)` (observe) and `obs_investigate(...)` (drill).** Everything
else is sources feeding those.

### Phase A — re-point at the current stack (unblocks everything)
- **`sources/victorialogs.py`** (LogsQL) replacing/aliasing `loki.py`'s role. Generic
  `query(logsql, window, limit)` + typed helpers: `events(event_type=…, filters=…)` so
  `llm_cost` / `pipeline_stage` / `search_query` are all reachable, `logs(surface=…, level=…)`.
- **`sources/victoriametrics.py`** (PromQL): `instant(query)` / `range(query, window)` + RED helpers
  keyed by `service`/`surface`.
- **`sources/victoriatraces.py`** (Jaeger API): `recent(service, window, limit)`,
  `trace(trace_id)`, `errors(service, window)`.
- **`sources/errors.py`**: add a `sentry_url` config field (default `sentry.io`, set to GlitchTip
  `:8090` in prod YAML) — Sentry-protocol compatible, so it's a base-URL + token change, not a rewrite.
- **Config:** add `victorialogs_url` / `victoriametrics_url` / `victoriatraces_url` / `sentry_url`
  to `TargetConfig`; ship a `config/observability.homelab.yaml` example pointed at the tailnet host.
- **Keep the legacy sources** behind config (so a Cloud-Loki target still works) — additive, not a
  breaking swap.

### Phase B — the two goal-shaped tools
- **`obs_surface(surface="api"|"pipeline"|…, window="1h")`** — ONE call returning that surface's
  five-signal snapshot: RED metrics (VictoriaMetrics), recent errors (GlitchTip), error-ish logs
  (VictoriaLogs), recent/slow traces (VictoriaTraces), and (pipeline) cost + `pipeline_stage` rollup.
  This is the literal *"observe the API / the pipeline"* verb.
- **`obs_investigate(trace_id? | run_id? | episode_id?, window)`** — the drill-down: given ANY join
  key, fan every backend and return the correlated bundle (logs + trace + spans + errors + cost +
  manifest/pipeline_stage). Generalizes `prod_correlate` to `trace_id` + `episode_id` + the new
  backends.

### Phase C — expose the new signals
- Wrap the existing app routes as tools: `prod_run_summary`, `prod_corpus_stats`,
  `prod_episode_manifest(episode_id)` (add the API route if absent), and register the already-written
  `enrichment.run_summary()`.
- `pipeline_stage` + manifest come "for free" through the VictoriaLogs `events()` helper +
  `obs_investigate`.

### Phase D — hygiene
- Split the surface: mark the two write tools clearly (a `mutating: true` annotation + a
  `PODCAST_OBS_ALLOW_WRITES` gate, default off) and fix the instructions/doc to stop claiming
  read-only.
- CLI parity (add `resilience` / `usage` / `surface` / `investigate` / `enrichment-*`).
- Surface `__version__` + a `tool_schema_version` in the MCP instructions; reconcile
  `OBSERVABILITY_CONTROL_PLANE.md` to 21 tools + the new ones (single source of truth).

### Sequencing
A (backends) → B (`obs_surface` + `obs_investigate`) → C (new-signal tools) → D (hygiene). A+B alone
delivers the goal; C/D make it complete and trustworthy.

---

## 4. Target E2E flow (what the goal looks like when done)

```
agent: observe the pipeline (last 2h)
  → obs_surface(surface="pipeline", window="2h")
    → { metrics: {run_rate, error_rate, stage p95}, errors: [GlitchTip issues tagged surface=pipeline],
        logs: [VictoriaLogs error lines], traces: [slow pipeline spans],
        cost: $X, pipeline_stage: {per-stage cost, failovers, flagged episodes} }

agent: episode 0006 looks wrong — investigate
  → obs_investigate(episode_id="…")
    → { manifest: <base>.manifest.json, pipeline_stage events, logs (episode_id=…),
        langfuse trace (GI/KG calls+cost), sentry issues (episode_id tag), the run it belonged to }

agent: this API request 500'd — investigate
  → obs_investigate(trace_id="…")
    → { VictoriaTraces span tree, correlated VictoriaLogs lines (trace_id=…), GlitchTip issue }
```

The join keys our recent work made real — `run_id`, `episode_id` (P0), `trace_id` (API OTEL, P2 #6)
— are exactly what makes `obs_investigate` cross-backend.

---

## 5. Repo boundary

- **This repo (`podcast_scraper`):** all of `src/podcast_obs/` (sources, tools, config), the app API
  routes it wraps, the `docker/observability/` image, and the docs. **All of the plan above is
  in-repo.**
- **Sibling `agentic-ai-homelab`:** the backends themselves (VictoriaMetrics/Logs/Traces, GlitchTip,
  Grafana) + their datasources/provisioning. The plan only needs their **URLs + read tokens** (tailnet
  reachable), not changes there.

## 6. Non-goals
- Not building a Grafana replacement — the doc already pairs `podcast_obs` (fast glance / agent verbs)
  with a separate Grafana MCP for deep dashboards; keep that split.
- Not making the control plane a write/ops tool — the two existing write probes get gated, not
  expanded.

## 7. Open questions for the operator
1. **GlitchTip API token** — is there a read token for `:8090`, or do we mint one? (blocks the errors
   re-point).
2. **VictoriaLogs/Metrics/Traces read auth** — tailnet-open, or is there a token/basic-auth in front?
3. **Deploy shape** — run the re-pointed `podcast_obs` as a homelab-side compose service (tailnet), so
   a cron/headless agent reaches it over http — confirm that's the intended host.
4. **Scope for now** — do Phase A+B (deliver the goal) first and land C/D after, or all four in one arc?
