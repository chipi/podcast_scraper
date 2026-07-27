# Observability Integration Review — pipeline signals ↔ Grafana / GlitchTip / Langfuse

**Date:** 2026-07-27
**Author:** Marko Dragoljevic
**Scope:** How the podcast_scraper pipeline emits logs / traces / errors / metrics / cost, where those
land (Grafana / GlitchTip / Langfuse / VictoriaMetrics-Logs-Traces), how the new per-episode
processing manifest (RFC-109 / ADR-130) plugs in, and the full-pipeline gaps to reach "measure
everything, correlate, evolve while running."
**Method:** two read-only mapping passes (emission surface + backend integration), findings
re-verified by hand. Related: [RFC-109](../rfc/RFC-109-per-episode-observability-manifest.md),
ADR-117 (multi-tenant o11y), `docs/guides/OBSERVABILITY_RUNBOOK.md` (authoritative current state).

---

## 0. TL;DR — the one finding that gates everything

**Per-episode correlation is effectively dead on a healthy run.** `correlation.set_episode_id()`
(#1053) is called from exactly one path — the **safety-net tail summarizer**
(`workflow/stages/summarization.py:230`), which by design processes ~0 episodes on a healthy run.
The **inline** `ProcessingProcessor` (`workflow/stages/processing.py`), which handles ~every episode,
**never sets it**. So for download / ASR / diarization / naming / normal-path summary, every log
line, Sentry tag, `emit_event`, and Langfuse span shows `episode_id=-`. Only GI/KG get an episode id
— and via a *different* mechanism (`llm_call_fuse.install_episode`), not the correlation module.

Everything the operator asked for — "correlate these things, query per episode, evolve while
running" — rests on a stable `episode_id` across all signals. **Fixing this is P0; it is the spine
the manifest and every dashboard hang off.**

---

## 1. Two systems both called "metrics" — do not conflate

| System | What | Sink | Relevance |
| --- | --- | --- | --- |
| **CI / quality metrics** (RFC-025/026/027/043, ADR-023) | test health, coverage, complexity, pipeline perf snapshot | static JSON → **GitHub Pages** | developer scoreboard; NOT prod o11y |
| **Production o11y** (ADR-117/119/120, OBSERVABILITY_*) | runtime logs/traces/errors/metrics/cost of the deployed app | **VictoriaMetrics/Logs/Traces + GlitchTip + Grafana** on the homelab box | this review |

`scripts/dashboard/generate_metrics.py` belongs to the first and is a *consumer*, not an emitter.
Don't expect its output to ever reach VictoriaMetrics.

---

## 2. Coverage matrix — signal × stage (production o11y)

Legend: ✅ emitted & sinks to a backend · 🟡 emitted to a file/sidecar only (no backend sink) ·
⚠️ partial / broken · ❌ none.

| Stage | Run metrics (`metrics.json`) | Per-episode | Cost | Traces | Errors | Quality signal |
| --- | --- | --- | --- | --- | --- | --- |
| Download | ✅ counters/timings | ⚠️ `episode_id=-` | n/a | ⚠️ httpx spans, no ep id | ⚠️ incidents JSONL, no Sentry | ❌ |
| ASR | ✅ | ⚠️ id | ✅ `llm_cost` event + Langfuse | ⚠️ | ⚠️ | 🟡 manifest (`speech_coverage`, failover) |
| Diarization | ❌ **no `diarization_*` in metrics.py** | ⚠️ id | ✅ (manifest + pricing) | ⚠️ | ⚠️ | 🟡 manifest (`num_speakers`, `unattributed`) |
| Naming | ✅ time only | ⚠️ id | ✅ (speaker-detect) | ⚠️ | ⚠️ | 🟡 manifest (detected-vs-named, flags) |
| Summary | ✅ | ⚠️ id | ✅ | ✅ Langfuse span | ⚠️ | 🟡 manifest (word_count, status) |
| GI | ✅ rich | ✅ (via `llm_call_fuse` ep id) | ✅ | ✅ | ⚠️ | 🟡 manifest (`insight_count`, `gi_all_gated`) |
| KG | ✅ rich | ✅ | ✅ | ✅ | ⚠️ | 🟡 manifest (node/edge counts) |
| Metadata-gen | ✅ count | ✅ `episode_finished` JSONL | — | — | ⚠️ | — |
| Search/index | ✅ + `search_query` event | n/a | n/a | ⚠️ | ⚠️ | ✅ query_log.jsonl |

**Backends that ARE wired:** GlitchTip per-surface DSN (`api`/`pipeline`, `sentry_init.py`);
Langfuse LLM spans via one choke point (`cost_monitoring.py:197`, deterministic `trace=sha256(run_id)`);
OTEL traces (pipeline on; **API + player OFF**); Prometheus `/metrics` on the API
(`server/app.py:397`, gated on `PODCAST_METRICS_ENABLED`); `emit_event` vendor-neutral JSONL
(`obs/events.py`) → VictoriaLogs. Cost funnels through a single choke point
(`provider_metrics.record_provider_call_cost` → `emit_llm_cost_event`).

---

## 3. Where the per-episode manifest fits

The manifest is the **per-episode quality+cost spine the o11y stack is missing** — but today it is a
🟡 **sidecar file with no backend sink**. It already holds, per episode, what no dashboard currently
shows in one place: actual ASR model + failover, diarization speakers/speech-seconds, naming
detected-vs-named + rework flags, GI/KG counts, and **all-six-stage per-episode cost** (via
`EpisodeCostProbe`).

It overlaps three existing artifacts, and is the most complete of them:

| Artifact | Per-episode cost | Diarization | Quality flags | Sinks to backend? |
| --- | --- | --- | --- | --- |
| `EpisodeMetrics` → `episode_finished` JSONL | ⚠️ **transcribe+summary only** | ❌ | ❌ | ✅ VictoriaLogs |
| `<base>.manifest.json` (new) | ✅ **all 6 stages** | ✅ | ✅ | ❌ **none** |
| `llm_cost` events | ✅ per-call | n/a | n/a | ✅ |

So a consumer trusting `episode_finished.estimated_cost` **systematically undercounts** any episode
with GI/KG. The manifest already fixes the *data* gap; it just isn't *emitted*.

**Integration seam:** have the manifest writer also `emit_event(event_type="episode_manifest", …)`
(reusing the existing vendor-neutral SDK) so each per-episode record flows to VictoriaLogs → Grafana,
addressable by `run_id`/`episode_id`. No new backend, no new dependency — the pipe already exists.

---

## 4. Gaps, prioritised

### P0 — correlation spine (unlocks everything else)
1. **`episode_id` unset on the inline path** (§0). Call `correlation.set_episode_id()` in
   `ProcessingProcessor` around each episode, matching what the safety-net summarizer does. Without
   it, per-episode correlation across logs/errors/traces/cost is dark for 5 of 7 stages.

### P1 — make the manifest observable + reconcile cost
2. **Manifest has no backend sink.** Emit an `episode_manifest` event per episode (§3) → queryable
   in Grafana. This is the operator's "expose + query + correlate" ask, cheaply.
3. **Per-episode cost SoT is split.** Make the manifest the per-episode cost record and either (a)
   write GI/KG cost back into `EpisodeMetrics`, or (b) point `episode_finished` at the manifest.
   Today the two disagree by the entire GI+KG cost.
4. **Diarization is invisible in run metrics.** No `diarization_*` in `metrics.py` / `metrics.json` /
   `run.jsonl` — it lives only in the manifest. Add run-level diarization counters (speakers,
   speech-seconds, cost, failover-adjacent) so it's dashboardable like the other stages.

### P1 — error observability blind spot
5. **No `sentry_sdk.capture_exception()` anywhere.** Every Sentry emission is `capture_message` /
   breadcrumb. The dominant pipeline pattern is `except Exception: logger.debug(...)` — caught
   errors never reach GlitchTip; only unhandled ones bubble via the SDK's auto-integrations. For an
   "errors" pillar this is a large hole: wire `capture_exception` at the stage-boundary catch sites
   (download, ASR, diarization, GI/KG) with `episode_id` scope.

### P2 — trace completeness + hygiene
6. **OTEL not initialised for the API server** (`init_otel()` only in `cli.py`, not
   `server/app.py`) → API errors/events never get a `trace_id`. (Pairs with the runbook's held
   "API access-log `trace_id`" G1 and "player traces OFF" G0.)
7. **`json_logs=True` is a dead config path** — imports `utils.json_logging` which **does not
   exist**; setting it raises `ModuleNotFoundError`. Either ship the module or remove the flag.
8. **`ml_inference` / `pipeline_stage` event types documented but never emitted** (`obs/events.py`
   docstring) — decide: emit them (a natural home for the manifest's per-stage records) or drop them
   from the catalog.
9. **`INFO`-level runs carry almost no per-stage detail** — every GI/KG/summary/search counter is
   DEBUG-gated in `log_metrics()`; only 3 scattered INFO lines. A one-line INFO per-stage summary
   would make tailing a prod run useful without DEBUG noise.
10. **GitOps not automated** — `make obs-sync` and `push-grafana-dashboards.sh` are manual/dry-run;
    no workflow runs them (infra-side, but worth a decision).

---

## 5. What's already good (don't rebuild)
- Vendor-neutral emission (`emit_event`, OTLP, Sentry/Langfuse SDKs) with pluggable backends — the
  right architecture (ADR-119/ADR-117). We extend it, not replace it.
- A **single cost choke point** — one place to trust for LLM cost + Langfuse spans.
- Deterministic `trace = sha256(run_id)` — a run's spans/logs/errors are addressable without search.
- The manifest gives us the per-episode quality dimension the stack lacked; it just needs a sink.

## 5b. Resolution status (all gaps actioned)

| Gap | Status |
| --- | --- |
| P0 #1 episode_id on inline path | **Done** — `correlation.episode_scope()` + bound at the download (`_process_episode_with_retry`), transcription (`transcribe_media_to_text`), and metadata-gen (`generate_episode_metadata`) seams. |
| P1 #2 manifest → backend sink | **Done** — `update_stage` emits a `pipeline_stage` event via `emit_event` → VictoriaLogs. |
| P1 #3 per-episode cost SoT | **Done** — GI/KG cost folded into `EpisodeMetrics` at the metadata-gen seam; `episode_finished` no longer undercounts. |
| P1 #4 diarization run metrics | **Done** — `Metrics.record_diarization` + `diarization_*` in `finish()` / `metrics.json`. |
| P1 #5 capture_exception | **Done** — `sentry_init.capture_stage_exception` wired at transcription / diarization / GI / KG catches. |
| P2 #6 API OTEL | **Done** — `_init_api_otel()` in `server/app.py`. |
| P2 #7 json_logs dead path | **Done** — shipped `utils/json_logging.py` (`JSONFormatter`, stdlib, correlation-stamped). |
| P2 #8 `pipeline_stage` event type | **Done** — now emitted (folded into P1 #2). |
| P2 #9 INFO per-stage summary | **Done** — `log_metrics` emits a compact `Stage summary — …` INFO line. |
| P2 #10 GitOps automation | **Decision, not code** — `make obs-sync` / `push-grafana-dashboards.sh` remain manual by design; automating them is a CI-workflow change in the sibling infra repo and needs operator authorization (never auto-add CI/deploy). Flagged for the operator. |
| Grafana surface | **Done (this repo)** — `config/grafana/dashboards/common/grafana-dashboard-pipeline-stage.json` (per-stage cost, time, **method-version inventory = the reprocess key**, flagged-episode drill-down). Auto-discovered by `grafana_sync.py`. Backend datasource stand-up stays infra-repo/operator-owned. |

## 6. Suggested sequence
P0 (#1) → P1 (#2 manifest→emit_event, #3 cost SoT, #5 capture_exception) → P1 (#4 diarization
metrics) → P2 hygiene (#6–#10). Each is independently shippable and measurable on the next night's
run. The MCP `podcast_obs` control-plane review (agent-facing surface) is the natural next step after
#2, since it queries these same signals.
