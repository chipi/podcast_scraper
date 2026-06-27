# Enrichment Layer API (RFC-088)

The enrichment layer adds a fourth artefact tier on top of GIL/KG/bridge:
typed, opt-in enrichers (deterministic / embedding / ML / LLM) that emit
their own envelopes under `enrichments/` and surface through three
co-operating APIs:

1. **HTTP** routes — `/api/jobs/enrichment`, `/api/enrichment/*` —
   mounted by `create_app(..., enable_jobs_api=True)`.
2. **MCP** tools — `enrichment_*` — registered by the
   `podcast_obs` observability control plane (`podcast-obs serve`).
3. **JSONL** events — append-only `enrichments/run.jsonl` — read by all
   tools and human operators.

This page is the authoritative reference; the canonical implementation
plan lives in
[`docs/wip/RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md`](../wip/RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md).

## HTTP API

### Job submission + listing

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/api/jobs/enrichment` | Enqueue a corpus-enrichment run. **202** + `{ job_id, status, corpus_path }`. Body (all optional): `only` (list of enricher ids — restrict to these), `skip` (list — exclude these), `corpus_only` (bool — skip episode-scope enrichers). |
| GET | `/api/jobs` | Shared registry — lists pipeline AND enrichment jobs; filter client-side by `command_type == "corpus_enrichment"`. |
| POST | `/api/jobs/{job_id}/cancel` | Command-type-agnostic — works on enrichment jobs because the cancel walks the registry by id only. |

### Enrichment surface

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/api/enrichment/status` | Last status snapshot. Fresh corpus → `{ available: false, reason: "no status yet" }`. |
| GET | `/api/enrichment/health` | Per-enricher health map: `{ enrichers: { <id>: { consecutive_failures, auto_disabled, auto_disabled_reason, last_error, circuit_state, circuit_opened_at } } }`. Pass `?enricher_id=<id>` for a single record. |
| GET | `/api/enrichment/metrics` | Rollup metrics: `{ window: "24h", per_enricher: { <id>: { runs_total, runs_ok, runs_failed, runs_timeout, runs_quarantined, runs_cancelled, retries_total, avg_duration_s, total_cost_usd } } }`. `?window=` accepts `1h`/`24h`/`7d`. |
| GET | `/api/enrichment/run-summary` | Last completed run summary: `{ status, run_id, profile, started_at, finished_at, duration_ms, per_enricher: {…} }`. Fresh corpus → `{ available: false }`. |
| GET | `/api/enrichment/events` | JSONL tail. `{ events: [...], count }`. Filters: `?enricher_id=`, `?event_type=`, `?limit=` (default 50). |
| POST | `/api/enrichment/health/{enricher_id}/re-enable` | Clear `auto_disabled` + zero `consecutive_failures` after a transient outage. Body (optional): `{ reason }` — appended to the health audit trail. Returns the updated health record. |

All routes are gated by `app.state.jobs_api_enabled`. When the router
is mounted but the flag is `false`, GETs return **500** with
`detail: "...jobs_api..."` so a misconfigured deploy is loud.

### Pydantic models

See [`src/podcast_scraper/server/routes/enrichment.py`](https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/routes/enrichment.py):

- `EnrichmentJobRequest` — POST `/api/jobs/enrichment` body.
- `EnrichmentJobAccepted` — 202 envelope.
- `HealthReEnableRequest` — POST re-enable body.

## MCP tools

The observability control plane (`src/podcast_obs/`) exposes eight new
read/write tools backing the same HTTP surface. Configure with
`PODCAST_OBS_API_BASE` (or a `targets.<name>.api_base` in
`podcast-obs.yaml`).

| Tool | What it answers | Implementation |
| ---- | --------------- | -------------- |
| `enrichment_run_status(target?)` | Last enrichment status snapshot | GET `/api/enrichment/status` |
| `enrichment_recent_runs(target?, limit=10)` | Recent enrichment-only jobs, newest first | GET `/api/jobs` filtered to `command_type == "corpus_enrichment"` |
| `enrichment_health(target?, enricher_id?)` | Per-enricher health (or one enricher) | GET `/api/enrichment/health` |
| `enrichment_metrics(target?, window="24h")` | Rollup metrics over a window | GET `/api/enrichment/metrics` |
| `enrichment_recent_events(target?, enricher_id?, event_type?, limit=50)` | JSONL event tail | GET `/api/enrichment/events` |
| `enrichment_eval_history(target?, eval_root?, limit=10)` | Enrichment-tagged eval runs from local `data/eval/runs/` | On-disk scan; no remote endpoint by design — eval artefacts are frozen-once-written operator-side data |
| `enrichment_re_enable(enricher_id, target?, reason?)` | Clear `auto_disabled` + zero `consecutive_failures` | POST `/api/enrichment/health/{id}/re-enable` |
| `enrichment_cancel(job_id, target?)` | Cancel a running/queued enrichment job | POST `/api/jobs/{id}/cancel` (command-type-agnostic) |

Both `prod_summary` and `prod_correlate` were extended so the enrichment
layer appears alongside `health/version/runs/cost/...`:

- `prod_summary` now includes `enrichment_status`, `enrichment_health`,
  and a compact 5-event tail under `enrichment_events`.
- `prod_correlate(run_id)` joins `enrichment_events` into the run-scoped
  cross-layer view (one call → trace + cost + errors + logs +
  enrichment events for one run_id).

## JSONL events

Events are appended to `<corpus>/enrichments/run.jsonl`, one line per
event. All event types are stable strings prefixed `enrichment.`:

| `event_type` | When | Key fields |
| ------------ | ---- | ---------- |
| `enrichment.run.started` | Executor opens a run | `run_id`, `parent_run_id`, `profile`, `enricher_set` |
| `enrichment.run.completed` | Executor finishes (any outcome) | `run_id`, `duration_ms`, `per_enricher_totals` |
| `enrichment.run.skipped` | Pipeline-attached run skipped (core pipeline failed) | `run_id`, `reason` |
| `enrichment.enricher.started` | One enricher's `enrich()` invocation begins | `run_id`, `enricher_id`, `attempt`, `scope` |
| `enrichment.enricher.completed` | One enricher's invocation ends (any status) | `run_id`, `enricher_id`, `status`, `duration_ms`, `records_written`, `retries` |
| `enrichment.enricher.retry` | A retry was scheduled after a retryable failure | `run_id`, `enricher_id`, `attempt`, `backoff_s`, `reason`, `error_class` |
| `enrichment.enricher.circuit_opened` | Per-enricher circuit breaker tripped during this run | `run_id`, `enricher_id`, `consecutive_failures`, `opened_at`, `cooldown_until` |
| `enrichment.enricher.auto_disabled` | Cross-run health crossed `auto_disable_threshold` | `run_id`, `enricher_id`, `consecutive_failed_runs`, `reason` |
| `enrichment.enricher.cancelled` | Enricher cancelled mid-run (cancel_event) | `run_id`, `enricher_id`, `reason`, `partial_records_written` |
| `enrichment.run.cost_cap_exceeded` | Per-enricher or run-wide cap fired | `run_id`, `enricher_id?`, `cost_usd`, `cap_usd` |

Every event payload carries the `RunContext` correlation envelope —
`run_id`, `parent_run_id`, `enricher_id`, `enricher_version`, `tier`,
`attempt`, `job_id` — so an agent's `prod_correlate(run_id)` join works
across pipeline + enrichment + LLM signals.

## Status vocabulary

`EnricherResult.status` ∈ `{ok, failed, timeout, quarantined, cancelled, skipped}` —
defined in `podcast_scraper.enrichment.protocol`. Run-level status
aggregates per-enricher outcomes:

- any `cancelled` outcome → run status `cancelled`
- else any `failed` / `timeout` / `quarantined` → run status `failed`
- else `ok`

## Per-tier resilience policy

| Tier | `max_retries` | Backoff | Circuit threshold | Auto-disable threshold | Concurrency |
| ---- | ------------- | ------- | ----------------- | ---------------------- | ----------- |
| Deterministic | 0 | n/a | n/a (no retries) | 5 | 4 |
| Embedding | 3 | 1s, 2× | 5 | 3 | 2 |
| ML | 2 | 5s, 2× | 3 | 2 | 1 |
| LLM | 5 | 2s, 2× | 3 | 2 | 4 |

`max_backoff_s` caps the per-retry sleep at 30s (embedding), 60s (ML),
or 120s (LLM). Backoff has 10% jitter.

## Configuration

Operator config under `viewer_operator.yaml`:

```yaml
enrichment:
  enabled: true
  max_total_cost_usd_per_run: 5.00       # run-wide cost cap (USD)
  fail_on_run_cost_cap: true             # set cancel_event when cap fires
  enrichers:
    topic_cooccurrence:
      enabled: true
      max_cost_usd_per_run: 0.10
      expected_duration_s: 30
    nli_contradiction:
      enabled: true
      opt_in: false                      # LLM tier: double opt-in
```

JSON Schema draft 2020-12 validation:
[`config/schema/enrichment.schema.json`](https://github.com/chipi/podcast_scraper/blob/main/config/schema/enrichment.schema.json).
The CLI exits non-zero on invalid config.

## CLI

```bash
python -m podcast_scraper.enrichment.cli \
  --output-dir <corpus> \
  [--only topic_cooccurrence,temporal_velocity] \
  [--skip nli_contradiction] \
  [--corpus-only] \
  [--re-enable nli_contradiction --re-enable-reason "transient HF outage"] \
  [--config viewer_operator.yaml] \
  [--log-level INFO]
```

The viewer surfaces the same CLI as the `POST /api/jobs/enrichment`
handler (spawns `python -m podcast_scraper.enrichment.cli` in a
subprocess and tracks it through the shared jobs registry).

## References

- [RFC-088 Enrichment Layer Architecture](../rfc/RFC-088-enrichment-layer-architecture.md) — protocol spec (status: **Active**)
- [Implementation plan](../wip/RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md) — chunk-by-chunk decomposition
- [Chunk-1 lock audit](../wip/RFC-088-CHUNK1-LOCK-AUDIT.md) — locked decisions for the foundation
- Source: [`src/podcast_scraper/enrichment/`](https://github.com/chipi/podcast_scraper/tree/main/src/podcast_scraper/enrichment)
- HTTP routes: [`src/podcast_scraper/server/routes/enrichment.py`](https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/routes/enrichment.py)
- MCP tools: [`src/podcast_obs/sources/enrichment.py`](https://github.com/chipi/podcast_scraper/blob/main/src/podcast_obs/sources/enrichment.py)
