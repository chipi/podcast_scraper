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

### Enrichment-config surface (RFC-088 v2)

Editable enrichment config — read/write the operator-side
``enrichment:`` block via fine-grained REST, with a JSON-Schema
companion for UI form generation.

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/api/enrichment/config?path=<corpus>` | Resolved view of the enrichment block. Response: `{ corpus_path, profile, profile_block, operator_block, resolved_block }` — profile block + operator override + their deep-merge. Operators / the UI see what's inherited vs what's been customised. |
| PUT | `/api/enrichment/config?path=<corpus>` | Persist the operator-side block to `<corpus>/viewer_operator.yaml`. Body: `{ enrichment_block: {...} }`. Validates against the JSON Schema first (400 on validation error). Atomic write; preserves every unrelated top-level key (e.g. `profile`, `feeds`). Returns the fresh resolved view. |
| GET | `/api/enrichment/config/schema` | Full JSON Schema for the `enrichment:` block, composed from the base schema + each enricher's `manifest.config_schema` (under `enrichers.<id>.properties`) + each provider type's `params_schema` (oneOf under `enrichers.<id>.provider`). The viewer's Configuration → Enrichment editor reads this and renders all form fields data-driven. |
| GET | `/api/enrichment/provider-types` | Registered provider types grouped by protocol: `{ by_protocol: { EmbeddingProvider: [{name, protocol, description, params_schema}, ...], NliScorer: [...] } }`. UI populates per-row provider dropdowns from this. |

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
| `enrichment.enricher.stall_warning` | Enricher's wall-clock exceeded `expected_duration_s` without finishing | `run_id`, `enricher_id`, `last_heartbeat_at`, `expected_interval_s` |
| `enrichment.health.re_enabled` | Operator manually re-enabled an auto-disabled enricher | `enricher_id`, `operator_id`, `reset_counter`, `cleared_cooldown`, `reason` |

Cost-cap fires (per-enricher quarantine + run-wide skip) do not have a
dedicated event type in the current build — they surface through the
`enrichment.enricher.completed` event with `status="quarantined"` or
`status="skipped"` and a structured `reason` field. A dedicated
`enrichment.run.cost_cap_exceeded` event may land in a future chunk
once LLM-tier query enrichers are wired.

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
  enrichers:                             # Shape B: per-enricher block,
                                         # block-present = enabled
    topic_cooccurrence:
      max_cost_usd_per_run: 0.10
      expected_duration_s: 30
    temporal_velocity:
      alpha: 0.7                         # per-enricher knob
      window_months: 6
    topic_similarity:
      top_k: 10
      provider:                          # ML enrichers declare provider
        type: sentence_transformer_local
        model: all-MiniLM-L6-v2
    nli_contradiction:
      threshold: 0.5
      opt_in: false                      # LLM tier: double opt-in
      provider:
        type: deberta_local
    insight_density:
      enabled: false                     # explicit opt-out (preserves block)
```

Presence of a block is the enable; ``enabled: false`` opts out
without losing the configuration (Shape B, RFC-088 v2). Knob keys
match each enricher's ``manifest.config_schema`` properties — see
[Enrichment Layer Guide → Per-enricher reference](../guides/ENRICHMENT_LAYER_GUIDE.md#per-enricher-reference).

JSON Schema draft 2020-12 validation:
[`config/schema/enrichment.schema.json`](https://github.com/chipi/podcast_scraper/blob/main/config/schema/enrichment.schema.json).
The CLI exits non-zero on invalid config. The viewer's
Configuration → Enrichment editor reads
[GET `/api/enrichment/config/schema`](#enrichment-config-surface-rfc-088-v2)
(which composes per-enricher fragments at request time) for form
generation.

## CLI

```bash
python -m podcast_scraper.cli enrich \
  --output-dir <corpus> \
  [--profile cloud_balanced] \
  [--enrichers topic_cooccurrence,temporal_velocity]    # alias for --only \
  [--only topic_cooccurrence,temporal_velocity] \
  [--skip nli_contradiction] \
  [--no-enrichers]                                      # disable everything \
  [--opt-in <id,id>]                                    # for requires_opt_in enrichers \
  [--with-ml]                                           # wire ML enrichers from provider blocks \
  [--corpus-only] \
  [--re-enable nli_contradiction --re-enable-reason "transient HF outage"] \
  [--config viewer_operator.yaml] \
  [--log-level INFO]
```

Resolution order (RFC-088 v2):

1. ``--profile`` provides the BASE EnricherSet from
   ``enricher_set_for_profile()``.
2. ``--config <yaml>`` deep-merges the operator-side ``enrichment:``
   block on top (Shape B: per-enricher dict with implicit-enabled
   default — see [Enrichment Layer Guide → Configuration](../guides/ENRICHMENT_LAYER_GUIDE.md#configuration)).
3. ``--no-enrichers`` / ``--enrichers`` / ``--only`` / ``--skip`` /
   ``--opt-in`` layer on top.
4. ``--with-ml``: walks the resolved EnricherSet and registers any
   enricher whose manifest declares a ``provider_requirement``,
   using the provider type named in
   ``enrichers.<id>.provider.type``. Without this flag, ML enrichers
   are skipped with a hinted WARNING (see the
   [provider-type registry](../guides/ENRICHMENT_LAYER_GUIDE.md#provider-type-registry)).

The workflow auto-passes ``--with-ml`` to the spawned-from-pipeline
CLI when ANY enricher in the resolved YAML has a ``provider:`` block.
Deterministic-only profiles get a plain spawn so the spawn log stays
honest about what runs.

The viewer surfaces the same CLI as the `POST /api/jobs/enrichment`
handler (spawns `python -m podcast_scraper.cli enrich` in a
subprocess and tracks it through the shared jobs registry).

## References

- [RFC-088 Enrichment Layer Architecture](../rfc/RFC-088-enrichment-layer-architecture.md) — protocol spec (status: **Completed**, 2026-06-27)
- [Enrichment Layer Guide](../guides/ENRICHMENT_LAYER_GUIDE.md) — operator + developer companion (CLI / viewer / writing a new enricher)
- [Implementation plan](../wip/RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md) — chunk-by-chunk decomposition
- [Chunk-1 lock audit](../wip/RFC-088-CHUNK1-LOCK-AUDIT.md) — locked decisions for the foundation
- Source: [`src/podcast_scraper/enrichment/`](https://github.com/chipi/podcast_scraper/tree/main/src/podcast_scraper/enrichment)
- HTTP routes: [`src/podcast_scraper/server/routes/enrichment.py`](https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/routes/enrichment.py) + [`corpus_enrichments.py`](https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/routes/corpus_enrichments.py)
- MCP tools: [`src/podcast_obs/sources/enrichment.py`](https://github.com/chipi/podcast_scraper/blob/main/src/podcast_obs/sources/enrichment.py)
