# RFC-088 Enrichment Layer — implementation plan

Working doc for landing the enrichment layer as RFC-088 designed it. Real
end-to-end build, not a paperwork promotion. Shape mirrors how RFC-097
landed (chunked PRs, each independently mergeable, ~9 chunks total).

**Target outcome:** RFC-088 → Completed; PRD-026 and PRD-027 unblocked; the
"4th artifact tier" of derived signals exists on disk and is consumable
from API + viewer.

**Chunk-1 lock audit (2026-06-26):** the 20-finding internal-consistency
audit of this plan after 7 iterative revisions lives at
[`RFC-088-CHUNK1-LOCK-AUDIT.md`](./RFC-088-CHUNK1-LOCK-AUDIT.md). The
fixes are inlined into this plan (B1–B10 + I1–I4); the audit lists
the open operator decisions (O1–O4) that block chunk-1 start.

**Working branch:** `feat/enrichment-layer` (new branch off main after the
current `feat/rfc-paperwork-promotions-v3` lands).

**Hard constraints set 2026-06-26:**

- **No DGX dependency anywhere in the shipping path.** DGX is an optional
  add-on for operator-side experimentation; the codebase, CI, and every
  default-enabled enricher must run without it. The NLI piece in chunk 4
  is a local DeBERTa-v3-small on CPU, full stop — no DGX fallback, no
  DGX preferred path. If a future LLM-tier enricher wants DGX, that's
  scoped as opt-in operator infrastructure, not as the default.
- **Every "smart" enricher (embedding, ml, llm tier) ships with an
  eval harness in the same chunk that introduces it.** Deterministic
  enrichers are validated by unit + integration tests; smart enrichers
  additionally need a labeled eval set under `data/eval/enrichment/`
  with structured metrics (P/R/F1, MRR, nDCG@k, calibration error —
  whichever fits the task) and a `scripts/eval/score/enrichment_*.py`
  scoring script. **Autoresearch wiring is optional per enricher** —
  required when there's a tunable param worth sweeping (e.g.
  `nli_contradiction.threshold`, `topic_similarity.top_k`), skipped
  when the enricher is parameter-free.
- **All enrichers are opt-in via profile presets, never per-config
  toggling alone.** Per `[[feedback_profiles_are_source_of_truth]]`,
  the `config/profiles/*.yaml` registry decides which enrichers run in
  which environment. Each profile preset names its enricher set
  explicitly (`airgapped`: deterministic + topic_similarity;
  `airgapped_thin`: deterministic only; `cloud_thin`: deterministic +
  topic_similarity + nli_contradiction; etc.). A new chunk (7) wires
  the `EnricherSet` into the existing `ProfilePreset` machinery.
- **Resilience is a core architectural property of the enrichment
  layer, not a layer added later.** Every enricher run goes through
  retry-with-backoff + circuit-breaker + auto-disable, mirroring the
  pattern we already use for cloud LLM providers
  (`utils/retry.py`, `utils/llm_circuit_breaker.py`,
  `utils/retryable_errors.py`, `utils/provider_metrics.py`). The
  executor is `asyncio.gather`-based with per-tier concurrency caps
  and cooperative cancel. Per-enricher health (consecutive failures,
  circuit state, auto-disable status) persists across runs in
  `.viewer/enrichment_health.json`. Full design lives in
  §"Resilience model" below.
- **Mock-server / E2E strategy is baked into the design, not bolted
  on.** Each smart enricher consumes a small scorer protocol
  (`NliScorer`, `EmbeddingProvider`, future `LLMScorer`). Real
  implementations call DeBERTa / LanceDB / providers; scenario-driven
  mocks under `tests/fixtures/enrichment/` exercise every resilience
  path (flaky-then-recovers, OOM, timeout, stall-forever, drift)
  without needing real models in CI. Same pattern as RFC-054 mock
  clients. Three-tier test pyramid: unit (stub) / integration
  (scenarios) / E2E stack-test (operator flow via `/api/jobs/enrichment`).

---

## Resilience model (core, not bolt-on)

Every enricher run goes through the same resilience pipeline. The
executor is `asyncio.gather`-based; failures are handled inside the
executor, never raised into the calling pipeline; per-enricher state
persists across runs.

### Per-enricher state machine

`READY → RUNNING → OK | (RETRY → RUNNING)* → QUARANTINED → (cooldown) → READY → AUTO-DISABLED`

```text
       enabled (config + profile preset)
                  │
                  ▼
              READY ─────────► RUNNING ───── ok ─────► OK
                ▲                 │
                │                 ├── transient failure ─► RETRY (backoff)
                │                 │
                │                 └── non-retryable / max retries ─┐
                │                                                   │
                │             cooldown elapsed                      ▼
                │ ◄─────────────────────────────────────────  QUARANTINED
                │                                                   │
                │                            N consecutive failures │
                │                                                   ▼
                │ ◄── operator re-enable ── (CLI / viewer) ── AUTO-DISABLED
```

### Per-tier policy (overridable per enricher via config)

| Tier | Max retries | Initial backoff | Backoff factor | Max backoff | Circuit threshold (consecutive fails in a run) | Auto-disable threshold (consecutive failed runs) |
| --- | --- | --- | --- | --- | --- | --- |
| `deterministic` | 0 | — | — | — | n/a | 5 |
| `embedding` | 3 | 1s | 2.0 | 30s | 5 | 3 |
| `ml` | 2 | 5s | 2.0 | 60s | 3 | 2 |
| `llm` | 5 | 2s | 2.0 | 120s | 3 | 2 |

Backoff: `min(initial * factor ** attempt + jitter, max_backoff)` —
same shape as `utils/retry.py`. Reuses `utils/llm_circuit_breaker.py`
state machine. Failure taxonomy reuses `utils/retryable_errors.py`
classification.

### Failure taxonomy

| Error | Class | Action |
| --- | --- | --- |
| `EnvelopeShapeError` (output fails schema validation) | non-retryable | fail fast; status `failed`; counts toward auto-disable |
| `BadInputError` (missing required artifact) | non-retryable | fail fast; status `failed: missing_input` |
| `DependencyAccessError` (LanceDB lock, file IO) | retryable | retry with backoff |
| `OutOfMemoryError` | retryable once | single retry after GC pause |
| `ModelLoadError` | retryable once | retry; if still fails → circuit opens immediately |
| `ScorerTimeoutError` (per-pair / per-batch) | retryable | retry; partial result preserved |
| `RunTimeoutError` (whole-enricher hard timeout) | non-retryable | cancel; status `timeout` |
| any other `Exception` | non-retryable (caught by safety net) | log + Sentry breadcrumb; status `failed: unexpected` |

Enrichers never raise into the executor — every run produces an
`EnricherResult(status: ok | failed | timeout | quarantined |
cancelled | skipped)`. The executor handles the result; the enricher
owns its retry loop.

### Async execution + concurrency caps

Both phases (episode-scope, corpus-scope) run as `asyncio.gather`
over enrichers with per-tier concurrency caps (config + profile
override):

| Tier | Default concurrency | Why |
| --- | --- | --- |
| `deterministic` | 4 | Pure CPU; bounded by core count |
| `embedding` | 2 | LanceDB connection contention |
| `ml` | 1 (sequential) | Single model in memory; avoid OOM |
| `llm` | matches provider rate limit | Same logic as existing LLM provider concurrency |

Per-enricher hard timeout via `asyncio.wait_for`. Defaults: 60s
episode-scope, 600s corpus-scope; overridable per enricher.

### Cooperative cancel + stall detection

- Cancel propagates from `/api/jobs/{job_id}/cancel` via the existing
  `cancel_requested` flag pattern. Executor passes an `asyncio.Event`
  into each enricher; long-running enrichers check between batches and
  bail cleanly. Partial output preserved with `status: cancelled`.
- Per-enricher heartbeat — enrichers emit a progress event every N
  pairs. No event for > 2× expected interval triggers a `WARNING`;
  configurable escalation to cancel.
- Pid-alive reconcile — existing
  `jobs.py:reconcile_jobs_inplace` (renamed per O4) catches dead
  enrichment-job processes the same way it catches dead pipeline-job
  processes.

### Cost-cap enforcement (per REPLAN-O7)

The cost-cap plumbing ships in chunk 1 next to the rest of the
resilience model. Chunks 4 + 5 just populate manifest fields; no
enforcement code lands there.

- **Per-enricher quarantine** — after each enricher run, the executor
  compares `EnrichmentMetrics.cost_usd` (set by the existing
  `record_provider_call_cost` chain when an enricher's scorer calls
  a provider) against `manifest.max_cost_usd_per_run`. Exceeded →
  status `quarantined`, reason `cost_cap_exceeded`. Other enrichers
  in the run continue normally.
- **Run-wide hard stop** — a `total_cost_usd` counter is incremented
  across every enricher's `EnrichmentMetrics.cost_usd`. When the total
  exceeds `enrichment.max_total_cost_usd_per_run`, the executor aborts
  the enrichment pass — subsequent enrichers in the queue marked
  `skipped`, reason `run_cost_cap_exceeded`. The whole run's `status`
  flips to `failed`; `exit_code` is non-zero unless
  `enrichment.fail_on_run_cost_cap: false` is set.
- Test cases in `test_resilience_scenarios.py`:
  - `cost_cap_per_enricher_quarantines_offender` — mocked scorer
    returns `cost_usd: 0.10` per call; enricher hits its
    `max_cost_usd_per_run = 0.50` budget after 5 calls; quarantine
    fires; other enrichers in the run continue.
  - `cost_cap_run_wide_aborts_pass` — total accumulated cost across
    enrichers exceeds the run-wide cap; subsequent enrichers in the
    queue marked `skipped`; run status `failed`.

### Health persistence

`.viewer/enrichment_health.json` (JSONL-style append-and-rotate same
as the jobs registry):

```json
{
  "topic_similarity": {
    "consecutive_failures": 0,
    "last_run_at": "2026-06-26T14:23:11Z",
    "last_run_id": "job_a1b2c3",
    "last_status": "ok",
    "auto_disabled": false,
    "circuit_state": "closed"
  },
  "nli_contradiction": {
    "consecutive_failures": 3,
    "last_run_at": "2026-06-26T15:01:42Z",
    "last_run_id": "job_d4e5f6",
    "last_status": "failed",
    "auto_disabled": true,
    "auto_disabled_at": "2026-06-26T15:01:42Z",
    "auto_disabled_reason": "3 consecutive failed runs (circuit opened twice)",
    "circuit_state": "open",
    "circuit_opened_at": "2026-06-26T15:01:42Z",
    "cooldown_until": "2026-06-26T16:01:42Z"
  }
}
```

Manual recovery: `podcast enrich --re-enable <id>` or
`POST /api/enrichment/health/<id>/re-enable`. Resets counter; clears
`auto_disabled`; optionally clears circuit cooldown.

### Observability

- Each run emits `enrichments/run_summary.json` with per-enricher
  outcome, duration, retry count, circuit state, model_id /
  model_version.
- Sentry breadcrumb on circuit-open + auto-disable events.
- Metrics surface via the existing pipeline metrics path
  (`workflow/metrics.py`): `enrichment.{id}.{ok|failed|retries|
  duration_ms}`.

---

## Mock-server / scorer-stub strategy (E2E + resilience coverage)

Same pattern as `tests/fixtures/mock_server/{gemini,mistral}_mock_client.py`
(RFC-054), applied at the **scorer protocol** layer because most
enrichment backends are in-process. HTTP mock server only needed
later for cloud LLM-tier query enrichers.

### Scorer protocols (injectable, swappable)

```text
src/podcast_scraper/enrichment/scorers/
  __init__.py
  protocol.py
    NliScorer            # async def score(premise, hypothesis) -> NliScore
    EmbeddingProvider    # async def topic_vector(topic_id) -> vec | None
    LLMScorer            # future, for LLM query enrichers
  nli_deberta.py         # real CPU DeBERTa implementation (chunk 4)
  lancedb_embeddings.py  # real LanceDB-backed implementation (chunk 3)
```

### Scenario-driven mocks under `tests/fixtures/enrichment/`

```text
tests/fixtures/enrichment/
  __init__.py
  mock_scorers.py
    ScenarioNliScorer(scenario=...)
    ScenarioEmbeddingProvider(scenario=...)
```

Scenarios per scorer:

| Scenario | Behaviour | Tests what |
| --- | --- | --- |
| `happy_high_contradiction` | Always returns 0.9 | happy path |
| `happy_no_contradiction` | Always returns 0.1 | happy path negative |
| `flaky_then_recovers` | Fails 2x with retryable error, then succeeds | retry + backoff path |
| `always_oom` | Raises `OutOfMemoryError` | OOM retry + circuit open |
| `always_dependency_lock` | Raises `DependencyAccessError` | retry + eventual circuit |
| `slow` | `await asyncio.sleep(timeout * 2)` | hard timeout |
| `stalls_forever` | Never returns | heartbeat watchdog escalation |
| `drift_calibration` | Returns mis-calibrated scores | eval drift detection |
| `intermittent_30pct` | Random fail-rate 30% | flake-tolerance |

### Three-tier test pyramid (mirrors ADR-095 viewer pyramid)

| Tier | Where | What it covers |
| --- | --- | --- |
| **Unit** | `tests/unit/enrichment/` | each enricher with stub scorer + synthetic fixture; deterministic outputs |
| **Integration** | `tests/integration/enrichment/` | scenario-driven scorers; assert retry counts, circuit state transitions, auto-disable persistence, cooperative cancel, hard timeout, per-tier concurrency caps, run_summary shape |
| **E2E stack-test** | `tests/stack-test/stack-enrichment-resilience.spec.ts` | enrichment job through `POST /api/jobs/enrichment` with scenario-injected scorers; viewer Operator tab observes status / quarantine / auto-disable / re-enable cycle |

### E2E test scenarios (concrete `stack-enrichment-resilience.spec.ts`)

1. **Happy path** — submit enrichment job; assert completion + output files + run_summary shape.
2. **Retry recovery** — submit with `flaky_then_recovers` scorer; assert success with `retry_count > 0`.
3. **Circuit open** — submit with `always_oom`; watch enricher quarantine; assert next run skips it during cooldown.
4. **Auto-disable across runs** — chain 3 failed runs; assert `enrichment_health.json` flips `auto_disabled: true`; viewer Operator tab shows the disabled badge.
5. **Manual re-enable** — operator clicks re-enable in viewer; assert health file resets, next run includes the enricher.
6. **Cancel** — submit long-running enrichment; send cancel via `/api/jobs/{id}/cancel`; assert graceful shutdown + partial output preserved.
7. **Stall watchdog** — `stalls_forever` scorer; assert heartbeat WARNING then escalation to cancel.

Scenario injection happens via a test-only `SCORER_OVERRIDE` env var
(same shape as the existing `PODCAST_MOCK_PROVIDER` overrides), so
production code paths never see the stub registry.

---

## Metrics, observability, analytics (core, mirrors the pipeline)

Same primitives the pipeline already uses — extended, not forked. The
goal: anything the operator can ask about a pipeline run, they can ask
about an enrichment run with identical surfaces (Metrics class, JSONL
events, run summary, live status, Sentry, Langfuse, Grafana,
dashboard JSON, viewer Operator tab).

### Inputs we extend (not reinvent)

| Pipeline surface | What we add for enrichment |
| --- | --- |
| `workflow/metrics.py` `Metrics` class | New `enrichment: dict[str, EnrichmentMetrics]` field; one record per enricher per run with `runs_total / runs_{ok,failed,timeout,quarantined,cancelled,skipped} / duration_seconds / retries_total / circuit_transitions / output_records_total / scorer_calls_total / scorer_failures_total / tokens_in / tokens_out / cost_usd`. Same JSON / CSV / dashboard export pipeline picks it up. |
| `workflow/jsonl_emitter.py` `JSONLEmitter` | New event types: `enrichment.run.{started,completed}`, `enrichment.enricher.{started,retry,completed,circuit_opened,auto_disabled,cancelled}`, `enrichment.health.re_enabled`. Each line atomically appended; same tail-friendly file (`run.jsonl`). |
| `workflow/run_summary.py` `create_run_summary` | Includes `enrichment_summary` when enrichment ran (per-enricher outcome / duration / retries / model_id / records_written / last_error / circuit state). |
| `workflow/run_manifest.py` | Manifest records which enrichers participated + their versions. |
| `workflow/cost_monitoring.py` | Reused as-is for future LLM-tier enrichers via the existing `record_provider_call_cost` hook — no enrichment-specific code needed for cost. |
| `monitor/status.py` `maybe_update_pipeline_status` | New `maybe_update_enrichment_status` writes `.viewer/enrichment_status.json` (mirrors `.pipeline_status.json` shape). Live monitor + viewer Operator tab subscribe. |
| `utils/sentry_init.py` | Reused; new breadcrumbs fired from `enrichment/resilience.py` on circuit-open + auto-disable + stall-escalation events. |
| `utils/langfuse_tracing.py` | Reused — when a future LLM-tier query enricher calls a provider, the existing tracing wraps it. No enrichment-specific Langfuse code. |
| `scripts/dashboard/generate_metrics.py` `detect_deviations()` | Extended to flag enrichment regressions: `ok_rate < 0.9` for 3+ runs, `p95_duration > 2× baseline`, `auto_disabled: true`. Surfaces as nightly `alerts[]`. |

### Per-enricher metric record (`EnrichmentMetrics` dataclass)

```python
# workflow/metrics.py (new dataclass alongside Metrics)
@dataclass
class EnrichmentMetrics:
    enricher_id: str
    enricher_version: str
    scope: str                 # "episode" | "corpus"
    tier: str                  # "deterministic" | "embedding" | "ml" | "llm"
    runs_total: int = 0
    runs_ok: int = 0
    runs_failed: int = 0
    runs_timeout: int = 0
    runs_quarantined: int = 0
    runs_cancelled: int = 0
    runs_skipped: int = 0
    duration_seconds: float = 0.0
    retries_total: int = 0
    circuit_transitions: dict[str, int] = field(default_factory=dict)
    output_records_total: int = 0
    scorer_calls_total: int = 0
    scorer_failures_total: dict[str, int] = field(default_factory=dict)
    tokens_in: int = 0           # ml / llm tier; deterministic stays 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    last_run_status: str = ""
    last_run_started_at: str = ""
    last_run_finished_at: str = ""
    model_id: str = ""
    model_version: str = ""
    error_samples: list[dict] = field(default_factory=list)  # most recent 5
```

Exposed via the existing `Metrics.to_json()` / `.to_csv()` /
`workflow.metrics.log_metrics()` paths — operators see enrichment
metrics anywhere they see pipeline metrics today.

### JSONL event vocabulary

| Event | Payload fields |
| --- | --- |
| `enrichment.run.started` | `run_id, profile, enricher_set, started_at` |
| `enrichment.enricher.started` | `enricher_id, scope, tier, attempt, started_at` |
| `enrichment.enricher.retry` | `enricher_id, attempt, backoff_s, reason, error_class` |
| `enrichment.enricher.completed` | `enricher_id, status, duration_ms, records_written, retries, finished_at` |
| `enrichment.enricher.circuit_opened` | `enricher_id, consecutive_failures, cooldown_until, opened_at` |
| `enrichment.enricher.auto_disabled` | `enricher_id, consecutive_failed_runs, reason, disabled_at` |
| `enrichment.enricher.cancelled` | `enricher_id, reason, partial_records_written` |
| `enrichment.enricher.stall_warning` | `enricher_id, last_heartbeat_at, expected_interval_s` |
| `enrichment.health.re_enabled` | `enricher_id, operator_id, reset_counter, cleared_cooldown` |
| `enrichment.run.completed` | `run_id, duration_ms, per_enricher_totals` |

Same `JSONLEmitter` instance, same `run.jsonl` file as the pipeline.

### Live status (`.viewer/enrichment_status.json`)

```json
{
  "schema_version": "1",
  "run_id": "job_d4e5f6",
  "started_at": "2026-06-26T15:01:42Z",
  "profile": "cloud_thin",
  "current_enricher": {
    "enricher_id": "nli_contradiction",
    "scope": "corpus",
    "tier": "ml",
    "attempt": 1,
    "progress": {
      "items_done": 247,
      "items_total": 1000,
      "eta_seconds": 142
    },
    "last_heartbeat_at": "2026-06-26T15:03:14Z"
  },
  "queue": ["nli_contradiction", "topic_similarity"],
  "completed": [
    {"enricher_id": "topic_cooccurrence_corpus", "status": "ok",
     "duration_ms": 412}
  ]
}
```

Atomic-write same as `pipeline_status.json`. Viewer Operator tab polls
it. CLI shows a Rich progress bar when run interactively (TTY) — same
pattern as `--monitor`.

### Sentry (errors + perf)

Three new event categories fired from `enrichment/resilience.py`:

1. `enrichment.circuit_opened` — breadcrumb + tag (`enricher_id`,
   `tier`, `consecutive_failures`). Not an issue alert; just a
   breadcrumb. Operators define their own threshold alert rules in
   Sentry (e.g. "more than 5 circuit-opens per hour" → page).
2. `enrichment.auto_disabled` — Sentry message at `warning` level
   (one-off event, not an exception). Operators alert on the message
   string in Sentry.
3. `enrichment.stall_escalation` — Sentry message at `error` level.
   Indicates an enricher had to be cancelled by the watchdog; likely a
   real bug or a corrupt input.

Unhandled exceptions inside the safety net continue to fire normal
Sentry issues via the existing pipeline init.

### Langfuse (LLM tracing)

No enrichment-specific Langfuse code. When a future LLM-tier query
enricher (chunk 5 / follow-on RFC) calls a provider, the existing
provider-level Langfuse tracing wraps the call. The enrichment layer
just passes `enricher_id` into the provider context so the Langfuse
trace carries an `enricher_id` tag.

### Grafana (operator-side dashboards)

`docs/guides/OBSERVABILITY_EXTENSIONS.md` §"Operator alerting — Sentry
+ Grafana" gains an "Enricher panels" subsection in chunk 8:

| Panel | Source | What it answers |
| --- | --- | --- |
| Enricher OK rate (24h, per id) | `metrics/history.jsonl` enrichment fields | Is any enricher unhealthy? |
| Enricher latency p50/p95 (per tier) | `metrics/history.jsonl` | Which enricher is slowest? |
| Circuit-state heatmap | `enrichment_health.json` (polled) | Which enrichers are quarantined? |
| Auto-disable events (timeline) | Sentry message search | When did we lose an enricher? |
| Eval drift (F1 / MRR over time) | `data/eval/enrichment/<id>/history.jsonl` | Is autoresearch silver drifting? |
| Cost per LLM enricher (future) | Langfuse | Spend by query enricher |

### Dashboard JSON (CI surface)

`metrics/latest.json` gains an `enrichment` block:

```json
{
  "enrichment": {
    "per_enricher": {
      "topic_similarity": {
        "ok_rate": 0.98, "p50_ms": 312, "p95_ms": 845,
        "retry_rate": 0.02, "circuit_opens_24h": 0,
        "auto_disabled": false, "model_id": "..."
      },
      "nli_contradiction": {
        "ok_rate": 0.85, "p50_ms": 4200, "p95_ms": 8900,
        "retry_rate": 0.15, "circuit_opens_24h": 1,
        "auto_disabled": false, "tokens_in": 14820,
        "tokens_out": 0, "cost_usd": 0.0
      }
    },
    "totals": {
      "runs_24h": 47, "ok_rate_24h": 0.94, "alerts_24h": 1
    }
  }
}
```

PR comments / nightly summary already render `metrics/latest.json` —
enrichment metrics show up there for free.

### Eval-time metrics

Each eval run (chunks 2–4) produces
`data/eval/enrichment/<id>/runs/<run_id>/metrics.json` with the gold-
scoring metrics (P / R / F1 / MRR / Brier / per-tier latency).
Aggregated into `data/eval/enrichment/<id>/history.jsonl` for trend
tracking. Autoresearch ratchet (chunks 3 & 4) reads these to detect
regressions vs champion baseline.

### Viewer Operator tab — Enrichment panel (chunk 6 lands the UI)

The chunk-6 viewer integration includes a new Enrichment panel on the
Operator tab. Surfaces:

- Per-enricher last-run status (✓ ok / ✗ failed / ⏳ running /
  💤 disabled).
- Health badges: auto-disabled flag + circuit state.
- Latency: p50 / p95 (last 24h).
- Output: records written, file size.
- Eval signal: latest F1 / MRR if eval data available.
- Drill-down: history graph (last 10 runs), recent errors, manual
  re-enable button.

Wires through the same `/api/jobs` + new `/api/enrichment/health` +
new `/api/enrichment/metrics` routes.

### Files added or extended

| File | Change |
| --- | --- |
| `workflow/metrics.py` | New `EnrichmentMetrics` dataclass + `Metrics.enrichment` field + record-methods |
| `workflow/jsonl_emitter.py` | New event-type constants |
| `workflow/run_summary.py` | `create_run_summary` reads `Metrics.enrichment` |
| `workflow/run_manifest.py` | Includes enricher set + versions |
| `monitor/status.py` | New `maybe_update_enrichment_status` + reader |
| `utils/sentry_init.py` | New `emit_enrichment_breadcrumb` helper |
| `enrichment/resilience.py` | Wires Sentry + JSONL emitter calls into state-machine transitions |
| `enrichment/executor.py` | Wires Metrics record-methods around every enricher call |
| `enrichment/health.py` | Persistence emits `enrichment.health.re_enabled` event |
| `scripts/dashboard/generate_metrics.py` | `detect_deviations()` extended |
| `docs/guides/OBSERVABILITY_EXTENSIONS.md` | New "Enricher panels" subsection (chunk 8) |
| `web/gi-kg-viewer/src/components/operator/EnrichmentPanel.vue` | Operator-tab UI (chunk 6) |

---

## Single-surface o11y: MCP server extension + correlation IDs

`src/podcast_obs/mcp_server.py` is the operator's single observability
surface today — one MCP server with 11 tools that join Sentry +
Langfuse + Loki + Grafana + GitHub + the prod API. An agent asks
`prod_correlate(run_id)` and gets every signal joined into one story.
Enrichment has to plug into the same surface, not become a second
one. Otherwise we have two stories and the agent has to stitch.

Two halves to this: the **MCP tool extensions** (what an agent can
ask) and the **correlation ID propagation** (how the answer stitches).

### Correlation ID propagation (the consistent story)

Every enrichment-emitted signal carries the `run_id` it belongs to,
plus enricher-scope identifiers. The schema:

| Field | Source | Set when |
| --- | --- | --- |
| `run_id` | UUID | the outermost job (pipeline run if attached; enrichment job if standalone) |
| `parent_run_id` | UUID or null | pipeline `run_id` for attached enrichment; null for standalone |
| `enricher_id` | str | every per-enricher event |
| `enricher_version` | str | every per-enricher event (helps diff before / after a version bump) |
| `tier` | str | every per-enricher event |
| `attempt` | int | every per-enricher event during retry loop |
| `job_id` | UUID | the jobs-API record id (== `run_id` for standalone enrichment) |

Where each correlation field lands:

| Surface | Carries | Format |
| --- | --- | --- |
| Pipeline run | `run_id` | already in `run_manifest.json`; unchanged |
| Enrichment job (pipeline-attached) | `run_id = pipeline.run_id`, `parent_run_id = null` | inherited at executor start |
| Enrichment job (standalone CLI / API) | `run_id = job_id`, `parent_run_id = null` | jobs API generates |
| `enrichments/run_summary.json` | `run_id`, `parent_run_id`, per-enricher records | written at end of run |
| `run.jsonl` events | every event line carries `run_id`, `enricher_id` (when applicable), `attempt`, `tier` | `JSONLEmitter` injects |
| Sentry breadcrumb / message | tags: `run_id`, `enricher_id`, `tier`, `enricher_version` | set by `enrichment/observability.py` |
| Langfuse trace (LLM-tier enrichers) | metadata: `run_id`, `parent_run_id`, `enricher_id`, `enricher_version`, `tier` | passed into provider context |
| Loki log lines | structured extra: `run_id`, `enricher_id`, `tier` | enrichment uses existing structured-logging adapter |
| Pipeline `Metrics` | `run_id` already present; enrichment per-id records share it | same emitter |
| Live status `.viewer/enrichment_status.json` | `run_id` at top level | written by `enrichment/status.py` |
| `enrichment_health.json` | `last_run_id` per enricher | persisted across runs |

Result: a single `run_id` lookup in Sentry / Langfuse / Loki / Grafana
returns the full enrichment chain alongside the pipeline chain.
`prod_correlate(run_id)` does the join.

### New `src/podcast_obs/sources/enrichment.py` source

Mirrors the existing `sources/{sentry,langfuse,loki,grafana,prod_api,github}.py`
modules. Reads:

- `/api/jobs?command_type=corpus_enrichment` for recent runs.
- `/api/enrichment/health` for per-enricher state.
- `/api/enrichment/metrics` for windowed counter snapshots.
- `/api/enrichment/status` (live, current run) — reads `.viewer/enrichment_status.json`.
- `/api/enrichment/events` for JSONL event slices.
- `/api/enrichment/eval-history?enricher_id=<id>` for eval ratchet data.

All routes ship in chunk 1 alongside the source module. Source uses
`httpx` only (no SDKs) — same constraint as the rest of `podcast_obs`.

### New MCP tools (added to `mcp_server.py`)

| Tool | Returns |
| --- | --- |
| `enrichment_run_status(target, run_id=None)` | Current live status: which enricher is running, items_done/total, eta_seconds, heartbeat freshness. If `run_id` omitted, returns the currently-active run; if specified, returns that run's final state from `enrichment/run_summary.json`. |
| `enrichment_recent_runs(target, limit=10, status=None)` | Recent enrichment jobs from `/api/jobs?command_type=corpus_enrichment`. Optional `status` filter. Each row carries `run_id`, `started_at`, `duration_ms`, per-enricher OK / failed counts. |
| `enrichment_health(target, enricher_id=None)` | Per-enricher health: `consecutive_failures`, `circuit_state`, `auto_disabled`, `cooldown_until`. Omitting `enricher_id` returns all enrichers; specifying narrows to one. |
| `enrichment_metrics(target, enricher_id=None, window="24h")` | Per-enricher windowed metrics: `ok_rate`, `p50_ms`, `p95_ms`, `retry_rate`, `circuit_opens`, `tokens_in`, `tokens_out`, `cost_usd`. Filter by enricher_id; window per Grafana convention (`1h`/`24h`/`7d`). |
| `enrichment_recent_events(target, enricher_id=None, event_type=None, window="1h", limit=50)` | JSONL event slice. Filter by enricher_id (e.g. `nli_contradiction`), event_type (e.g. `enrichment.enricher.retry`), time window. Useful for "what failed and why in the last hour." |
| `enrichment_eval_history(target, enricher_id, metric="f1", limit=20)` | Eval ratchet history for a smart enricher. Returns `[{run_id, started_at, champion_threshold, dev_f1, held_out_f1, drift_from_baseline}]`. Wired in chunks 3 / 4 once the eval JSONL ships. |
| `enrichment_re_enable(target, enricher_id, reason)` | Operator-side reactivation. Calls `POST /api/enrichment/health/<id>/re-enable` with `reason` (audit trail in the health file). Returns the reset health record. |
| `enrichment_cancel(target, job_id, reason)` | Cancel a running enrichment job. Proxies `POST /api/jobs/{job_id}/cancel` with `reason` recorded in the cancel envelope. Useful when an agent detects a runaway pattern via `enrichment_recent_events` (per chunk-1 lock audit §O2). |

### Extensions to existing tools

| Tool | Extension |
| --- | --- |
| `prod_correlate(run_id, target)` | Joins enrichment per-enricher outcomes + retry counts + circuit transitions + LLM-tier provider calls under that `run_id`. The cross-layer view stays one tool. |
| `prod_summary(target)` | New `enrichment` subsection: counts of active / quarantined / auto-disabled enrichers; OK rate over last 24h; alerts pending. |
| `prod_recent_logs(target, service=...)` | Accepts `service="enrichment.<id>"` to filter Loki for enrichment events by enricher_id. |
| `prod_recent_errors(target, ...)` | Sentry filter by `tag:enricher_id` supported via existing `contains` field. |
| `prod_recent_traces(target, ...)` | Langfuse traces tagged with `enricher_id` carry the tag through; no schema change. |
| `prod_cost_today(target)` | LLM-tier enricher costs roll up via Loki `llm_cost` events — already include `run_id`; gain `enricher_id` field. |

### Test surface for the MCP additions (in chunk 1)

| Test | Asserts |
| --- | --- |
| `tests/unit/podcast_obs/sources/test_enrichment_source.py` | All 6 routes return the expected shape; httpx error handling matches the existing source pattern. |
| `tests/unit/podcast_obs/test_mcp_server_enrichment_tools.py` | All 7 new tools registered; each closes over config correctly; arguments validate. |
| `tests/integration/podcast_obs/test_mcp_correlate_with_enrichment.py` | Run a pipeline + enrichment fixture; query `prod_correlate(run_id)`; assert pipeline + enrichment + LLM signals all join under one run_id. |
| `tests/integration/podcast_obs/test_mcp_summary_includes_enrichment.py` | `prod_summary` includes the enrichment subsection with the configured enrichers' health. |

### Failure handling

Same as the existing MCP tools: when an upstream surface is
unconfigured (no Sentry DSN, no Langfuse keys, no Grafana endpoint),
the tool returns `{"available": false, "reason": "..."}` rather than
raising. Mirrors `_run` in `mcp_server.py:38`. Agents handle the
absence gracefully.

### Files added or extended

| File | Change |
| --- | --- |
| `src/podcast_obs/sources/enrichment.py` | New source module; httpx-only reads against the new `/api/enrichment/*` routes |
| `src/podcast_obs/sources/__init__.py` | Re-export the new module |
| `src/podcast_obs/mcp_server.py` | **8** new tool closures (incl. `enrichment_cancel` per O2) + extensions to `prod_correlate` / `prod_summary` |
| `src/podcast_obs/aggregate.py` | `_correlate` adds the enrichment join; `_summary` adds the enrichment subsection |
| `src/podcast_obs/config.py` | New `EnrichmentEndpoint` config field (per target) |
| `src/podcast_obs/cli.py` | Optional `--tool enrichment_*` test shortcuts |
| `src/podcast_scraper/enrichment/correlation.py` | Surface-specific correlation extras for the o11y emit paths: `correlation_extras_for_logging(ctx)` (Loki structured fields), `sentry_tags_for_context(ctx)` (string-coerced Sentry tags), `langfuse_metadata_for_context(ctx)` (Langfuse trace metadata), `jsonl_event_extras(ctx)` (per-line JSONL extras). The `RunContext` dataclass itself lives in `enrichment/protocol.py` next to the rest of the foundation types. The Sentry tag-setter helper is `utils/sentry_init.set_correlation_tags(tags)`; consumed by `enrichment/observability.stamp_sentry_correlation(ctx)`. |
| `src/podcast_scraper/enrichment/executor.py` | Establishes `RunContext` once at run start; passes through to every enricher + scorer call |
| `src/podcast_scraper/utils/sentry_init.py` | New helper `set_correlation_tags(run_id, enricher_id, tier)` (no-op when SDK absent) |
| `src/podcast_scraper/utils/langfuse_tracing.py` | New helper `with_enrichment_metadata(run_id, enricher_id, ...)` decorator (no-op when SDK absent) |
| `docs/api/ENRICHMENT_LAYER_API.md` | New "Correlation IDs + MCP surface" section |
| `docs/guides/OBSERVABILITY_EXTENSIONS.md` | Mention the new MCP tools in the §"Operator alerting" section (chunk 8 polish) |
| `src/podcast_scraper/server/jobs.py` (renamed from `pipeline_jobs.py` per O4) | Adds `COMMAND_ENRICHMENT = "corpus_enrichment"`; `build_enrichment_argv` / `enqueue_enrichment_job` / `spawn_enrichment_subprocess` helpers; module docstring updated to "Job queue, subprocess spawn, and registry updates — serves pipeline and enrichment job kinds via `command_type`." 4 internal src imports updated (`jobs_log_path.py`, `scheduler.py`, `routes/jobs.py`, `pipeline_run_prometheus.py`); 4 test files renamed for symmetry (`test_pipeline_jobs_*.py` → `test_jobs_*.py`); doc references find-and-replaced. |
| `src/podcast_scraper/enrichment/protocol.py` `EnricherManifest` | Adds `max_cost_usd_per_run: float \| None = None` and `expected_duration_s: int \| None = None` fields per O1 cost-cap decision. Chunk 4 (NLI) and chunk 5 (LLM query enrichers) consume these; chunk 1 wires the JSON Schema to accept them. |
| `config/schema/enrichment.schema.json` | Top-level `enrichment.max_total_cost_usd_per_run: float \| null` and `enrichment.fail_on_run_cost_cap: bool = true` fields per O1 cost-cap decision. Chunk 1 ships the schema; chunk 5 wires enforcement. |

### Per-chunk MCP extension responsibility

The MCP server is the single surface — each chunk owns the part of
the surface its data populates. Locks in the consistent-story
guarantee chunk-by-chunk:

| Chunk | MCP additions |
| --- | --- |
| **1 (foundation)** | `enrichment_run_status`, `enrichment_recent_runs`, `enrichment_health`, `enrichment_metrics`, `enrichment_recent_events`, `enrichment_re_enable`, **`enrichment_cancel` (per O2)**, extended `prod_correlate` (joins enrichment by run_id even with no enrichers registered — surface is wired through the no-op path), extended `prod_summary` (enrichment subsection with empty counts) |
| **2 (deterministic)** | each enricher's metrics flow through `enrichment_metrics` automatically; no MCP code change |
| **3 (topic_similarity)** | `enrichment_eval_history` for `topic_similarity` |
| **4 (nli_contradiction)** | `enrichment_eval_history` for `nli_contradiction`; Langfuse pass-through tested via `prod_recent_traces` |
| **5 (QueryEnricher)** | query enrichers carry `run_id` (per-request, derived from the search request id); join in `prod_correlate` |
| **6 (server + viewer)** | viewer Operator-tab Enrichment panel consumes the same `/api/enrichment/*` routes that the MCP source module uses — single read surface for both |
| **7 (profile presets)** | none |
| **8 (docs)** | `OBSERVABILITY_EXTENSIONS.md` + `ENRICHMENT_LAYER_GUIDE.md` document the MCP tools |

---

## Architectural prerequisite — resolve the RFC-097 ↔ RFC-088 divergence

Before any code, decide and document this once:

RFC-097 chunk 9 (`src/podcast_scraper/kg/topic_clustering.py`) writes
**concept-Topic nodes + `RELATED_TO` edges into the KG artifacts
directly**. That contradicts RFC-088 Key Decision #1 ("Enrichers never
modify core artifacts").

Two coherent paths:

- **(A) Co-exist, different audiences.** The KG-direct path stays as the
  airgapped/typed-connectivity story (the v3 KG ontology, deterministic
  CI, what an LLM grounds against). RFC-088's `topic_similarity` enricher
  writes the same signal as derived data under `enrichments/` for surface
  consumers that want raw scores/ranks. Two outputs, two purposes, both
  honest.
- **(B) Retract RFC-097 chunk 9's KG mutation.** Move concept-Topic +
  `RELATED_TO` out of the KG artifacts into the enrichment layer; KG
  v3 ontology drops `RELATED_TO`. Cleaner architecturally but it
  breaks every consumer that already reads `RELATED_TO` from KG (viewer
  graph, ABOUT∩MENTIONS_PERSON joins, NER post-pass downstream) and
  invalidates corpora generated since #1094.

**Recommendation: (A).** Document the divergence in a new ADR
("Enrichment Layer Boundary vs KG-Direct Connectivity"). This becomes
chunk 0 of the implementation.

---

## Chunk 0 — Architectural decision (1 small PR, doc-only)

**Deliverables:**

- New ADR: `docs/adr/ADR-104-enrichment-layer-boundary-vs-kg-direct.md`.
  Captures: KG-direct path is for airgapped + LLM-grounding; enrichment
  layer is for derived/scored/rankable signals; both can name the same
  underlying signal; reconciliation rule (KG is canonical for connectivity,
  enrichment is canonical for scores/ranks); Decision #1 from RFC-088
  amended to "Enrichers never modify core artifacts produced by core
  pipeline stages — the RFC-097 chunk 9 KG mutation is part of core, not
  enrichment".
- RFC-088 body amended: cross-ref the new ADR, scope clarification at the
  top.
- RFC-097 body amended: cross-ref the new ADR, scope clarification at the
  top.

**Why first:** every subsequent chunk needs this boundary settled. No code
risk; pure design alignment.

**Acceptance:** `make docs` strict green, operator review on the
divergence framing.

---

## Chunk 1 — Foundation: protocol + registry + executor + paths (1 medium PR)

**Module:** `src/podcast_scraper/enrichment/`

**Files:**

```text
src/podcast_scraper/enrichment/
  __init__.py
  protocol.py          # Enricher (PEP 544, @runtime_checkable) —
                       # async def enrich(*, bundle, corpus_root,
                       # all_bundles, config, ctx: RunContext) ->
                       # EnricherResult. Amends RFC-088 §Protocol
                       # (was: sync def enrich(...) -> dict). Sync
                       # deterministic bodies use @sync_enricher
                       # decorator (runs in default thread executor).
                       # EnricherManifest, EnricherScope, EnricherTier,
                       # EpisodeArtifactBundle. EnricherResult frozen
                       # dataclass (status: ok|failed|timeout|
                       # quarantined|cancelled|skipped, data, error,
                       # error_class, retry_count, circuit_state,
                       # duration_ms, records_written). Minimal
                       # EnricherSet stub (enabled_enrichers,
                       # per_enricher_config, opt_in_flags) — chunk 7
                       # wires profile presets to it.
  registry.py          # register(), get(), list_enabled(); double-opt-in
                       # gate for LLM tier; tests use direct registry
                       # construction via pytest fixtures (not profile
                       # presets) per chunk-1 lock audit §B7.
  envelope.py          # Output validation: derived: true required,
                       # computed_at, enricher_id, enricher_version,
                       # schema_version, status, error?, data
  paths.py             # _episode_enrichment_path(),
                       # _corpus_enrichment_path(); multi-feed-aware;
                       # invariants from RFC-088 §Directory Structure
  executor.py          # two-phase pass: phase 1 = EPISODE enrichers
                       # over all bundles; phase 2 = CORPUS enrichers;
                       # never raises; one WARNING per failed enricher
  cli.py               # `podcast enrich --output-dir ... [--corpus-only]
                       # [--only <id>,<id>] [--skip <id>,<id>]
                       # [--re-enable <id>]`
  resilience.py        # per-enricher state machine, retry-with-backoff
                       # (reuses utils/retry.py), circuit-breaker (reuses
                       # utils/llm_circuit_breaker.py), failure taxonomy
                       # (reuses utils/retryable_errors.py + enrichment-
                       # specific additions), heartbeat watchdog
  health.py            # .viewer/enrichment_health.json persistence,
                       # auto-disable / re-enable, cross-run state
  metrics.py           # EnrichmentMetrics record helpers; wraps every
                       # enricher call to update workflow/metrics.py
                       # Metrics.enrichment field; emits JSONL events
                       # via the existing JSONLEmitter
  status.py            # .viewer/enrichment_status.json live-status
                       # writer (mirrors monitor/status.py shape);
                       # heartbeat + progress publishing for viewer
                       # Operator tab + CLI Rich progress bar
  observability.py     # Sentry breadcrumb + message helpers
                       # (circuit_opened / auto_disabled /
                       # stall_escalation); reuses utils/sentry_init.py;
                       # Langfuse context tag passthrough for future
                       # LLM enrichers (enricher_id tag)
  correlation.py       # RunContext dataclass (run_id, parent_run_id,
                       # enricher_id, enricher_version, tier, attempt,
                       # job_id); correlation_extras_for_logging();
                       # set_sentry_correlation_tags();
                       # langfuse_trace_metadata() — threads
                       # correlation IDs through every emit path so
                       # prod_correlate(run_id) stitches a single story
  scorers/
    __init__.py
    protocol.py        # NliScorer, EmbeddingProvider, LLMScorer
                       # protocols (injectable)
```

Real scorer implementations land with their consuming enricher:
`scorers/lancedb_embeddings.py` with chunk 3 `topic_similarity`,
`scorers/nli_deberta.py` with chunk 4 `nli_contradiction`. The
protocols + injection plumbing ship in chunk 1 so the framework is
complete.

**Operator entry points — three layers, dev/ops separation explicit:**

| Layer | Entry point | Audience | Notes |
| --- | --- | --- | --- |
| **CLI** (one-shot, scriptable) | `podcast enrich --output-dir <corpus> [--only id,id] [--skip id,id] [--corpus-only]` | Operators on a terminal, automation scripts, dev iteration | The canonical operator-facing entry point. Standalone runs without re-running the core pipeline. |
| **Jobs API** (server-managed, viewer-driven) | `POST /api/jobs/enrichment` with body `{"only": [...], "skip": [...], "corpus_only": false}` | Viewer Operator tab; remote / cron-driven runs | New job type alongside the existing `"full_incremental_pipeline"`. Same JSONL registry, same stale/PID-orphan reconcile, same cancel + status flow as RFC-077 pipeline jobs. |
| **Pipeline-attached** (free, automatic) | Runs as Step N+1 of `workflow/orchestration.py` | Every full pipeline run | Always-on; covered by the core pipeline integration in this chunk. |

**No new `make enrich` target.** Per operator: Makefile targets are for
dev convenience, not ops. Dev iteration uses
`.venv/bin/python -m podcast_scraper enrich --output-dir <test-corpus>`
directly — same as how we don't have a `make scrape` for one-off
pipeline runs. The two production operator paths are the CLI and the
jobs API.

**Jobs-API integration (the new job type):**

- `src/podcast_scraper/server/jobs.py` (renamed from `pipeline_jobs.py`
  in sub-commit 1) gains a new
  `COMMAND_ENRICHMENT = "corpus_enrichment"` constant alongside the
  existing `COMMAND_FULL`. The `command_type` field on the JSONL job
  record is already parametric — `_new_job_record` just takes whatever
  is passed in.
- New helpers next to the existing pipeline ones:
  - `build_enrichment_argv(corpus_root, operator_yaml, *, only=None, skip=None, corpus_only=False) -> list[str]`
  - `enqueue_enrichment_job(corpus_root, operator_yaml, *, only=None, skip=None, corpus_only=False) -> dict`
  - `spawn_enrichment_subprocess(corpus_root, operator_yaml, job_record, *, only=None, skip=None, corpus_only=False)`
- The promote-queued / cancel / stale-reconcile / pid-alive logic is
  agnostic to `command_type` — works for the new kind without code
  changes there.
- New server route `POST /api/jobs/enrichment` (mirrors the existing
  pipeline-job route shape). Reuses the `app_operator_guard` auth
  layer.
- Viewer Operator tab gains a "Run enrichment" action that submits
  this job type — same status / progress / cancel UI as pipeline
  jobs.

**Config schema additions** (per chunk-1 lock audit §B8 + §I1 + §I4):

- New `config/schema/enrichment.schema.json` formal JSON Schema for the `enrichment:` block; validation runs at `enrichment/cli.py` startup with a clear error on malformed config.
- New server routes module `src/podcast_scraper/server/routes/enrichment.py` carrying all 6 new routes (`POST /api/jobs/enrichment` + `POST /api/enrichment/health/<id>/re-enable` + 5 read routes); mounted in `server/app.py`.
- `src/podcast_scraper/server/pipeline_jobs.py` keeps its filename; module docstring notes it serves multiple `command_type` values. Rename to `jobs.py` is a deferred cleanup.

New top-level `enrichment:` block in operator YAML / corpus config:

```yaml
enrichment:
  enabled: true            # master switch; default true
  enrichers:
    topic_cooccurrence:
      enabled: true        # deterministic, on by default
    nli_contradiction:
      enabled: false
      opt_in: false        # LLM/ML tiers require both
```

**Pipeline wiring:** add an enrichment-pass step to
`workflow/orchestration.py` that runs after all core artifacts are
written. No-op when `enrichment.enabled: false` or no enrichers
registered. Pure addition — does not alter core stage signatures.

**Pipeline-attached failure semantics** (per chunk-1 lock audit §B4):
the enrichment step runs **only when the core pipeline completes
successfully** (`run.status == "ok"` in `run_summary`). Partial-
failure pipelines emit `enrichment.run.skipped { reason:
"core_pipeline_failed" }` JSONL event and skip the enrichment step.
Operators can still run enrichment standalone via CLI / jobs API on a
partial corpus — that's a deliberate operator choice.

**`.viewer/` directory creation** (per chunk-1 lock audit §B6):
`enrichment/health.py` and `enrichment/status.py` both do
`Path.mkdir(parents=True, exist_ok=True)` on the `.viewer/` parent
before the first write. Standalone runs against corpora that lack a
`.viewer/` directory work without the operator pre-creating it.

**Cancel + concurrency interaction** (per chunk-1 lock audit §B10):
the executor creates ONE `asyncio.Event` per run; every enricher
receives a reference via `ctx.cancel_event`. Cancel sets the event
once; every parallel enricher checks it between batches and bails.
Test: `test_executor.py::test_cancel_during_parallel_run_bails_all_workers`.

**`runs_skipped` flow path** (per chunk-1 lock audit §B5):
`EnrichmentMetrics.runs_skipped` increments when an enricher is
**configured + enabled by the active profile preset** but skipped this
run for one of: circuit open + cooldown active; auto-disabled;
`--skip <id>` CLI flag; `enabled: false` in operator YAML override;
query-time enricher whose precomputed-file dependency is missing.
Profile-preset says off → enricher isn't registered at all → no
`EnrichmentMetrics` record (distinct state).

**Tests** (`tests/unit/enrichment/`):

- `test_protocol.py` — `EnricherManifest` completeness, runtime_checkable
  conformance, scope/tier enum coverage.
- `test_registry.py` — register/lookup, double-opt-in gate (WARNING +
  skip when `requires_opt_in=True` and `opt_in` missing), missing
  enricher id → KeyError.
- `test_envelope.py` — `derived: true` enforced; failed status passes
  through; schema_version required.
- `test_paths.py` — single-feed vs multi-feed layout, episode vs
  corpus scope path resolution.
- `test_executor.py` — two-phase ordering, no-op when nothing registered,
  per-enricher failure isolation, byte-identical core artifacts before
  and after.
- `tests/integration/enrichment/test_enrichment_pass.py` — full pass
  against a 3-episode fixture; asserts directory layout, `derived: true`
  on every output, core artifacts unchanged.
- `tests/unit/server/test_enrichment_jobs.py` — new job-type coverage:
  `enqueue_enrichment_job` produces the right `command_type`, queue/
  promote/cancel/reconcile work identically to pipeline jobs.
- `tests/integration/server/test_enrichment_job_route.py` — `POST
  /api/jobs/enrichment` round-trip: enqueue → poll status → completion
  → log surfacing. Reuses the existing pipeline-job route fixtures.
- `tests/unit/enrichment/test_resilience.py` — per-tier retry policy,
  backoff calculation, circuit-breaker state transitions, failure
  taxonomy classification, heartbeat watchdog behaviour.
- `tests/unit/enrichment/test_health.py` — `enrichment_health.json`
  shape, atomic write, recovery from corrupt file, auto-disable
  threshold, manual re-enable.
- `tests/unit/enrichment/test_config_schema.py` — `enrichment.schema.json`
  validates the operator YAML `enrichment:` block; malformed config
  rejected with clear error; valid config round-trips through
  `enrichment/cli.py` startup (per chunk-1 lock audit §B8).
- `tests/integration/enrichment/test_resilience_scenarios.py` —
  scenario-driven scorer mocks exercise the full retry-then-recover,
  circuit-open, OOM, timeout, stall, intermittent-30%, drift paths
  end-to-end through the executor + **enricher-side synthetic
  failure cases** (per chunk-1 lock audit §B9):
  - `enricher_emits_malformed_envelope` — envelope fails schema
    validation → `EnvelopeShapeError` (non-retryable, auto-disable
    candidate).
  - `enricher_missing_required_input` — bundle missing `bridge_path`
    → `BadInputError` (non-retryable, status `failed: missing_input`).
  - `enricher_raises_unexpected` — bare `RuntimeError` from enricher
    body → caught by safety net, logged + Sentry breadcrumb, status
    `failed: unexpected`.
  Scorer scenarios exercise the retry + circuit-breaker path;
  enricher-side cases exercise the executor's safety-net path. Both
  kinds needed. **One of the most important test files in chunk 1.**
- `tests/unit/enrichment/test_metrics.py` — `EnrichmentMetrics`
  dataclass round-trips through JSON / CSV; all run_status values
  produce the right counter increments; cost / token accounting for
  the ml + llm tiers; `Metrics.to_json()` carries enrichment fields;
  **`error_samples` capped at 5 via `__post_init__` (per chunk-1
  lock audit §I2)** — older samples popped on push.
- `tests/unit/enrichment/test_status.py` — `.viewer/enrichment_status.json`
  atomic write, heartbeat publishing, progress payload validation,
  reader round-trip.
- `tests/unit/enrichment/test_observability.py` — Sentry breadcrumb
  fired on circuit-open / auto-disable / stall-escalation; Langfuse
  context tag carries `enricher_id` when wrapped around a provider
  call; o11y extensions are no-op when SDKs absent.
- `tests/unit/workflow/test_jsonl_emitter_enrichment_events.py` —
  every new enrichment event type emits a parseable JSONL line with
  the documented payload fields.
- `tests/integration/dashboard/test_enrichment_in_metrics_latest.py`
  — `scripts/dashboard/generate_metrics.py` writes the new
  `enrichment` block in `metrics/latest.json` and
  `detect_deviations()` flags the expected regressions
  (ok_rate < 0.9, p95 > 2× baseline, auto_disabled).
- `tests/stack-test/stack-enrichment-resilience.spec.ts` — 7 E2E
  resilience scenarios (happy / retry-recovery / circuit-open /
  auto-disable / re-enable / cancel / stall-watchdog) driven through
  `POST /api/jobs/enrichment` with `SCORER_OVERRIDE` env-var injection.
- `tests/stack-test/stack-enrichment-live-monitor.spec.ts` — live
  status flow: long-running scenario emits heartbeats, viewer
  Operator tab reads `enrichment_status.json` and renders the
  current-enricher progress + ETA; status switches to `idle` on
  completion.
- `tests/unit/enrichment/test_correlation.py` — `RunContext`
  construction; pipeline-attached vs standalone parent_run_id
  semantics; correlation extras propagate through every emit point
  (logging extras, Sentry tags, Langfuse metadata, JSONL event
  fields).
- `tests/unit/podcast_obs/sources/test_enrichment_source.py` — 6
  new routes return expected shape; httpx errors match the existing
  source pattern.
- `tests/unit/podcast_obs/test_mcp_server_enrichment_tools.py` — 7
  new MCP tools registered; argument validation; "available: false"
  graceful-degrade when surfaces unconfigured.
- `tests/integration/podcast_obs/test_mcp_correlate_with_enrichment.py`
  — run pipeline + enrichment fixture; query
  `prod_correlate(run_id)`; assert pipeline + enrichment + LLM
  signals all join under one run_id (the consistent-story
  guarantee).
- `tests/integration/podcast_obs/test_mcp_summary_includes_enrichment.py`
  — `prod_summary` includes the enrichment subsection.
- `tests/unit/server/test_enrichment_read_routes.py` — new server
  routes `/api/enrichment/health`, `/api/enrichment/metrics`,
  `/api/enrichment/status`, `/api/enrichment/events`,
  `/api/enrichment/eval-history` return the expected shape on a
  populated corpus + sensible 404 when empty.

**Mock-scorer fixtures** (`tests/fixtures/enrichment/mock_scorers.py`):
`ScenarioNliScorer` + `ScenarioEmbeddingProvider` with the 9 scenarios
listed in §"Mock-server / scorer-stub strategy" above. Ship in
chunk 1 — every later chunk consumes them.

**Docs:** `docs/api/ENRICHMENT_LAYER_API.md` — output envelope schema,
filename conventions, discovery rules, opt-in semantics, **CLI usage
+ jobs-API endpoint shape + viewer-tab integration note + resilience
model + health-file shape + manual recovery procedures**.

**Acceptance:** ci-fast green; integration tests pass (including all 9
scenario-driven resilience cases + metrics/o11y/status round-trips +
the `prod_correlate(run_id)` cross-layer join test);
stack-test green (7 resilience scenarios + live-monitor scenario);
`make docs` strict green; no enrichers shipped yet but the framework
is fully exercised by the "no-op" path through CLI, jobs API, and
pipeline-attached entry points + the full resilience pipeline through
the scorer-stub scenarios. **Every operator observability surface
(Metrics class, JSONL events, run_summary, live status,
enrichment_status.json, Sentry breadcrumbs, dashboard metrics, MCP
tools) is populated even with no real enricher running — the
framework owns the surfaces, enrichers just write through them.**
**`prod_correlate(run_id)` returns a consistent enrichment-included
story even with zero enrichers configured (empty enrichment block in
the join, not a 500).**

**Est size:** ~2250–3050 LOC code + ~2350–3050 LOC tests + ~600 LOC
mock-scorer fixtures + ~600 LOC E2E spec. ~3 reviewer days.

Chunk-1 lock audit (2026-06-26) added ~50 LOC code + ~150 LOC tests
on top of the prior estimate for: explicit `EnricherResult` dataclass,
minimal `EnricherSet` stub in chunk 1, `config/schema/enrichment.schema.json`,
`server/routes/enrichment.py` module, enricher-side failure
scenarios (B9), cancel-shared-event test (B10), `.viewer/` directory
creation (B6), pipeline-attached failure path (B4), `runs_skipped`
flow tests (B5). See `docs/wip/RFC-088-CHUNK1-LOCK-AUDIT.md`.

---

## Chunk 2 — Deterministic enrichers (1 medium PR)

**Module:** `src/podcast_scraper/enrichment/builtin/`

**Enrichers** (all `tier=DETERMINISTIC`, default `enabled: true`):

1. **`topic_cooccurrence`** (episode scope) — wraps the existing
   `kg/corpus.py:topic_cooccurrence` count function into the enricher
   envelope; reads `*.bridge.json`, writes
   `metadata/enrichments/{stem}.topic_cooccurrence.json`.
2. **`topic_cooccurrence_corpus`** (corpus scope) — aggregates Chunk 2.1
   outputs into a single `enrichments/topic_cooccurrence_corpus.json`
   with cross-corpus pair counts, ranked.
3. **`temporal_velocity`** (corpus scope) — monthly topic mention counts
   over a 12-month window from `kg + bridge`, with a 3-period EWMA
   trend signal per topic. Writes `enrichments/temporal_velocity.json`.
4. **`grounding_rate`** (corpus scope) — for each Person, % of their
   Insights that are `grounded: true` across the corpus. Writes
   `enrichments/grounding_rate.json`.
5. **`guest_coappearance`** (corpus scope) — pairs of Persons appearing
   in the same episode; ranked by episode count. Writes
   `enrichments/guest_coappearance.json`.
6. **`insight_density`** (episode scope) — Insight count per segment
   (early/mid/late, 1/3 splits of episode duration). Writes
   `metadata/enrichments/{stem}.insight_density.json`.

**Tests:** one unit test file per enricher with synthetic 2–3 episode
fixtures asserting numerics. One integration test runs all six against
the eval `curated_5feeds_smoke_v1` corpus and asserts shape + status.

**Eval harness (no autoresearch — these are parameter-free):**

- `data/eval/enrichment/deterministic/gold/` — hand-authored expected
  outputs for the 6 enrichers over a 3–5 episode synthetic corpus.
- `scripts/eval/score/enrichment_deterministic.py` — diffs current
  outputs against gold; emits per-enricher accuracy table + any
  drift. Direct Python entry point: `python scripts/eval/score/enrichment_deterministic.py`.
- The chunk-2 test suite wraps this script as a unit test so the
  gold-fixture exact-match runs in CI on every PR. No Make target
  needed — per REPLAN-O6, this matches the existing `scripts/eval/score/`
  convention (39 scoring scripts; none have Make wrappers).
- Acceptance: exact match against gold on the synthetic corpus.

**Acceptance:** every enricher idempotent, deterministic, < 5s for the
100-episode reference corpus; CI runs them by default (the fixture is
real-corpus-tiny, no LLM, no network); eval harness exact-match passes.

**Est size:** ~600–900 LOC code + ~700–1000 LOC tests + ~200 LOC eval.

---

## Chunk 3 — Embedding tier (1 medium PR)

**Enricher:** `topic_similarity` (corpus scope, `tier=EMBEDDING`,
`enabled: false` by default — opt-in via config).

**Implementation:** reuses the LanceDB hybrid index built by RFC-090.
Reads topic embeddings from the existing index (no re-embedding); writes
pair-wise cosine similarity matrix (top-K per topic, K=20 default) to
`enrichments/topic_similarity.json`. Cross-references the RFC-097
concept-Topic ids when present so downstream consumers can join.

**Tests:** unit test with stub embeddings; integration test against a
LanceDB-indexed corpus with a small embedder. Cost-of-validation: real
sentence-transformers must run in integration only (already a dependency).

**Eval harness:**

- `data/eval/enrichment/topic_similarity/gold/` — operator-curated
  labels of "these two topics are related" pairs over the prod corpus
  (start with ~50 pairs, grow over time). Per-pair fields: `topic_a`,
  `topic_b`, `expected_related: bool`, `notes`.
- `scripts/eval/score/enrichment_topic_similarity.py` — at every
  threshold candidate, computes precision / recall / F1 against the
  labels + MRR@10 of the top-K neighbours per topic. Direct Python
  entry point: `python scripts/eval/score/enrichment_topic_similarity.py
  --threshold 0.7 --top-k 20`. NOT in CI (needs the real embedder +
  indexed corpus).
- Per REPLAN-O6 — no `make eval-enrichment-*` wrapper for the scoring
  script; matches the existing `scripts/eval/score/` 39-script
  convention.

**Autoresearch wiring (required for this enricher):** the two tunable
parameters — `similarity_threshold` (default 0.7) and `top_k` (default
20) — are wired into the existing autoresearch v2 framework as a small
sweep: dev/held-out split on the curated pairs, F1 as the ratchet
signal. Same shape as RFC-073's Track A loop. Adds:

- `autoresearch/enrichment_topic_similarity/eval/score.py` program
  entry point (same shape as `autoresearch/bundled_prompt_tuning/eval/score.py`).
- New `make autoresearch-enrichment-topic-similarity [CONFIG=…]
  [REFERENCE=…]` Makefile target (same shape as existing
  `make autoresearch-score` / `make autoresearch-score-bundled`).
- Champion params land in the default `topic_similarity.yaml` config
  block on accept.

**ADR-104 anchor:** explicitly documents that `topic_similarity` (this
enricher) and RFC-097 chunk 9 `RELATED_TO` (KG-direct) can coexist —
this enricher is the rankable/scored variant, KG is the
typed-connectivity variant.

**Est size:** ~300–500 LOC code + ~400–600 LOC tests + ~400 LOC eval +
~300 LOC autoresearch wiring.

---

## Chunk 4 — ML tier: `nli_contradiction` (1 medium PR)

**Enricher:** `nli_contradiction` (corpus scope, `tier=ML`,
`enabled: false` by default — opt-in via config).

**Implementation:**

- Local NLI model via `[ml]` extra. **Confirmed model:**
  `cross-encoder/nli-deberta-v3-small` (DeBERTa-v3-small fine-tuned on
  MNLI; ~80MB, runs on CPU at ~50–200ms/pair). **CPU-only — no DGX
  fallback, no DGX preferred path.** Added to `pyproject.toml` `[ml]`
  extra and to the preload manifest.
- For each topic, take all Insights from different Persons mentioning
  that topic (via `MENTIONS_PERSON ∩ ABOUT`), score every cross-Person
  pair with NLI (`contradiction` probability), keep pairs with score ≥
  configurable threshold (default 0.5).
- Output: `enrichments/nli_contradiction.json` — list of `{topic_id,
  person_a_id, person_b_id, insight_a_id, insight_b_id,
  contradiction_score, model_id, model_version}`.

**CI hygiene** (per `[[feedback_no_llm_in_ci]]`): NLI model must NOT run
in CI; integration test uses a stub `NliScorer` protocol that returns
fixed scores per pair. Real-model test gated by environment marker, runs
only on operator demand. **Crucial — the enricher infrastructure must be
testable deterministically without the model.**

**Eval harness (this is the biggest one):**

- `data/eval/enrichment/nli_contradiction/gold/` — operator-curated
  contradiction labels over the prod corpus. Start ~100 cross-Person
  Insight pairs on shared topics, each labelled
  `{contradiction, neutral, entailment}` plus a `confidence` field.
  Grow as the corpus grows. JSONL for diffability.
- `scripts/eval/score/enrichment_nli_contradiction.py` — at every
  threshold candidate, computes:
  - precision / recall / F1 against the contradiction class
  - Brier score (calibration of probability estimates)
  - error analysis: false positives + false negatives with insight text
  Direct Python entry point: `python scripts/eval/score/enrichment_nli_contradiction.py
  --model nli-deberta-v3-small --threshold 0.5`. NOT in CI.
- Splits the labels dev/held-out per `[[feedback_silver_judge_vendor_bias]]`
  — though there's no judge here, the dev/held-out hygiene applies
  whenever the loop is closed via autoresearch.
- Per REPLAN-O6 — no `make eval-enrichment-*` wrapper; matches the
  existing 39-script convention.

**Autoresearch wiring (required for this enricher):** two tunable
dimensions worth a sweep:
1. `threshold` — anywhere from 0.3 to 0.8 in 0.05 steps; ratchet on
   dev-set F1, validate on held-out.
2. `model_variant` — `nli-deberta-v3-small` (default) vs
   `nli-deberta-v3-base` vs `nli-deberta-v3-large` — useful if F1 on
   dev is below acceptable. Each is a one-line config change; the eval
   loop spawns the contender against the current champion.

Adds:

- `autoresearch/enrichment_nli_contradiction/eval/score.py` program
  entry point (same shape as `autoresearch/bundled_prompt_tuning/eval/score.py`).
- New `make autoresearch-enrichment-nli-contradiction [CONFIG=…]
  [REFERENCE=…]` Makefile target (same shape as existing
  `make autoresearch-score`).
- Champion params land in the default `nli_contradiction.yaml` config
  block on accept.

**Edge type:** does NOT introduce a `CONTRADICTS` edge type in KG v2.0
(per ADR-104 boundary — derived data lives in enrichments). A future v3
KG decision can promote validated contradiction pairs to typed edges if
warranted; not in scope here.

**Est size:** ~400–600 LOC code + ~600–800 LOC tests + ~500 LOC eval +
~600 LOC autoresearch wiring + ~100-row gold-label JSONL.

---

## Chunk 5 — `QueryEnricher` protocol (Phase 4) (1 medium PR)

Unblocks PRD-027 Enriched Search.

**Module additions:**

```text
src/podcast_scraper/enrichment/
  query_protocol.py    # QueryEnricher protocol — runs at request time;
                       # signature: enrich_query_result(*, query, results,
                       # config) -> decorated results
  query_registry.py    # parallel registry for query enrichers;
                       # double-opt-in for LLM tier preserved
```

**Concrete query enricher(s):**

- `query_topic_relatedness` (deterministic) — decorates each search hit
  with the precomputed `topic_similarity` ranks from chunk 3 output.
  Trivial, demonstrates the protocol, ships enabled by default.
- (LLM-tier `query_synthesis` deferred to a follow-up RFC per RFC-088
  §Phase 4 — not in this chunk.)

**Server integration:** new `enrich_results: bool` parameter on
`/api/search`. When true, runs registered query enrichers in order over
the response. Wired into `src/podcast_scraper/server/routes/search.py`.

**Tests:** unit tests for the protocol + registry; integration test
against the search route asserts decoration round-trip.

**Est size:** ~400–600 LOC code + ~400–600 LOC tests.

---

## Chunk 6 — Server + viewer consumption (1 medium PR)

Cross-references PRD-026 + PRD-027 explicitly.

**Server (`src/podcast_scraper/server/routes/`):**

- New `/api/corpus/enrichments/{enricher_id}` route — serves a corpus-
  scope enrichment file by id (404 with `{ available: false }` when the
  file is absent; mirrors the RFC-075 `topic-clusters` route shape).
- New `/api/corpus/episode/{episode_id}/enrichments/{enricher_id}` route
  — episode-scope variant.
- Catalog row (`CatalogEpisodeRow`) gains `enrichments: list[str]` —
  ids of enrichers that have output for this episode.

**Viewer (`web/gi-kg-viewer/src/`):**

- `corpusEnrichmentsApi.ts` — fetch helpers + tests.
- Topic Entity View tab consumes `topic_cooccurrence_corpus` +
  `temporal_velocity` (renders RELATED_TO chips + trend sparkline).
- Person Profile rail gains `grounding_rate` badge + `guest_coappearance`
  rail section.

**Tests:** vitest for the new viewer APIs + composables; Playwright stack
test exercises Topic Entity View consumption (gated as the existing
stack-test job).

**Est size:** ~700–1000 LOC code (server + viewer split roughly evenly)
+ ~800–1100 LOC tests.

---

## Chunk 7 — Profile-preset wiring (1 medium PR)

Per `[[feedback_profiles_are_source_of_truth]]`, the
`config/profiles/*.yaml` registry is the authoritative answer to "which
enrichers run in this environment." Every enricher ships off by default;
profile presets are the only thing that turns the deterministic six on,
and the way operators opt into the smart tiers.

**Module additions:**

- New `EnricherSet` dataclass alongside the existing `StageOption` /
  `ProfilePreset` machinery (`src/podcast_scraper/profiles/`).
  Carries: `enabled_enrichers: list[str]`, per-enricher config overrides,
  opt-in flags for the `llm` tier.
- `ProfilePreset` gains an `enrichments: EnricherSet` field. Legacy
  presets default to a sentinel "all-off" set so older profiles keep
  current behaviour.

**Preset assignments (the "what runs where" matrix):**

| Profile | Deterministic 6 | topic_similarity | nli_contradiction | Query enrichers |
|---|---|---|---|---|
| `test_default` | off | off | off | off |
| `airgapped_thin` | **on** | off | off | off |
| `airgapped` | **on** | **on** | off | **on** (deterministic only) |
| `cloud_thin` | **on** | **on** | **on** | **on** |
| `dev` | **on** | **on** | off | **on** |
| `prod` | **on** | **on** | **on** | **on** |

Concrete rationale:
- `test_default`: nothing — tests opt in per-test when they exercise an
  enricher.
- `airgapped_thin`: cheap deterministic only; this profile is the
  always-runs-on-CI variant.
- `airgapped`: adds `topic_similarity` because the LanceDB index exists
  in airgapped runs; adds the deterministic query enricher.
- `cloud_thin` / `prod`: full stack including NLI.
- `dev`: matches `airgapped` (we don't want devs accidentally running
  the NLI model on every save; they can override locally).

**Override mechanism:** CLI `--enrichers <id>,<id>` and
`--no-enrichers <id>,<id>` flags layer over the profile choice (matches
existing `--feed` / `--profile` override semantics).

**Tests:**

- `tests/unit/profiles/test_enricher_set.py` — preset assignments match
  the matrix above; CLI override semantics; sentinel default for legacy
  presets.
- Drift test: every profile preset must declare an `enrichments` field
  (cannot silently inherit), enforced by existing profile-validator.
- Pipeline integration test per profile: run the enrichment pass with
  each preset, assert only the expected outputs appear on disk.

**Acceptance:** every preset's enricher set is explicit; legacy presets
still pass the existing drift test; the matrix is documented in the
profiles README.

**Est size:** ~400 LOC code + ~500 LOC tests + ~100 LOC docs.

---

## Chunk 8 — Promotion + ADR + docs (1 small PR)

- RFC-088 Status → Completed (with the implementation-time amendments
  from chunks 0/3 inline).
- ADR-104 promoted Proposed → Accepted (was Proposed in chunk 0).
- PRD-026 Status: Draft → Implemented (or Partial if Topic Entity View
  surface is intentionally tail-end work; assess on landing).
- PRD-027 Status: Draft → Partial (LLM query synthesis is a follow-on
  RFC, not in scope).
- `docs/architecture/ARCHITECTURE.md` enrichment-layer section added.
- `docs/guides/ENRICHMENT_LAYER_GUIDE.md` operator-facing how-to:
  enabling enrichers, opt-in semantics, output discovery, re-run, opt-in
  for ML tier, profile-preset matrix.

**Est size:** doc-only, ~300 LOC across files.

---

## Cross-cutting concerns

**CI hygiene** ([[feedback_no_llm_in_ci]]):

- No enricher calls a paid LLM in CI.
- NLI (chunk 4) ships with a stub scorer for tests; real model only
  fires under an explicit operator-driven workflow.
- `make ci-fast` must remain under its current budget after each chunk
  (the deterministic enrichers run in ~seconds on the smoke corpus).

**Performance:**

- Two-phase executor optimised for the common case (deterministic
  enrichers, corpus < 1000 episodes) — single-threaded is fine.
- Parallelism inside a phase deferred (RFC-088 Open Question #2). Add
  when corpus size or ML tier latency demands it.
- Pipeline wall-clock budget for enrichment pass: ≤ 5% of full pipeline
  on the reference corpus (sub-5s on 100 episodes for the
  deterministic 6).

**Versioning + backwards compatibility:**

- Each enricher carries `manifest.version` (semver). Schema bumps go via
  new `schema_version` field on output; readers must handle older
  versions.
- v1 readers must treat a missing enrichment file as "not configured",
  never as an error (RFC-088 Decision: ungated graceful degrade).

**Operator config defaults:**

- The 6 deterministic enrichers ship enabled by default.
- All embedding/ml/llm tier enrichers ship `enabled: false` by default.
- The double-opt-in (`enabled: true` AND `opt_in: true`) gate is the
  only thing the LLM tier can ever pass.

**Re-run semantics** (RFC-088 Decision #8): full recompute on every run.
Incremental updates deferred. Re-run on stale enrichments after a core
pipeline rebuild is the operator's responsibility (matches the existing
"corpus is on disk, what you see is what's there" stance from RFC-072).

**Migration story:** zero migration. Existing corpora work unchanged.
Enrichments are additive; their absence is silent.

---

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Enrichment pass adds non-trivial pipeline wall-clock | Default to 6 deterministic enrichers only; measure on real corpus before declaring done; opt out via config |
| NLI model is too big for CI / dev laptops | DeBERTa-v3-small (~80MB, CPU); ship stub mode for tests; real model only runs under operator-side workflow. No DGX dependency anywhere. |
| RFC-097 / RFC-088 divergence (RELATED_TO) breaks consumers if naive | Settle as chunk 0 with ADR-104 before any code |
| `topic_similarity` duplicates LanceDB work | Reuse existing index, don't re-embed; the enricher only consumes and projects |
| Operator config schema sprawl | Match RFC-077 viewer-operator YAML conventions; one block, one source of truth |
| Viewer changes cross stack-test boundary | Each chunk that touches the viewer ships its own Playwright spec; reuse the dev-hook pattern from `__GIKG_SUBJECT__` |

---

## Decisions resolved 2026-06-26

1. **ADR-104 framing** — Path (A): KG-direct + enrichment-layer paths
   coexist. KG owns typed connectivity (airgapped + LLM grounding);
   enrichment owns scored/rankable signals (UI consumption + autoresearch
   tuning).
2. **NLI model** — DeBERTa-v3-small, CPU only. **No DGX fallback.**
   DGX is operator-side experimentation, never the shipping path.
3. **Phasing cadence** — single `feat/enrichment-layer` integration
   branch with chunks 0–8 as separate commits, one PR at the end.
   Matches the RFC-097 chunked-PR shape that worked well for review.
4. **Operator entry points** — three layers, all in chunk 1: CLI
   (`podcast enrich`), Jobs API (new job type `"corpus_enrichment"`
   alongside `"full_incremental_pipeline"`, viewer Operator tab gets a
   "Run enrichment" action), and pipeline-attached (always-on
   Step N+1 of `workflow/orchestration.py`). **No `make enrich`
   target** — Makefile is for dev convenience, not ops; dev iteration
   uses `.venv/bin/python -m podcast_scraper enrich` directly.
5. **GH-issue tracking** — Epic + 9 child issues, one per chunk. Epic
   is the umbrella; chunks reference the Epic and each other. Operator
   explicitly authorized creation 2026-06-26 (overrides the default
   `[[feedback_never_open_gh_issues]]` for this work only).
6. **Eval per smart enricher** — every embedding/ml/llm-tier enricher
   ships an eval harness in the same chunk; autoresearch wiring lands
   alongside when the enricher has tunable params. Per REPLAN-O6
   (revised):
   - **Scoring scripts** ship under `scripts/eval/score/enrichment_*.py`
     as direct Python entry points (matches the existing 39-script
     convention; **no `make eval-enrichment-*` Make wrappers**).
   - **Autoresearch program entry points** ship under
     `autoresearch/enrichment_<id>/eval/score.py` (same shape as
     `autoresearch/bundled_prompt_tuning/eval/score.py`) and **get**
     `make autoresearch-enrichment-<id>` Make wrappers (matches existing
     `make autoresearch-score` / `make autoresearch-score-bundled`
     convention).
   - Eval data lives under `data/eval/enrichment/`.
7. **Profile-preset wiring** — chunk 7 wires `EnricherSet` into
   `ProfilePreset`. Every preset names its enricher set explicitly;
   per-CLI overrides supported (`--enrichers`, `--no-enrichers`).

---

## Estimated total

9 chunks (was 7; +eval-per-chunk integrated, +profile-preset chunk
added). Doc/code split roughly 25/75. Total ~3500–5500 LOC of code +
~4500–6500 LOC of tests + ~1500 LOC of eval scripts/scoring + ~1200 LOC
of autoresearch wiring. Mostly mergeable in a 3–4-week elapsed window
on a single integration branch.

The deterministic baseline (chunks 0–2) remains the highest-value
milestone — `topic_cooccurrence_corpus` / `temporal_velocity` /
`grounding_rate` on disk and consumable; PRD-026 Topic Entity View work
unblocked. Chunks 3–4 add the smart tiers with full eval coverage.
Chunks 5–6 wire query-time + viewer consumption. Chunk 7 makes the
operator's profile-preset control over the whole thing real. Chunk 8
closes the paperwork.

If we have to land less than the full plan: stopping after chunk 2 +
chunk 7 (preset wiring scoped to deterministic only) gives the
deterministic enrichers usable in every profile without rework. Smart
tiers (chunks 3–4) and query-time (chunk 5) layer in additively.
