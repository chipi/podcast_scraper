# RFC-088 Enrichment Layer — implementation plan

Working doc for landing the enrichment layer as RFC-088 designed it. Real
end-to-end build, not a paperwork promotion. Shape mirrors how RFC-097
landed (chunked PRs, each independently mergeable, ~9 chunks total).

**Target outcome:** RFC-088 → Completed; PRD-026 and PRD-027 unblocked; the
"4th artifact tier" of derived signals exists on disk and is consumable
from API + viewer.

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
  `pipeline_jobs.py:reconcile_jobs_inplace` catches dead enrichment-
  job processes the same way it catches dead pipeline-job processes.

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
  protocol.py          # Enricher, EnricherManifest, EnricherScope,
                       # EnricherTier, EpisodeArtifactBundle (PEP 544,
                       # @runtime_checkable, mirrors RFC-088 §Protocol)
  registry.py          # register(), get(), list_enabled(); double-opt-in
                       # enforcement for LLM tier; YAML-driven discovery
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

- `src/podcast_scraper/server/pipeline_jobs.py` gains a new
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

**Config schema additions:** new top-level `enrichment:` block in operator
YAML / corpus config:

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
- `tests/integration/enrichment/test_resilience_scenarios.py` —
  scenario-driven scorer mocks exercise the full retry-then-recover,
  circuit-open, OOM, timeout, stall, intermittent-30%, drift paths
  end-to-end through the executor. **One of the most important
  test files in chunk 1** — it pins resilience behaviour against
  representative failure modes.
- `tests/stack-test/stack-enrichment-resilience.spec.ts` — 7 E2E
  resilience scenarios (happy / retry-recovery / circuit-open /
  auto-disable / re-enable / cancel / stall-watchdog) driven through
  `POST /api/jobs/enrichment` with `SCORER_OVERRIDE` env-var injection.

**Mock-scorer fixtures** (`tests/fixtures/enrichment/mock_scorers.py`):
`ScenarioNliScorer` + `ScenarioEmbeddingProvider` with the 9 scenarios
listed in §"Mock-server / scorer-stub strategy" above. Ship in
chunk 1 — every later chunk consumes them.

**Docs:** `docs/api/ENRICHMENT_LAYER_API.md` — output envelope schema,
filename conventions, discovery rules, opt-in semantics, **CLI usage
+ jobs-API endpoint shape + viewer-tab integration note + resilience
model + health-file shape + manual recovery procedures**.

**Acceptance:** ci-fast green; integration tests pass (including all 9
scenario-driven resilience cases); stack-test green (7 E2E scenarios);
`make docs` strict green; no enrichers shipped yet but the framework
is fully exercised by the "no-op" path through CLI, jobs API, and
pipeline-attached entry points + the full resilience pipeline through
the scorer-stub scenarios.

**Est size:** ~1500–2100 LOC code + ~1500–2000 LOC tests + ~600 LOC
mock-scorer fixtures + ~400 LOC E2E spec. Added ~600 LOC code +
~800 LOC tests for the resilience primitives, scorer protocols,
health persistence, and scenario mocks. ~2 reviewer days.

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
  drift. Runs on every PR via `make eval-enrichment-deterministic`.
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
  labels + MRR@10 of the top-K neighbours per topic.
- Runs locally + on demand via `make eval-enrichment-topic-similarity`.
  NOT in CI (needs the real embedder + indexed corpus).

**Autoresearch wiring (optional, recommended):** the two tunable
parameters — `similarity_threshold` (default 0.7) and `top_k` (default
20) — are wired into the existing autoresearch v2 framework as a small
sweep: dev/held-out split on the curated pairs, F1 as the ratchet
signal. Same shape as RFC-073's Track A loop. Adds
`autoresearch/enrichment_topic_similarity/` directory; produces a
champion params recommendation that lands in the default
`topic_similarity.yaml` config block.

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
- Splits the labels dev/held-out per `[[feedback_silver_judge_vendor_bias]]`
  — though there's no judge here, the dev/held-out hygiene applies
  whenever the loop is closed via autoresearch.
- Runs locally + on demand via `make eval-enrichment-nli-contradiction`.
  NOT in CI.

**Autoresearch wiring (required for this enricher):** two tunable
dimensions worth a sweep:
1. `threshold` — anywhere from 0.3 to 0.8 in 0.05 steps; ratchet on
   dev-set F1, validate on held-out.
2. `model_variant` — `nli-deberta-v3-small` (default) vs
   `nli-deberta-v3-base` vs `nli-deberta-v3-large` — useful if F1 on
   dev is below acceptable. Each is a one-line config change; the eval
   loop spawns the contender against the current champion.

Lives at `autoresearch/enrichment_nli_contradiction/` mirroring
`autoresearch/bundled_prompt_tuning/`. Champion params land in the
default `nli_contradiction.yaml` config block on accept.

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
   alongside when the enricher has tunable params. Eval lives under
   `data/eval/enrichment/`, scoring scripts under
   `scripts/eval/score/enrichment_*.py`, Makefile targets
   `make eval-enrichment-*`.
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
