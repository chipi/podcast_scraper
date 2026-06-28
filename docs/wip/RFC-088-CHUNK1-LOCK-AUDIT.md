# RFC-088 chunk-1 lock audit

The implementation plan grew organically across seven commits (the
foundation, the eval-per-chunk + profile-preset + DGX-drop revision,
chunk 0 ADR-104, the jobs-API job type, the resilience + mock-scorer
engine, the metrics + o11y + analytics layer, and the MCP server
extension + correlation IDs). The frame: **the iterated plan IS the
reference now — it does better than the original RFC-088 alone — and
this audit is about making sure that improved reference is internally
consistent before chunk 1 starts.**

Findings categorized: **Blocking** (must resolve before code starts),
**RFC-088 amendments** (small targeted edits to bring the RFC body
in line with the improved plan, same pattern as ADR-104 chunk 0),
**Important** (resolve in the plan but acceptable to nail down at
impl-time), **Open** (operator-decision items), **Non-gaps**
(checked, no action needed).

---

## Blocking (resolved in this audit's plan amendment)

### B1. Enricher protocol must be `async def`

**Found:** RFC-088 §Enricher Protocol defines `def enrich(...) -> dict`
(sync, line 218 of the RFC). The plan's executor is `asyncio.gather`-
based and the scorer protocols (`NliScorer.score`, etc.) are async.
Sync enricher in an async executor either forces a thread-pool
adapter (extra ceremony) or can't await async scorers cleanly.

**Resolution:** plan + RFC-088 §Enricher Protocol amend to:

```python
@runtime_checkable
class Enricher(Protocol):
    @property
    def manifest(self) -> EnricherManifest: ...

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict,
        ctx: RunContext,
    ) -> EnricherResult:
        ...
```

Note added: deterministic enrichers can be sync-implemented and
wrapped via a tiny `sync_enricher` decorator that runs them in the
default thread executor. Authors of deterministic enrichers don't pay
the async ceremony tax; the executor pays it for them.

**Files touched in chunk 1 PR:** RFC-088 body §Enricher Protocol +
plan body + chunk-1 module list + chunk-1 issue acceptance criteria.

### B2. `EnricherResult` shape must be defined in the protocol

**Found:** the plan mentions `EnricherResult` with `status: ok |
failed | timeout | quarantined | cancelled | skipped` in passing but
never defines the dataclass. The RFC currently says `enrich()`
returns `dict`. Two outputs mean the framework doesn't know how to
unwrap.

**Resolution:** plan + RFC-088 amend to add:

```python
@dataclass(frozen=True)
class EnricherResult:
    status: str            # "ok" | "failed" | "timeout" | "quarantined"
                           # | "cancelled" | "skipped"
    data: dict | None      # the enricher's output dict; None when
                           # status != "ok"
    error: str | None      # short reason; None when status == "ok"
    error_class: str | None  # e.g. "OutOfMemoryError",
                             # "ScorerTimeoutError"
    retry_count: int = 0
    circuit_state: str | None = None  # at result time
    duration_ms: int = 0
    records_written: int = 0   # only meaningful when status == "ok"
```

The enricher returns this; the executor reads `status` to drive the
state machine and writes `data` into the envelope file when `status
== "ok"`.

### B3. `EnricherSet` chunk-1 stub vs chunk-7 full preset wiring

**Found:** the executor in chunk 1 needs to know which enrichers to
run. Chunk 7 ships the full profile-preset → `EnricherSet` wiring.
Without chunk 1 owning a minimal `EnricherSet`, the executor would
need to either be re-written in chunk 7 or use ad-hoc config keys
chunk 7 then refactors.

**Resolution:** chunk 1 ships a minimal `EnricherSet` dataclass —
`enabled_enrichers: list[str]`, `per_enricher_config: dict[str, dict]`,
`opt_in_flags: dict[str, bool]`. Chunk 1's executor consumes it.
Chunk 7's profile-preset machinery just constructs the `EnricherSet`
from preset YAML and hands it to the executor.

Plan + issue updated.

### B4. Pipeline-attached failure semantics

**Found:** "Step N+1 of `workflow/orchestration.py`" implies after
`_finalize_pipeline`, but the plan never says explicitly what happens
when core pipeline partially fails.

**Resolution:** plan amend — pipeline-attached enrichment **runs only
when the core pipeline completes successfully** (`run.status == "ok"`
in `run_summary`). Partial-failure pipelines emit
`enrichment.run.skipped { reason: "core_pipeline_failed" }` JSONL
event and skip the enrichment step. Operators can still run
enrichment standalone via CLI or jobs API on a partial corpus —
that's a deliberate operator choice, not the default.

### B5. `runs_skipped` flow path

**Found:** `EnrichmentMetrics.runs_skipped` counter exists but the
plan doesn't say when it increments vs when no record is written at
all.

**Resolution:** `runs_skipped` increments when an enricher is
**configured + enabled by the active profile preset** but skipped this
run for one of:
- circuit currently open + cooldown active
- auto-disabled flag set
- `--skip <id>` CLI flag
- `enabled: false` in operator YAML override
- query-time enricher whose dependency (precomputed file) is missing

Profile-preset says off → enricher isn't even registered in this run
→ no `EnrichmentMetrics` record at all. That's a distinct state.

### B6. `.viewer/` directory creation

**Found:** `enrichment_health.json` and `enrichment_status.json` are
written under `.viewer/`. Pipeline-attached path inherits an existing
`.viewer/`; standalone path may run against a corpus that has none.

**Resolution:** `enrichment/health.py` and `enrichment/status.py`
both do `Path.mkdir(parents=True, exist_ok=True)` on the
`.viewer/` parent before the first write. Test coverage in
`test_health.py` and `test_status.py` includes the "no .viewer/
exists" path.

### B7. Test enricher registration

**Found:** profile preset matrix says `test_default` has all
enrichers off. But unit + integration tests need to register specific
enrichers under test. The gap: HOW do tests register?

**Resolution:** tests use **direct registry construction via pytest
fixtures**, bypassing profile presets entirely. Standard pattern:

```python
@pytest.fixture
def registered_topic_cooccurrence(registry):
    registry.register(TopicCooccurrenceEnricher())
    yield registry
    registry.clear()
```

The profile-preset path is one of several ways enrichers register;
tests use the direct path because that's what's under test. The
`test_default` profile being all-off means "tests get nothing by
default unless they explicitly register" — exactly the safety
property the matrix is designed for.

Plan amend in chunk-1 acceptance criteria + chunk-1 module docstring
on `registry.py`.

### B8. Operator YAML JSON schema

**Found:** pipeline config has a formal JSON Schema for validation
(`config/schema/*.json`). The `enrichment:` YAML block has none.

**Resolution:** chunk 1 ships `config/schema/enrichment.schema.json`
covering the `enrichment:` block (master switch + per-enricher
overrides + per-tier policy overrides). Schema validation runs in
`enrichment/cli.py` startup and rejects malformed config with a
clear error. Test in `tests/unit/enrichment/test_config_schema.py`.

### B9. Enricher-side failure scenarios (not just scorer-side)

**Found:** mock-scorer scenarios cover scorer failures
(`always_oom`, `flaky_then_recovers`, etc.) but the enricher itself
can fail too — e.g., output envelope shape error, missing required
input artifact, KeyError in data extraction.

**Resolution:** plan amend `test_resilience_scenarios.py` to include
enricher-level synthetic failure cases alongside scorer scenarios:

- `enricher_emits_malformed_envelope` — envelope fails schema
  validation → `EnvelopeShapeError` (non-retryable, auto-disable
  candidate).
- `enricher_missing_required_input` — bundle missing `bridge_path` →
  `BadInputError` (non-retryable, status `failed: missing_input`).
- `enricher_raises_unexpected` — bare `RuntimeError` from the
  enricher body → caught by safety net, logged + Sentry breadcrumb,
  status `failed: unexpected`.

These exercise the executor's safety-net path; the scorer scenarios
exercise the retry + circuit-breaker path. Both kinds needed.

### B10. Cancel + concurrency interaction

**Found:** per-tier concurrency caps (deterministic 4 / embedding 2 /
ml 1 / llm = rate limit) run multiple enrichers in parallel within
a phase. The plan says cancel propagates via `asyncio.Event` but
doesn't explicitly say the **same** event is shared across all
parallel enrichers.

**Resolution:** the executor creates ONE `asyncio.Event` per run and
passes a reference into every enricher's `enrich()` call. Cancel sets
the event once; every parallel enricher checks it between batches
and bails. Test in `test_executor.py`:
`test_cancel_during_parallel_run_bails_all_workers`.

---

## RFC-088 amendments needed (same pattern as ADR-104)

Two amendments land in the chunk-1 PR alongside the foundation code:

### A1. Async protocol amendment (B1 above)

RFC-088 body §Enricher Protocol gets the `async def enrich` rewrite.
Header gets a new boundary-clarification note:

> "**Async protocol amendment (2026-06-26)**: §Enricher Protocol's
> `def enrich(...) -> dict` is replaced with `async def enrich(...)
> -> EnricherResult` to match the async-executor implementation
> shipping in RFC-088 Epic #1101 chunk 1. Sync deterministic
> enricher bodies are supported via the `@sync_enricher` decorator
> (runs in the default thread executor). See chunk 1 (#1103) for the
> protocol shape and `enrichment/protocol.py` for the canonical
> implementation."

### A2. `EnricherResult` shape amendment (B2 above)

RFC-088 §Enricher Protocol gains a new subsection defining the
`EnricherResult` dataclass. Cross-ref in the header.

Both amendments are small (~20 lines each in the RFC body). Same
ADR-104 spirit: bring the spec in line with the improved shipping
shape, don't rewrite the RFC from scratch.

---

## Important (resolve in plan, accept impl-time nail-down)

### I1. `pipeline_jobs.py` naming becomes misleading

**Found:** the file already serves "pipeline" jobs. Adding
`COMMAND_ENRICHMENT` makes the name historical.

**Resolution:** keep the file name (less churn, fewer import edits).
Add a module docstring noting the file handles multiple
`command_type` values now. Renaming to `jobs.py` is a deferred
follow-up not worth the diff size.

### I2. `error_samples` cap

**Found:** `EnrichmentMetrics.error_samples: list[dict]` has
unbounded growth risk.

**Resolution:** `__post_init__` enforces `len <= 5`; record-helpers
push + pop-oldest. Test in `test_metrics.py`.

### I3. Search request_id plumbing for query enrichers (chunk 5)

**Found:** plan says query enrichers carry a "per-request run_id"
but search routes don't emit request_ids today.

**Resolution:** chunk 5 issue (#1107) updated — add small request_id
generation to `server/routes/search.py` (UUID per request) and pass
it into the QueryEnricher invocation as `RunContext.run_id`. Tiny
addition; lives with chunk 5 since that's where the consumer is.

### I4. Server route layout

**Found:** plan adds 6 new routes (`/api/jobs/enrichment`,
`/api/enrichment/{health,metrics,status,events,eval-history}`).
Where do they live?

**Resolution:** new file
`src/podcast_scraper/server/routes/enrichment.py` holds all 6 routes.
Mounted in `server/app.py` alongside the existing routes. Pattern
matches RFC-095 MCP server module layout.

---

## Open (operator decision) — RESOLVED 2026-06-26

### O1. Cost cap policy for LLM-tier enrichment → **BOTH**

**Decision:** **(c) — both per-enricher quarantine AND run-wide
hard stop.** Belt-and-suspenders model:

- **Per-enricher soft budget** (default off; settable per enricher in
  YAML, e.g. `nli_contradiction.max_cost_usd_per_run: 0.50`). When
  exceeded, quarantine that enricher (status `quarantined`,
  reason `cost_cap_exceeded`); other enrichers continue.
- **Run-wide hard cap** (default off; settable in the `enrichment:`
  top-level YAML, e.g. `enrichment.max_total_cost_usd_per_run: 5.00`).
  When exceeded, abort the whole enrichment pass with status
  `failed: run_cost_cap_exceeded` — pipeline-attached path bubbles
  to `enrichment.run.completed { status: failed, reason: cost_cap }`
  and does NOT fail the parent pipeline run (enrichment is additive).
- Standalone CLI / Jobs API runs surface non-zero `exit_code` on
  run-wide hard cap hit; operators can opt in to soft-cap-only by
  setting `enrichment.fail_on_run_cost_cap: false`.

No code wiring in chunk 1 (no LLM enricher ships in chunk 1) but the
JSON Schema (B8) and `EnricherManifest` carry the budget fields from
day one so chunk 4 + chunk 5 just consume them.

### O2. `enrichment_cancel` MCP tool → **ADD**

**Decision:** **add `enrichment_cancel(target, job_id, reason)`** to
chunk 1's MCP surface. Proxies `POST /api/jobs/{job_id}/cancel`;
records `reason` in the cancel envelope for audit. Costs ~30 LOC code
+ a unit test + an integration test.

Updates: MCP tools count in chunk 1 goes from 7 to 8. Issue #1103
amended.

### O3. RFC-088 Status transition during implementation → **Active**

**Decision:** **move RFC-088 from Draft to Active** when chunk 1
foundation lands. Reflects the live-implementation state. Chunk 8
promotes Active → Completed at the end.

Changes:

- RFC-088 body Status line: Draft → Active (chunk 1 PR carries the
  edit alongside the foundation code).
- `docs/rfc/index.md` gap-analysis text updates: RFC-088 moves from
  the "Draft RFCs (not indexed)" list to a new "Active RFCs (in
  implementation)" line.
- Chunk 8 (#1110) issue updated: Status promotion now Active →
  Completed (was Draft → Completed).

### O4. Module rename: `pipeline_jobs.py` → `jobs.py` → **NOW**

**Decision:** **rename in chunk 1**, alongside the new job-type
addition (less awkward to add `COMMAND_ENRICHMENT` to a file already
called `pipeline_jobs`).

Rename impact (verified):

- 1 src file renamed: `src/podcast_scraper/server/pipeline_jobs.py`
  → `src/podcast_scraper/server/jobs.py`.
- 4 internal src imports updated:
  - `src/podcast_scraper/server/jobs_log_path.py`
  - `src/podcast_scraper/server/scheduler.py`
  - `src/podcast_scraper/server/routes/jobs.py` (HTTP route module
    — different file, different package, no name clash)
  - `src/podcast_scraper/server/pipeline_run_prometheus.py`
- 4 test files renamed for symmetry:
  - `test_pipeline_jobs_argv.py` → `test_jobs_argv.py`
  - `test_pipeline_jobs_log_pump.py` → `test_jobs_log_pump.py`
  - `test_pipeline_jobs_helpers.py` → `test_jobs_helpers.py`
  - `test_pipeline_jobs_reconcile.py` → `test_jobs_reconcile.py`
- Docs touched (find-and-replace): `RFC-079`, `RFC-098`,
  `CODEQL_DISMISSALS.md`.

Module docstring updates: "Pipeline job queue, subprocess spawn,
and registry updates" → "Job queue, subprocess spawn, and registry
updates — serves pipeline and enrichment job kinds via `command_type`."

The HTTP route module `server/routes/jobs.py` already exists with that
name; no clash (different package).

---

## Confirmed non-gaps

- **mkdocs nav** — auto-includes via index pattern; ADR-104 and any
  future docs land without nav edits. Verified.
- **Chunk numbering** — searched plan + issues; all references 0–8
  consistent post-revision. No stale "old chunk 7" references.
- **GH-issue dependency chain** — chunk 0 → 1 → (2 ∥ 3 ∥ 4 ∥ 5) → 6
  → 7 → 8. Acyclic, child-issue dependency edges match. Verified.
- **Eval data not in CI** is intentional per
  `[[feedback_no_llm_in_ci]]` — chunks 3 + 4 acceptance criteria
  distinguish CI gates (stub-mode unit tests) from operator-driven
  gates (real-model eval). Confirmed.
- **No DGX dependency** — searched plan + issues; the only mentions
  of DGX are explicit "no DGX in shipping path" guardrails. None
  smuggled back in. Confirmed.
- **ADR-104 boundary** — searched plan + issues; no enrichment chunk
  modifies core artifacts. RFC-097 chunk 9's KG-direct `RELATED_TO`
  is acknowledged as core (per ADR-104), `topic_similarity` writes
  ranks to enrichments, `nli_contradiction` writes pairs to
  enrichments, no enricher writes to `*.kg.json` or `*.gi.json`.
  Confirmed.

---

## Lock state — LOCKED 2026-06-26

All four open operator decisions resolved:

- [x] **O1** — both per-enricher quarantine + run-wide hard stop cost cap
- [x] **O2** — `enrichment_cancel` MCP tool added (chunk 1 MCP surface = 8 tools, was 7)
- [x] **O3** — RFC-088 Status: Draft → **Active** (chunk 1 PR carries the edit); chunk 8 promotes Active → Completed
- [x] **O4** — `pipeline_jobs.py` → `jobs.py` rename ships in chunk 1 alongside the `COMMAND_ENRICHMENT` addition

Chunk 1 is implementable as written:

- [x] B1–B10 inlined in the plan (commit `dba0c38a`).
- [x] I1–I4 inlined in the plan (commit `dba0c38a`).
- [x] O1–O4 decisions inlined in the plan (this commit).
- [x] A1 + A2 RFC-088 §Protocol amendments queued for chunk-1 PR body.

**Chunk-1 size (final):** ~2300–3100 LOC code + ~2400–3100 LOC tests
+ ~600 LOC mocks + ~600 LOC E2E. ~3 reviewer days.

Delta from O1–O4 decisions: ~50 LOC code + ~50 LOC tests for the
`enrichment_cancel` MCP tool + the cost-cap fields in `EnricherManifest`
+ the `pipeline_jobs.py → jobs.py` rename + cascading import updates.

**Status:** chunk 1 is locked. Implementation can start.
