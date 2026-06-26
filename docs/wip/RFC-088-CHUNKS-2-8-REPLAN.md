# RFC-088 chunks 2–8 replan (post chunk-1 expansion)

Chunk 1 grew significantly across the resilience / mock-scorer /
metrics-o11y-analytics / MCP / correlation / cost-cap iterations. The
foundation now absorbs cross-cutting concerns that the original
chunk plan had distributed across later chunks. This doc walks
chunks 2–8 to record what stays / what moves / what shrinks, plus
the 3 follow-on operator decisions surfaced by the replan.

**Source of truth:** plan body at
`docs/wip/RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md`. This doc
is the rationale; the plan is the authoritative shape.

---

## What chunk 1 absorbed (the things chunks 2–8 no longer build)

| Surface | Was originally in | Now in chunk 1 |
| --- | --- | --- |
| `Enricher` protocol (async + EnricherResult + RunContext) | Implied across all chunks | Defined once in `enrichment/protocol.py` |
| Retry-with-backoff, circuit-breaker, auto-disable | Each smart-enricher chunk | Single `enrichment/resilience.py` consumed by every enricher |
| Heartbeat watchdog + hard timeout + cooperative cancel | Each smart-enricher chunk | Single `enrichment/executor.py` mechanism |
| `EnrichmentMetrics` + JSONL events + run_summary + live status + Sentry breadcrumbs + Langfuse tags | Implicit / per-chunk | Single set of surfaces in `enrichment/{metrics,status,observability}.py` |
| `metrics/latest.json.enrichment` dashboard block + `detect_deviations` regression checks | Chunk 8 / future | Chunk 1 ships the framework writing through, then every later chunk's enricher populates it for free |
| MCP server extension (`enrichment_*` tools, `prod_correlate` join, `prod_summary` subsection) | Chunk 8 / future | Chunk 1 ships 8 tools wired through the no-op path |
| Correlation IDs across Sentry / Langfuse / Loki / JSONL / run_summary / status / health | Implicit | Single `enrichment/correlation.py` `RunContext` envelope |
| Mock-scorer scenario engine (9 scenarios × 2 scorer protocols) | Chunk 4 / future | Single `tests/fixtures/enrichment/mock_scorers.py` consumed by every later chunk |
| Jobs-API job type `corpus_enrichment` + 6 server routes + CLI + viewer Operator-tab integration | Distributed | Single new `server/routes/enrichment.py` + CLI in chunk 1 (viewer Operator-tab Enrichment panel lands in chunk 6) |
| `.viewer/enrichment_health.json` persistence + auto-disable + manual recovery | Chunk 4 / future | Single `enrichment/health.py` |
| Per-enricher + run-wide cost-cap enforcement plumbing (O1) | Chunk 5 / future | **See §"Chunk-1 follow-on amendment" below** — moving the *plumbing* to chunk 1 makes chunks 4 + 5 trivially benefit; only the *populating* (per-enricher manifest values) stays in 4+5 |
| `pipeline_jobs.py` → `jobs.py` rename + multi-`command_type` docstring | Deferred follow-up | Chunk 1 (O4 decision) |

Net: chunks 2–8 are smaller in scope and more focused on the
**domain content** of each chunk (the enricher's algorithm, the
gold-set labels, the autoresearch sweep config, the viewer surface
wiring) rather than rebuilding cross-cutting infrastructure.

---

## Chunk-1 follow-on amendment (cost-cap enforcement plumbing)

**Discovered during replan:** the O1 decision put cost-cap *fields*
in chunk 1 (`EnricherManifest.max_cost_usd_per_run` + top-level
`enrichment.max_total_cost_usd_per_run`), but enforcement was
deferred to chunks 4 + 5. That's awkward — the *plumbing* (when
does the executor check? what does it do?) is generic and belongs
next to the resilience model. Only the *manifest values + per-tier
provider cost wiring* needs to wait for the LLM enricher to land.

**Resolution:** chunk 1 ships the enforcement plumbing in
`enrichment/executor.py` + `enrichment/resilience.py`:

- After each enricher run, the executor compares
  `EnrichmentMetrics.cost_usd` (set by the existing
  `record_provider_call_cost` chain when an enricher's scorer calls a
  provider) against `manifest.max_cost_usd_per_run`. Exceeded →
  quarantine that enricher only with status `quarantined`, reason
  `cost_cap_exceeded`.
- A run-wide `total_cost_usd` counter is incremented across every
  enricher's `EnrichmentMetrics.cost_usd`. When the total exceeds
  `enrichment.max_total_cost_usd_per_run`, the executor aborts the
  enrichment pass (subsequent enrichers in the queue marked
  `skipped`, reason `run_cost_cap_exceeded`). The whole run's
  `status` flips to `failed`; `exit_code` is non-zero unless
  `enrichment.fail_on_run_cost_cap: false` is set.
- Chunk 1 unit test: `test_resilience_scenarios.py` gains
  `cost_cap_per_enricher_quarantines_offender` and
  `cost_cap_run_wide_aborts_pass` cases driven via the scorer mock
  that returns a `cost_usd: 0.10` field per call.

Chunks 4 + 5 just populate the manifest fields. No enforcement code
lands there.

This is a ~50 LOC code + ~80 LOC tests addition to chunk 1. New
chunk-1 size: ~2350–3150 LOC code + ~2500–3200 LOC tests (was
~2300–3100 / ~2400–3100).

---

## Per-chunk delta

### Chunk 2 — Deterministic enrichers + gold-fixture eval

**Smaller, more focused.** Was: "ship 6 enrichers + eval harness."
Now: ship the 6 algorithms; everything else (registry / executor /
resilience / metrics / MCP) is consumed from chunk 1.

**Concrete deltas:**

- Each enricher is `async def enrich(...) -> EnricherResult`
  using the `@sync_enricher` decorator on a sync body (deterministic
  enrichers don't need async semantics internally).
- Each enricher writes through `Metrics.enrichment` automatically;
  no per-enricher metrics wiring code.
- No new resilience code; the deterministic-tier policy (0 retries,
  auto-disable at 5 consecutive failed runs) already enforced by
  chunk-1 `resilience.py`.
- Eval harness simplified: `data/eval/enrichment/deterministic/gold/`
  + `scripts/eval/score/enrichment_deterministic.py` (direct Python
  entry point, no Make wrapper — matches the existing 39-script
  convention per REPLAN-O6). The chunk-2 test suite wraps this script
  as a unit test so the gold-fixture exact-match runs in CI on every
  PR.
- Acceptance unchanged: every enricher idempotent, <5s on the
  reference corpus, gold-fixture exact-match.

**Est size (was ~600-900 LOC code + ~700-1000 LOC tests + ~200 LOC
eval). Now:** ~500-800 LOC code (less framework boilerplate) +
~600-900 LOC tests + ~200 LOC eval. ~2 reviewer days.

### Chunk 3 — `topic_similarity` (embedding tier) + eval + autoresearch sweep

**Smaller, more focused.**

**Concrete deltas:**

- Concrete `enrichment/scorers/lancedb_embeddings.py` implements the
  `EmbeddingProvider` protocol from chunk 1. Real-embedder
  integration test exercises it; the chunk-1
  `ScenarioEmbeddingProvider` covers all resilience scenarios.
- All resilience behaviour (3 retries, 30s max backoff, circuit at
  5 consecutive failures within a run, auto-disable at 3 failed
  runs) inherited from chunk 1; no new code.
- Cost-cap manifest fields stay at None (no provider cost for
  LanceDB local index).
- Eval harness: `data/eval/enrichment/topic_similarity/gold/` +
  `scripts/eval/score/enrichment_topic_similarity.py` (direct Python,
  no Make wrapper) + autoresearch ratchet at
  `autoresearch/enrichment_topic_similarity/eval/score.py` with
  `make autoresearch-enrichment-topic-similarity` Make wrapper
  (matches existing `make autoresearch-score` convention).
- The `enrichment_eval_history` MCP tool from chunk 1 starts
  returning real data once chunk 3's eval JSONL ships.

**Est size (was ~300-500 LOC code + ~400-600 LOC tests + ~400 LOC
eval + ~300 LOC autoresearch). Now:** ~250-400 LOC code + ~350-500
LOC tests + ~400 LOC eval + ~300 LOC autoresearch. ~2 reviewer days.

### Chunk 4 — `nli_contradiction` (ml tier) + NLI eval set + autoresearch sweep

**Smaller code, eval still substantial.**

**Concrete deltas:**

- Concrete `enrichment/scorers/nli_deberta.py` implements the
  `NliScorer` protocol from chunk 1. Real-model integration test
  exercises it; the chunk-1 `ScenarioNliScorer` covers all
  resilience scenarios (including OOM, stall, drift).
- Cost-cap manifest fields populated: `max_cost_usd_per_run = None`
  (NLI is local CPU, free). Run-wide cost cap unchanged.
- Eval set is the biggest single deliverable in chunk 4 (~100
  labelled pairs JSONL) — this stays substantial.
- Autoresearch sweep stays substantial (threshold ratchet +
  model_variant rotation across `nli-deberta-v3-small / base /
  large`).
- All resilience behaviour (2 retries, 60s max backoff, circuit at
  3 consecutive failures, auto-disable at 2 failed runs) inherited
  from chunk 1.

**Est size (was ~400-600 LOC code + ~600-800 LOC tests + ~500 LOC
eval + ~600 LOC autoresearch). Now:** ~300-500 LOC code + ~500-700
LOC tests + ~500 LOC eval + ~600 LOC autoresearch + ~100-row
labelled JSONL. ~3 reviewer days (eval + autoresearch hold size).

### Chunk 5 — `QueryEnricher` protocol (RFC-088 Phase 4)

**Significantly smaller.**

**Concrete deltas:**

- Two new protocol modules: `enrichment/query_protocol.py` +
  `enrichment/query_registry.py`. Define `QueryEnricher` async
  protocol with `enrich_query_result(*, query, results, config,
  ctx)`.
- Per-request `RunContext` derived from a new `request_id` UUID on
  the search route (chunk-1 lock audit §I3).
- One concrete query enricher: `query_topic_relatedness`
  (deterministic) — decorates each search hit with precomputed
  `topic_similarity` ranks from chunk 3 output.
- `/api/search` gains `enrich_results: bool` parameter; when true
  the query enricher chain runs over the response.
- LLM-tier query enrichers stay out of scope (separate follow-on RFC
  per RFC-088 Phase 4). Cost-cap enforcement plumbing from chunk 1
  is reused — when LLM query enrichers eventually ship, they only
  populate manifest fields, no new enforcement code.

**Est size (was ~400-600 LOC code + ~400-600 LOC tests). Now:**
~300-450 LOC code + ~350-550 LOC tests. ~2 reviewer days.

### Chunk 6 — Server routes (user-facing) + viewer integration

**Mostly viewer work now.** Server read routes mostly shipped in
chunk 1 (consumed by the MCP source module); chunk 6 adds the
user-facing ones (`/api/corpus/enrichments/<id>`,
`/api/corpus/episode/<id>/enrichments/<id>`) and the
catalog-row extensions.

**Open decision (REPLAN-O5):** chunk 6 currently bundles:
- (i) user-facing server routes + catalog row extension
- (ii) viewer Operator-tab Enrichment panel
- (iii) Topic Entity View consumption (PRD-026)
- (iv) Person Profile consumption (PRD-029 adds)

Each is a separate viewer surface. Options:

- **(A) Keep monolithic** — one chunk 6 PR, several internal
  sections. Faster wall-clock, longer review cycle.
- **(B) Split into 6a / 6b / 6c** — server + Operator panel /
  Topic Entity View / Person Profile. Three smaller PRs.

**Recommendation:** (A) for the integration branch (everything
lands together) but commit-internally split (one commit per
sub-section). RFC-097 shape — works for review.

**Concrete deltas:**

- Server routes module already exists from chunk 1; chunk 6 adds
  the user-facing routes alongside.
- Viewer Operator-tab Enrichment panel consumes
  `/api/jobs?command_type=corpus_enrichment` + `/api/enrichment/*`
  read routes (already shipped). Panel renders per-enricher last-run
  status / health badges / latency / drill-down.
- Topic Entity View tab consumes `topic_cooccurrence_corpus` +
  `temporal_velocity` (from chunks 2 and 3) — RELATED_TO chips +
  trend sparkline.
- Person Profile rail consumes `grounding_rate` +
  `guest_coappearance` (from chunk 2).
- vitest + Playwright stack-test cover the viewer additions.

**Est size (was ~700-1000 LOC code + ~800-1100 LOC tests). Now:**
~700-1000 LOC code (viewer-heavy) + ~800-1100 LOC tests. ~3-4
reviewer days. Size unchanged because viewer surfaces are the actual
deliverables here.

### Chunk 7 — Profile-preset wiring (`EnricherSet`)

**Smaller.** Chunk 1 ships the minimal `EnricherSet` stub; chunk 7's
job is profile-preset matrix integration.

**Concrete deltas:**

- New `EnricherSet` extended fields on the chunk-1 stub:
  per-enricher config overrides + opt-in flags (the stub only had
  `enabled_enrichers`).
- `ProfilePreset.enrichments: EnricherSet` field.
- Profile-preset matrix codified in `config/profiles/*.yaml`:
  `test_default` (none), `airgapped_thin` (deterministic only),
  `airgapped` (deterministic + topic_similarity + query enrichers),
  `cloud_thin` (full stack), `dev`, `prod`.
- CLI `--enrichers` / `--no-enrichers` overrides.
- Drift test: every profile preset must declare an `enrichments`
  field.

**Est size (was ~400 LOC code + ~500 LOC tests). Now:** ~300 LOC
code + ~400 LOC tests + ~100 LOC docs. ~1-2 reviewer days.

### Chunk 8 — Promotion + ADR-104 Accepted + ENRICHMENT_LAYER_GUIDE

**Unchanged.** Same scope: RFC-088 Active → Completed (was Draft →
Completed before O3); ADR-104 Proposed → Accepted; PRD-026
implemented; PRD-027 partial; ENRICHMENT_LAYER_GUIDE published.

Adds operator-facing documentation of the 8 MCP tools (from chunk
1) in OBSERVABILITY_EXTENSIONS.md.

**Est size:** ~300 LOC across docs. ~1 reviewer day.

---

## Total revised estimate

| Chunk | Code | Tests | Eval / Misc | Reviewer days |
| --- | ---: | ---: | ---: | ---: |
| 0 (ADR-104) | 0 | 0 | 1 ADR doc | 0.5 (shipped) |
| 1 (foundation + amendment) | ~2400-3200 | ~2500-3200 | ~600 mocks + ~600 E2E | ~3 |
| 2 (deterministic) | ~500-800 | ~600-900 | ~200 eval | ~2 |
| 3 (topic_similarity) | ~250-400 | ~350-500 | ~400 eval + ~300 autoresearch | ~2 |
| 4 (nli_contradiction) | ~300-500 | ~500-700 | ~500 eval + ~600 autoresearch + ~100-row JSONL | ~3 |
| 5 (QueryEnricher) | ~300-450 | ~350-550 | — | ~2 |
| 6 (server + viewer) | ~700-1000 | ~800-1100 | — | ~3-4 |
| 7 (profile presets) | ~300 | ~400 | ~100 docs | ~1-2 |
| 8 (promotion + docs) | — | — | ~300 docs | ~1 |
| **Total** | **~4750-6650** | **~5500-7350** | **~3700 incidentals** | **~17-19 days** |

Previous total estimate (before replan): ~3500–5500 LOC code +
~4500–6500 LOC tests + ~1500 LOC eval + ~1200 LOC autoresearch
~3-4 weeks. New total: ~10–12k LOC code + tests + ~3.7k incidentals,
~3.5–4 weeks. **The total grew because chunk 1 grew, not because
chunks 2–8 grew.** Chunks 2–8 are individually smaller than the
original plan.

---

## Follow-on operator decisions — partial RESOLVED 2026-06-26

### REPLAN-O5. Chunk 6 split → **CONFIRMED: monolithic PR, commit-per-chunk** ✅

**Decision:** **all 9 chunks land on the single `feat/rfc-paperwork-promotions-v3` branch as separate commits; one PR at the end.** Matches the operator's earlier directive ("keep going on this branch") and the RFC-097 chunked-commit shape that worked well for review. Chunk 6 internally still splits into server-routes / Operator-panel / Topic-Entity-View / Person-Profile commits, but they all ride the same PR.

### REPLAN-O7. Cost-cap enforcement plumbing in chunk 1 → **CONFIRMED** ✅

**Decision:** **plumbing in chunk 1.** The executor checks + per-enricher quarantine + run-wide abort all ship in chunk 1's `enrichment/executor.py` + `enrichment/resilience.py`. Chunks 4 + 5 populate manifest fields only. Test cases added to chunk-1 `test_resilience_scenarios.py`:
- `cost_cap_per_enricher_quarantines_offender` — scorer mock returns `cost_usd: 0.10` per call; enricher hits its `max_cost_usd_per_run` budget; status flips to `quarantined`, reason `cost_cap_exceeded`; other enrichers continue.
- `cost_cap_run_wide_aborts_pass` — total accumulated `cost_usd` across enrichers exceeds `enrichment.max_total_cost_usd_per_run`; subsequent enrichers in the queue marked `skipped`, reason `run_cost_cap_exceeded`; run status `failed` (unless `enrichment.fail_on_run_cost_cap: false`).

**Chunk-1 size delta:** ~50 LOC code + ~80 LOC tests added.

### REPLAN-O6. Make targets for eval workflows → **CONFIRMED option (2): scoring scripts only, Make wrappers for autoresearch programs** ✅

**Decision (revised after operator analysis):** **match the established API-LLM eval pattern.** The 39 existing `scripts/eval/score/*.py` files have NO Make wrappers — they're called directly. Make targets are reserved for *programs* (recurring multi-step workflows like `make autoresearch-score`, `make benchmark`, `make silver-pairwise`, `make run-compare`).

**Applied to enrichment:**

| Surface | Lives at | How called |
| --- | --- | --- |
| Per-enricher scoring scripts | `scripts/eval/score/enrichment_deterministic.py`, `enrichment_topic_similarity.py`, `enrichment_nli_contradiction.py` | Direct Python: `python scripts/eval/score/enrichment_topic_similarity.py --threshold 0.7 --top-k 20`. **No Make wrappers.** |
| Deterministic gold-fixture check | Same script as above (`enrichment_deterministic.py`) | Called from chunk-2 test suite as a unit test → runs in CI on every PR |
| Autoresearch ratchet programs (chunks 3 + 4) | `autoresearch/enrichment_topic_similarity/eval/score.py`, `autoresearch/enrichment_nli_contradiction/eval/score.py` (same shape as existing `autoresearch/bundled_prompt_tuning/eval/score.py`) | Direct Python + Make wrappers: `make autoresearch-enrichment-topic-similarity`, `make autoresearch-enrichment-nli-contradiction` (same shape as existing `make autoresearch-score`) |

**Rationale:** the operator was right to question Make wrappers per scoring script — the 39-script convention shows scoring is a Python-script-direct affair; Make wrappers are reserved for programs. Per-enricher scoring scripts get the convention-matching shape; the autoresearch programs that wrap them get the convention-matching Make wrappers.

---

## What this replan does NOT change

- **Architectural boundaries** — ADR-104 still governs; KG-direct
  RELATED_TO stays in core, enrichment-layer signals stay in
  `enrichments/`. No re-litigation.
- **Profile-preset matrix** — the chunk 7 matrix unchanged.
- **Eval set sizes / quality bars** — chunks 3 + 4 acceptance F1
  thresholds unchanged.
- **No DGX dependency** — still the hard constraint.
- **No-LLM-in-CI** — still the hard constraint.
- **`prod_correlate(run_id)` consistent-story guarantee** — chunk 1
  acceptance criterion unchanged.

---

## Next steps

1. Operator confirms REPLAN-O5 / O6 / O7.
2. Plan body updated per this delta (already in progress alongside
   this doc).
3. GH issues #1104–#1110 updated with the revised scopes.
4. Chunk 1 implementation starts.
