# RFC-088 Enrichment Layer — implementation plan

Working doc for landing the enrichment layer as RFC-088 designed it. Real
end-to-end build, not a paperwork promotion. Shape mirrors how RFC-097
landed (chunked PRs, each independently mergeable, ~9 chunks total).

**Target outcome:** RFC-088 → Completed; PRD-026 and PRD-027 unblocked; the
"4th artifact tier" of derived signals exists on disk and is consumable
from API + viewer.

**Working branch:** `feat/enrichment-layer` (new branch off main after the
current `feat/rfc-paperwork-promotions-v3` lands).

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
                       # [--only <id>,<id>] [--skip <id>,<id>]`
```

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

**Docs:** `docs/api/ENRICHMENT_LAYER_API.md` — output envelope schema,
filename conventions, discovery rules, opt-in semantics.

**Acceptance:** ci-fast green; integration test passes; `make docs`
strict green; no enrichers shipped yet but the framework is fully
exercised by the "no-op" path.

**Est size:** ~700–1100 LOC code + ~600–900 LOC tests. One reviewer day.

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

**Acceptance:** every enricher idempotent, deterministic, < 5s for the
100-episode reference corpus; CI runs them by default (the fixture is
real-corpus-tiny, no LLM, no network).

**Est size:** ~600–900 LOC code + ~700–1000 LOC tests.

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

**ADR-104 anchor:** explicitly documents that `topic_similarity` (this
enricher) and RFC-097 chunk 9 `RELATED_TO` (KG-direct) can coexist —
this enricher is the rankable/scored variant, KG is the
typed-connectivity variant.

**Est size:** ~300–500 LOC code + ~400–600 LOC tests.

---

## Chunk 4 — ML tier: `nli_contradiction` (1 medium PR)

**Enricher:** `nli_contradiction` (corpus scope, `tier=ML`,
`enabled: false` by default — opt-in via config).

**Implementation:**

- Local NLI model via `[ml]` extra. Candidate model:
  `cross-encoder/nli-deberta-v3-small` (DeBERTa-v3-small fine-tuned on
  MNLI; ~80MB, runs on CPU). Add to `pyproject.toml` `[ml]` extra and
  to the preload manifest.
- For each topic, take all Insights from different Persons mentioning
  that topic (via `MENTIONS_PERSON ∩ ABOUT`), score every cross-Person
  pair with NLI (`contradiction` probability), keep pairs with score ≥
  configurable threshold (default 0.5).
- Output: `enrichments/nli_contradiction.json` — list of `{topic_id,
  person_a_id, person_b_id, insight_a_id, insight_b_id,
  contradiction_score}`.

**CI hygiene** (per `[[feedback_no_llm_in_ci]]`): NLI model must NOT run
in CI; integration test uses a stub `NliScorer` protocol that returns
fixed scores per pair. Real-model test gated by environment marker, runs
only on operator demand. **Crucial — the enricher infrastructure must be
testable deterministically without the model.**

**Edge type:** does NOT introduce a `CONTRADICTS` edge type in KG v2.0
(per ADR-104 boundary — derived data lives in enrichments). A future v3
KG decision can promote validated contradiction pairs to typed edges if
warranted; not in scope here.

**Est size:** ~400–600 LOC code + ~600–800 LOC tests (most of it stub-
mode coverage).

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

## Chunk 7 — Promotion + ADR + docs (1 small PR)

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
  for ML tier.

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
| NLI model is too big for CI / dev laptops | Pick a small distilled model (DeBERTa-v3-small ~80MB); ship stub mode; gate real run behind operator workflow |
| RFC-097 / RFC-088 divergence (RELATED_TO) breaks consumers if naive | Settle as chunk 0 with ADR-104 before any code |
| `topic_similarity` duplicates LanceDB work | Reuse existing index, don't re-embed; the enricher only consumes and projects |
| Operator config schema sprawl | Match RFC-077 viewer-operator YAML conventions; one block, one source of truth |
| Viewer changes cross stack-test boundary | Each chunk that touches the viewer ships its own Playwright spec; reuse the dev-hook pattern from `__GIKG_SUBJECT__` |

---

## Open decisions for operator

These are the ones to settle before chunk 0 lands:

1. **ADR-104 phrasing** — is "Co-exist, two outputs, two purposes"
   acceptable, or do you want me to push harder on retiring RFC-097
   chunk 9's KG-direct path? (Latter is more invasive but architecturally
   cleaner. Recommendation: keep both per (A).)
2. **NLI model choice** — DeBERTa-v3-small (~80MB, CPU-OK, MNLI) is the
   default recommendation. If you'd rather host on DGX via the existing
   vLLM autoresearch stack (Qwen3-30B in-context NLI), say so — that's
   a different chunk-4 plan.
3. **Phasing cadence** — 7 chunks, each its own PR, merged sequentially
   into main? Or one big `feat/enrichment-layer` integration branch with
   chunks as commits, single PR at the end? RFC-097 used the latter and
   it worked well for review; default to that here unless you prefer
   smaller PRs.
4. **`make enrich` standalone target** — RFC-088 §Open Question #1 says
   "likely yes for Phase 2". Confirm: ship the standalone target in
   chunk 1 with the CLI, or defer to chunk 7?
5. **GitHub issues** — per `[[feedback_never_open_gh_issues]]`, I won't
   open any. Want one umbrella issue for the layer + 7 child issues per
   chunk? Or stay branch-only and let the PRs carry the narrative?

---

## Estimated total

7 chunks. Doc/code split roughly 25/75. Total ~3000–4500 LOC of code +
~3500–5000 LOC of tests. Mostly mergeable in a 2–3-week elapsed window
if chunks land sequentially with one-day reviewer turnaround. The
deterministic baseline (chunks 0–2) is the highest-value milestone —
gets `topic_cooccurrence_corpus` / `temporal_velocity` /
`grounding_rate` on disk and consumable, and unblocks PRD-026 Topic
Entity View work without needing the embedding/ml/llm tiers.

If we have to land less than the full plan: stopping after chunk 2 gives
the bulk of the user-visible value; chunks 3–6 layer in capability
without rework.
