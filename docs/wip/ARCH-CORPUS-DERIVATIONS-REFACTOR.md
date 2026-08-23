# Corpus-derivations refactor — the hang + the Frankenstein

**Date:** 2026-08-23 · **Trigger:** the `substack:post:178618026` repair exercise (#1757) —
first successful episode repair, which surfaced two structural problems in the pipeline's
corpus-level finalize. Both are `src/` changes → require a full image rebuild → Stack test →
redeploy, so they are batched into one "next cycle" (see § Batched scope).

Status: DRAFT for advisor (Fable 5) stress-test, then scope.

---

## Design principle (operator, 2026-08-23) — the governing rule

**Every corpus-level derivation (vector index, topic clusters, corpus enrichment) has TWO modes,
symmetric across all three, exactly like the index already does:**

1. **Incremental / partial — the DEFAULT, in-pipeline.** Operates only on the *set of episodes in
   the current run*. A 1-episode repair → delta reindex + delta recluster + delta reenrich. Fast,
   per-episode, no whole-corpus work. This is what a normal run/repair does.
2. **Full / global — an EXPLICIT, invocable operation.** Like a full reindex today, we also want a
   `re-topic-cluster` and a `re-corpus-enrich` that rebuild the whole corpus, run *only when asked*
   (model/threshold change, schema migration, corruption recovery).

The bug today is that clustering (and, in effect, enrichment) run the **full/global** path on
**every** run — a whole-corpus rebuild triggered by a 1-episode change. The index does not make
this mistake (it deltas by default). Bring clustering + enrichment to the same shape.

The two problems below are both instances of "this derivation is missing its incremental mode /
runs the global mode by default."

---

## Problem 1 — topic clustering hangs the finalize (O(n³) whole-corpus rebuild)

### Elaboration

`build_topic_clusters_for_corpus` → `topic_clusters.cluster_indices_by_threshold`
(`src/podcast_scraper/search/topic_clusters.py:277–300`) is a **hand-rolled agglomerative
clustering** with no memoization:

```python
clusters = [{i} for i in range(n)]          # n singleton clusters
def mean_inter_cluster(ci, cj):             # O(|ci|·|cj|)
    for a in ci:
        for b in cj: tot += sim[a, b]       # recomputed from scratch every time
while len(clusters) > 1:                     # ~n merges
    for i in range(len(clusters)):           # × O(clusters²) pairs
        for j in range(i+1, len(clusters)):
            s = mean_inter_cluster(...)       # × O(|ci|·|cj|), never cached
    # merge best pair
```

- Complexity ≈ **O(n³)** (worse with recomputation), over the **whole corpus's** topic vectors
  (678 episodes → thousands of topic vectors).
- Observed live: **100% CPU, flat memory (~1 GiB), zero output for 24+ min** → the n×n similarity
  matrix is built once, then the merge loop just burns CPU. Classic CPU-bound O(n³) signature.
- **Not** a deadlock/infinite loop — it *would* finish, in hours-to-forever, and it degrades every
  time the corpus grows. At current scale it is indistinguishable from hung. Every prior run was
  killed here; nobody has ever let it complete.
- Root smell: it is a **whole-corpus rebuild triggered by a 1-episode change**, run **inline** on
  every pipeline run (a repair, a nightly, anything).

### Solution outline

1. **Replace the algorithm** with a library routine: `scipy.cluster.hierarchy.linkage` +
   `fcluster` (average linkage on the cosine-distance condensed matrix) or sklearn
   `AgglomerativeClustering(metric=…, linkage="average", distance_threshold=…)`. O(n²) memory,
   O(n² log n) time, C-optimized. This alone removes the hang at current scale.
2. **Make it incremental (delta):** only recompute clustering for topics whose vectors changed
   this run (mirror `IndexRunStats`: `topics_scanned / skipped_unchanged / reclustered`). A
   1-episode repair should recluster only the affected neighborhood, not all 678 episodes.
3. **Bound it:** a hard cap / timeout with a logged "clustering skipped: N topics over budget"
   so a pathological input degrades to a warning, never a silent 24-min spin.
4. Guardrail test: a Tier-2 matrix row that runs clustering on a corpus-scale fixture and asserts
   it completes under a wall-clock bound (regression guard for the O(n³) return).

---

## Problem 2 — the "Frankenstein": mixed patterns for corpus-level derivations

### Elaboration

The pipeline derives several corpus-level artifacts after per-episode processing, and each uses a
**different execution pattern**:

| Derivation | How it runs | Incremental? | State |
|---|---|---|---|
| per-episode (ASR/diar/summary/GI/KG edges) | inline | yes (per episode) | ✅ good |
| vector index | inline finalize | **yes** (`reindexed=1, skipped_unchanged=677`) | ✅ good |
| topic clustering | inline finalize | **no — full-corpus O(n³)** | ✗ Problem 1 |
| corpus enrichment | **async side-car** (`_maybe_spawn_enrichment_after_pipeline` enqueues a `corpus_enrichment` job) | delta-capable (`enrichment/staleness.py` mirrors `IndexRunStats`) | ✗ **enqueued but did not run** |

Two are inline+delta, one is inline+full-rebuild, one is decoupled-async. Worse — the async one is
**silently unreliable**: this run enqueued `corpus_enrichment` at 22:30:19, but the latest
enrichment `run_summary.json` is **2026-08-14** (8 days stale). The episode was transcribed and
indexed but **never corpus-enriched**, and nothing surfaced that. (Relates to open issue #1811,
"reprocessing silently drops enrichment.")

### Solution outline

**Apply the two-mode principle (above) to every derivation: delta-by-default in-pipeline, full as
an explicit op — all three symmetric, all reporting the same `{scanned / skipped_unchanged /
changed}` shape.**

1. The index already has both modes; enrichment's `staleness.py` already has the delta machinery —
   so most of the pattern exists.
2. **Clustering (Problem 1):** add the incremental mode (delta on the run's episode set) as the
   default; keep the full rebuild ONLY as the explicit `re-topic-cluster` global op (and fix its
   algorithm, Problem 1).
3. **Enrichment:** run the **delta** enrichment **inline** in the finalize (on the run's episode
   set) — retire the always-on async side-car that silently no-ran. Provide `re-corpus-enrich` as
   the explicit full/global op. A repair then does, in one finalize: `reindex(+1)`,
   `recluster(delta)`, `reenrich(+1)`.
4. **The one real trade-off to stress-test:** enrichment can be heavy (LLM calls via the gateway),
   which is *why* it was pushed async originally. But delta-inline means a 1-episode repair is 1
   episode of enrichment (cheap) — the heavy case is only the explicit full/global op, which *can*
   stay async **but with a reliable, observed promotion + a "did it run?" gate** (never a silent
   side-car). Advisor: is inline-delta enrichment fast enough for the normal N-episode run, or does
   even the delta path need the async tier?

---

## Advisor (Fable 5) stress-test — verdict (2026-08-23)

The review changed premises. Key findings:

- **F1 — enrichment is NOT LLM-heavy.** No enricher calls the gateway; it's local ML (MiniLM +
  DeBERTa NLI + sentence-transformers). Cost is model load + corpus-wide passes, not LLM latency.
- **F2 — "delta reenrich(+1)" is ill-defined.** 7 of 9 enrichers are `EnricherScope.CORPUS`; only
  `insight_density` + `insight_sentiment` are episode-scope. A 1-episode "delta" still pays the full
  corpus-scope Phase 2 (~7-8 min by manifest). So inline-delta enrichment is NOT cheap.
- **F3 — the silent no-run is (mostly) a paused queue, CONFIRMED on prod.** `.viewer/jobs.paused`
  has existed since **Aug 19 12:18**; the sweeper logs "drain paused… 0 queued jobs waiting" every
  30s. Enrichment is 8-days-stale because the drain was paused — a **visibility bug, not an
  architecture bug**. Deeper wrinkle found on unpause (2026-08-23): removing the flag promoted
  nothing because **the pipeline's "enqueued corpus_enrichment" never persists a queued row** — the
  real silent-drop mechanism (a `docker compose run` reprocess enqueues to a registry the API
  sweeper doesn't see, or the enqueue is dropped). This is the teeth behind #1811.

Answers (condensed):

1. **Delta clustering** is well-defined only as a conservative one-directional approximation and is
   *mostly unnecessary* — once the algorithm is O(n²), a full rebuild over ~5-15k topics is
   seconds. Right "incremental" = a **skip-gate** (fingerprint topic rows; skip if unchanged), the
   index's shape, no partition-maintenance risk. ~~Delta-assign is a later scale escape hatch
   only.~~ *(Superseded 2026-08-23: RFC-118 formalises the delta backbone; clustering keeps the
   skip-gate — an outer empty-delta gate in the finalize plus the inner row-fingerprint gate —
   and delta-assign stays a non-goal there too.)*
2. **scipy `linkage(average, cosine)` + `fcluster`** is a **true UPGMA drop-in** (proven from the
   math: mean-cosine-dist = 1−mean-cosine-sim, average linkage is monotonic → cut at `1−threshold`
   = the greedy loop's partition). Swap only `cluster_indices_by_threshold` internals. **Hazard:**
   tie-break / iteration-order changes reshuffle `tc:` slug suffixes → break stored per-user
   interest keys → add **deterministic cluster ordering**. Declare `scipy` in `[search]` (likely
   already transitive via sentence-transformers; verify).
3. **Do NOT inline enrichment.** The side-car isn't the bug, the unobserved paused queue is. Keep
   queued-async; fix reliability with a **gate**: record `job_id` in `run_summary`; alert when a
   `corpus_enrichment` row is queued too long OR when enrichment `run_summary` is older than the
   newest gi/kg mtime; make `jobs.paused` LOUD (not a 30s INFO nobody sees).
4. **Ordering:** `enrich-edges → index → clusters`; enrichment depends only on gi/kg, reads neither
   clusters nor index. No cycle. Async-after-finalize avoids the repair race by design (#1653).
5. **No corpus migration.** Exposed surface = `tc:` id churn (mitigate with deterministic ordering +
   an old-vs-new cluster-id diff printed by the explicit `re-topic-cluster` op). MVP = clustering
   swap + enrichment observability gate; the full symmetric refactor is only-if-measured.

**Biggest risk:** inlining corpus-scope enrichment on the false "delta = 1 episode" premise.
**Two-mode verdict:** sound as an operator invariant, wrong as a literal "symmetric delta for all
three" — it's "incrementality where the math supports it, cheap-full where it doesn't, explicit full
op everywhere, skip-stats + did-it-run gate everywhere." Strike "retire the async side-car."

## Batched scope — the next rebuild cycle

Both fixes are `src/` → image rebuild → Stack test → redeploy (hours). Since we pay that once,
batch the other **rebuild-gated AND related** items. Filter: needs a `src/` change *and* touches
the pipeline / corpus-derivation / cost / reliability surface this run exercised.

### A. Core — split by the advisor's verdict (NO enrichment refactor)
- **A1 — Topic clustering (rebuild, `src/`).** Swap `cluster_indices_by_threshold` internals to
  scipy `linkage(average,cosine)`+`fcluster` (true UPGMA drop-in) + **deterministic cluster
  ordering** (prevents `tc:` id churn) + a **changed-topics skip-gate** + a Tier-2 wall-clock
  guardrail row (ADR-095). Kills the hang; no schema/migration. Declare `scipy` in `[search]`.
- **A2 — Enrichment reliability slice** (advisor Fable 5, 2026-08-23). NOT a refactor; keep the
  queued-async design. But it's bigger than first thought: the ADR-108-validated `topic_consensus`
  has **never run in prod** (incomplete rollout, not a deliberate disable). Five defects, ordered:
  1. **Honest coverage (do first — the meta-defect).** Enabled-but-not-run enrichers leave NO row
     in `run_summary` and the run still reports `status=ok` (`registry.py:75-96`, `run_summary.py`,
     `executor.py:861-880` only fixed the zero-enricher case). An ENABLED enricher that didn't run
     → a `not_registered`/`not_admitted`/`timeout` row + the run must NOT aggregate to `ok`. This is
     what let the partial reenrich report green. `src/` → rebuild.
  2. **Ship gate_metrics.json into the image.** `.dockerignore:41` excludes `data/`; confirmed
     `/app/data/eval NOT in image` → `topic_consensus` (`on_missing_data="reject"`) self-rejects
     in-container forever. Ship the eval data into the wheel/image OR fix `_default_eval_root`
     (`admission.py:122-124`). `src/`+build → rebuild.
  3. **`--with-ml` on reenrich-prod + a survivable ML timeout.** Add `--with-ml` (the CLI's designed
     contract — the auto-path passes it via `needs_ml`), but PAIR it with raising/scaling the ML
     manifests' `expected_duration_s` (120s/180s → won't survive 678 episodes + cold model
     download) and an **HF cache volume** for pipeline-llm (no cache mount today → downloads every
     run inside the timeout). Workflow + compose → mostly box-deployable.
  4. **Enqueue-never-persists** (defect 2): box shows 0 today rows. Root cause open — get the next
     reprocess's job log (`orchestration.py:1923-1937` enqueued-vs-could-not; `jobs.py:682-688`
     coalesce). Investigation first, then fix.
  5. **Pause alert** (lowest): `.viewer/jobs.paused` blocks drain with only a 30s INFO nobody sees —
     surface it + alert when enrichment `run_summary` is older than newest gi/kg mtime. o11y config.
  **Subsumes #1811.** Removing `--litellm-api-base` (already done) was correct, not a bug.
- **Kept as explicit ops:** `re-topic-cluster` / `re-corpus-enrich` full/global ops; the
  `{scanned/skipped_unchanged/changed}` stats symmetry (`EnrichmentRunStats` already mirrors
  `IndexRunStats`).

### B. Batch-in — rebuild-gated + directly related (do in the same image)
- **#1752** — "reprocess ran 6h, 1600+ LLM calls, changed NOTHING." Same no-op class; the two-mode
  refactor + the delta-reporting (`changed=0` is loud) is the fix. Core-adjacent.
- **#1809** — Deepgram retries billed per attempt but priced once (run-budget ledger undercounts).
  We leaned on `cost_cap_usd=2` this run; the ledger must be honest. Cost-accounting `src/`.
- **#1808** — audio sweep cost never measured. Cost/o11y `src/`.
- **#1810** — nothing stops two pipeline runs sharing one corpus. Concurrency guard on the
  ingest/finalize path. Safety `src/`.
- **#1757** — safety release: the cost cap didn't fire in a prior run. Cost-cap correctness — same
  ingest path; verify with our run (priced $0.32 under $2, but confirm the gate math).
- **#1789** — audio provenance only stamped by `archive backfill`, not download/reprocess. Storage
  `src/`; our repair downloaded fresh audio, so provenance applies.

### C. New learnings from THIS run (not yet issues — for the batch)
- **Topic-clustering O(n³) hang** → Problem 1 (the concrete, live-repro'd cause). File as issue?
- **Enrichment enqueued-but-never-ran, confirmed live** (8-day-stale run_summary) → hard evidence
  for #1811; the "did it run?" gate is mandatory.
- **Noisy red-herring log**: `reprocess_existing_only is set without reprocess_source … (no-op)`
  fires for every feed even in worklist mode where the work-list DID match — it misdirected 4
  debugging sessions. Suppress/scope it in worklist mode. Trivial `src/`.
- **o11y structured labels + traces** for pipeline runs: `run_id` / `episode_ids` as log labels +
  VictoriaTraces stage spans (currently `trace=-`). `src/` (emit side). Batch as an o11y-quality
  slice.

### D. NOT this cycle — box-deployable (no rebuild; do separately, anytime)
- Drop the box-log tees in `reprocess-prod.yml` / `sweep-prod-audio.yml` (workflow YAML).
- Per-container CPU metric labeled by container name / a hang alert (Alloy/cAdvisor + Grafana config).
- **#1804** flock unification (workflow). **#1805** ci-fast local-green (dev-box). Deploy-side.

### Deliberately out of scope for this cycle (bigger, own cycle)
- ops-api epic (#1687–#1692, #1785) — converting reprocess/gi-repair to queued jobs is a large
  orchestration change; it *rhymes* with the two-mode refactor (one execution model) but is its own
  arc. Note the overlap; don't fold it in blind.
- Wave/quality epics (#1682–#1686, #1801) — content quality + entity resolution; separate.
