# Open backlog → v3 corpus + 10k episodes (post-hygiene, 2026-08-02)

The docs-hygiene pass pruned every DONE doc, so what remains under `docs/wip/` is genuinely open or
reference. This is the intent-vs-reality map: what we *planned and haven't built*, what we *believe
shipped but hasn't*, and how each relates to the north star — **the v3 corpus and scaling to 10,000
episodes**. It seeds the next-session plans; it is not itself a plan.

## The 6 critical-path gaps (the real next work)

1. **DGX-local model decision is not made.** `BAKEOFF-18EP-RESULTS.md`'s own ordering says the
   Ollama/DGX-local wave runs *after* the cost-instrumentation fix — and that arm, the one that
   actually decides the v2.5 Gemini→DGX swap, **has not run**. This is Phase D of the v2.5 handover.
   → run the DGX-local bake-off arm; it gates the whole v2.5 corpus.
2. ~~**Two unreconciled "north star" plans.**~~ **RESOLVED 2026-08-02** — the v2.5 handover was
   folded into `1000-EPISODES-REPROCESS-PLAN.md` (now the single canonical arc doc; the 2.5 corpus
   is its current stage, not a separate plan) and deleted.
3. **Corpus-growth strategy is undecided** — `ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md` has no target
   size, no curated-overlap-vs-broad-ingest call, no onboarding mechanics. This is the actual lever
   from ~100/209 episodes to 10,000. → decide size + show-selection strategy (feeds Corpus Scout).
4. **Host identity is unbuilt, not just imperfect.** `EPIC-HOST-IDENTIFICATION.md` is fully specced,
   **zero code** (`person→HOSTS→podcast`, `/api/relational/shows`, host scorecard). At 10k eps the
   per-episode coverage heuristic it replaces won't scale. → build Slice-0 gold set + scorecard.
5. **The reprocess acceptance gate isn't finished.** `CORPUS-V4-FIXTURE-LADDER.md` / #1189 (OPEN):
   the eval↔production transcript-variant parity gap is unresolved. This is the gate meant to catch
   regressions during the 1000-episode reprocess. → freeze the v4 fixture ladder + close the parity gap.
6. ~~**Full-corpus diarization run status is UNVERIFIED.**~~ **RESOLVED 2026-08-02** — operator
   confirmed the full-corpus community-1 diarization was done at the ~90-episode scale in v2.2
   (PR #1335); the corpus carries uniform RTTM / speaker counts. `diarize-full-corpus-run.md` deleted.

## QUALITY tier — corpus quality levers (not blocking, high leverage)

- **Enricher hardening mid-flight** — `ENRICHER-HARDENING-ROADMAP.md`, PR-C = #1168 (OPEN); per-enricher
  emission tests + fast-CI wiring remain.
- **Speaker-resolution tail** — `SPEAKER-RESOLUTION-ROADMAP.md`: the ~113 un-introduced panel guests
  (this session's #1286 measured them embeddings-only) + promote the measurement scripts to a `make` gate.
- **Diarization split/merge cause** — `DIARIZATION-SPLIT-HOST-CLUSTER-MERGE.md`: #1330 contained it at
  the naming layer; the diarization-layer cause is deferred (needs a Tier-2 matrix row).
- **Graph perf debt (#1219)** — `graph-tech-debt.md`: fcose/topDown costs compound as node counts grow
  toward 10k episodes.
- **Super-theme signal comparison**, **dual-Whisper future uses (#1046)** — parked QUALITY experiments.

## ADJACENT workstreams (parallel, not on the corpus critical path)

- **Go-live (Goal-1)** — further along than memory suggested: player + operator LIVE on `main`, orrery
  split with its own edge. Remaining is a narrow ops tail (CF WAF apply, alert wiring, RBAC split —
  #1160/#1161/#1164/#1165 open, INFRA-HARDENING #1160). **None argued as blocking corpus/scale.**
- **Observability** — `observability-app-surface-plan` + `-correlation-id-enhancement`: phased, in
  progress; matters operationally at scale but doesn't gate the corpus.
- **Knowledge-retention package (12 docs, all PLANNED-UNBUILT, zero code)** — a whole future product.
  Every placeholder PRD/RFC number **collides with a real, different doc** — must be renumbered before
  any merges. One piece is a forward dep worth pulling toward the arc: **Corpus Scout** (`PRD-037` /
  `RFC-088` in that package) is feed-prospecting tooling that directly serves gap #3 (10k growth) but
  is currently gated behind the rest of the unbuilt cluster.
- **One sleeper** — `wip-concurrent-pipeline-http-retry-metrics.md`: the metrics bug that forces
  one-run-per-process; relevant if the 10k throughput plan wants concurrent same-process runs.

## Additional prune / re-triage candidates the deep read surfaced

- `1175-LOCAL-CORPUS-PORT.md` — **DELETED 2026-08-02** (#1175 closed, scripts + Make targets exist).
- `EXPLORE_EXPANSION_IDEAS.md` — **DELETED 2026-08-02** (all 5 ideas in `cli.py`, #601/#597 closed).
- `SEARCH-V3-IMPLEMENTATION-PLAN.md` — **DELETED 2026-08-02** (S0–S8 shipped `e590887f`/#1274; RFC-107 is the permanent home; 3 referrers repointed to it).
- `LORA_HYBRID_PIPELINE_PLAN.md` — parent epic #907 CLOSED, LoRA out of scope per memory. **(Kept open per operator ask.)**
- `SPEC_KG_GI_ONTOLOGY_V3_WISHLIST` — its gating (#1036 closed) passed; a *different* v3 (KG/GI schema, not corpus). → re-triage.

## Kept OPEN for your review (not deleted)

- `SPEAKER-PIPELINE-SUBSYSTEM-AUDIT.md` — 3/4 Tier-1 items fixed by naming-4; D1/D2 + Tier-3/4 unverified.
- `LORA_HYBRID_PIPELINE_PLAN.md` — see above.

## NOT verified (equal weight)

Sub-agent flags not independently re-checked: DGX SSH reachability, Storage Box provisioning state,
whether the full diarization run executed, whether `AUTORESEARCH_EVAL_PLAYBOOK` issue numbers still map
to real issues, and the content-compatibility of the real docs the knowledge-retention placeholders
collide with. Treat these as open, not resolved.
