# Open backlog → v3 corpus + 10k episodes (post-hygiene, 2026-08-02)

The docs-hygiene pass pruned every DONE doc, so what remains under `docs/wip/` is genuinely open or
reference. This is the intent-vs-reality map: what we *planned and haven't built*, what we *believe
shipped but hasn't*, and how each relates to the north star — **the v3 corpus and scaling to 10,000
episodes**. It seeds the next-session plans; it is not itself a plan.

## The 4 live critical-path gaps (2 of 6 closed — see foot of section)

1. **DGX-local model decision is not made** — the current front (v2.5 stage D). The bake-off arm
   that decides the v2.5 Gemini→DGX swap **has not run**. → run it; it gates the whole v2.5 corpus.
   **Entry-state verified 2026-08-02 (read-only SSH):** DGX `dgx-llm-1` reachable (key auth OK, ping
   0% loss). **The DGX was re-provisioned since the plan was written** — the autoresearch-vLLM
   bring-up path Phase D assumes is GONE: no `~/agentic-ai-homelab` repo, no `gpu-mode-swap.sh`
   anywhere under `~`//opt//srv, and NO autoresearch/coder-next vLLM container; `:8003` serves nothing.
   What IS running (persistent containers): `moss`, `pyannote`, `faster-whisper` (all ~10 d),
   `librechat-*` (3 d), obs stack; `/opt` holds native `moss-server`/`pyannote-server`/`faster-whisper`/
   `speaches-gb10`/`llm-models`/`actions-runner`; ollama `:11434` up (llama3.1:8b).
   **BLOCKER for the fresh session: how the bake-off LLM is served on the re-provisioned box is an
   open operator call** (moss? librechat? a new vLLM?) — Phase D's serving assumptions + the stale
   `reference_gpu_mode_swap_script` / `project_dgx_vllm_distinction` / `project_dgx_tailscale_acl`
   memories need re-confirmation before the bake-off runs.
   Local side present & verified: `autoresearch/` harness (JUDGING.md, PER_MODEL_OPTIMAL_PARAMS.md,
   bundled_prompt_tuning), `scripts/backfill/relabel_corpus.py`, DGX profiles (`prod_dgx_balanced`,
   `prod_dgx_full_with_fallback`, `cloud_with_dgx_primary`), parity baseline corpus
   `.test_outputs/manual/prod-v2/corpus` (4.5G, 90 audio, 10 feeds).
2. **Corpus-growth strategy is undecided** — `ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md` has no target
   size, no curated-overlap-vs-broad-ingest call, no onboarding mechanics. This is the actual lever
   from ~90 episodes to 10,000. → decide size + show-selection strategy (feeds Corpus Scout `PRD-037`/`RFC-088`).
3. **Host identity is unbuilt, not just imperfect.** `EPIC-HOST-IDENTIFICATION.md` is fully specced,
   **zero code** (`person→HOSTS→podcast`, `/api/relational/shows`, host scorecard). At 10k eps the
   per-episode coverage heuristic it replaces won't scale. → build Slice-0 gold set + scorecard.
4. **The reprocess acceptance gate isn't finished.** `CORPUS-V4-FIXTURE-LADDER.md` / #1189 (OPEN):
   the eval↔production transcript-variant parity gap is unresolved. This is the gate meant to catch
   regressions during the 1000-episode reprocess. → freeze the v4 fixture ladder + close the parity gap.

**Closed since first draft (2026-08-02):** ~~north-star reconcile~~ (v2.5 handover folded into the
canonical `1000-EPISODES-REPROCESS-PLAN.md`, handover deleted) · ~~full-corpus diarization run~~
(done at ~90 eps in v2.2 / #1335; `diarize-full-corpus-run.md` deleted).

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
- `SPEC_KG_GI_ONTOLOGY_V3_WISHLIST` — **PROMOTED 2026-08-02** to `docs/architecture/corpus/ontology-v3-forward-look.md` (RFC-097 depends on it as v3 spec input; RFC-097 repointed to the permanent home).

## Kept OPEN for your review (not deleted)

- `SPEAKER-PIPELINE-SUBSYSTEM-AUDIT.md` — 3/4 Tier-1 items fixed by naming-4; D1/D2 + Tier-3/4 unverified.
- `LORA_HYBRID_PIPELINE_PLAN.md` — see above.

## NOT verified (equal weight)

Sub-agent flags not independently re-checked: DGX SSH reachability, Storage Box provisioning state,
whether the full diarization run executed, whether `AUTORESEARCH_EVAL_PLAYBOOK` issue numbers still map
to real issues, and the content-compatibility of the real docs the knowledge-retention placeholders
collide with. Treat these as open, not resolved.
