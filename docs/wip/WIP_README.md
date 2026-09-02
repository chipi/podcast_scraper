# Work In Progress (WIP) Documentation

This folder holds early-stage notes, backlog plans, and reference material that has
not been promoted to a PRD, RFC, ADR, runbook, or release note. Content here is
**not authoritative** — when an item is ready, it moves to its proper home and the
WIP file is removed.

## Promotion targets

- Long-term feature ideas → **PRD** in `docs/prd/` and the relevant **RFC** in `docs/rfc/`
- Architectural decisions → **ADR** in `docs/adr/`
- Operator-facing procedures → **guide** in `docs/guides/`
- Post-release retrospective → **fold the substance into** the corresponding release note in
  `docs/releases/` (do NOT leave it in WIP and link to it — see the rule below)

## References are ONE-WAY: permanent docs must never point back into `docs/wip/`

A WIP doc is deleted the moment its work lands, so anything permanent that links to it becomes a
dead reference. **No ADR / RFC / PRD / release note / guide / `docs/api/` doc, and no code / test /
README / comment / docstring, may reference a `docs/wip/…` path.** (WIP↔WIP is fine — a WIP set
travels together; and a WIP doc pointing *at* a permanent artifact is just naming its promotion
target.) When a permanent doc needs WIP content, **promote it** — copy the substance inline or cite
the durable source (shipped code, commit hash, issue/PR, or the ADR/RFC/PRD that superseded it).
This is the rule the 2026-08-02 hygiene pass had to retrofit; the templates and AGENTS.md carry it.

## Current contents

| File | Description | Status |
| ---- | ----------- | ------ |
| `ONBOARDING-RUNBOOK-200-700-EPISODES-2026-08.md` | Operator runbook for onboarding 200–700 episodes across new shows (`feat/onboarding-readiness` arc). Preconditions in order (merge → deploy → backfill → snapshot → health drill), measured budget math ($0.238/ep → $60–210 with retry allowance), cost-chunked batch sizing (~$10–25/dispatch, batch one small), per-batch 7-point verification checklist, abort paths (#1785 STOP endpoint → stop-prod-pipeline.yml → SSH), and an explicit NOT-covered section (ML enricher re-enable #1817, retroactive fingerprints, feed selection, consumer-side capacity). Every prod dispatch operator-gated. | Active (runbook draft) |
| `1192-1286-1285-guest-recall-arc-findings.md` | Measured findings for the guest-recall tail. **#1192**: the transcript-intro guest lever names **0 new guest voices** and causes **3 ASR name-flips** on the prod-v2 90-ep relabel — the roster's active on-air-intro + self-intro paths already exhaust text-reachable recall → **not shipped** (validated negative, like the #1228 revert). **#1286** (embeddings) is the only lever that reaches the ~113 un-introduced-panel tail; needs DGX. **#1285** (ASR name canonicalization) confirmed real by the 3 flips (Cerisier/Serissier, Karnal/Carnell, Allardice/Allardyce). | Active (findings) |
| `2026-07-26-production-security-ops-review.md` | Fable 5 advisor holistic production security + ops review across the three surfaces (player / operator / tailnet-only privileged plane). P0 origin-lock (subsequently verified applied), P1s (operator viewer rate-limit gap, `?grant=creator` authZ footgun, no operator-appdata/corpus backup schedule, public-repo bcrypt doorman), P2s, and an equal-weight not-covered/unknowns section. Analysis only. | Active (review) |
| `2026-07-26-security-ops-mitigation-plan.md` | Phased, owner-tagged mitigation plan for the review. Phase 0 verification done (origin-lock CONFIRMED applied via TF state — `:443` locked to CF ranges; `OPERATOR_SECRETS_VIA_FILES` off; `.com` in admin emails), Phase 1 in-repo P1 quick wins, Phase 2 hardening (mix me/operator), Phase 3 maturity. Drives the `security/prod-hardening-2026-07` PR. | Active (plan) |
| `observability-app-surface-plan.md` | Phased plan for full self-hosted o11y (metrics + logs) over the podcast operator surface (pipeline, LLM, ML, enrichers, player, search) plus orrery, grounded in a telemetry-emission recon. Phase 0 (infra collector, api `/metrics`, node + edge-security dashboards, Cloud-agent removal) done; Phase 1 = ship scoped structured app logs → VictoriaLogs + log-derived dashboards (LLM cost, pipeline, search); Phase 2 = app instrumentation for real-time ML/search/enricher metrics; Phase 3 = orrery; Phase 4 = alerts + retention. | Active (plan) |
| `1000-EPISODES-REPROCESS-PLAN.md` | **THE canonical arc doc** (reconciled 2026-08-02 — absorbed the former v2.5 handover). Reprocess v2→v3 with a fully-local (DGX) pipeline + expand to 500–1000 eps / 20–30 podcasts (10k horizon). Incremental single-variable arc v2.1→v2.5; **v2.2/v2.3/v2.4 MERGED** (#1335, #1355), **v2.5 = current front** = Gemini→DGX-local LLM swap (stages D bake-off → E swap → F freeze; gate = disjoint-vendor scalar judge-panel parity vs Gemini). Organizing principle = **reprocess-once economics**. v4 fixtures (#1189) as a growing harness. Open: go-live-vs-2.5 sequencing, parity-gate signal, expansion feeds (#630). | Active (plan) |
| `OPEN-BACKLOG-v3-10k.md` | Intent-vs-reality map of the remaining WIP docs vs the v3-corpus / 10k-episode north star. The 6 critical-path gaps (2 now RESOLVED: north-star reconcile + full-corpus diarization), quality/adjacent tiers, prune log. Seeds the next-session plans. | Active (map) |
| `DIARIZATION-SPLIT-HOST-CLUSTER-MERGE.md` | Follow-up behind the #1330 naming fabrications. The v2.1.x re-cascade surfaced two "fabricated person" classes that are DIARIZATION artifacts the naming layer now CONTAINS (verified by deterministic replay): a cold-open **montage cluster** ("I'm Kevin Russo… I'm Casey Noon…" merged into one 13s SPEAKER_NN → refused a self-intro name when short, via `distinct_self_introductions` + `MONTAGE_CLIP_MAX_TALK_S`), and a detected-guest **forced onto a bumper** ("Robert Pape" onto a 30s "We'll be right back" → surname-aware spare, honorific-restricted). Both fail toward unnamed. The real cause (merge/split of clusters) is a diarization-layer normalisation deferred to v2.2 (community-1); blast radius rewrites voice ids so it's its own branch. | Active (follow-up) |
| `SPEAKER-RESOLUTION-ROADMAP.md` | Roadmap to reduce unknown diarized speakers, with **before/after measurement per step** on the real prod-v2 corpus (90 diarized eps / 579 voices). Shipped: #1a (+41 voices), #1b (episode-scope), #2 (publisher denylist), #3 (host/guest role), voice_type classification, Step B known_hosts (+19 voices, cumulative 36.6%→47.0%), Step C (unnamed host→"Host" label), Step D (intro NER guest detection, +7 voices). 466/579 (80%) handled; 113 truly unknown remain (un-introduced panel guests). Open decisions: promote measure scripts to make target, chase talk-time threshold. | Active (plan + measured) |
| `GOAL1-ORRERY-CRITICAL-PATH.md` | Trimmed scope for the operator's real goal order (1 orrery on minimal common infra, 2 player, 3 operator surface). Pins what Goal-1 needs (live rollout + T-11 + orrery vhost coordination) vs what is PARKED (all player work) vs deferred (goal-3 privilege-split). | Active (scope) |
| `GOAL1-GO-LIVE-PLAN.md` | Owner-tagged (🧑 You / 🤖 Me / 🤝 Both) phased plan to take orrery live on the shared edge. Config is built but not applied to the running box (cloud-init ran pre-config). Recommends Option A (imperative-once, no rebuild → no data-loss); phases: safety net → converge box (firewall closed) → o11y live → pre-public gate → open firewall → orrery onboard → Cloudflare → post-live. Open decisions: apply path A/B/C, corpus location (volume vs boot disk), re-rebase timing. | Active (plan) |
| `GOAL1-PHASE3-PREP.md` | Phase 3 hard-gate prep: (a) pre-public-gate pre-walk for orrery — 8 items mapped to done (substrate: catch-all Caddy, metadata-egress, T-11 alerting) vs pending (orrery cap_drop / digest-pin / rollback rehearsal); (b) ADR-115 secrets cutover exact cutover-gate-ordered steps (age key → prod.enc.yaml → decrypt on box → add secrets overlay → drop plaintext keys). | Active (prep) |
| `GOAL1-AUDIO-ARCHIVE-ROLLOUT.md` | Audio-archive Storage Box (#1199) rollout/transition plan: current state (storage_box.tf gated off, `audio_storage_backend=local/remote`,`archive pull` CLI, reprocess-prod), data-at-risk analysis (corpus backed up daily; audio archive NOT covered + best-effort-irreplaceable), 7-step rollout after DR-green (provision → verify → backfill+verify → flip backend → e2e → backup policy → prune), new-command runbook. Folds in as go-live Phase 7. | Active (rollout) |
| `GOAL1-MERGE-TO-LIVE-ROADMAP.md` | Living from-here roadmap: the gated sequence from "infra PR merges to main" → orrery live (harden → push → PR → merge → DR-drill-green → go-live phases). Holds the invariant (merge applies nothing live) + the open-decisions tracker (apply path, corpus location, DR-drill gate, branch hygiene, merge style). Links `GOAL1-GO-LIVE-PLAN.md` for phase detail. | Active (roadmap) |
| `INFRA-HARDENING-PLAN.md` | Sequenced execution plan for the infra security hardening effort (T-01…T-12 from `docs/security/THREAT_MODEL.md`). Infra-first: Docker/orchestration → Host → Edge → App. Precedes the shared public edge (ADR-114). Tracked in one hardening issue. | Active (plan) |
| `1161-API-SEPARATION-ROUTE-INVENTORY.md` | #1161 reassessment **v2** (corrected): three surfaces — consumer player (`/api/app/*`, OAuth, settled), kg/gi web app (`gi-kg-viewer` = today the operator console driving the FULL privileged `/api/*`), operator control plane. Two gaps: (1) `/api/*` has NO per-request authz (role system `listener<creator<admin` wired only into `/api/app/*`); (2) `docker.sock` behind a public RBAC gate reintroduces T-01. Recommends splitting the backend by privilege (public-api no-sock vs control-api tailnet-sock, enqueue→drain). Key fork: does admin trigger privileged actions from the public web app? | Reassessment v2 |
| `EVAL_1016_metrics/vllm_metrics_*_phase2c.log` | Raw vLLM `/metrics` polls per candidate (input data for the canonical per-model param compendium, which lives at `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md`) | Reference |
| `PUBLIC-EXPOSURE-AND-PRIVATE-SPLIT.md` | What the **public** repo actually exposes, and the private-split options. Separates two boundaries that get conflated: **prod content** (transcripts, segments, GI artifacts — *never public, boundary holds*; the 30 "committed episodes" in `prod_validation_v1/` are symlinks, 4.8 KB of paths) versus **method** (456 eval configs, the 2,175-line autoresearch playbook, 49 scorecards, 148 WIP notes — *fully public*, operator accepted 2026-07-14, will revisit). Records the enforcement gap: the content boundary rests on `.gitignore` alone, with no pre-commit check — a measured shape+allowlist guard (0 false positives across 203 synthetic-fixture hits) is proposed but NOT built. Notes that history is permanent, so forward-only and retroactive purge are different decisions with wildly different costs. | Active (analysis, no action taken) |
| `1046-WHISPER-DUAL-MODEL-FUTURE-USES.md` | Parks 5 alternative uses of the dual-model machinery (dual-pass reconciliation, confidence-weighted NER, sniff-driven NER pre-pass, speculative pipeline, cross-model dispatch) — all align with the intelligence-extraction goal that the skip-deep gate violated. None queued; planning material for next session pickup. Includes the offline-prototype-able subset using saved transcripts under `data/eval/runs/1046-measurement-pass-2/`. | Backlog |
| `DIGEST-TOPICBAND-THREAD-UNSAFETY-ARM64.md` | Follow-up to the parallelisation ranking above. Concrete SIGSEGV (api container `exit 139`) inside `ThreadPoolExecutor.map` around `run_corpus_search` on macOS arm64 stack-test, faulthandler pointing at sentencepiece / torch C extensions. Documents the workaround (unconditional sequential `map()` in `corpus_digest.py` — same code path in prod and stack-test) and four options for a real fix (warm the tokenizer on api startup — recommended first; module lock; process pool; upstream wheel patch). | Followup |
| `super-theme-signal-comparison.md` | Queued: compare 3 signals for the super-theme rollup on top of `topic_theme_clusters` — cross-cluster topic-lift (shipped default in v1.1.0), centroid cosine, member Jaccard on 1-hop lift neighbourhood. Pick winner by editorial-read of super-theme labels on prod-v2. graph-v3 tier 7-1a follow-up. | Queued |
| `DGX_NEXT_STEPS.md` | Living strategic doc on what runs on DGX vs local, vLLM vs Ollama, offload decision frame | Living |
| `LORA_HYBRID_PIPELINE_PLAN.md` | LoRA + hybrid pipeline exploration | Idea |
| `manual-test-plan-gi-kg.md` | Manual GI/KG smoke checklist | Reference |
| `wip-concurrent-pipeline-http-retry-metrics.md` | Open documentation gap for `http_urllib3_retry_events` | Open |
| `wip-topic-clusters-validation-reference.yaml` | Reference topic-cluster validation config | Reference |
| `player/mockups/` | Phone mockups (HTML + PNG) of the three explored Player aesthetics; **Direction B (Editorial Bold)** adopted → UXS-011. Design aids, not shipped assets. | Reference |
| `NER_FP_SAMPLE_LABELLED_2026-06-24.json` | Operator-labellable 50-row sample produced by `scripts/dev/measure_ner_mentions_diff.py` against the prod-v2 corpus; backs the TP/FP claim in the determinism investigation doc | Reference |
| `enrichment-visual-inspection-plan.md` | 4-stage plan for restarting the viewer against a freshly-enriched small corpus and inspecting where each RFC-088 enrichment signal surfaces in the player/viewer UI (uncovering UX gaps). | Active |
| `ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md` | **THE canonical corpus-expansion doc** (current state refreshed 2026-08-29). How to grow the **eval** corpus with more real shows so the enrichers produce visible value. Value model per enricher; highest lever = topic OVERLAP across shows. **Read order: §5f** (final 15+10 feed list, all RSS verified) → **§5g** (smoke→assess→deepen protocol + buckets) → **§5i** (evidence-based thresholds; `bridge_partition.both` is the primary signal) → **§5j** (live corpus = 14 feeds / 765 eps; probe group 1 all DEEPEN; 10 Batch A feeds still to go). **§1 and §6 are STALE** and marked as such — §6 is the pre-verification sketch and sits after §5f purely by append order. Expansion vehicle = `#630`. | Active |
| `HANDOVER-2026-08-29-batch-a-remainder.md` | Executable runbook for the next batch: ingest **10 episodes each** for the **10 remaining §5f Batch A feeds**. **Unit of work = 100 episodes (10 eps x 10 feeds), a REPEATABLE pass — not "onboard ten feeds".** Run three times: 08-30 08:33, 08-31 09:05, 09-01 22:41 (in flight). `episode_selection=unprocessed` means each pass takes 10 NEW episodes, so the ten feeds sit at 20 eps and go to 30 — corpus 24 feeds / 966 eps, fully local at $0. **Steps 1–2 are marked superseded**: the `PUT /api/feeds` merge was never a prerequisite, and the selection control is **`episode_selection`** (per-request, `998d5312`), not `episode_order` — see `docs/guides/INGESTION_RUNBOOK.md`, now canonical. **Steps 3–4 are the live work**: no Batch A feed has a §5i grade or a §5g bucket yet. Open + unexplained: KG `node_count` pinned at exactly 29 across 7 episodes; `insight_salvage` discarding insights 26–30 by arrival order. | Active |
| `consumer-node-view-backend-followups.md` | Two backend follow-ups from the `feat/consumer-remember` node-view review round, each shipped as a viewer-side approximation: (1) **precise per-show host/guest** — viewer infers Host from episode coverage (`3809300a`); accurate version needs a pipeline `person→hosts→show` edge from feed/author metadata. (2) **out-of-slice insight rendering** — mention drill (`566a9f69`) resolves in-slice insights; corpus-wide CIL mentions need a `GET /api/relational/insight-detail` endpoint (insight text + supporting quotes off the full server graph) + a small `InsightNodeView` + gate extension to `Insight`. | Backlog |
| `EPIC-HOST-IDENTIFICATION.md` | Epic spec: reliably know every show's host(s), persist as metadata + a `person→HOSTS→podcast` graph edge, and render host-aware behaviours (graph/digest/library/person card). TDD spine = a **host scorecard** (Slice 0: gold set + coverage/precision/recall/sample-bias metrics) that every path must move toward target. Paths A(per-show edge)→C(show-notes parsing)→B(voice embeddings)→D(multi-guest)→E(role taxonomy). Supersedes the "precise per-show host/guest" follow-up. | Specced |
| `PLAYER-GOLDEN-WALKTHROUGH-v3.md` | **v3** — supersedes v2. Same exhaustive player walk, folding in the new consumer **"Where they agree"** consensus row (`topic_consensus` is no longer operator-only): §2.3 person card + §3/§4/§5/appendix updated, with a **fresh** person-card capture (Nic Harrigan, live consensus row) in `assets/player-walkthrough-v3/` (8 unchanged surfaces reused). Cross-refs the operator show-landing Signals band (RFC-104/UXS-015 Phase 2). | Active |
| `ENRICHER-HARDENING-ROADMAP.md` | Fixtures→coverage→surfaces roadmap grading every enricher surface/cross-link against a weak/good/excellent rubric. PR-A (v3 `topic_similarity` fixture + invariants) + PR-B (per-enricher emission non-degeneracy + consensus executor smoke) **landed** on `fix/adr108-enricher-surfaces`; PR-C = **#1168** (operator-viewer served-corpus e2e harness); **PR-D reconciled 2026-07-09** — the "thin" surfaces were already built + tested (data-absence, not code-absence), so PR-D is capture+verify, not build. | Active |
| `graph-v3/HARDEN-FOLLOWUPS-2026-07-17.md` | 2026-07-17 harden pass on `feat/graph-v3` surfaced 3 items self-noted as "untracked" inside other WIP docs (USERPREFS-1 three deferred items, aggregatedEdges V1 enricher-gate, Speaker + Quote shape live-verify). Consolidates them so they're findable by title; no GH issues opened per operator's standing rule. HD1–HD9 harden fixes landed on the branch and are documented in `graph-v3/SUMMARY.md` § "Post-tier-8 harden follow-ups". | Active |
| `graph-tech-debt.md` | Running log of graph-viewer improvements surfaced during PR work but deferred as out-of-scope. Template + convention for adding new items; two entries from PR #1207 (residual +408 ms wave-1 fcose cost, `quality: 'draft'` as documented off-lever). Launching pad for future perf sessions. | Active |

## Guidelines

- Documents here are **not** part of the official documentation site.
- Documents may be incomplete, outdated, or experimental.
- Periodically review and either:
  1. Promote to the appropriate doc category (PRD / RFC / ADR / guide / release note).
  2. Delete if obsolete or superseded.
  3. Keep as backlog / reference if it still has signal.

## Recent cleanups

- **2026-06-24 second pass** — Removed 7 more notes for shipped/superseded
  work (47 → 40 files):
  - **Autoresearch programme planning docs** (2):
    `AUTORESEARCH_NEXT_PHASE_AGENT_PLAN.md`,
    `AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md` —
    #907 + #927 epics + children all closed per
    `[[project_autoresearch_programme]]`; plans no longer live-bearing.
  - **Plans for shipped features** (3): `RESEARCH_POWERED_REGISTRY_PLAN.md`
    (registry promotion shipped in #1060),
    `VLLM_RELOCATION_TO_HOMELAB_REPO.md` (homelab repo already owns
    `/opt/vllm-autoresearch/docker-compose.yml`),
    `EVAL_1016_LANDSCAPE_2026_06.md` (#1016 children all closed; raw
    metrics dir retained as historical input under `EVAL_1016_metrics/`).
  - **Superseded measurement notes** (2):
    `VLLM_GB10_TUNING_VALIDATION_2026-06-18.md` (canonical compendium
    is `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md`),
    `FIXTURE_AUDIO_TOOLING_COMPARISON_2026_06_13.md` (tooling choice
    landed in shipped fixture pipeline).
  - **Ghost-row cleanup**: 2 entries (`967-interaction-cost-trace.md`,
    `POST-MIGRATION-GRAPH-VALIDATION-967-974-876.md`) were in the README
    table but their files were deleted on 2026-06-16. Table now matches
    disk.
- **2026-06-24** — Pruned 25 stale notes (71 → 47 files):
  - **Done/closed work** (4): `V27-OBSERVABILITY-SCOPE-803-805-426_2026-06-21.md`
    (#803/#805 in #1047, #426 → #1052), `CLOUD-PROVIDER-RESILIENCE-E2E-GAP-1003.md`
    (#1003 closed), `1046-WHISPER-MULTI-MODEL-DESIGN.md` (gate rejected,
    parked ideas live in sibling future-uses doc), `RFC097_CHUNK9_PLAN.md`
    (closed by #1073, see ADR-101).
  - **Session/handoff notes ≥3 days old** (5): `SESSION_BRIEFING_2026-06-19.md`,
    `SESSION_2026-06-20_1033_FOLLOWUPS_STATUS.md`,
    `SESSION_2026-06-20_FINAL_STATUS.md`,
    `OPERATOR_HANDOFF_NOTES_2026-06-21.md`, `NEXT_SESSION_PLAN.md`
    (2026-06-13).
  - **Planning docs for shipped work** (5):
    `NEXT_BATCH_REGISTRY_RUNTIME.md` (registry promotion in #1060),
    `SPEC_1035_NER_PREPASS_DESIGN.md` + `EVAL_1035_NER_PREPASS_VERDICT.md`
    (#1035 shipped), `KILL_CODESPACE_COLLAPSE_TO_DEV_PROD.md`
    (subsumed by `docs/guides/DEV_PROD_ENV_DETECT_REMOVAL.md`), `WAVE-3-PLAN.md`
    (audio waves 1–3 covered by `docs/guides/AUDIO_PIPELINE_GUIDE.md`).
  - **Eval verdicts for closed/superseded cohorts** (8):
    `EVAL_1016_FINAL_REPORT_2026_06_17.md` (superseded by §11 in itself),
    `EVAL_1016_OVERNIGHT_REPORT.md`, `EVAL_1016_ROUND3_REVIEW.md`,
    `EVAL_1033_COHORT_RERUN_2026-06-19.md`,
    `EVAL_112_ENTITY_FOCUSED_KG_2026-06-19.md`,
    `EVAL_113_SMALL_MODEL_STANDOFF.md`,
    `EVAL_116_CELL_C_REBASELINE_2026-06-20.md`,
    `EVAL-hybrid-search-validation.md` (#1010 LanceDB shipped).
  - **Pre-impl reviews for shipped features** (3):
    `SPEAKER-ATTRIBUTION-PIPELINE-REVIEW.md` (#876 shipped),
    `TEST-SUITE-REVIEW.md` (post-migration scan; action items addressed),
    `CHUNK7_SILVER_REBUILD_RUNBOOK.md` (chunk 7 audit landed in
    `docs/guides/eval-reports/`).
- **2026-06-16** — Removed 11 notes for shipped/closed work: `APPROACH-913-909-964-993.md`,
  `AUDIO-WAVES-HARDENING-AUDIT.md` (#964 done; #913/#400 closed), `967-interaction-cost-trace.md`,
  `974-adfree-validation.md`, `POST-MIGRATION-GRAPH-VALIDATION-967-974-876.md` (#967/#974 in #1010),
  `DEP-EXTRAS-SEPARATION-1019-SCOPE.md` (#1019), `BATCH-PLAN-diarization-followups.md`,
  `SPOKEN_BY-REPROCESS-876.md`, `REHEARSAL-876-findings-20260609.md`,
  `NEXT_SESSION_HANDOFF-feat-946.md` (#876/#946 done), `COVERAGE-DEBT-deepgram-diarization-pr908.md`
  (PR #908). Working notes now live in the git-ignored `.journal/` (see AGENTS.md → Document location).
- **2026-05-22** — Removed `GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md`; superseded by
  shipped graph handoff orchestrator ([ADR-094](../adr/ADR-094-graph-handoff-orchestrator-fsm.md),
  [RFC-085](../rfc/RFC-085-graph-handoff-orchestrator-retrospective.md)).
- [CORPUS-V4-FIXTURE-LADDER.md](CORPUS-V4-FIXTURE-LADDER.md) — every failure the v3 speaker/ads arc hit, as a taxonomy, and the fixture ladder to catch them next time. Headline: every bug was found by a human reading output, not by a test — and the tests that would have saved us were already red and already ignored.
- [BAKEOFF-18EP-RESULTS.md](BAKEOFF-18EP-RESULTS.md) — final medium-tier multi-provider bake-off on the 18-episode prod-v3 corpus, 8 clean arms scored with grok-4.3 as the single vendor-disjoint judge (gemini-2.5-flash-lite leads on surf/ep; sonnet-5 and gpt-5.5 dropped — empty-content failures on bundled evidence calls, the latter also triggering the entailment-fallback money-guardrail fix).
- [BAKEOFF-OVERNIGHT-2026-08-05.md](BAKEOFF-OVERNIGHT-2026-08-05.md) — v2.5 10-model bake-off frontier (9-ep control): qwen3.7-flash (value) + deepseek-v4-flash (quality) beat every Gemini tier/GPT/pro-reasoning model; gemini-2.5-pro GI-bug fixed + fairly re-judged. **Active** front.
- [BAKEOFF-FINALE-INVENTORY.md](BAKEOFF-FINALE-INVENTORY.md) — **living** work-inventory for the 100-ep finale (fix/improve/auto-research, tagged provider-agnostic vs deepseek-specific) + the iterate-deepseek-vs-proceed-to-qwen decision framework. **Active**.
- [CORPUS_COMPARE_V2_V3_PILOT.md](CORPUS_COMPARE_V2_V3_PILOT.md) — deterministic-only metric comparison of v2-cloud vs v3-dgx over 9 shared episodes (joined by GUID): v3 names far more voices (`voices_named` +3.44) and nearly eliminates timeline drift (`timeline_error_pct` 1.92% → 0.19%), but surfaces fewer quotes and insights. Text-quality judging is out of scope here — tracked separately via the cross-vendor judge panel.
- **OPEN — airgapped/local ML grounder at 8% coverage** ([EVAL_GROUNDING_WHO_FINDS_THE_QUOTE_2026_07](../guides/eval-reports/EVAL_GROUNDING_WHO_FINDS_THE_QUOTE_2026_07.md)) — the local extractive-QA + NLI grounder finds evidence for only **8%** of insights vs **82%** for the LLM (qwen) grounder. Two structural faults: no retrieval step, and the NLI verifier demands strict entailment that insights don't satisfy. Fix = add embedding retrieval **and** replace the verifier. Only affects the **local/offline (airgapped) profiles** — cloud/DGX profiles ground with the summarising LLM. This is the "tracked separately" item from the report; parked here (no GH issue opened) until scheduled.
- [DGX-SERVICE-PERF-BASELINE-PLAN.md](DGX-SERVICE-PERF-BASELINE-PLAN.md) — isolated load/perf harness + baseline plan for each DGX inference service (diarization/Whisper/vLLM) after the 2026-08-04 OOM (#1397); `scripts/perf/dgx_service_loadtest.py` ramps concurrency + correlates homelab memory/`Shmem`. Per-service issues #1398/#1399/#1400.
- [ISSUE-1540-ml-optional-assessment.md](ISSUE-1540-ml-optional-assessment.md) — blast-radius assessment for making local-ML opt-in (#1540). Split: CI wins (lever F, already implemented) stay in #1540; the torch-free runtime image (A), CI path-narrowing (E, re-opens #1527), and cloud NER prepass (D, latent-quality question) move to a discussion issue. `[search]`→torch is the whole cost; Search is hard-coupled to a locally-built LanceDB index (no keyword fallback). Gateway-embeddings path already exists — gated on whether LiteLLM serves `/embeddings`.
- [RETRO-stack-publish-qemu-hang-2026-08.md](RETRO-stack-publish-qemu-hang-2026-08.md) — retrospective (#1800): the Stack-test `publish` job hit GitHub's 6h cap repeatedly because four images built sequentially multi-arch and the learning-app arm64 build wedged under QEMU; fix (already shipped `def2dd503`) = amd64-only per-image matrix split. **Closed by the doc.**
- [ALLOY-DOUBLE-SHIP-FIX-2026-08-25.md](ALLOY-DOUBLE-SHIP-FIX-2026-08-25.md) — pipeline container logs reach VictoriaLogs twice (operator.alloy + legacy base.alloy docker sources; measured ×2.0). Podcast-repo half done (dashboards repointed); homelab half (remove the legacy block, root-owned on-box copy) documented with apply + verify steps. **Open until the homelab half lands.**
- [SPEND-ESTIMATOR-OVERSTATEMENT-2026-08-28.md](SPEND-ESTIMATOR-OVERSTATEMENT-2026-08-28.md) — run-reported LLM cost overstates gateway truth ~3.5× ($2.2685 estimated vs $0.6518 in SpendLogs over one sweep). Cause: LiteLLM returns cost in the `x-litellm-response-cost` HEADER, which the plain `openai` SDK never surfaces, so `_openai_response_cost_usd` always falls back to the direct-DeepSeek pricing table. Consequence: the $10 run cap halts runs ~3.5× earlier than intended. **Open — fix not chosen (prefers reading the header).**
