# Work In Progress (WIP) Documentation

This folder holds early-stage notes, backlog plans, and reference material that has
not been promoted to a PRD, RFC, ADR, runbook, or release note. Content here is
**not authoritative** — when an item is ready, it moves to its proper home and the
WIP file is removed.

## Promotion targets

- Long-term feature ideas → **PRD** in `docs/prd/` and the relevant **RFC** in `docs/rfc/`
- Architectural decisions → **ADR** in `docs/adr/`
- Operator-facing procedures → **guide** in `docs/guides/`
- Post-release retrospective → reference from the corresponding release note in `docs/releases/`

## Current contents

| File | Description | Status |
| ---- | ----------- | ------ |
| `PROD_V2_VS_V3_DGX_BATTLE_2026-07-12.md` | Running observations for the cloud-vs-DGX corpus battle (prod-v2 vs prod-v3). Reframes the ROI question honestly — the cloud built the whole 99-episode corpus for **$0.51**, so the DGX cannot win on API savings; the case must rest on quality parity, unmetered eval sweeps, and privacy. Records three bugs found by running the real pipeline on the real corpus (silence-removal timeline break; ad narrator crowned host; transcript cache replaying the drifted transcripts so a 60-second run "succeeded" doing nothing), the v2 deterministic baseline (3.05% timeline error, only 37% of voices named), live DGX throughput, and the two-arm judge design (as-shipped vs same-transcript controlled) with the cross-vendor rule enforced in code. | Active (live run) |
| `DIARIZATION-MODEL-COMPARISON-PLAN.md` | Plan to answer "model limit vs bad ground truth" for the diarization failures. pyannote 3.1 (5/8) and community-1/v4 (4/8) both cap on the 8-fixture set and fail the SAME fixtures — but they share the `segmentation-3.0`+`wespeaker` backbone, so that's not independent. Parallel experiments on the same fixtures+scoring: (A) pyannote embedding swap (wespeaker→ECAPA/TitaNet), (B) Deepgram engine, (C) NeMo engine, (D) fixture audit (is `expected=3` on the multi_accent cases even reachable?). Related #1170/#1171. | Active (plan) |
| `DIARIZATION-TUNING-EVAL.md` | Runbook for tuning pyannote out of over-segmentation (one episode gave 15 labels for ~2 speakers). Covers the `clustering_threshold` knob, the 44 v3 ground-truth sidecars (`expected_diarized_voices`, panel + 30 ad-voice guardrails), the sweep harness (`scripts/eval/score/diarization_tuning_sweep_v1.py`), and Round 1 (fixtures → 100% count-match on DGX GPU) → Round 2 (real episodes). Includes the "always check `FIXTURES_VERSION`, v1/v2 are dead" gotcha. | Active (runbook) |
| `SPEAKER-RESOLUTION-ROADMAP.md` | Roadmap to reduce unknown diarized speakers, with **before/after measurement per step** on the real prod-v2 corpus (90 diarized eps / 579 voices). Shipped: #1a (+41 voices), #1b (episode-scope), #2 (publisher denylist), #3 (host/guest role), voice_type classification, Step B known_hosts (+19 voices, cumulative 36.6%→47.0%), Step C (unnamed host→"Host" label), Step D (intro NER guest detection, +7 voices). 466/579 (80%) handled; 113 truly unknown remain (un-introduced panel guests). Open decisions: promote measure scripts to make target, chase talk-time threshold. | Active (plan + measured) |
| `RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md` | 9-chunk plan to implement RFC-088 Enrichment Layer (protocol + registry + executor + 6 deterministic enrichers + topic_similarity + nli_contradiction + QueryEnricher + server/viewer + profile presets + promotion). Includes resilience model, mock-scorer scenario engine, metrics/o11y/analytics surfaces, MCP server extension with correlation IDs, and per-chunk eval harnesses. Epic #1101, children #1102–#1110. | Active (plan) |
| `RFC-088-CHUNK1-LOCK-AUDIT.md` | Honest review of chunk-1 readiness after 7 iterative plan revisions. 20 findings (10 blocking, 4 RFC-088 amendments, 4 important, 4 open). Resolves async-vs-sync protocol drift, defines `EnricherResult` shape, pipeline-attached failure semantics, runs_skipped flow, test enricher registration, YAML JSON schema, and 4 more. **LOCKED 2026-06-26** — all four operator decisions (O1–O4) resolved. | Audit (locked) |
| `RFC-088-CHUNKS-2-8-REPLAN.md` | Per-chunk delta after chunk 1 grew. Documents what chunk 1 absorbed, per-chunk shrinkage / focus shifts, revised LOC + reviewer-day estimates, and 3 follow-on operator decisions (REPLAN-O5 chunk-6 split / REPLAN-O6 make-targets-for-eval / REPLAN-O7 cost-cap plumbing in chunk 1). | Active |
| `RFC-088-real-corpus-validation-findings.md` | Post-chunks-0-9 real-corpus validation (item #8) against `.test_outputs/manual/prod-v2/corpus` (209 episodes, 99 latest-run bundles). All five bugs closed on this branch: Bug 1 — CLI was a no-op (wired `register_deterministic_enrichers` + `discover_episode_bundles` + force-include semantics); Bug 2 — `temporal_velocity` uses `effective_last_month` so stale / partial current-month data doesn't collapse velocity to 0; Bug 3 — shared `is_unresolved_speaker_placeholder` helper filters `SPEAKER_NN` placeholders from `guest_coappearance` + `grounding_rate`; Bug 4 — `insight_density` now reads `episode.duration_seconds` and Quote `timestamp_start_ms`; Bug 5 — `@sync_enricher` infers `records_written` from the largest top-level list in the dict. | Closeable |
| `ADR-108-REAL-CORPUS-EVAL-2026-07.md` | Real-DeBERTa eval of the reimagined `topic_consensus` + `stance_timeline` (ADR-108) over prod-v2 (99 bundles). Symmetric NLI entailment failed (1 pair / 2 903, recall ≈ 0); the tuning loop found the winning **composite** — embedding cosine ≥ 0.70 AND NLI contradiction ≤ 0.5 → **precision 0.91**, so `topic_consensus` was rewritten to it and **ACTIVATED** (auto-promoted). `stance_timeline` stays dark — its stance signal is genuinely ~0 on factual insights (also fixed a `sign_flips`-on-noise deadzone bug). No metrics fabricated. | Findings (topic_consensus activated) |
| `RFC097_FOLLOWUPS_HANDOFF.md` | Session-handoff for `feat/rfc097-followups` branch. **2026-06-23 update:** all 6 issues closed — #1060 (`36ed9274` + 5 FU commits) registry promotion + clean-reference WER + DGX portability + cross-vendor judges; #1048 (`ceeb0485`) Person Landing shared shell; #1049 (`25ab4db9`) Position Tracker timeline; #1050 (`279d9569`) Person Profile aggregate (UXS-010 sections); #1073 (`691d72c4`) chunk 9 — most pre-shipped in PR #1039, this branch closes the JSON schema tightening + code-site fallouts; #1074 already shipped in PR #1036 (`b1ef7046..a31b4e9a`, 7 fingerprint gap-closure commits). Branch is ready for review / push when operator approves. | Closeable |
| `PHASE1-OPEN-WEIGHT-LLM-LANDSCAPE-2026-06.md` | #928 reframe Phase 1: open-weight LLM landscape (≤35B) → 6-candidate tier-1 shortlist (Qwen3.5:35b incumbent + Qwen3-30B-A3B-Instruct-2507 + DeepSeek-R1-Distill-32B + Magistral 1.2 + Mistral Small 4 + gemini-2.5-flash-lite cloud anchor). Phase 2 eval → #1016. | Active |
| `EVAL_1016_metrics/vllm_metrics_*_phase2c.log` | Raw vLLM `/metrics` polls per candidate (input data for the canonical per-model param compendium, which lives at `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md`) | Reference |
| `PUBLIC-EXPOSURE-AND-PRIVATE-SPLIT.md` | What the **public** repo actually exposes, and the private-split options. Separates two boundaries that get conflated: **prod content** (transcripts, segments, GI artifacts — *never public, boundary holds*; the 30 "committed episodes" in `prod_validation_v1/` are symlinks, 4.8 KB of paths) versus **method** (456 eval configs, the 2,175-line autoresearch playbook, 49 scorecards, 148 WIP notes — *fully public*, operator accepted 2026-07-14, will revisit). Records the enforcement gap: the content boundary rests on `.gitignore` alone, with no pre-commit check — a measured shape+allowlist guard (0 false positives across 203 synthetic-fixture hits) is proposed but NOT built. Notes that history is permanent, so forward-only and retroactive purge are different decisions with wildly different costs. | Active (analysis, no action taken) |
| `RFC097_CHUNK8_FOLLOWUP_TICKETS.md` | Draft ticket bodies for the 3 chunk-8 follow-up items split off the v2 foundation PR (Person Landing + Position Tracker + Person Profile viewer surfaces). Operator opens GH issues from these. | Active |
| `FINGERPRINT_GAPS_ANALYSIS_2026-06-22.md` | Audit of `fingerprint.json` generation surfacing 8 concrete gaps (generation_params={} for GI/KG, backing-model-id hidden behind `autoresearch` alias, vLLM server flags absent, container image absent, postprocessor + extraction_src absent, bullets-vs-paragraph upstream provenance not threaded, hash incomplete). 7-commit work plan included. | Active |
| `HOMELAB_COMPOSE_DRIFT_SYNC_2026-06-22.md` | Cross-repo apply instructions for op-Q #6: Phase 2c flags (`--max-num-seqs=4 --enforce-eager`) applied to `agentic-ai-homelab/infra/vllm/autoresearch/docker-compose.yml`. Also flags 4 additional drift dimensions between homelab source and live DGX for operator decision. | Active |
| `DGX_OBSERVABILITY_910_942_PLAN.md` | Disposition of #910 (closed with subscope 4 split to #1046, subscope 2 closed as already-covered by dcgm-exporter+alloy, subscopes 1+3 planned with acceptance criteria) + full plan for #942 (Sentry SDK init inside pyannote-server, cross-repo homelab edit). | Active |
| `942-PYANNOTE-SENTRY-APPLY.md` | Cross-repo apply doc for #942 — paste-ready edits for `agentic-ai-homelab/infra/dgx/pyannote-server/{app.py,Dockerfile}` + operator env setup + Sentry verification. Runbook delta in `docs/guides/DGX_RUNBOOK.md` § "In-process Sentry on DGX services (#942)". | Active |
| `1046-WHISPER-DUAL-MODEL-FUTURE-USES.md` | Parks 5 alternative uses of the dual-model machinery (dual-pass reconciliation, confidence-weighted NER, sniff-driven NER pre-pass, speculative pipeline, cross-model dispatch) — all align with the intelligence-extraction goal that the skip-deep gate violated. None queued; planning material for next session pickup. Includes the offline-prototype-able subset using saved transcripts under `data/eval/runs/1046-measurement-pass-2/`. | Backlog |
| `DASHBOARD-PERF-ANALYSIS-digest-99ep.md` | Dashboard perf root-cause: `corpus/digest` runs ~6 sequential topic-band semantic searches (4.6s, no cache); ranked options (parallelise + cache + lazy-load) + cold-init segfault caveat | Active |
| `RUNBOOK-876-corpus-rediarization.md` | #876 DGX re-diarization operational runbook (health gate → pilot → backup → full run → rollback); gated on #944 | Ready |
| `MULTI-USER-AND-GRAPH-FSM-ANALYSIS.md` | Multi-user/FSM analysis + graph-viewer diarization-support gaps | Backlog |
| `AUTORESEARCH_LEARNINGS_FOR_V3.md` | Rolling failure-mode catalogue from #907 children — spec input for v3 (#921) | Reference |
| `DGX_NEXT_STEPS.md` | Living strategic doc on what runs on DGX vs local, vLLM vs Ollama, offload decision frame | Living |
| `EXPLORE_EXPANSION_IDEAS.md` | Explore-tab feature brainstorming | Idea |
| `HOME_AI_HARDWARE_PLAN.md` | Local hardware / on-prem AI plan | Idea |
| `LORA_HYBRID_PIPELINE_PLAN.md` | LoRA + hybrid pipeline exploration | Idea |
| `METRICS_DOCS_AND_DASHBOARD_V2.md` | Metrics docs / dashboard redesign | Partial |
| `POST_REINGESTION_PLAN.md` | Post-pipeline-rev validation plan | Reference |
| `PROD_RUN_ANALYSIS_100EP.md` | 100-episode production run retrospective | Reference |
| `QUALITY_IMPROVEMENTS_BACKLOG.md` | Quality / GI / KG improvement backlog | Backlog |
| `UNIFIED_QUALITY_PLAN.md` | Unified quality framework draft | Draft |
| `issue-382-transformers-v5-upgrade-plan.md` | transformers v5 upgrade plan (2026-04-02) — superseded by the 2026-07-05 deep analysis below | Superseded |
| `ISSUE-382-TRANSFORMERS-V5-DEEP-ANALYSIS-2026-07-05.md` | Deep analysis + refined phased plan for #382. Maps every removed-API touchpoint (`pipeline("summarization"/"text2text-generation"/"question-answering")`, `transformers.file_utils`), scores modernization opportunities (drop `pipeline()` entirely for `generate()`/QA-head forward, adopt `GenerationConfig`, bump `sentence-transformers>=5.6.0`, verify SDPA on MPS/CUDA), defines 7 phases with test gates + rollback | Active |
| `ISSUE-382-TRANSFORMERS-V5-EXECUTION-PLAN.md` | 13-phase execution manual for #382 (transformers v5 + architectural unification epic). Phases 0-9+E/F/G all landed; see docs/adr/ADR-068 post-impl for the shipped shape. | Closeable |
| `ISSUE-382-AI-PROVIDER-AUDIT-2026-07-05.md` | Mid-Phase-E audit answering "we all want to look at AI providers" — confirmed HFEvidenceBackend sits one layer below SummarizationProvider, no conflict. One follow-up opportunity documented (bundled-inference dedup across cloud providers, orthogonal to #382). | Closeable |
| `manual-test-plan-gi-kg.md` | Manual GI/KG smoke checklist | Reference |
| `REMEMBER-half-scope.md` | Consumer Learning Platform "Remember" half scope — PRD-040 Capture (P2) + PRD-041 Consolidation (P3). P3 rebased to **consume the Enrichment Layer (RFC-088)** envelopes (co-occurrence / similarity / velocity / contradiction). Decisions: per-user files + scoped recall, `/api/app/*`, in-app digest, MD-only export. **RFC-088 landed on `main` 2026-06-27; P3 (#1113) shipped on top of it.** | Closeable |
| `P3-CONSOLIDATION-EXECUTION-PLAN.md` | Detailed P3 execution plan (epic #1113): what already exists to unify on (P2 captures + RFC-090/094/072 + the **shipped RFC-088 envelopes & `/api/corpus/enrichments*` readers**), the consumer enrichment read-surface design (`/api/app/episodes/{slug}/enrichment` + `/api/app/corpus/enrichment`, heard-set-scoped), the #1120–#1126 child decomposition, sequencing, and D6/ADR-104 guardrails. **RFC-088 is on `main` (2026-06-27); envelope/route names reconciled — P3 (#1113) shipped.** | Closeable |
| `wip-concurrent-pipeline-http-retry-metrics.md` | Open documentation gap for `http_urllib3_retry_events` | Open |
| `wip-topic-clusters-validation-reference.yaml` | Reference topic-cluster validation config | Reference |
| `VIEWER-UI-709-695-694-PLAN.md` | UI batch plan (`feat/viewer-ui-709-695-694`): widen+restructure the Config modal to a sub-nav rail, in-app log-viewer modal (#695), per-feed override drill-in (#694), Scheduled rail item (#709), built on a small shared-primitive layer (AppDialog/ToggleSwitch/RelativeTime/clipboard) | Active |
| `CLOUD-PROVIDER-RESILIENCE-E2E-GAP-1003.md` | Audit of cloud-LLM resilience E2E coverage; matrix + 5 follow-up gaps surfaced during #1003 (closed 2026-06-15) | Closed |
| `player/EPIC-2-CONSUMER-APP-PLAN.md` | Epic 2 (consumer Learning App, P1) plan + task breakdown (S1–S2 server, C1–C7 client) with a cross-cutting Definition of Done (mobile-first + i18n + a11y + full test pyramid on every task). Mirrors GH Epic 2. Player-first vertical slice over local content; defers #1069/#1070/Capture. | Active |
| `player/SERVER-SIDE-GAP-ANALYSIS.md` | Server-side gap analysis for the consumer platform — what `/api/app/*` shipped in Epic 1 vs what the consumer surfaces need (surfaced the catalog-list endpoint gap). | Active |
| `player/LEARNING-PLATFORM-GAP-ANALYSIS-2026-07.md` | Refresh of umbrella #1062 after PR #1141: workstream ledger (P0–P3 + Epic 3 ~all shipped), gap analysis to the PRD-035 vision, and 4 pivots. Net finding: the functional vision is essentially shipped; the one real build gap is the `rank_discover` eval (personalization is dark / flag-off). Enrichment #1101: 6 chunks closed as tracking debt, but #1105/#1106 smart-enricher accuracy eval+sweep are deferred (kept open) — see the RFC-088 audit. | Active |
| `RFC-088-ENRICHMENT-EPIC-1101-AUDIT-2026-07.md` | Spec-vs-main audit of enrichment epic #1101: all 9 RFC-088 chunks shipped + promoted (2026-06-27), evals CI-gated via `tests/unit/enrichment/test_eval_scripts_smoke.py`; 6 chunks closed 2026-07-05 as tracking debt; #1105/#1106 smart-enricher accuracy eval+sweep were deferred (kept open, carry solo); #1101 umbrella closed. Corrects the gap-analysis doc's Pivot 1. | Active |
| `player/RFC-LANDSCAPE-FOR-PLATFORM.md` | Survey of in-flight/adjacent RFCs the platform builds on (RFC-075/023/068/088/094/069) and older ideas to dismiss or fold in. | Active |
| `player/1069-SCRAPE-ON-DEMAND-SCOPE-ANALYSIS.md` | Scope decision for #1069 (last open P0): discovery-over-local already shipped, so #1069 only blocks *corpus growth*. Primitives already exist (single-feed pipeline #807, `ContentSource` seam, jobs API), so the operator-ingest slice is small; the consumer-discovery slice (Podcast Index + scrape UI + guardrails) is large + premature pre-user-scale. Recommends **re-scope** (thin ingest now, defer the rest), with **close/fold-into-scheduled-ingestion** as fallback. **Update 07-07:** phase-1 ingestion = the pipeline (the standalone ingest primitive was built then dropped as redundant); the #1069 work that shipped is ingestion↔enrichment consistency (enrich verb + scheduler `kind` + spawn-bug fix). | Active |
| `player/1144-DISAGREEMENT-DETECTOR-FEASIBILITY.md` | #1144 (07-07): **BUILT, no-LLM path measured non-viable, ships gated dark.** `stance_disagreement` (stance-aggregate DeBERTa, no LLM per operator constraint) scored vs `gold_v1.jsonl` (40 prod-v2 + designed v3 Cho-vs-Bessent disagreement) → **0% precision** (aggregate) / **10%** (atomic-max): DeBERTa can't separate opposition from topic-adjacency, so the shared-question gate needs an LLM (ruled out). Enricher wired + gate-guarded, stays dark (measured `gate_metrics.json` precision 0.0); `perspectives` (#1146) is the live no-LLM surface; `gold_v1.jsonl` is the regression bar (auto-promotes any future non-LLM scorer ≥ 0.5). | Active |
| `player/mockups/` | Phone mockups (HTML + PNG) of the three explored Player aesthetics; **Direction B (Editorial Bold)** adopted → UXS-011. Design aids, not shipped assets. | Reference |
| `MENTIONS_PERSON_DETERMINISM_INVESTIGATION.md` | #1076 chunk 4 path-A/B decision doc + post-shipment status banner: NER pass validated via 50-row operator-labelled sample (47 TP / 3 AMBIGUOUS / 0 FP), enabled in airgapped + airgapped_thin profiles, prod-v2 retro sweep landed (marker `#1076-ner-2026-06-24`) | Closed |
| `NER_FP_SAMPLE_LABELLED_2026-06-24.json` | Operator-labellable 50-row sample produced by `scripts/dev/measure_ner_mentions_diff.py` against the prod-v2 corpus; backs the TP/FP claim in the determinism investigation doc | Reference |
| `enrichment-visual-inspection-plan.md` | 4-stage plan for restarting the viewer against a freshly-enriched small corpus and inspecting where each RFC-088 enrichment signal surfaces in the player/viewer UI (uncovering UX gaps). | Active |
| `CORPUS-EVOLUTION-FOR-COMPLEX-ENRICHERS.md` | Living notes (2026-07-06): the fixture-generated corpora are too thin/interview-shaped to exercise the cross-person enrichers (contradiction/disagreement/perspectives) — seeded with this session's hard evidence (#1106 0/150 contradictions, #1144 0/40 disagreements, #1146 0 perspectives in the validation corpus). Two evolution tracks: a committed "rich pocket" for e2e/CI + dialogue/time-span/scale for realistic evals. | Active |
| `ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md` | Living notes (2026-07-06): how to grow the **eval** corpus (prod-v2, ~100 real episodes) with more real shows so the newly-shipped enrichers produce visible value. Value model per enricher; highest lever = topic OVERLAP across shows (compounds perspectives + disagreement + co-appearance). Distinct from the test-fixture evolution note. | Active |
| `FIXTURE-CAPABILITY-GAP-ANALYSIS.md` | #1148 analysis: the fake-feed fixtures were built for the pipeline-quality era (ASR/canon/sponsor/voice, per EVAL_FIXTURES_V3); they do NOT exercise the enrichers, topic/person surfaces, or digest/discovery we since added. Capability→fixture gap matrix + six concrete structures v3+ must add (time spread, cross-speaker topic overlap, engineered opposition, multi-guest, grounding variation, seeded users) — each with gold labels. | Active |
| `RFC-088-ENRICHER-ACCURACY-GATE-2026-07.md` | Design + shipped mechanism (2026-07-06): give the enrichment eval/gold/gating side the modularity the runtime already has. New `enrichment/eval/` package — AccuracyScorer protocol + registry, generic `expected_enrichment[<id>]` gold (no per-enricher field names), and an accuracy gate declared on each manifest that reads `data/eval` and drives admission → registry → profiles → UI config, mirroring the provider `RegressionRule` gate. `nli_contradiction`'s exclusion is now a data-driven gate decision (0% precision, #1106), not the hand-commented `profile_sets` disable. `GET /api/enrichment/config/admission` surfaces the gate reason to the UI. | Active |
| `handover-theme-clusters.md` | Theme-clusters (co-occurrence) feature handover: DONE = enricher `topic_theme_clusters` + PMI/lift + `GET /api/corpus/theme-clusters` (4 commits, tests green). REMAINING (execute-ready) = server attach at 4 call sites, player KnowledgePanel/EntityCard render + "Theme/Similar" rename, `--lp-theme` token, graph `thc:` node theming. BLOCKED on non-empty theme-cluster data (prod-pilot too small; prod-v2 discovers 0 bundles) — get real clusters + eyeball (value test) before building the UI. | Active |
| `consumer-node-view-backend-followups.md` | Two backend follow-ups from the `feat/consumer-remember` node-view review round, each shipped as a viewer-side approximation: (1) **precise per-show host/guest** — viewer infers Host from episode coverage (`3809300a`); accurate version needs a pipeline `person→hosts→show` edge from feed/author metadata. (2) **out-of-slice insight rendering** — mention drill (`566a9f69`) resolves in-slice insights; corpus-wide CIL mentions need a `GET /api/relational/insight-detail` endpoint (insight text + supporting quotes off the full server graph) + a small `InsightNodeView` + gate extension to `Insight`. | Backlog |
| `handover-enrichment-tabs-session.md` | Session handover (2026-07-01): Enrichment 3rd-tab (Details·Enrichment·Neighbourhood) added to graph topic/person cards (`NodeDetail` + new `NodeEnrichmentSection`) and the episode rail (`SubjectRail` + `EpisodeDetailPanel`), typecheck + 84 vitests green, uncommitted awaiting operator visual check. Also captures branch `feat/consumer-remember` state (#1128 auth epic + enrichment + serve fix, NOT pushed), the open `topic_cooccurrence` PMI/lift enhancement thread, and pre-push checklist. | Active |
| `EPIC-HOST-IDENTIFICATION.md` | Epic spec: reliably know every show's host(s), persist as metadata + a `person→HOSTS→podcast` graph edge, and render host-aware behaviours (graph/digest/library/person card). TDD spine = a **host scorecard** (Slice 0: gold set + coverage/precision/recall/sample-bias metrics) that every path must move toward target. Paths A(per-show edge)→C(show-notes parsing)→B(voice embeddings)→D(multi-guest)→E(role taxonomy). Supersedes the "precise per-show host/guest" follow-up. | Specced |
| `PROD-V2-ADR108-VALIDATION-PLAN.md` | Local validation plan for the ADR-108 work (`topic_consensus` + `insight_sentiment` + read-time conversation/position arcs) on the **real prod-v2 corpus**. Discovery finding: prod-v2 was NOT re-enriched — it still carries the old `nli_contradiction` and has **0** `insight_sentiment` sidecars + no `topic_consensus.json`, so the new data points don't surface yet. Phases: 0 re-enrich (`make enrich CORPUS=… PROFILE=local WITH_ML=1 ONLY=topic_consensus,insight_sentiment`) → 1 verify artifacts → 2 API contracts → 3 both frontends → 4 coverage matrix → 5 edges. Verdict: **code-ready, data-not-ready** until Phase 0 runs. | Active |
| `PROD-V2-GOLDEN-WALKTHROUGH.md` | Golden end-to-end walkthrough of every enricher surface + cross-link on the re-enriched prod-v2 corpus (2026-07-09), with screenshots (`assets/prod-v2-walkthrough/`). Doubles as the user guide (v1) and the test reference for e2e/tier-3 flows. §5 surface-coverage matrix + §6 qualitative read (3 end-analysis Qs: fixture completeness, coverage/assertions, data happiness). Surfaces not screenshotted are tagged *v2 to-capture* (= not captured, **not** unbuilt). | Active |
| `PLAYER-GOLDEN-WALKTHROUGH-v2.md` | **v2** — consumer-player-only, exhaustive. The entity model + every cross-link (show→episode→topic→person), **every** place the player surfaces enrichment (each with the computing mechanism explained), walked as an interaction journey with 9 screenshots (`assets/player-walkthrough-v2/`), on prod-v2 signed-in. §3 surface table, §4 per-enricher mechanism, §5 qualitative read, §6 test map. Internal reference for "how it connects / does it make sense / testing". | Superseded by v3 |
| `PLAYER-GOLDEN-WALKTHROUGH-v3.md` | **v3** — supersedes v2. Same exhaustive player walk, folding in the new consumer **"Where they agree"** consensus row (`topic_consensus` is no longer operator-only): §2.3 person card + §3/§4/§5/appendix updated, with a **fresh** person-card capture (Nic Harrigan, live consensus row) in `assets/player-walkthrough-v3/` (8 unchanged surfaces reused). Cross-refs the operator show-landing Signals band (RFC-104/UXS-015 Phase 2). | Active |
| `ENRICHER-HARDENING-ROADMAP.md` | Fixtures→coverage→surfaces roadmap grading every enricher surface/cross-link against a weak/good/excellent rubric. PR-A (v3 `topic_similarity` fixture + invariants) + PR-B (per-enricher emission non-degeneracy + consensus executor smoke) **landed** on `fix/adr108-enricher-surfaces`; PR-C = **#1168** (operator-viewer served-corpus e2e harness); **PR-D reconciled 2026-07-09** — the "thin" surfaces were already built + tested (data-absence, not code-absence), so PR-D is capture+verify, not build. | Active |
| `1175-LOCAL-CORPUS-PORT.md` | #1175 design note for local corpus export/import (`make export-corpus` + `make import-corpus`) — instance-to-instance portability without going through the backup repo. Same tarball format as the CI backup flow, no `gh` dependency in the local path. New scripts under `scripts/ops/corpus_snapshot/` wrap the existing manifest primitives; docs deliverables extend the SSOT guide + add an airgap runbook. | Active (in progress) |
| `1180-MPS-AUDIT.md` | #1180 MPS-exclusive audit — one silent-over-serialize edge case in `_both_providers_use_mps`'s `torch.backends.mps.is_available()` fallback. Fix: emit `WARNING` line when the fallback fires so operators seeing a low `processing_overlap_ratio` can trace it back. Explains what stays unchanged (fallback itself, `mps_exclusive` default) and why. | Closeable |
| `E2E-WHISPER-FAILURES-DIAGNOSIS.md` | Diagnosis of 5 pre-existing e2e Whisper failures. Root cause: Whisper succeeds, then `apply_diarization_to_result` fails on uncached pyannote HF model → OSError bubbles to outer transcription catch → mislabeled as "Whisper transcription failed" AND drops the successfully computed transcript. Fix: broaden the diarization catch to include OSError/RuntimeError so transcription output is preserved. All 5 tests now green. | Closeable |

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
    (subsumed by POST_RFC097_DEV_PROD_REMOVAL), `WAVE-3-PLAN.md`
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
- [SESSION_LOG_GI_2026-07-12_13.md](SESSION_LOG_GI_2026-07-12_13.md) — grounded-insights session: nine silent defects, every claim made and retracted, and the twelve gaps still open. Headline: production runs a configuration we never measured (gate OFF, ceiling 12, temperature 0.3).
- [GI_TWO_MODULES.md](GI_TWO_MODULES.md) — the insight pipeline is two modules (write the insight / find the quote). They must be measured separately, and who owns module 2 is an open decision before prod v3.
- [GI_WHAT_TO_SURFACE.md](GI_WHAT_TO_SURFACE.md) — what an insight should BE: route (surface/connect/drop) instead of rank-and-truncate; no cutoff baked into the corpus; why a stance needs a speaker.
- [CORPUS-V4-FIXTURE-LADDER.md](CORPUS-V4-FIXTURE-LADDER.md) — every failure the v3 speaker/ads arc hit, as a taxonomy, and the fixture ladder to catch them next time. Headline: every bug was found by a human reading output, not by a test — and the tests that would have saved us were already red and already ignored.
