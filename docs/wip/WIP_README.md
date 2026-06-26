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
| `RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md` | 9-chunk plan to implement RFC-088 Enrichment Layer (protocol + registry + executor + 6 deterministic enrichers + topic_similarity + nli_contradiction + QueryEnricher + server/viewer + profile presets + promotion). Includes resilience model, mock-scorer scenario engine, metrics/o11y/analytics surfaces, MCP server extension with correlation IDs, and per-chunk eval harnesses. Epic #1101, children #1102–#1110. | Active (plan) |
| `RFC-088-CHUNK1-LOCK-AUDIT.md` | Honest review of chunk-1 readiness after 7 iterative plan revisions. 20 findings (10 blocking, 4 RFC-088 amendments, 4 important, 4 open). Resolves async-vs-sync protocol drift, defines `EnricherResult` shape, pipeline-attached failure semantics, runs_skipped flow, test enricher registration, YAML JSON schema, and 4 more. Operator decisions outstanding: cost cap policy / enrichment_cancel MCP tool / RFC-088 status during impl / module rename. Lock state at bottom. | Audit |
| `RFC097_FOLLOWUPS_HANDOFF.md` | Session-handoff for `feat/rfc097-followups` branch. **2026-06-23 update:** all 6 issues closed — #1060 (`36ed9274` + 5 FU commits) registry promotion + clean-reference WER + DGX portability + cross-vendor judges; #1048 (`ceeb0485`) Person Landing shared shell; #1049 (`25ab4db9`) Position Tracker timeline; #1050 (`279d9569`) Person Profile aggregate (UXS-010 sections); #1073 (`691d72c4`) chunk 9 — most pre-shipped in PR #1039, this branch closes the JSON schema tightening + code-site fallouts; #1074 already shipped in PR #1036 (`b1ef7046..a31b4e9a`, 7 fingerprint gap-closure commits). Branch is ready for review / push when operator approves. | Closeable |
| `PHASE1-OPEN-WEIGHT-LLM-LANDSCAPE-2026-06.md` | #928 reframe Phase 1: open-weight LLM landscape (≤35B) → 6-candidate tier-1 shortlist (Qwen3.5:35b incumbent + Qwen3-30B-A3B-Instruct-2507 + DeepSeek-R1-Distill-32B + Magistral 1.2 + Mistral Small 4 + gemini-2.5-flash-lite cloud anchor). Phase 2 eval → #1016. | Active |
| `EVAL_1016_metrics/vllm_metrics_*_phase2c.log` | Raw vLLM `/metrics` polls per candidate (input data for the canonical per-model param compendium, which lives at `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md`) | Reference |
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
| `issue-382-transformers-v5-upgrade-plan.md` | transformers v5 upgrade plan | Reference |
| `manual-test-plan-gi-kg.md` | Manual GI/KG smoke checklist | Reference |
| `wip-concurrent-pipeline-http-retry-metrics.md` | Open documentation gap for `http_urllib3_retry_events` | Open |
| `wip-topic-clusters-validation-reference.yaml` | Reference topic-cluster validation config | Reference |
| `VIEWER-UI-709-695-694-PLAN.md` | UI batch plan (`feat/viewer-ui-709-695-694`): widen+restructure the Config modal to a sub-nav rail, in-app log-viewer modal (#695), per-feed override drill-in (#694), Scheduled rail item (#709), built on a small shared-primitive layer (AppDialog/ToggleSwitch/RelativeTime/clipboard) | Active |
| `CLOUD-PROVIDER-RESILIENCE-E2E-GAP-1003.md` | Audit of cloud-LLM resilience E2E coverage; matrix + 5 follow-up gaps surfaced during #1003 (closed 2026-06-15) | Closed |
| `player/EPIC-2-CONSUMER-APP-PLAN.md` | Epic 2 (consumer Learning App, P1) plan + task breakdown (S1–S2 server, C1–C7 client) with a cross-cutting Definition of Done (mobile-first + i18n + a11y + full test pyramid on every task). Mirrors GH Epic 2. Player-first vertical slice over local content; defers #1069/#1070/Capture. | Active |
| `player/SERVER-SIDE-GAP-ANALYSIS.md` | Server-side gap analysis for the consumer platform — what `/api/app/*` shipped in Epic 1 vs what the consumer surfaces need (surfaced the catalog-list endpoint gap). | Active |
| `player/RFC-LANDSCAPE-FOR-PLATFORM.md` | Survey of in-flight/adjacent RFCs the platform builds on (RFC-075/023/068/088/094/069) and older ideas to dismiss or fold in. | Active |
| `player/mockups/` | Phone mockups (HTML + PNG) of the three explored Player aesthetics; **Direction B (Editorial Bold)** adopted → UXS-011. Design aids, not shipped assets. | Reference |
| `MENTIONS_PERSON_DETERMINISM_INVESTIGATION.md` | #1076 chunk 4 path-A/B decision doc + post-shipment status banner: NER pass validated via 50-row operator-labelled sample (47 TP / 3 AMBIGUOUS / 0 FP), enabled in airgapped + airgapped_thin profiles, prod-v2 retro sweep landed (marker `#1076-ner-2026-06-24`) | Closed |
| `NER_FP_SAMPLE_LABELLED_2026-06-24.json` | Operator-labellable 50-row sample produced by `scripts/dev/measure_ner_mentions_diff.py` against the prod-v2 corpus; backs the TP/FP claim in the determinism investigation doc | Reference |

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
