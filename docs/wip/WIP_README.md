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
| `V27-OBSERVABILITY-SCOPE-803-805-426_2026-06-21.md` | Value-focused close-out scope for 3 v2.7 issues: #803 full deploy-observability build (deploy events + Grafana panel + Sentry release boundary + 7-tool prod-monitor MCP), #805 static handoff audit + runbook PR (live drill stays operator-owned), #426 pivot to Langfuse AI-metrics tracing coexisting with the existing cost/Loki solution. **Shipped:** #803 + #805 merged in #1047 (both closed); #426 → spawned **#1052** (Langfuse LLM o11y) built on `feat/langfuse-observability` (core tracing + control-plane traces probe, live-validated). | Done |
| `PHASE1-OPEN-WEIGHT-LLM-LANDSCAPE-2026-06.md` | #928 reframe Phase 1: open-weight LLM landscape (≤35B) → 6-candidate tier-1 shortlist (Qwen3.5:35b incumbent + Qwen3-30B-A3B-Instruct-2507 + DeepSeek-R1-Distill-32B + Magistral 1.2 + Mistral Small 4 + gemini-2.5-flash-lite cloud anchor). Phase 2 eval → #1016. | Active |
| `EVAL_1016_LANDSCAPE_2026_06.md` | #1016 Phase 2 per-stage eval (summary / GI / KG) of the 6-candidate cohort; dual-silver methodology (Opus 4.7 + Sonnet 4.6) + multi-vendor judge panel (Sonnet 4.6 + GPT-5.4). Promoted to `docs/guides/eval-reports/` when complete. | Active |
| `EVAL_1016_metrics/vllm_metrics_*_phase2c.log` | Raw vLLM `/metrics` polls per candidate (input data for the canonical per-model param compendium, which lives at `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md`) | Reference |
| `RFC097_CHUNK8_FOLLOWUP_TICKETS.md` | Draft ticket bodies for the 3 chunk-8 follow-up items split off the v2 foundation PR (Person Landing + Position Tracker + Person Profile viewer surfaces). Operator opens GH issues from these. | Active |
| `RFC097_CHUNK9_PLAN.md` | Implementation plan for chunk 9 (strict KG v2.0 + GI v3.0 + ADR-101). Bake gate dropped per operator 2026-06-22; deferred to follow-up PR for scope reasons (~15 test/fixture migration files). Plan + ADR-101 outline + verification commands ready to act on. | Active |
| `FINGERPRINT_GAPS_ANALYSIS_2026-06-22.md` | Audit of `fingerprint.json` generation surfacing 8 concrete gaps (generation_params={} for GI/KG, backing-model-id hidden behind `autoresearch` alias, vLLM server flags absent, container image absent, postprocessor + extraction_src absent, bullets-vs-paragraph upstream provenance not threaded, hash incomplete). 7-commit work plan included. | Active |
| `HOMELAB_COMPOSE_DRIFT_SYNC_2026-06-22.md` | Cross-repo apply instructions for op-Q #6: Phase 2c flags (`--max-num-seqs=4 --enforce-eager`) applied to `agentic-ai-homelab/infra/vllm/autoresearch/docker-compose.yml`. Also flags 4 additional drift dimensions between homelab source and live DGX for operator decision. | Active |
| `DGX_OBSERVABILITY_910_942_PLAN.md` | Disposition of #910 (closed with subscope 4 split to #1046, subscope 2 closed as already-covered by dcgm-exporter+alloy, subscopes 1+3 planned with acceptance criteria) + full plan for #942 (Sentry SDK init inside pyannote-server, cross-repo homelab edit). | Active |
| `942-PYANNOTE-SENTRY-APPLY.md` | Cross-repo apply doc for #942 — paste-ready edits for `agentic-ai-homelab/infra/dgx/pyannote-server/{app.py,Dockerfile}` + operator env setup + Sentry verification. Runbook delta in `docs/guides/DGX_RUNBOOK.md` § "In-process Sentry on DGX services (#942)". | Active |
| `1046-WHISPER-MULTI-MODEL-DESIGN.md` | Design doc + measurement passes for #1046 (Whisper sniff-pass gate). Pass 2 (32-episode corpus, 2026-06-23): geomean ratio 4.98×, break-even r*=0.80, sniff r=0.75 (just below break-even), FN=9% (1 in 11 episodes silently lossy). Under the operator's "best intelligence" goal, gate is **REJECTED** — 9% FN unacceptable and 12% wallclock saving has no upside. Orchestrator stays in tree as opt-in plumbing; default disabled. Per-call `model_override` capability preserved for future uses → see sibling `1046-WHISPER-DUAL-MODEL-FUTURE-USES.md`. | Closeable |
| `1046-WHISPER-DUAL-MODEL-FUTURE-USES.md` | Parks 5 alternative uses of the dual-model machinery (dual-pass reconciliation, confidence-weighted NER, sniff-driven NER pre-pass, speculative pipeline, cross-model dispatch) — all align with the intelligence-extraction goal that the skip-deep gate violated. None queued; planning material for next session pickup. Includes the offline-prototype-able subset using saved transcripts under `data/eval/runs/1046-measurement-pass-2/`. | Backlog |
| `967-interaction-cost-trace.md` | #967 live devtools trace: fcose removed the layout wall; pan/zoom FPS vs node count → cap=50 is the smooth ceiling. Plus a side finding (FAISS-fallback segfault on a legacy-schema corpus) | Active |
| `POST-MIGRATION-GRAPH-VALIDATION-967-974-876.md` | Post-migration graph validation on re-diarized prod-v2: 2 bugs found+fixed (focus reconciliation, prefix-tolerant rail), Tier-3 data scan (offset 1.0, SPOKEN_BY, adfree) + test tiers all green, open findings | Active |
| `DASHBOARD-PERF-ANALYSIS-digest-99ep.md` | Dashboard perf root-cause: `corpus/digest` runs ~6 sequential topic-band semantic searches (4.6s, no cache); ranked options (parallelise + cache + lazy-load) + cold-init segfault caveat | Active |
| `RUNBOOK-876-corpus-rediarization.md` | #876 DGX re-diarization operational runbook (health gate → pilot → backup → full run → rollback); gated on #944 | Ready |
| `MULTI-USER-AND-GRAPH-FSM-ANALYSIS.md` | Multi-user/FSM analysis + graph-viewer diarization-support gaps | Backlog |
| `AUTORESEARCH_LEARNINGS_FOR_V3.md` | Rolling failure-mode catalogue from #907 children — spec input for v3 (#921) | Reference |
| `AUTORESEARCH_NEXT_PHASE_AGENT_PLAN.md` | 3-phase 2-agent parallelization plan for post-v2.1 autoresearch work | Reference |
| `AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md` | Dependency map across #907 next-phase open issues (#921/#924/#927/#928-31/#932/#933/#923) | Reference |
| `DGX_NEXT_STEPS.md` | Living strategic doc on what runs on DGX vs local, vLLM vs Ollama, offload decision frame | Living |
| `EXPLORE_EXPANSION_IDEAS.md` | Explore-tab feature brainstorming | Idea |
| `HOME_AI_HARDWARE_PLAN.md` | Local hardware / on-prem AI plan | Idea |
| `LORA_HYBRID_PIPELINE_PLAN.md` | LoRA + hybrid pipeline exploration | Idea |
| `METRICS_DOCS_AND_DASHBOARD_V2.md` | Metrics docs / dashboard redesign | Partial |
| `POST_REINGESTION_PLAN.md` | Post-pipeline-rev validation plan | Reference |
| `PROD_RUN_ANALYSIS_100EP.md` | 100-episode production run retrospective | Reference |
| `QUALITY_IMPROVEMENTS_BACKLOG.md` | Quality / GI / KG improvement backlog | Backlog |
| `RESEARCH_POWERED_REGISTRY_PLAN.md` | Research-driven registry concept | Idea |
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

## Guidelines

- Documents here are **not** part of the official documentation site.
- Documents may be incomplete, outdated, or experimental.
- Periodically review and either:
  1. Promote to the appropriate doc category (PRD / RFC / ADR / guide / release note).
  2. Delete if obsolete or superseded.
  3. Keep as backlog / reference if it still has signal.

## Recent cleanups

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
