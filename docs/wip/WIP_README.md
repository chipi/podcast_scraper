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
| `PHASE1-OPEN-WEIGHT-LLM-LANDSCAPE-2026-06.md` | #928 reframe Phase 1: open-weight LLM landscape (≤35B) → 6-candidate tier-1 shortlist (Qwen3.5:35b incumbent + Qwen3-30B-A3B-Instruct-2507 + DeepSeek-R1-Distill-32B + Magistral 1.2 + Mistral Small 4 + gemini-2.5-flash-lite cloud anchor). Phase 2 eval → #1016. | Active |
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
| `CLOUD-PROVIDER-RESILIENCE-E2E-GAP-1003.md` | Audit of cloud-LLM resilience E2E coverage; matrix + 5 follow-up gaps surfaced during #1003 (closed 2026-06-15) | Closed |

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
