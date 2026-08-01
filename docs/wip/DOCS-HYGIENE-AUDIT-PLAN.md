# Docs-hygiene audit — WIP tree (2026-08-01)

Classification of `docs/wip/` docs, each verified against closed issues / commits / code /
file existence. **This is the audit; execution (deletion) is gated on operator approval** —
`classify → approve → execute`, UNCERTAIN stays. Scope here is the WIP tree (~63 docs); the
ADR/RFC/guides sweep (rest of the ~439) is still pending.

## DONE — superseded/shipped, safe to delete (verified)

| file | why DONE (verification) |
|---|---|
| player/1069-SCRAPE-ON-DEMAND-SCOPE-ANALYSIS.md | #1069 CLOSED; doc says superseded |
| 1129-1130-OUTBOUND-HTTP-FACTORY.md | table "AS-SHIPPED on main", rows ✅ |
| player/1144-DISAGREEMENT-DETECTOR-FEASIBILITY.md | build+measurement record; enricher shipped gated-dark |
| 968-SPEACHES-RESEARCH.md | recommendation landed in `902c3be6` |
| ADVISOR-REVIEW-FIXES-naming4.md | shipped in `b24608fa` (naming-4) |
| AUTORESEARCH_JUDGE_ITERATION_ROADMAP.md | round-1 actions landed; superseded by promoted eval-report |
| DOC-FRESHNESS-AUDIT-2026-07.md | self-describes ephemeral; superseded by this hygiene effort |
| player/EPIC-2-CONSUMER-APP-PLAN.md | Epic 2 (#1077) CLOSED/shipped |
| player/EPIC2-playtest-backlog.md | folded into Epic 3 (#1093 CLOSED) |
| player/EPIC3-parked-decisions.md | all items DECIDED; Epic 3 shipped |
| player/EPIC3-proposal.md | became Epic 3 (#1093 CLOSED, PRD-043) |
| EVAL-1191-salience-ranking-100ep-2026-07-29.md | #1191 CLOSED 2026-07-30 |
| FIXTURE-CORPUS-FULL-PRODUCT-SUBSTRATE-2026-07.md | superseded by V3 promotion |
| GB10_GPU_ISOLATION_RESEARCH_2026-06-19.md | recommends close #1000; #1000 CLOSED |
| ISSUE-382-PR-BODY-DRAFT.md | #382 CLOSED |
| JSON-RELIABILITY-DEEP-RESEARCH-2026-06-18.md | #912 CLOSED |
| player/LEARNING-PLATFORM-GAP-ANALYSIS-2026-07.md | 2026-07-05 snapshot; superseded |
| MOSS-PRODUCTION-READY-PLAN.md | #1177 + #1174 CLOSED |
| ORRERY-ALLOY-MIGRATION-NOTE.md | matching commit in orrery repo |
| PLAN-llm-host-guest-role.md | implemented in `resolution.py`/`pipeline.py` |
| POST-LAUNCH-FIXLIST.md | items verified done / #1263 CLOSED |
| POST_RFC097_DEV_PROD_REMOVAL.md | committed `ce029849` + follow-up |
| RELABEL-ONLY-OPTION-PROPER-JOB.md | `relabel_only` present in `config.py` |
| player/RFC-LANDSCAPE-FOR-PLATFORM.md | surveyed RFCs shipped |
| player/SERVER-SIDE-GAP-ANALYSIS.md | Foundation/Epic 2/3 shipped |
| SLICE-1-1173-1191-PLAN.md | #1173 + #1191 CLOSED |
| SPEC_KG_GI_ONTOLOGY_REVIEW_2026-06-20.md | superseded by round-3; #1036 CLOSED |
| SPEC_KG_GI_ONTOLOGY_V2_2026-06-20.md | superseded by round-3; #1036 CLOSED |
| SPEC_KG_GI_ONTOLOGY_V2_ROUND3_2026-06-20.md | adopted spec, now shipped (#1036 CLOSED) |
| search-v3/TEST-PYRAMID-AUDIT-2026-07-21.md | follow-ups resolved (doc's own note) |
| V2.4-STATUS-overnight.md | superseded by `b24608fa` |
| V3-ENRICHER-CONTENT-DESIGN-2026-07.md | v3 fixtures built with real enrichment |
| V3-PROMOTION-MIGRATION-2026-07.md | `FIXTURES_VERSION`=v3, no orphaned v2 |

## KEEP — live reference / unbuilt-but-planned (do not touch)
knowledge-retention/* (unbuilt package — 00-HANDOFF, 00-VISION, PLAN, PRD-034/035/036/037,
RFC-081/086/087/088, UXS), 1273-largev3-int8 (#1273 OPEN), 2026-07-26-cloudflare-waf-ratelimit,
CODEBASE-REVIEW-2026-07-17 (#1162 OPEN), CORPUS-UPGRADE-2.7-RUNBOOK (v2.7.0.dev0), 
DGX-WHISPER-STALL-INVESTIGATION (paused), EVAL-DATA-PRIVATE-REPO-SCAFFOLD, FLIGHTCAST-PANEL-LIMITATIONS,
graph-v3/HARDEN-FOLLOWUPS + REPRODUCIBILITY, LABELING-RESIDUAL-UNKNOWNS-CENSUS, LABELING-TIER3-COMPLEXITY,
MCP-O11Y-REALIGNMENT-GAP-ANALYSIS, OBSERVABILITY-INTEGRATION-REVIEW (P0 still true),
SPEAKER-PIPELINE-SUBSYSTEM-AUDIT, SPEC_KG_GI_ONTOLOGY_V3_WISHLIST, VISION-search-and-intelligence,
AUTORESEARCH_EVAL_PLAYBOOK, observability-correlation-id-enhancement, and this session's
naming-arc + guest-recall findings docs.

## Notable finding (needs reconcile, not deletion)
All four `knowledge-retention/PRD-03x` and `RFC-08x` **placeholder numbers collide with real,
shipped docs** (e.g. `RFC-088` is the shipped enrichment-layer-architecture RFC, not corpus-scout).
The package's own docs flag "confirm and renumber before merge" — real and unreconciled.

## NOT covered (equal weight)
- ADR/RFC/PRD/guides trees (the other ~376 docs) — not classified this pass.
- Dangling cross-refs / unindexed enumeration — the strict `make docs` build (green, `MAKE_DOCS_EXIT=0`)
  only surfaces INFO-level anchor warnings (CORPUS_REPROCESSING, OBSERVABILITY_RUNBOOK, PROD_RUNBOOK,
  RFC-082); a full dangling-ref sweep is not done.
- **Execution (deletion of the DONE list) is NOT done** — gated on operator approval per the documented process.
