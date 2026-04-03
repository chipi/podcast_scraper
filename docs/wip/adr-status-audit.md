# ADR Status Audit

**Date**: 2026-04-03
**Purpose**: Verify that each ADR's status field accurately reflects
whether the decision has been implemented in the codebase.

## Summary

- **66 ADRs** total (001–066)
- **60 Accepted** — most verified as implemented
- **3 Proposed** — assessed below
- **3 Accepted but NOT YET implemented** — flagged below

Numbers below use the **post-renumber** ADR numbering (2026-04-03).

## Proposed ADRs (need status decision)

| ADR | Title | Current | Assessed | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| 056 | Composable E2E Mock Response Strategy | Proposed | **Keep Proposed** | E2E mock infrastructure exists but the full composable ResponseProfile/Router architecture from this ADR is NOT implemented. The RFC (054) is still Draft/Partial. |
| 055 | Adaptive Summarization Routing | Proposed | **Keep Proposed** | No episode profiling or routing logic exists. RFC-053 is Not Started. |
| 048 | Centralized Model Registry | ~~Proposed~~ → **Accepted** | **Promoted** | `ModelRegistry` with `ModelCapabilities` fully implemented in `model_registry.py`, used across summarizer, ML provider, hybrid provider, embedding loader, and config. Status updated 2026-04-03. |

## Accepted ADRs That Are NOT YET Implemented

These ADRs were marked Accepted (based on the RFC decision) but the
corresponding code does not yet exist:

| ADR | Title | Status | Evidence | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| 062 | Sentence-Boundary Transcript Chunking | Accepted | No sentence-boundary chunking for search indexing in `src/podcast_scraper/search/`. Summarizer has sentence splitting but that is a different context. RFC-061 is Partial. | **Keep Accepted** (decision is sound; implementation in progress with RFC-061) |
| 063 | Transparent Semantic Upgrade for gi explore | Accepted | `gi explore` does NOT use `VectorStore`/FAISS. Code explicitly treats semantic search as future. RFC-061 is Partial. | **Keep Accepted** (decision is sound; blocked on RFC-061 completion) |
| 064 | Canonical Server Layer | Accepted | No `src/podcast_scraper/server/` exists. No `podcast serve` CLI. RFC-062 is Not Started. | **Keep Accepted** (decision is sound; implementation is RFC-062 scope) |
| 065 | Vue 3 + Vite + Cytoscape Frontend | Accepted | No `.vue` files in repo. Only legacy `web/gi-kg-viz/`. RFC-062 is Not Started. | **Keep Accepted** (decision is sound; implementation is RFC-062 scope) |
| 066 | Playwright for UI E2E Testing | Accepted | No Playwright config in repo. RFC-062 is Not Started. | **Keep Accepted** (decision is sound; implementation is RFC-062 scope) |
| 021 | Acceptance Test Tier as Final CI Gate | Accepted | `tests/acceptance/` with pytest marker does NOT exist. Script-based acceptance exists but is a different shape. RFC-023 is Partial. | **Keep Accepted** (decision is sound; pytest-marker implementation pending) |
| 049 | Materialization Boundary for Eval Inputs | Accepted | Materialization code exists (`materialize_dataset.py`, `data/eval/materialized/`). The boundary concept is implemented in practice. RFC-046 is Completed. | **Correctly Accepted** (actually implemented) |
| 050 | Single Code Path for Eval and App | Accepted | Fingerprinting exists, single-path architecture enforced. RFC-048 is Completed. | **Correctly Accepted** (actually implemented) |
| 054 | Relational Postgres Projection | Accepted | No Postgres code exists. RFC-051 is Not Started. | **Keep Accepted** (decision is sound; implementation is future) |
| 057 | AutoResearch Thin Harness | Accepted | `autoresearch/prompt_tuning/` exists with program.md and score.py. Partial implementation. | **Keep Accepted** (partially implemented) |
| 058 | Additive pyannote Diarization | Accepted | No diarization code, no pyannote, no `[diarize]` extra. RFC-058 is Not Started. | **Keep Accepted** (decision is sound; implementation is future) |
| 059 | Confidence-Scored Multi-Signal Commercial Detection | Accepted | No `CommercialDetector`, no `cleaning/commercial/`. RFC-060 is Not Started. | **Keep Accepted** (decision is sound; implementation is future) |

**Note**: "Accepted" for ADRs means the architectural decision has been
*accepted* (ratified), not necessarily *implemented*. This is correct
ADR semantics — an ADR records a decision, not a delivery. The table
above is for visibility into implementation readiness.

## Accepted ADRs — Verified Implemented (spot-check)

All other Accepted ADRs (001–044, 048–054) were spot-checked and
confirmed implemented:

| ADR | Title | Verified |
| :--- | :--- | :--- |
| 001–010 | Core pipeline decisions | Yes — workflow, RSS, filesystem, Whisper, speaker detection, metadata, summarization all in codebase |
| 011–013 | Provider pattern decisions | Yes — unified provider, protocol discovery, technology naming all in `providers/` |
| 014 | Externalized Prompt Management | Yes — `prompts/` with Jinja2 templates, `store.py` |
| 015 | Secure Credential Injection | Yes — env-based secrets throughout |
| 016–020 | Dev workflow decisions | Yes — worktrees, CI tiers, isolated envs, squash-merge |
| 021 | Standardized Test Pyramid | Yes — `tests/unit/`, `tests/integration/`, `tests/e2e/` |
| 022–023 | Test health & metrics | Yes — flaky defense, dashboards |
| 024–025 | Experiment configuration | Yes — `data/eval/configs/`, baselines |
| 026 | Golden Dataset Versioning | Yes — `data/eval/references/` |
| 027–029 | Provider fingerprinting & profiles | Yes — `fingerprint.py`, `profiles.py` |
| 030–031 | Benchmarking strategy & quality gates | Yes — eval scripts, heuristic checks |
| 032–035 | Audio preprocessing | Yes — `preprocessing/audio/`, ffmpeg, caching, opus |
| 036–038 | Hybrid summarization | Yes — MAP-REDUCE, local LLM backends, prompt contract |
| 039–041 | Continuous review tooling | Yes — Dependabot, pydeps, (pre-release partial) |
| 042 | Proactive Metric Regression Alerting | Yes — `generate_metrics.py` with alerts |
| 043 | Unified Provider Metrics Contract | Yes — `ProviderCallMetrics` pattern |
| 044 | Unified Retry Policy with Metrics | Yes — centralized retry |
| 048 | MPS Exclusive Mode | Yes — `mps_exclusive` in config, workflow serialization |
| 049 | Per-Capability Provider Selection | Yes — independent provider fields in config |
| 050 | Per-Episode JSON Artifacts | Yes — `*.gi.json`, `*.kg.json` pattern |
| 051 | Separate GIL and KG Layers | Yes — `gi/` and `kg/` packages |
| 052 | Grounding Contract | Yes — `grounding.py`, `contracts.py` |
| 053 | VectorStore Protocol | Yes — `search/protocol.py` |
| 054 | FAISS Phase 1 | Yes — `search/faiss_store.py` |

## Action Items

1. **ADR-048** (was ADR-047): ~~Update status from "Proposed" to "Accepted"~~ — **DONE 2026-04-03** (updated during renumber; stale "not yet implemented" note also fixed)
2. **ADR-055, 056, 057, 058, 059, 060, 063, 065, 066**: Keep Accepted
   (decision ratified; implementation pending per their respective RFCs)
3. **ADR-045, 046**: Keep Proposed (not yet implemented)

*Note: ADR numbers in this document use the **post-renumber** scheme.*
