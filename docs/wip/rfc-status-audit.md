# RFC Status Audit

**Date**: 2026-04-03
**Purpose**: Assess whether each RFC's status field accurately reflects
implementation reality.

## Summary

- **61 RFCs** total (001–062, no 014)
- **35 Completed** (status already correct)
- **2 Narrative** (RFC-015, RFC-041 — phases complete, CI pending)
- **24 Draft** — assessed below

## Completed RFCs (no action needed)

These RFCs are already marked Completed/Implemented and verified:

| RFC | Title | Status |
| :--- | :--- | :--- |
| 001 | Workflow Orchestration | Completed |
| 002 | RSS Parsing & Episode Modeling | Completed |
| 003 | Transcript Download Processing | Completed |
| 004 | Filesystem Layout & Run Management | Completed |
| 005 | Whisper Integration Lifecycle | Completed |
| 006 | Whisper Screenplay Formatting | Completed |
| 007 | CLI Interface & Validation | Completed |
| 008 | Configuration Model & Validation | Completed |
| 009 | Progress Reporting Integration | Completed |
| 010 | Automatic Speaker Name Detection | Completed |
| 011 | Per-Episode Metadata Document Generation | Completed |
| 012 | Episode Summarization (Local Transformers) | Completed |
| 013 | OpenAI Provider Implementation | Completed |
| 016 | Modularization for AI Experiments | Complete |
| 017 | Prompt Management and Loading | Completed |
| 018 | Test Structure Reorganization | Completed |
| 019 | E2E Test Infrastructure | Complete |
| 020 | Integration Test Infrastructure | Completed |
| 021 | Modularization Refactoring Plan | Completed |
| 022 | Environment Variable Candidates Analysis | Completed |
| 024 | Test Execution Optimization | Completed |
| 025 | Test Metrics and Health Tracking | Completed |
| 026 | Metrics Consumption and Dashboards | Completed |
| 028 | ML Model Preloading and Caching | Completed |
| 029 | Provider Refactoring Consolidation | Completed |
| 030 | Python Test Coverage Improvements | Completed |
| 031 | Code Complexity Analysis Tooling | Completed |
| 032 | Anthropic Provider Implementation | Implemented |
| 033 | Mistral Provider Implementation | Completed |
| 034 | DeepSeek Provider Implementation | Completed |
| 035 | Google Gemini Provider Implementation | Completed |
| 036 | Grok Provider Implementation | Completed |
| 037 | Ollama Provider Implementation | Completed |
| 039 | Development Workflow with Worktrees & CI | Completed |

## Narrative-Status RFCs

| RFC | Title | Current Status | Assessment | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| 015 | AI Experiment Pipeline | Phase 1–3 complete, CI pending | Core pipeline, experiment runner, configs, eval scoring all exist. CI integration (auto-run on PR) not done. | **Partial** — flag "CI integration pending" |
| 041 | ML Benchmarking Framework | Phase 0–1 complete, CI pending | `data/eval/`, `scripts/eval/`, golden datasets, baselines, comparison scripts all exist. Automated CI benchmarking not wired. | **Partial** — flag "CI integration pending" |

## Draft RFC Audit

| RFC | Title | Current Status | Assessed Status | Evidence / Gaps |
| :--- | :--- | :--- | :--- | :--- |
| 023 | README Acceptance Tests | Draft | **Completed (alt impl)** | Script-based acceptance (`scripts/acceptance/`, `config/acceptance/`, `make test-acceptance`) implements the decision (fourth acceptance tier as final CI gate) via a more powerful YAML-config system instead of pytest markers. |
| 027 | Pipeline Metrics Improvements | Draft | **Partial** | Rich metrics in `workflow/metrics.py` with JSON export, `JSONLEmitter`, per-episode status. CSV export NOT found. Two-tier logging partially present. |
| 038 | Continuous Review Tooling | Draft | **Partial** | Dependabot config EXISTS. pydeps/coupling analysis EXISTS (`scripts/tools/analyze_dependencies.py`). Pre-release checklist script (`make pre-release`) NOT found. |
| 040 | Audio Preprocessing Pipeline | Draft | **Completed** | Full implementation: `preprocessing/audio/ffmpeg_processor.py`, opus codec, audio caching (`cache.py`), factory pattern, integration + E2E tests. Should be marked Completed. |
| 042 | Hybrid Summarization Pipeline | Proposed | **Completed** | Full hybrid MAP-REDUCE in `summarizer.py`, `hybrid_ml_provider.py`, Ollama/local backends, eval configs. Should be marked Completed. |
| 043 | Automated Metrics Alerts | Draft | **Partial** | `scripts/dashboard/generate_metrics.py` produces alerts, nightly CI appends to `$GITHUB_STEP_SUMMARY`. Automated PR comments NOT implemented. |
| 044 | Model Registry | Draft | **Completed** | `model_registry.py` with `ModelRegistry`, `ModelCapabilities`, used across summarizer, ML provider, hybrid provider, embedding loader, config. Should be marked Completed. |
| 045 | ML Model Optimization Guide | Draft | **Completed** | `cleaning_v4` profile, `is_junk_line`, `strip_episode_header`, `anonymize_speakers`, `register_profile` all implemented. RFC itself serves as the guide. |
| 046 | Materialization Architecture | Draft | **Completed** | `data/eval/materialized/` with datasets, `scripts/eval/materialize_dataset.py`, `materialize_baseline.py`, experiment config integration. Should be marked Completed. |
| 047 | Run Comparison Visual Tool | Draft | **Completed** | Streamlit app at `tools/run_compare/app.py`, `compare_runs.py` script, Makefile targets. Should be marked Completed. |
| 048 | Eval-App Alignment | Draft | **Completed** | Fingerprinting in `evaluation/fingerprint.py`, single-path provider architecture, explicit params in eval configs, scorer read-only pattern. Should be marked Completed. |
| 049 | GIL Core Concepts & Data Model | Draft | **Completed** | `src/podcast_scraper/gi/` package: pipeline, grounding, contracts, schema, explore, load, quality metrics. `gi.schema.json` exists. Should be marked Completed. |
| 050 | GIL Use Cases & Consumption | Draft | **Partial** | `gi explore` and `gi query` implemented. `gi list` NOT found. Insight Explorer pattern partially present. |
| 051 | Database Projection (GIL & KG) | Draft | **Not Started** | No Postgres integration, no SQL migrations, no `gi export --target postgres`. Only JSON/NDJSON export exists. |
| 052 | Locally Hosted LLMs with Prompts | Draft | **Completed** | Prompt directories under `src/podcast_scraper/prompts/` including `ollama/`, Ollama provider implemented, prompt store exists. Should be marked Completed. |
| 053 | Adaptive Summarization Routing | Draft | **Not Started** | No episode profiling, no routing logic as described. Only basic MAP-REDUCE short-input routing exists (generic, not RFC-053). |
| 054 | E2E Mock Response Strategy | Draft | **Partial** | E2E mock server and per-URL error behavior exist. Full composable ResponseProfile/Router architecture NOT implemented. |
| 055 | KG Core Concepts & Data Model | Draft | **Completed** | `src/podcast_scraper/kg/` package: pipeline, llm_extract, schema, corpus. `kg.schema.json` exists. CLI wired. Should be marked Completed. |
| 056 | KG Use Cases & Consumption | Draft | **Partial** | `kg validate`, `kg inspect`, `kg export`, `kg entities`, `kg topics` implemented. `kg explore` and `kg list` NOT found. |
| 057 | AutoResearch Optimization Loop | Draft | **Partial** | `autoresearch/prompt_tuning/` exists with `program.md`, `eval/score.py`. Track B (ML params) NOT set up. No evidence of completed runs. |
| 058 | Audio Speaker Diarization | Draft | **Not Started** | No diarization code, no pyannote integration, no `[diarize]` extra in pyproject.toml. |
| 059 | Speaker Detection Refactor | Draft | **Partial** | Package stub at `src/podcast_scraper/speaker_detectors/` (empty: "Stage 0"). Original `speaker_detection.py` unchanged. |
| 060 | Diarization-Aware Commercial Cleaning | Draft | **Not Started** | No `CommercialDetector`, no `cleaning/commercial/` module. Only the four-phrase `remove_sponsor_blocks` exists. |
| 061 | Semantic Corpus Search | Draft | **Completed (Phase 1 CLI)** | `search/` package: FAISS store, chunker, `index_corpus`, `search`/`index` CLIs; config + YAML; `gi explore --topic` uses `<output_dir>/search` when present; docs `SEMANTIC_SEARCH_GUIDE.md`. Sentence-boundary chunking (ADR-055) still tokenizer-based only. Phase 2 (Qdrant/service) out of scope. |
| 062 | GI/KG Viewer v2 | Implemented | **Completed (M1–M7)** | Server (`src/podcast_scraper/server/`), Vue SPA (`web/gi-kg-viewer/`), `podcast serve` CLI, Playwright E2E, v1 removed. |

## Recommendations Summary

**Updated to Completed (11 RFCs) — DONE 2026-04-03:**
RFC-023, RFC-040, RFC-042, RFC-044, RFC-045, RFC-046, RFC-047, RFC-048,
RFC-049, RFC-052, RFC-055
*(Status fields updated in RFC files; moved from Open to Completed in
`docs/rfc/index.md`)*

**Remain as Open / Partial (flagged gaps) (9 RFCs):**
RFC-015, RFC-027, RFC-038, RFC-041, RFC-043, RFC-050,
RFC-054, RFC-056, RFC-057, RFC-059

**Remain as Draft / Not Started (5 RFCs):**
RFC-051, RFC-053, RFC-058, RFC-060, RFC-062
