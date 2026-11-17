# Guides

Practical guides for using and developing Podcast Scraper.

## Quick Start

| Guide | Description |
| ------- | ------------- |
| [Installation Guide](INSTALLATION_GUIDE.md) | Install paths (see README), first **`--profile` + `--config` + `--feeds-spec`** run |
| [Quick Reference](QUICK_REFERENCE.md) | Common commands cheat sheet |
| [Troubleshooting](TROUBLESHOOTING.md) | Common issues and solutions |
| [Glossary](GLOSSARY.md) | Key terms and concepts |

## Configuration

| Reference | Description |
| --------- | ----------- |
| [Configuration API](../api/CONFIGURATION.md) | `Config`, env vars, YAML — [Twelve-factor (config)](../api/CONFIGURATION.md#twelve-factor-app-alignment-config), [Download resilience](../api/CONFIGURATION.md#download-resilience) (retries, Issue #522, presets, CLI parity), `failure_summary` in `run.json` |
| [CLI](../api/CLI.md) | Flags; [Quick Start](../api/CLI.md#quick-start) (**`--profile`**, **`--config`**, **`--feeds-spec`**); `--http-retry-total`, `--episode-retry-max`, etc. |

## Development

| Guide | Description |
| ------- | ------------- |
| [Development Guide](DEVELOPMENT_GUIDE.md) | Development environment setup, workflow, and [Workbench](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) — `make serve` / `serve-api` / `serve-ui`, `make test-ui-e2e` |
| [Release Playbook](RELEASE_PLAYBOOK.md) | Standing plan before a public tag: eval/profiles policy (major vs minor), docs gates, release notes pattern, alignment with `vX.Y.Z` tags |
| [Polyglot repository guide](POLYGLOT_REPO_GUIDE.md) | Python root vs `web/workbench/`, env files, Makefile targets for the viewer |
| [Server Guide](SERVER_GUIDE.md) | FastAPI: `/api/*` (artifacts, search, Corpus Library and Digest, index rebuild, feeds, operator config, jobs), OpenAPI `/docs`, static SPA, tests under `tests/integration/server/` |
| [Pipeline and Workflow Guide](PIPELINE_AND_WORKFLOW.md) | Pipeline flow, module roles, quirks, run tracking |
| [Git Worktree Guide](GIT_WORKTREE_GUIDE.md) | Git worktree-based development workflow |
| [Dependencies Guide](DEPENDENCIES_GUIDE.md) | Third-party dependencies and rationale |
| [Markdown Linting](MARKDOWN_LINTING_GUIDE.md) | Markdown style and linting practices |
| [Engineering Process](ENGINEERING_PROCESS.md) | How the lifecycle of features, architectural changes and technical decisions is managed through the PRD / UXS / RFC / ADR chain; read before editing PRDs, UXS, RFCs, ADRs or architecture narrative |
| [Dev/prod env-detect removal](DEV_PROD_ENV_DETECT_REMOVAL.md) | Why the runtime "is this a test run?" heuristic that silently flipped 24 model defaults between TEST and PROD values was removed (2026-06-23) |
| [Mounting the private eval research repo](EVAL_RESEARCH_MOUNT.md) | How to check out the private eval research repo alongside the application so one agent or person sees both in one tree |
| [Splitting a surface into a private repo](SPLITTING_A_SURFACE_INTO_A_PRIVATE_REPO.md) | Rules for moving a whole surface (infra, ops, a corpus, a research area) out of this repo into a private one, taken from the eval research split |
| [Segfault Mitigation](SEGFAULT_MITIGATION.md) | Diagnosing and mitigating segmentation faults during pipeline execution, particularly at process shutdown |

## Corpus operations

| Guide | Description |
| ------- | ------------- |
| [SRE book infra critique](SRE_BOOK_INFRA_CRITIQUE.md) | Reliability rubric (SRE themes): SLIs/SLOs, error budget, toil, alerting, change risk, incidents — for reviewing runbooks, workflows, and ops design |
| [Corpus change method](CORPUS_CHANGE_METHOD.md) | How every change to what the corpus serves is planned, judged and shipped |
| [Corpus integrity repair runbook](CORPUS_INTEGRITY_REPAIR_RUNBOOK.md) | Ordered, gated sequence for repairing a corpus damaged by the defects fixed in epic #1657: which detector finds each, which repair fixes it, and which repairs fix nothing |
| [Corpus reprocessing runbook](CORPUS_REPROCESSING.md) | How to rebuild an existing corpus's artifacts by re-running the real profile-driven pipeline (stages an installed extension provides, such as diarization, cleaning and GI, apply only with that extension) |
| [Corpus upgrade framework](CORPUS_UPGRADE.md) | Managed, idempotent migrations for moving a deployed corpus across releases (#862) |
| [Onboarding a new language](LANGUAGE_ONBOARDING_RUNBOOK.md) | Measurement runbook: the order of work and the measurement traps for onboarding a new language |

## Testing

| Guide | Description |
| ------- | ------------- |
| [Testing Strategy](../architecture/TESTING_STRATEGY.md) | Pyramid, pytest layers, and **Playwright** as additive browser UI E2E |
| [Testing Guide](TESTING_GUIDE.md) | Commands, markers, and [Browser E2E](TESTING_GUIDE.md#browser-e2e-workbench) (`make test-ui-e2e`) |
| [Unit Testing Guide](UNIT_TESTING_GUIDE.md) | Unit test patterns and mocking |
| [Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md) | Integration test guidelines; [FastAPI / CIL / bridge / lift](INTEGRATION_TESTING_GUIDE.md#fastapi-cil-bridge-search-lift) |
| [E2E Testing Guide](E2E_TESTING_GUIDE.md) | pytest E2E server/ML; [Playwright](E2E_TESTING_GUIDE.md#browser-e2e-playwright) for the viewer |
| [Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md) | Test prioritization |
| [Gemini E2E Testing](GEMINI_E2E_TESTING.md) | How Gemini E2E tests use a fake SDK client that routes calls to the E2E mock server over HTTP |
| [Testing strategy — ML providers](testing-strategy-ml.md) | Where to mock and where not to for ML providers and backends: mock at our abstraction boundary, never at a library-internal name |
| [Real-Episode Validation](REAL_EPISODE_VALIDATION.md) | The "test with real episodes before push" pattern: unit tests prove code paths route correctly, not that a feature works against real provider responses and transcripts |
| [GI/KG Manual Test Plan](GI_KG_MANUAL_TEST_PLAN.md) | Close-the-loop validation: run the real pipeline on 2–3 episodes of a familiar feed, then inspect `*.kg.json` (and `*.gi.json` where an extension writes it) for plausibility, schema health and CLI ergonomics |
| [Operator post-deploy smoke test](OPERATOR_SMOKE_TEST.md) | Live smoke test of the operator surface after a deploy (unauthed and authed specs under `web/workbench/e2e/live/`) |
| [Pipeline check runbook](PIPELINE_CHECK.md) | `make pipeline-check` runs the pipeline under variants, records every decision and compares runs; the standard check before merging pipeline routing changes (#2287) |
| [Validating Corpus Fixes Fast](VALIDATING_CORPUS_FIXES_FAST.md) | Reference guide for validating changes whose correctness is a property of the corpus (speaker attribution, entity resolution, enrichment) rather than of a function |

## Provider System

| Guide | Description |
| ------- | ------------- |
| [Provider Deep Dives](PROVIDER_DEEP_DIVES.md) | Per-provider reference cards, benchmarks, and magic quadrant |
| [ML Model Comparison](ML_MODEL_COMPARISON_GUIDE.md) | Compare ML models: Whisper, spaCy, Transformers (BART/LED) |
| [Provider Configuration](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md) | Quick provider configuration reference |
| [Ollama Provider Guide](OLLAMA_PROVIDER_GUIDE.md) | Ollama installation, setup, troubleshooting, and testing |
| [Provider Implementation](PROVIDER_IMPLEMENTATION_GUIDE.md) | Implementing new providers |
| [ML Provider Reference](ML_PROVIDER_REFERENCE.md) | Technical reference for local ML models |
| [Protocol Extension](PROTOCOL_EXTENSION_GUIDE.md) | Extending protocols |

## Features

| Guide | Description |
| ------- | ------------- |
| [RSS and feed ingestion](RSS_GUIDE.md) | How RSS URLs become `RssFeed` and `Episode` objects: HTTP, caches, parsing, selection, multi-feed; entry point for future non-RSS ingestion docs |
| [Semantic Search](SEMANTIC_SEARCH_GUIDE.md) | RFC-061 corpus vector index; **`GET /api/search`** over transcript chunks, summary bullets and KG entities and topics; the chunk-to-Insight lift applies only with an extension that writes GIL |
| [Knowledge Graph](KNOWLEDGE_GRAPH_GUIDE.md) | KG: **`kg.json`**, entities/topics/relationships; the **bridge** that aligns KG with GIL needs an extension that writes GIL; same [browser viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) |
| [Preprocessing Profiles](PREPROCESSING_PROFILES_GUIDE.md) | Preprocessing profiles (`cleaning_v4`, `cleaning_hybrid_after_pattern`, …) for transcript cleaning and hybrid_ml MAP input (RFC-042 / Issue #419) |
| [Docker Compose Guide](DOCKER_COMPOSE_GUIDE.md) | **Recommended** end-to-end stack (Workbench + API + on-demand pipeline jobs, mock RSS feeds for trying it) |
| [Docker Service Guide](DOCKER_SERVICE_GUIDE.md) | Running podcast_scraper as a single-container service (supervisor / systemd / scheduler-driven) |
| [Docker Variants Guide](DOCKER_VARIANTS_GUIDE.md) | LLM-only vs ML-enabled pipeline image tiers |
| [Audio Pipeline Guide](AUDIO_PIPELINE_GUIDE.md) | Operator and developer reference for transcription and API audio chunking; speaker attribution and commercial/sponsor cleaning are provided by an installed extension (Audio Wave 1 and Wave 2) |
| [Enrichment Layer Guide](ENRICHMENT_LAYER_GUIDE.md) | Operator/developer reference for the enrichment layer (RFC-088): typed, opt-in enrichers on top of KG and whatever installed extensions add; this repository registers one enricher |
| [Graph Visualization Guide](GRAPH_VISUALIZATION_GUIDE.md) | Every visual rule and business decision of the Workbench graph canvas, with the code path where it lives |

## Evaluation and baselines

| Guide | Description |
| ------- | ----------- |
| [Chip Huyen ML / AI critique](CHIP_HUYEN_ML_AI_CRITIQUE.md) | ML/AI rubric (seven themes); **experiments vs production** lenses and optional short output tables — inspired by *Designing Machine Learning Systems* and *AI Engineering* |
| [Live Pipeline Monitor](LIVE_PIPELINE_MONITOR.md) | Dev tooling: `--monitor`, RSS/CPU/stage dashboard or `.monitor.log`, `.pipeline_status.json`; optional `.[monitor]` memray + py-spy (RFC-065, #512) |
| [Tufte Chart Critique](TUFTE_CHART_CRITIQUE.md) | Rubric for critiquing a data visualization against Tufte's principles and returning specific fixes |

## AI Coding

| Guide | Description |
| ------- | ------------- |
| [Cursor AI Best Practices](CURSOR_AI_BEST_PRACTICES_GUIDE.md) | AI-assisted development |
| [Agent-Browser Closed Loop](AGENT_BROWSER_LOOP_GUIDE.md) | Browser loops: automated E2E (`make test-ui-e2e`) + live co-development (Chrome DevTools MCP); **user-reported UI bugs:** **symmetry** — re-validate in the same channel you used to reproduce (MCP ≠ replaced by pytest alone); plus tests ([obligatory validation](AGENT_BROWSER_LOOP_GUIDE.md#obligatory-validation-when-fixing-a-reported-ui-bug)) |
| [Agent-Pipeline Feedback Loop](AGENT_PIPELINE_LOOP_GUIDE.md) | Python pipeline loops: CI diagnosis, acceptance testing, `--monitor` real-time feedback, `metrics.json` post-mortem |
| [Documentation Agent Guide](DOCUMENTATION_AGENT_GUIDE.md) | Documentation workflows |
