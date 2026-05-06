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
| [Development Guide](DEVELOPMENT_GUIDE.md) | Development environment setup, workflow, and [GI/KG viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) — `make serve` / `serve-api` / `serve-ui`, `make test-ui-e2e` |
| [Release Playbook](RELEASE_PLAYBOOK.md) | Standing plan before a public tag: eval/profiles policy (major vs minor), docs gates, release notes pattern, alignment with `vX.Y.Z` tags |
| [Prod operator cheat sheet](PROD_OPERATOR_CHEAT_SHEET.md) | Daily production operations shortcut: deploy, health checks, incident triage, rollback, and key credential map |
| [Polyglot repository guide](POLYGLOT_REPO_GUIDE.md) | Python root vs `web/gi-kg-viewer/`, env files, Makefile targets for the viewer |
| [Server Guide](SERVER_GUIDE.md) | FastAPI: `/api/*` (artifacts, CIL, search with optional **`lifted`**, explore, Corpus Library, index rebuild), OpenAPI `/docs`, static SPA, tests under `tests/integration/server/` |
| [Pipeline and Workflow Guide](PIPELINE_AND_WORKFLOW.md) | Pipeline flow, module roles, quirks, run tracking |
| [Git Worktree Guide](GIT_WORKTREE_GUIDE.md) | Git worktree-based development workflow |
| [Dependencies Guide](DEPENDENCIES_GUIDE.md) | Third-party dependencies and rationale |
| [Markdown Linting](MARKDOWN_LINTING_GUIDE.md) | Markdown style and linting practices |

## Testing

| Guide | Description |
| ------- | ------------- |
| [Testing Strategy](../architecture/TESTING_STRATEGY.md) | Pyramid, pytest layers, and **Playwright** as additive browser UI E2E |
| [Testing Guide](TESTING_GUIDE.md) | Commands, markers, and [Browser E2E](TESTING_GUIDE.md#browser-e2e-gi-kg-viewer-v2) (`make test-ui-e2e`) |
| [Unit Testing Guide](UNIT_TESTING_GUIDE.md) | Unit test patterns and mocking |
| [Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md) | Integration test guidelines; [FastAPI / CIL / bridge / lift](INTEGRATION_TESTING_GUIDE.md#fastapi-cil-bridge-search-lift) |
| [E2E Testing Guide](E2E_TESTING_GUIDE.md) | pytest E2E server/ML; [Playwright](E2E_TESTING_GUIDE.md#browser-e2e-playwright) for the viewer |
| [Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md) | Test prioritization |

## Provider System

| Guide | Description |
| ------- | ------------- |
| [AI Provider Comparison](AI_PROVIDER_COMPARISON_GUIDE.md) | Compare all 9 providers: cost, quality, speed, privacy |
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
| [GIL / KG / CIL cross-layer](GIL_KG_CIL_CROSS_LAYER.md) | **RFC-072** map: **`bridge.json`**, CIL HTTP routes, semantic **lift**, offset verification, CLI/Make, and test entry points |
| [RSS and feed ingestion](RSS_GUIDE.md) | How RSS URLs become `RssFeed` and `Episode` objects: HTTP, caches, parsing, selection, multi-feed; entry point for future non-RSS ingestion docs |
| [Semantic Search](SEMANTIC_SEARCH_GUIDE.md) | RFC-061 corpus vector index; **`GET /api/search`**; **chunk-to-Insight lift** and **`verify-gil-chunk-offsets`** when GIL + index share transcript space |
| [Grounded Insights](GROUNDED_INSIGHTS_GUIDE.md) | GIL: **`gi.json`**, quotes, schema, CLI; **`bridge.json`** sibling for canonical ids; optional [browser viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) |
| [Knowledge Graph](KNOWLEDGE_GRAPH_GUIDE.md) | KG: **`kg.json`**, entities/topics/relationships; **bridge** aligns KG with GIL for APIs; same [browser viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) |
| [Preprocessing Profiles](PREPROCESSING_PROFILES_GUIDE.md) | Preprocessing profiles (`cleaning_v4`, `cleaning_hybrid_after_pattern`, …) for transcript cleaning and hybrid_ml MAP input (RFC-042 / Issue #419) |
| [Docker Compose Guide](DOCKER_COMPOSE_GUIDE.md) | **Recommended** way to run the platform end-to-end: viewer + API + on-demand pipeline jobs, all from your browser |
| [Docker Service Guide](DOCKER_SERVICE_GUIDE.md) | Running podcast_scraper as a single-container service (supervisor / systemd / scheduler-driven) |
| [Docker Variants Guide](DOCKER_VARIANTS_GUIDE.md) | LLM-only vs ML-enabled pipeline image tiers |

## Evaluation and baselines

| Guide | Description |
| ------- | ----------- |
| [Experiment Guide](EXPERIMENT_GUIDE.md) | Datasets, baselines, experiments, promotion, metrics, and quality evaluation (RFC-041) |
| [Evaluation Reports](eval-reports/index.md) | Quality sweeps: ROUGE, embeddings, report library |
| [Performance Guide](PERFORMANCE.md) | Performance considerations, optimization, and troubleshooting |
| [Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md) | Frozen release profiles: RSS, CPU%, wall time per stage (RFC-064) |
| [Optimization Workflow](OPTIMIZATION_WORKFLOW_GUIDE.md) | Data-driven process for investigating and solving performance/cost problems |
| [Live Pipeline Monitor](LIVE_PIPELINE_MONITOR.md) | Dev tooling: `--monitor`, RSS/CPU/stage dashboard or `.monitor.log`, `.pipeline_status.json`; optional `.[monitor]` memray + py-spy (RFC-065, #512) |
| [Performance Reports](performance-reports/index.md) | Published profile snapshots (tables, caveats) |

## AI Coding

| Guide | Description |
| ------- | ------------- |
| [Cursor AI Best Practices](CURSOR_AI_BEST_PRACTICES_GUIDE.md) | AI-assisted development |
| [Agent-Browser Closed Loop](AGENT_BROWSER_LOOP_GUIDE.md) | Browser loops: automated E2E (`make test-ui-e2e`) + live co-development (Chrome DevTools MCP); **user-reported UI bugs:** **symmetry** — re-validate in the same channel you used to reproduce (MCP ≠ replaced by pytest alone); plus tests ([obligatory validation](AGENT_BROWSER_LOOP_GUIDE.md#obligatory-validation-when-fixing-a-reported-ui-bug)) |
| [Agent-Pipeline Feedback Loop](AGENT_PIPELINE_LOOP_GUIDE.md) | Python pipeline loops: CI diagnosis, acceptance testing, `--monitor` real-time feedback, `metrics.json` post-mortem |
| [Documentation Agent Guide](DOCUMENTATION_AGENT_GUIDE.md) | Documentation workflows |
