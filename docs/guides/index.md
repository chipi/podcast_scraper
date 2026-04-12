# Guides

Practical guides for using and developing Podcast Scraper.

## Quick Start

| Guide | Description |
| ------- | ------------- |
| [Quick Reference](QUICK_REFERENCE.md) | Common commands cheat sheet |
| [Troubleshooting](TROUBLESHOOTING.md) | Common issues and solutions |
| [Glossary](GLOSSARY.md) | Key terms and concepts |

## Configuration

| Reference | Description |
| --------- | ----------- |
| [Configuration API](../api/CONFIGURATION.md) | `Config`, env vars, YAML — [Twelve-factor (config)](../api/CONFIGURATION.md#twelve-factor-app-alignment-config), [Download resilience](../api/CONFIGURATION.md#download-resilience) (retries, Issue #522, presets, CLI parity), `failure_summary` in `run.json` |
| [CLI](../api/CLI.md) | Flags including `--http-retry-total`, `--episode-retry-max`, etc. (Control Options) |

## Development

| Guide | Description |
| ------- | ------------- |
| [Development Guide](DEVELOPMENT_GUIDE.md) | Development environment setup, workflow, and [GI/KG viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) — `make serve` / `serve-api` / `serve-ui`, `make test-ui-e2e` |
| [Polyglot repository guide](POLYGLOT_REPO_GUIDE.md) | Python root vs `web/gi-kg-viewer/`, env files, Makefile targets for the viewer |
| [Server Guide](SERVER_GUIDE.md) | FastAPI: `/api/*` (artifacts, search, explore, Corpus Library, index rebuild), OpenAPI `/docs`, static SPA, tests under `tests/integration/server/` |
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
| [Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md) | Integration test guidelines |
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
| [RSS and feed ingestion](RSS_GUIDE.md) | How RSS URLs become `RssFeed` and `Episode` objects: HTTP, caches, parsing, selection, multi-feed; entry point for future non-RSS ingestion docs |
| [Semantic Search](SEMANTIC_SEARCH_GUIDE.md) | RFC-061 corpus vector index: config (`vector_search`), `search` / `index` CLIs, semantic `gi explore --topic` |
| [Grounded Insights](GROUNDED_INSIGHTS_GUIDE.md) | Grounded insights (insights + evidence quotes), enabling GIL, gi.json, CLI, schema; optional [browser viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) |
| [Knowledge Graph](KNOWLEDGE_GRAPH_GUIDE.md) | KG (entities, topics, relationships): PRD-019 / RFC-055–056, artifacts, `kg` CLI; same [browser viewer](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) for `kg.json` |
| [Preprocessing Profiles](PREPROCESSING_PROFILES_GUIDE.md) | Preprocessing profiles (`cleaning_v4`, `cleaning_hybrid_after_pattern`, …) for transcript cleaning and hybrid_ml MAP input (RFC-042 / Issue #419) |
| [Docker Service Guide](DOCKER_SERVICE_GUIDE.md) | Running podcast_scraper as a service-oriented Docker container |
| [Docker Variants Guide](DOCKER_VARIANTS_GUIDE.md) | LLM-only vs ML-enabled Docker image variants |

## Evaluation and baselines

| Guide | Description |
| ------- | ----------- |
| [Experiment Guide](EXPERIMENT_GUIDE.md) | Datasets, baselines, experiments, promotion, and quality evaluation (RFC-041) |
| [Metrics Guide](METRICS_GUIDE.md) | Pipeline `metrics.json`, download-resilience counters, links to run tracking and CI metrics |
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
| [Agent-Browser Closed Loop](AGENT_BROWSER_LOOP_GUIDE.md) | Browser loops: automated E2E (`make test-ui-e2e`) + live co-development (Chrome DevTools MCP) |
| [Agent-Pipeline Feedback Loop](AGENT_PIPELINE_LOOP_GUIDE.md) | Python pipeline loops: CI diagnosis, acceptance testing, `--monitor` real-time feedback, `metrics.json` post-mortem |
| [Documentation Agent Guide](DOCUMENTATION_AGENT_GUIDE.md) | Documentation workflows |
