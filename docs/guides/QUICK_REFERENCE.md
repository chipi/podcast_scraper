# Quick Reference

One-page cheat sheet for common podcast_scraper commands.

---

## Setup

```bash

# Initial setup

git clone https://github.com/chipi/podcast_scraper.git && cd podcast_scraper
python3 -m venv .venv && source .venv/bin/activate
make init                    # Install all dependencies (auto-uses wheels/spacy if *.whl present)
# Slow ML downloads once: make download-spacy-wheels  →  then make init picks it up

# ML model preloading (for tests)

make preload-ml-models       # Download Whisper, spaCy, transformers models

# Cache management

make backup-cache            # Backup .cache directory (saves to ~/podcast_scraper_cache_backups/)
make restore-cache           # Restore cache from backup (interactive)
make backup-cache-list       # List available backups
```

**Polyglot repo:** Python and `Makefile` at the root; GI/KG viewer UI under `web/gi-kg-viewer/`
(npm). See [Polyglot repository guide](POLYGLOT_REPO_GUIDE.md) for env file locations and viewer
commands.

**Twelve-factor style config:** Prefer **environment variables** for secrets and deploy-specific
values; use **YAML/JSON** for shared defaults. See [Configuration API — Twelve-factor alignment](../api/CONFIGURATION.md#twelve-factor-app-alignment-config).

---

## Diagnostic Commands (Issue #379)

```bash
# Run diagnostic checks
python -m podcast_scraper.cli doctor

# Include network connectivity check
python -m podcast_scraper.cli doctor --verbose

# LLM pricing YAML: show path, metadata, staleness (see docs/api/CLI.md)
python -m podcast_scraper.cli pricing-assumptions
make check-pricing-assumptions
```

## Daily Development

```bash

# Before coding

source .venv/bin/activate

# After making changes

make format                  # Auto-format code (black + isort)
make lint                    # Check style (flake8)
make type                    # Type check (mypy)
make quality                 # Code quality (complexity, docstrings, dead code, spelling)

# Before committing

make ci-fast                 # Fast CI gate (~6-10 min, default before push)
make ci                      # Full CI suite (+ Playwright, coverage enforce)
```

---

## Testing

| Command | What it does | Time |
| ------- | ------------ | ---- |
| `make test-unit` | Unit tests (parallel, network blocked) | ~30s |
| `make test-integration` | Integration tests (serial) | ~2min |
| `make test-e2e` | E2E tests (serial, real ML) | ~5min |
| `make test-fast` | Critical path only | ~1min |
| `make test` | All tests | ~8min |
| `make test-nightly` | Nightly tests (production models) | ~4hrs |
| `make test-ui` | Vitest unit tests for `web/gi-kg-viewer` TS utils (no browser) | ~1s |
| `make test-ui-e2e` | Playwright E2E for `web/gi-kg-viewer` (Firefox; installs browsers) | ~1–3 min |

```bash

# Run specific test file

pytest tests/unit/test_config.py -v

# Run specific test

pytest tests/unit/test_config.py::test_config_validation -v

# Run with output visible

pytest tests/ -v --no-header

# Debug failing test

pytest tests/path/to/test.py -x -v --tb=short
```

---

## Documentation

```bash
make docs                    # Build docs site
make fix-md                  # Auto-fix markdown issues
make lint-markdown           # Check markdown style

# Local preview

mkdocs serve                 # http://localhost:8000 (docs site; same default port as `podcast serve` — use `-a 127.0.0.1:8001` for one of them if both run)
```

---

## GI / KG Viewer (v2, RFC-062)

| Command | What it does |
| ------- | ------------ |
| `make serve SERVE_OUTPUT_DIR=…` | FastAPI + Vite dev (API **8000**, UI **5173**) |
| `make serve-api SERVE_OUTPUT_DIR=…` | API only on **8000** |
| `make serve-ui` | Vite only on **5173** (proxies `/api` → 8000) |
| `make test-ui` | Vitest unit tests for TS utils (fast, no browser) |
| `make test-ui-e2e` | Playwright tests (Firefox; Vite on **5174** inside config) |

**Prerequisites:** `pip install -e ".[server]"`; once per clone, `cd web/gi-kg-viewer && npm install && npm run build` to serve the built SPA from `serve`.

**Docs:**
[Server Guide](SERVER_GUIDE.md) (`/api/*`, **`/docs`**)
· [web/gi-kg-viewer/README.md](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/README.md)
· [Development Guide](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype)

---

## Provider Selection

| Need | Guide |
| ---- | ----- |
| Compare providers (cost, quality, speed, privacy) | [AI Provider Comparison](AI_PROVIDER_COMPARISON_GUIDE.md) |
| Per-provider specs, benchmarks, magic quadrant | [Provider Deep Dives](PROVIDER_DEEP_DIVES.md) |
| Implement a new provider | [Provider Implementation](PROVIDER_IMPLEMENTATION_GUIDE.md) |
| Quick provider config | [Provider Configuration](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md) |
| Ollama setup | [Ollama Provider Guide](OLLAMA_PROVIDER_GUIDE.md) |

---

## Evaluation and Profiles

```bash
# Run an experiment against a baseline
make experiment-run \
  CONFIG=data/eval/configs/my_config.yaml \
  BASELINE=baseline_prod_authority_v1

# Promote a successful run to baseline
make run-promote RUN_ID=run_xxx \
  --as baseline PROMOTED_ID=baseline_v2 \
  REASON="New production baseline"

# Capture a performance profile
make profile-freeze VERSION=v2.6-openai \
  PIPELINE_CONFIG=config/profiles/capture_e2e_openai.yaml

# Compare two profiles
make profile-diff FROM=v2.6-wip-openai TO=v2.6-wip-gemini
```

**Docs:**
[Experiment Guide](EXPERIMENT_GUIDE.md)
· [Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md)
· `data/eval/README.md`
· `data/profiles/README.md`

---

## CLI Usage

```bash

# Basic transcript download

python3 -m podcast_scraper.cli https://example.com/feed.xml

# With Whisper transcription

python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --transcribe-missing --whisper-model base

# Full processing

python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --transcribe-missing \
  --generate-metadata \
  --generate-summaries \
  --output-dir ./output

# From config file

python3 -m podcast_scraper.cli --config config.yaml
```

---

## Git Workflow

```bash

# Create feature branch

git checkout -b feature/my-feature

# Make changes, then:

make format && make ci-fast  # Format and verify (ci-fast is the default gate)

# Commit

git add <specific-files>
git commit -m "feat: add my feature"

# Push and create PR

git push -u origin feature/my-feature
```

---

## Debugging

```bash

# Enable debug logging

export LOG_LEVEL=DEBUG

# Check test coverage

make test-unit
open htmlcov/index.html

# Check for linting issues

make lint 2>&1 | head -50

# Validate config file

python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

---

## Grounded Insights

Grounded insights are key takeaways linked to verbatim quotes (evidence). When the GIL pipeline is enabled:

- **Config**: `generate_gi: true`; optional evidence stack: `embedding_model`, `extractive_qa_model`, `nli_model` (and `*_device`).
- **Output**: One `gi.json` per episode (co-located with transcript/summary).
- **CLI**: `gi inspect`, `gi show-insight`, `gi explore` (see Grounded Insights Guide).
- **Browser viewer (v2):** `python -m podcast_scraper.cli serve --output-dir …` with built `web/gi-kg-viewer/dist/` — or `make serve` for dev ([Development Guide](DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype)).

See the [Grounded Insights Guide](GROUNDED_INSIGHTS_GUIDE.md) and [GIL Ontology](../architecture/gi/ontology.md).

---

## Key Files

| File | Purpose |
| ---- | ------- |
| `pyproject.toml` | Dependencies, pytest config |
| `Makefile` | Development commands |
| `.pre-commit-config.yaml` | Pre-commit hooks |
| `tests/conftest.py` | Shared test fixtures |
| `.markdownlint.json` | Markdown linting rules |

---

## Key Directories

| Directory | Purpose |
| --------- | ------- |
| `src/podcast_scraper/` | Main source code |
| `tests/unit/` | Unit tests |
| `tests/integration/` | Integration tests |
| `tests/e2e/` | End-to-end tests |
| `docs/` | Documentation |
| `config/examples/` | Config examples |

---

## Common Issues

| Issue | Fix |
| ----- | --- |
| spaCy wheels re-download often | `make download-spacy-wheels` then `export PIP_FIND_LINKS="$(pwd)/wheels/spacy"` (see [Dependencies Guide](DEPENDENCIES_GUIDE.md#optional-local-wheel-cache-for-spacy-models)) |
| Tests skip "model not cached" | `make preload-ml-models` |
| Import errors | `pip install -e ".[dev,ml,llm]"` |
| Whisper fails | `brew install ffmpeg` |
| `make visualize` / dependency graphs fail | `brew install graphviz` (macOS) or `apt install graphviz` (Linux) |
| CI fails locally | `make ci` |

**See:** [Troubleshooting Guide](TROUBLESHOOTING.md)

---

## Docker

```bash
# Build variants
make docker-build-llm      # LLM-only variant (~200MB)
make docker-build          # ML-enabled variant (~1-3GB)
make docker-build-fast     # ML variant, no model preloading

# Test both variants
make docker-test

# Run container
docker run -v ./config.yaml:/app/config.yaml \
           -v ./output:/app/output \
           podcast-scraper:latest

# Docker Compose
docker-compose up -d       # ML-enabled variant
docker-compose -f docker-compose.llm-only.yml up -d  # LLM-only variant
```

**See:** [Docker Service Guide](DOCKER_SERVICE_GUIDE.md), [Docker Variants Guide](DOCKER_VARIANTS_GUIDE.md)

---

## Links

- [Development Guide](DEVELOPMENT_GUIDE.md) - Full development workflow
- [Testing Guide](TESTING_GUIDE.md) - Detailed test information
- [AI Provider Comparison](AI_PROVIDER_COMPARISON_GUIDE.md) - Provider decision guide
- [Provider Deep Dives](PROVIDER_DEEP_DIVES.md) - Per-provider benchmarks
- [Experiment Guide](EXPERIMENT_GUIDE.md) - Eval datasets and baselines
- [Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md) - Release timing snapshots
- [Docker Service Guide](DOCKER_SERVICE_GUIDE.md) - Docker usage and deployment
- [Docker Variants Guide](DOCKER_VARIANTS_GUIDE.md) - LLM-only vs ML-enabled
- [CLI Reference](../api/CLI.md) - All CLI options
- [Configuration](../api/CONFIGURATION.md) - Config file options
