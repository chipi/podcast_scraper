# Scripts

Utility scripts for the podcast scraper project, organized by purpose.

## Folder Structure

```text
scripts/
├── acceptance/     # E2E acceptance test scripts
├── cache/          # Cache management scripts
├── dashboard/      # Metrics and dashboard generation scripts
├── eval/           # Evaluation and experiment scripts
├── tools/          # Development tooling scripts
└── setup_venv.sh   # Virtual environment setup
```

---

## Acceptance Test Scripts (`acceptance/`)

Scripts for running E2E acceptance tests, analyzing results, and generating performance benchmarks.

### Acceptance Test Scripts

- **`run_acceptance_tests.py`** - Run multiple config files sequentially and collect structured data
- **`analyze_bulk_runs.py`** - Analyze acceptance test results and generate reports
- **`generate_performance_benchmark.py`** - Generate performance benchmarking reports grouped by provider/model

### Acceptance Test Usage

See the **[Testing Guide](../docs/guides/TESTING_GUIDE.md)** for complete usage instructions.

**Quick examples:**

```bash
# Run acceptance tests
make test-acceptance CONFIGS="config/examples/config.example.yaml"

# Analyze results
make analyze-acceptance SESSION_ID=20260208_101601

# Generate performance benchmark
make benchmark-acceptance SESSION_ID=20260208_101601
```

---

## Evaluation Scripts (`eval/`)

Evaluation scripts for the AI quality and experimentation platform:

### Core Scripts

- **`run_experiment.py`** - Run experiments with complete evaluation loop (runner + scorer + comparator)
- **`materialize_baseline.py`** - Create frozen baseline artifacts from current system state
- **`materialize_dataset.py`** - Materialize datasets from dataset JSON definitions
- **`promote_run.py`** - Promote runs to baselines or references

### Dataset Management

- **`create_dataset_json.py`** - Create canonical dataset JSON files from source data
- **`generate_episode_metadata.py`** - Generate episode metadata from RSS XML files
- **`generate_source_index.py`** - Generate source inventory index.json files

### Usage

See the **[Experiment Guide](../docs/guides/EXPERIMENT_GUIDE.md)** for complete usage instructions.

**Quick examples:**

```bash
# Create a dataset
make dataset-create DATASET_ID=my_dataset_v1

# Materialize a baseline
make baseline-create BASELINE_ID=my_baseline_v1 DATASET_ID=my_dataset_v1

# Run an experiment
make experiment-run CONFIG=data/eval/configs/my_experiment.yaml

# Promote a run
make run-promote RUN_ID=run_xxx --as baseline PROMOTED_ID=baseline_v2 REASON="..."
```

### Evaluation Dataset Structure

See `data/eval/README.md` for details on the evaluation dataset structure.

---

## Development Tools (`tools/`)

### fix_markdown.py

Automatically fixes common markdown linting issues that can be corrected programmatically.

#### Usage (fix_markdown.py)

Fix all markdown files in the project:

```bash
python scripts/tools/fix_markdown.py
```

**What it fixes:**

1. **Table Separator Formatting** (MD060):
   - Adds spaces around pipes in table separator rows
   - Converts `|-------------|--------|` to `| ----------- | ------ |`

2. **Trailing Spaces** (MD009):
   - Removes trailing whitespace from all lines

3. **Blank Lines Around Lists** (MD032):
   - Adds blank lines before and after lists when missing

4. **Code Block Language Specifiers** (MD040):
   - Adds language specifiers to code blocks when detectable (Python, JavaScript, etc.)

#### When to Use

Run this script regularly, especially:

- Before committing markdown files
- After bulk edits to documentation
- When CI fails on markdown linting errors
- As part of your pre-commit workflow

#### Integration with Pre-commit

The pre-commit hook runs `markdownlint` which catches issues, but doesn't auto-fix them. Use this script to fix issues before committing:

```bash
# Fix all markdown issues
python scripts/tools/fix_markdown.py

# Then commit
git add -A
git commit -m "your message"
```

---

### Other Tools

- **`analyze_dependencies.py`** - Analyze module dependencies and detect architectural issues (circular imports, import thresholds)
- **`analyze_test_memory.py`** - Analyze test suite memory usage and resource consumption
- **`check_unit_test_imports.py`** - Verify unit tests can import modules without ML dependencies
- **`profile_e2e_test_memory.py`** - Profile individual E2E tests to identify memory-intensive tests

See `docs/guides/DEVELOPMENT_GUIDE.md` for detailed usage of these tools.

---

## Cache Management (`cache/`)

### backup_cache.py

Backs up the `.cache` directory containing ML models and other cached resources.

**Make targets:**

- `make backup-cache` - Create backup
- `make backup-cache-dry-run` - Preview what would be backed up
- `make backup-cache-list` - List existing backups
- `make backup-cache-cleanup` - Clean up old backups (keeps last N)

### restore_cache.py

Restores cache from a backup archive.

**Make targets:**

- `make restore-cache` - Restore from backup (interactive)
- `make restore-cache-dry-run` - Preview restore operation

See `docs/guides/DEVELOPMENT_GUIDE.md` for detailed usage.

---

## Dashboard & Metrics (`dashboard/`)

Scripts used in CI workflows to generate metrics and dashboards:

- **`collect_pipeline_metrics.py`** - Collect pipeline performance metrics by running a minimal pipeline
- **`generate_metrics.py`** - Extract metrics from test artifacts (JUnit XML, coverage reports)
- **`generate_dashboard.py`** - Generate HTML dashboard from metrics
- **`consolidate_dashboard_data.py`** - Build `dashboard-data.json` (CI + nightly `latest` + `history` arrays) for a single-fetch unified dashboard; **`make build-metrics-dashboard-preview`** runs it automatically
- **`metrics_jsonl.py`** - Parse metrics history (JSONL); tolerates legacy multi-line appends
- **`append_metrics_history_line.py`** - Emit one compact JSON line for `history-*.jsonl` (used by CI; do not use `echo "$(cat latest.json)"`)
- **`repair_metrics_jsonl.py`** - Rewrite `history-ci.jsonl` / `history-nightly.jsonl` as proper one-object-per-line JSONL
- **`capture_quality_for_metrics.py`** - Run interrogate, vulture, and codespell; emit `docstrings.json`, `vulture.json`, and `codespell.txt` for `generate_metrics.py` (replaces removed CLI flags in interrogate 1.7+ / vulture 2.x)
- **`fetch_ci_metrics_artifacts.sh`** - Download the `metrics/` workflow artifact from recent successful `python-app.yml` runs on `main` (requires [GitHub CLI](https://cli.github.com/) `gh`). Default **40** run IDs attempted (many runs lack the artifact — use **`N=100`** or higher for ~20–40 bundles). Skips re-download if **`run-<id>/latest-ci.json`** already exists. **`make fetch-ci-metrics`** / **`make fetch-ci-metrics N=80`**
- **`fetch_metrics_file_from_pages.sh`** - CI helper: download one file from the live site **`.../metrics/<file>`** (default URL from **`GITHUB_REPOSITORY`**) via **`curl`**, else **`git show gh-pages:metrics/...`**. Used by **`nightly.yml`** and **`python-app.yml`** so history accumulates when Pages is deployed with **`deploy-pages`** (the **`gh-pages`** git ref may be stale). Optional env **`METRICS_PAGES_BASE`**, **`GITHUB_REPOSITORY`**.
- **`fetch_nightly_metrics_artifacts.sh`** - Download **`nightly-metrics`** from up to **`LIMIT`** successful **`nightly.yml`** runs on **`main`** into **`artifacts/nightly-metrics-runs/run-<id>/`** (needs **`gh`**). Default **`LIMIT=25`**. **`merge_nightly_metrics_runs_to_history.py`** merges **`latest-nightly.json`** from each bundle into **`metrics/history-nightly.jsonl`**. **`make fetch-nightly-metrics N=25`**
- **`merge_nightly_metrics_runs_to_history.py`** - Merge **`run-*/latest-nightly.json`** → one JSONL + copy newest **`latest-nightly.json`**; used by fetch ( **`N`≥1 ) and **`build_local_metrics_preview.sh`** when **`artifacts/nightly-metrics-runs/`** has bundles
- **`fetch_nightly_metrics.sh`** - **`make fetch-nightly-metrics`** (no **`N`**): latest artifact or Pages **`curl`**. **`make fetch-nightly-metrics N=25`**: multi-download + merge (many nightly chart points). Env: **`GHPAGES_METRICS_BASE`**, **`GITHUB_BRANCH`**, **`METRICS_DIR`**, **`FETCH_NIGHTLY_PREFER_PAGES=1`**
- **`fetch_and_validate_ci_metrics.sh`** - Runs the fetch script, then **`validate_metrics_bundle.py`** on every `artifacts/ci-metrics-runs/run-*` directory. **`make fetch-ci-metrics-validate`** / **`make fetch-ci-metrics-validate N=80`**
- **`validate_metrics_bundle.py`** - Sanity-check a downloaded or local `latest-ci.json` / `latest-nightly.json` plus optional `history-*.jsonl`. **`make validate-metrics-bundle BUNDLE=artifacts/ci-metrics-runs/run-<id>`**

**Validate the dashboard HTML from a download:** `cd artifacts/ci-metrics-runs/run-<id>` then `python -m http.server 8765` and open `http://127.0.0.1:8765/index.html` (CI data only unless you also copy nightly JSON/JSONL into that folder).

**Both segments (CI + Nightly) locally:** `make build-metrics-dashboard-preview` merges the newest `artifacts/ci-metrics-runs/run-*` with `metrics/latest-nightly.json` and `metrics/history-nightly.jsonl`, writes `artifacts/dashboard-preview/dashboard-data.json`, then `make serve-metrics-dashboard` → `http://127.0.0.1:8777/`. **`make metrics-preview-check`** rebuilds with strict JSONL validation (fails if `history-*.jsonl` looks like pretty-printed JSON).

**One shot (download + validate + preview + server):** `make metrics-dashboard-live` or `make metrics-dashboard-live N=10` (requires `gh` auth; blocks on HTTP server until Ctrl+C). This is the **metrics** dashboard only, not `make docs` / MkDocs.

**Re-run `generate_metrics.py` like CI** needs a full `reports/` tree (merged `pytest.json`, `coverage.xml`, radon JSON, etc.); that comes from downloading **`coverage-unified`** plus **`pytest-unit`**, **`pytest-integration`**, **`pytest-e2e`** from the *same* run and re-running the merge step from `python-app.yml` locally—not bundled in the `metrics` artifact.

These scripts are primarily used in CI workflows (see `.github/workflows/`). See `docs/ci/WORKFLOWS.md` for details.

---

## Setup Scripts

### setup_venv.sh

Creates a Python virtual environment and installs the package in editable mode.

#### Usage (setup_venv.sh)

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

**What it does:**

1. Creates `.venv/` virtual environment
2. Upgrades pip
3. Installs package in editable mode (`pip install -e .`)

#### Next Steps

After running `setup_venv.sh`:

1. **Activate the virtual environment:**

   ```bash
   source .venv/bin/activate
   ```

2. **Install development dependencies:**

   ```bash
   make init
   # Or manually: pip install -e .[dev,ml]
   ```

3. **Set up environment variables (optional, for OpenAI providers):**

   ```bash
   cp config/examples/.env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. **Run the CLI:**

   ```bash
   python -m podcast_scraper.cli <rss_url> [options]
   ```

See `README.md` and `CONTRIBUTING.md` for more details.

---

### preload_ml_models.py

Preloads ML models into cache for faster CI runs. Used automatically in CI workflows.

**Make targets:**

- `make preload-ml-models` - Preload models (development)
- `make preload-ml-models-production` - Preload models (production)

**Usage:**

```bash
python scripts/cache/preload_ml_models.py
python scripts/cache/preload_ml_models.py --production
```
