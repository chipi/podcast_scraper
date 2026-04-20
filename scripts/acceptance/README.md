# Acceptance Test Scripts

This directory contains scripts for **end-to-end (E2E) acceptance testing**: full pipeline runs (RSS → download → transcribe → summarize → metadata) under real or fixture conditions. Acceptance tests verify that the application runs correctly with different configs and providers, as opposed to unit/integration tests or the evaluation framework (metrics, baselines, experiments).

## What Acceptance Tests Are

- **Full pipeline runs** using the same code path as production (**`python -m podcast_scraper.service`** per config).
- **Config-driven**: each run uses a YAML config (RSS URL, providers, `generate_metadata`, `generate_summaries`, etc.). The tracked matrix lives under **`config/acceptance/`** (see that folder’s **`README.md`**).
- **Full pipeline**: the fast matrix merges **official `config/profiles/`** bundles so each row exercises summaries, GI/KG, and semantic index when the profile enables them (see `config/acceptance/README.md`). The E2E test suite also covers GI/KG CLI paths (e.g. `tests/e2e/test_gi_cli_e2e.py`, `tests/e2e/test_kg_cli_e2e.py`).
- **Optional fixtures**: with `--use-fixtures`, runs use a local E2E server (test feeds, mock APIs) so tests can run without network or real provider keys.

They are **not** the same as the **evaluation framework** in `scripts/eval/` (experiments, baselines, datasets, metrics, regression). Use evaluation for model/prompt comparison and quality metrics; use acceptance for “does the pipeline run and produce expected outputs?”.

## Core Scripts

### Test Runner

- **`run_acceptance_tests.py`** – Run E2E acceptance tests
  - Takes one or more config **glob patterns** in a single `--configs` / `CONFIGS` string: separate patterns with whitespace (shell quoting as usual). Each pattern is expanded; **paths are deduped and sorted** before running.
  - Examples: paths to your operator YAMLs, or materialized `sessions/session_*/materialized/*.yaml` from **`--from-fast-stems`**.
  - Runs each config sequentially: full pipeline (transcribe, summarize, write metadata, and GIL when `generate_gi: true`).
  - Writes outputs under `--output-dir` (default: `.test_outputs/acceptance`): logs, timing, exit codes, optional baseline comparison.
  - **Session data is written after each config** so you can run `make analyze-acceptance SESSION_ID=...` even if the run is interrupted.
  - With `--use-fixtures`, RSS and cloud provider bases point at the E2E mock server; dummy keys are set where validation requires them. **Ollama** uses `OLLAMA_API_BASE` toward the same server; the mock `/api/tags` response includes model names used by acceptance YAMLs (e.g. `llama3.1:8b`, hybrid reduce models, `gemma2:9b`) so `USE_FIXTURES=1` runs do not need a local `ollama serve`.
  - **`--from-fast-stems`**: load **`config/acceptance/FAST_CONFIG.yaml`**, merge each enabled row (`defaults` + feeds fragment + **`profile:`**), write **`sessions/session_*/materialized/{id}.yaml`**, and run those paths. Pair with **`--use-fixtures`** for CI fixture smoke.
  - Options: `--use-fixtures` (E2E server), `--compare-baseline`, `--save-as-baseline`, `--no-auto-analyze`, `--no-auto-benchmark`, `--timeout SECONDS` (per-run timeout), `--fast-only` (filter glob results to fast stems), `--stream-debug` (show DEBUG lines on the console; default is **INFO and above** only while streaming—full logs are still saved to `stdout.log` / `stderr.log`).
  - **RSS feed cache:** Each session sets `PODCAST_SCRAPER_RSS_CACHE_DIR` to `sessions/session_*/rss_cache` so the first fetch of a given feed URL is written to disk and later configs in the same run reuse it (less load on real RSS hosts). Normal CLI use leaves this unset (no caching). Episode media is separate; see `reuse_media` in config.
  - **Multiple configs:** After one **session (batch) header** (config count, mode, paths), each run logs a single line **── Run *i*/*n*: `filename.yaml` ──** and **feeds (this config, …)** or **rss (this config): …** for that YAML only (no repeated batch-wide messages). Multi-feed YAMLs with `feeds:` / `rss_urls` get per-run fixture URL substitution when `USE_FIXTURES=1`. After each multi-feed service run, **`corpus_manifest.json`** and **`corpus_run_summary.json`** are written at the corpus parent when applicable (#506); use **`python -m podcast_scraper.cli corpus-status --output-dir …`** to inspect.
  - Usage: `make test-acceptance FROM_FAST_STEMS=1 USE_FIXTURES=1`, or `make test-acceptance CONFIGS="path/to/operator.yaml"`, or `run_acceptance_tests.py` with `--configs` / `--from-fast-stems` and `--output-dir`.

### Analysis & Benchmarking

- **`analyze_bulk_runs.py`** – Analyze one or more acceptance sessions
  - Compares runs, summarizes pass/fail, timing, resource usage.
  - Use after a session or with `make analyze-acceptance SESSION_ID=...`.

- **`generate_performance_benchmark.py`** – Generate performance benchmarking report from acceptance results
  - Builds a report from session data (e.g. run times, memory).
  - Useful for tracking performance across configs or over time.

## How It Works

1. **Configs**: You choose one or more pipeline YAML paths, or **`--from-fast-stems`** to materialize rows from **`config/acceptance/FAST_CONFIG.yaml`**. Each file specifies feeds + **`profile:`** (or full inline operator keys) per `config/acceptance/README.md`.
2. **Run**: `run_acceptance_tests.py` invokes **`python -m podcast_scraper.service --config …`** once per config (same pipeline as production). Outputs go to `--output-dir` with a session subfolder (timestamped).
3. **Results**: Logs, exit codes, and optional metrics are stored. With `--compare-baseline` or `--save-as-baseline`, the runner can compare or save a baseline for simple regression checks.
4. **Analysis**: After a run, `analyze_bulk_runs.py` (or the runner’s built-in analysis) can summarize results; `generate_performance_benchmark.py` can produce a performance report.

## Usage

Run a single full-pipeline config (path is yours — often under **`config/playground/`** or a temp file):

```bash
make test-acceptance CONFIGS="path/to/operator.yaml"
```

Run the **fast matrix** with fixtures (materializes **`FAST_CONFIG.yaml`** rows):

```bash
make test-acceptance-fixtures-fast
```

Run the same matrix with explicit flags:

```bash
make test-acceptance FROM_FAST_STEMS=1 USE_FIXTURES=1 NO_AUTO_ANALYZE=1 NO_AUTO_BENCHMARK=1
```

**Feeds lists** for **`--feeds-spec`** are RFC-077 YAML/JSON you supply (shape: **`config/examples/feeds.spec.example.yaml`**). Combine with an operator YAML that sets **`profile:`** and corpus options.

**Append / resume (GitHub #444):** set **`append: true`** on the operator YAML, re-run twice under **`USE_FIXTURES=1`** to validate **`run_append_*`** skip behavior.

**`FAST_ONLY=1`:** after **`CONFIGS=`** expands to multiple paths, keep only stems that match **`id:`** rows in **`FAST_CONFIG.yaml`** (see **`config/acceptance/README.md`**).

```bash
make test-acceptance CONFIGS="path/to/session/materialized/*.yaml" USE_FIXTURES=1 FAST_ONLY=1
```

Fast matrix row **`id`** values are defined in **`config/acceptance/FAST_CONFIG.yaml`** (see **`config/acceptance/README.md`**).

**Full fast matrix + fixtures** (recommended smoke: every enabled **`FAST_CONFIG.yaml`** row, E2E mock server; CI uses a higher per-run timeout):

```bash
make test-acceptance-fixtures-fast
```

Equivalent:

```bash
make test-acceptance FROM_FAST_STEMS=1 USE_FIXTURES=1 NO_AUTO_ANALYZE=1 NO_AUTO_BENCHMARK=1 TIMEOUT=1500
```

**Main / release CI** runs `make test-acceptance-fixtures-fast` on push (see `.github/workflows/python-app.yml`, job `test-acceptance-fixtures`). The same job then runs **`make verify-gil-offsets-after-acceptance`**: for the latest session under `.test_outputs/acceptance/sessions/`, every `runs/run_*` that contains **`search/metadata.json`** is checked with **`verify-gil-chunk-offsets --strict`** so **GIL Quote** character spans align with **FAISS transcript** chunks (RFC-072 / issue #528; semantic **lift** contract). Override the acceptance output root with **`OUTPUT_DIR=…`** if you use a non-default `--output-dir`. **Scheduled nightly** (`.github/workflows/nightly.yml`) does not run this acceptance matrix; it uses **`make test-nightly`** instead.

To run the offset pass locally after a fixture session:

```bash
make verify-gil-offsets-after-acceptance
```

Run with a per-run timeout (e.g. 600 seconds) so long configs are killed and reported as failed:

```bash
python scripts/acceptance/run_acceptance_tests.py --from-fast-stems --use-fixtures --timeout 600
```

See `config/acceptance/README.md` for the fast matrix, fragments, and provider prerequisites. For the full experiment and evaluation workflow (datasets, baselines, metrics), see `docs/guides/EXPERIMENT_GUIDE.md` and `scripts/eval/README.md`.
