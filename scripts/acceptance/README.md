# Acceptance Test Scripts

This directory contains scripts for **end-to-end (E2E) acceptance testing**: full pipeline runs (RSS → download → transcribe → summarize → metadata) under real or fixture conditions. Acceptance tests verify that the application runs correctly with different configs and providers, as opposed to unit/integration tests or the evaluation framework (metrics, baselines, experiments).

## What Acceptance Tests Are

- **Full pipeline runs** using the same code path as production (CLI → service → workflow).
- **Config-driven**: each run uses a YAML config (RSS URL, providers, `generate_metadata`, `generate_summaries`, etc.). Configs live in `config/acceptance/` (see that folder’s README for the config list).
- **Full pipeline**: configs under `config/acceptance/full/` set `generate_summaries`, `generate_gi`, `generate_kg`, and semantic index (`vector_search`) so each acceptance run exercises the same end-to-end artifact set (see `config/acceptance/README.md`). The E2E test suite also covers GI/KG CLI paths (e.g. `tests/e2e/test_gi_cli_e2e.py`, `tests/e2e/test_kg_cli_e2e.py`).
- **Optional fixtures**: with `--use-fixtures`, runs use a local E2E server (test feeds, mock APIs) so tests can run without network or real provider keys.

They are **not** the same as the **evaluation framework** in `scripts/eval/` (experiments, baselines, datasets, metrics, regression). Use evaluation for model/prompt comparison and quality metrics; use acceptance for “does the pipeline run and produce expected outputs?”.

## Core Scripts

### Test Runner

- **`run_acceptance_tests.py`** – Run E2E acceptance tests
  - Takes one or more config **glob patterns** in a single `--configs` / `CONFIGS` string: separate patterns with whitespace (shell quoting as usual). Each pattern is expanded; **paths are deduped and sorted** before running.
  - Examples: `config/acceptance/full/acceptance_planet_money_*.yaml`, `config/acceptance/full/*.yaml`.
  - Runs each config sequentially: full pipeline (transcribe, summarize, write metadata, and GIL when `generate_gi: true`).
  - Writes outputs under `--output-dir` (default: `.test_outputs/acceptance`): logs, timing, exit codes, optional baseline comparison.
  - **Session data is written after each config** so you can run `make analyze-acceptance SESSION_ID=...` even if the run is interrupted.
  - With `--use-fixtures`, RSS and cloud provider bases point at the E2E mock server; dummy keys are set where validation requires them. **Ollama** uses `OLLAMA_API_BASE` toward the same server; the mock `/api/tags` response includes model names used by acceptance YAMLs (e.g. `llama3.1:8b`, hybrid reduce models, `gemma2:9b`) so `USE_FIXTURES=1` runs do not need a local `ollama serve`.
  - Options: `--use-fixtures` (E2E server), `--compare-baseline`, `--save-as-baseline`, `--no-auto-analyze`, `--no-auto-benchmark`, `--timeout SECONDS` (per-run timeout), `--fast-only` (run only configs in `config/acceptance/FAST_CONFIGS.txt`), `--stream-debug` (show DEBUG lines on the console; default is **INFO and above** only while streaming—full logs are still saved to `stdout.log` / `stderr.log`).
  - **RSS feed cache:** Each session sets `PODCAST_SCRAPER_RSS_CACHE_DIR` to `sessions/session_*/rss_cache` so the first fetch of a given feed URL is written to disk and later configs in the same run reuse it (less load on real RSS hosts). Normal CLI use leaves this unset (no caching). Episode media is separate; see `reuse_media` in config.
  - **Multiple configs:** After one **session (batch) header** (config count, mode, paths), each run logs a single line **── Run *i*/*n*: `filename.yaml` ──** and **rss (this config): …** for that YAML only (no repeated batch-wide messages).
  - Usage: `make test-acceptance CONFIGS="config/acceptance/full/acceptance_planet_money_ml_dev.yaml"` or use `run_acceptance_tests.py` directly with `--configs` and `--output-dir`.

### Analysis & Benchmarking

- **`analyze_bulk_runs.py`** – Analyze one or more acceptance sessions
  - Compares runs, summarizes pass/fail, timing, resource usage.
  - Use after a session or with `make analyze-acceptance SESSION_ID=...`.

- **`generate_performance_benchmark.py`** – Generate performance benchmarking report from acceptance results
  - Builds a report from session data (e.g. run times, memory).
  - Useful for tracking performance across configs or over time.

## How It Works

1. **Configs**: You choose one or more pipeline configs from `config/acceptance/full/`. Each config specifies RSS feed, providers (whisper, spacy, openai, etc.), and enables summaries, GI, KG, and semantic index per `config/acceptance/README.md`.
2. **Run**: `run_acceptance_tests.py` invokes the application (CLI) once per config. Outputs go to `--output-dir` with a session subfolder (timestamped).
3. **Results**: Logs, exit codes, and optional metrics are stored. With `--compare-baseline` or `--save-as-baseline`, the runner can compare or save a baseline for simple regression checks.
4. **Analysis**: After a run, `analyze_bulk_runs.py` (or the runner’s built-in analysis) can summarize results; `generate_performance_benchmark.py` can produce a performance report.

## Usage

Run a single full-pipeline config:

```bash
make test-acceptance CONFIGS="config/acceptance/full/acceptance_planet_money_ml_dev.yaml"
```

Run all Planet Money full-pipeline configs:

```bash
make test-acceptance CONFIGS="config/acceptance/full/acceptance_planet_money_*.yaml"
```

Run all full acceptance configs (Planet Money + The Journal, ML and LLM providers):

```bash
make test-acceptance CONFIGS="config/acceptance/full/*.yaml"
```

Run the **full acceptance matrix** with fixtures and without auto analysis/benchmark:

```bash
make test-acceptance CONFIGS="config/acceptance/full/*.yaml" \
  USE_FIXTURES=1 NO_AUTO_ANALYZE=1 NO_AUTO_BENCHMARK=1
```

Run with E2E fixtures (no real RSS or API keys):

```bash
make test-acceptance CONFIGS="config/acceptance/full/acceptance_planet_money_ml_dev.yaml" USE_FIXTURES=1
```

Run only the fast subset (for CI on PRs):

```bash
make test-acceptance CONFIGS="config/acceptance/full/*.yaml" USE_FIXTURES=1 FAST_ONLY=1
```

The list of fast configs is in `config/acceptance/FAST_CONFIGS.txt`.

Run with a per-run timeout (e.g. 600 seconds) so long configs are killed and reported as failed:

```bash
python scripts/acceptance/run_acceptance_tests.py --configs "config/acceptance/full/*.yaml" --use-fixtures --timeout 600
```

See `config/acceptance/README.md` for the list of acceptance configs and provider prerequisites. For the full experiment and evaluation workflow (datasets, baselines, metrics), see `docs/guides/EXPERIMENT_GUIDE.md` and `scripts/eval/README.md`.
