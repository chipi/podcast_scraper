# Evaluation Scripts

Scripts for the AI quality and experimentation platform. Organised into five
domain subfolders (#633):

| Subfolder | Purpose |
| --- | --- |
| `data/` | Prepare *inputs* to evals: datasets, silver references, baselines. |
| `experiment/` | Produce *runs*: invoke the pipeline with some variation (sweeps, provider runs, validation, fingerprint smoke). |
| `score/` | Compute metrics on a *single* run against a silver/gold reference. |
| `compare/` | Read *multiple* runs — list, diff, aggregate, promote. |
| `profile/` | RFC-064 performance-profile capture (companion to `config/profiles/freeze/`). |

Separate from the **eval-reports docs** under `docs/guides/eval-reports/` and
the silver/dataset *data* under `data/eval/` — this directory holds only the
executable scripts.

## Core Scripts

### Experiment Execution

- **`experiment/run_experiment.py`** - Run experiments with complete evaluation loop
  - Executes experiments using experiment config YAML files
  - Computes metrics (intrinsic and vs_reference)
  - Generates comparisons against baselines
  - See `docs/guides/EXPERIMENT_GUIDE.md` for usage

### Baseline & Dataset Management

- **`data/materialize_baseline.py`** - Create frozen baseline artifacts
  - Generates baselines from current system state
  - Includes comprehensive fingerprinting
  - Creates structured outputs (predictions.jsonl, metrics.json, fingerprint.json)
  - Usage: `make baseline-create BASELINE_ID=... DATASET_ID=...`

- **`data/materialize_dataset.py`** - Materialize datasets from JSON definitions
  - Validates source paths and hashes
  - Copies transcripts to materialized directory
  - Generates per-episode metadata
  - Usage: `make dataset-materialize DATASET_ID=...`

- **`data/create_dataset_json.py`** - Create canonical dataset JSON files
  - Discovers episodes from source directories
  - Generates dataset definitions with episode metadata
  - Usage: `make dataset-create DATASET_ID=...`

### Source Data Management

- **`data/generate_episode_metadata.py`** - Generate episode metadata from RSS XML
  - Extracts episode information from RSS feeds
  - Creates `.metadata.json` files per episode
  - Usage: `make metadata-generate INPUT_DIR=...`

- **`data/generate_source_index.py`** - Generate source inventory index.json
  - Creates inventory of all episodes in a source directory
  - Computes SHA256 hashes for drift detection
  - Usage: `make source-index SOURCE_DIR=...`

### Promotion Workflow

- **`compare/promote_run.py`** - Promote runs to baselines or references
  - Moves runs from `data/eval/runs/` to `baselines/` or `references/`
  - Marks artifacts as immutable
  - Creates README.md with promotion metadata
  - Usage: `make run-promote RUN_ID=... --as baseline|reference ...`

### Fingerprint Validation

- **`experiment/test_fingerprint_smoke.py`** - Smoke test for fingerprint validation
  - Validates that fingerprints change for semantic changes (model, prompt, preprocessing, chunking)
  - Validates that fingerprints stay stable for non-semantic changes (re-runs, paths)
  - Implements automated assertions for fingerprint correctness
  - Usage: `python scripts/eval/test_fingerprint_smoke.py`

- **`experiment/test_fingerprint_matrix.py`** - Minimal test matrix for fingerprint validation
  - Tests fingerprint changes for prompt, preprocessing, chunk size changes
  - Tests fingerprint stability for re-runs and different output paths
  - Includes isolation checks to ensure only intended parameters change
  - Usage: `python scripts/eval/test_fingerprint_matrix.py`

## Usage

All scripts are typically invoked via `make` targets. See:

- `docs/guides/EXPERIMENT_GUIDE.md` - Complete experiment workflow
- `Makefile` - All available make targets

## Script Organization

These scripts are organized here because they are all part of the evaluation and experimentation system. They work together to:

1. Prepare source data (generate metadata, create indexes)
2. Create datasets (create_dataset_json, materialize_dataset)
3. Create baselines (materialize_baseline)
4. Run experiments (run_experiment)
5. Promote artifacts (promote_run)
