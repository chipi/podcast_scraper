# Experiment Guide

**Status:** 🚧 Work in Progress - This guide will evolve as the experiment system matures.

This guide explains how to run AI experiments using the podcast_scraper benchmarking framework. Experiments allow you to test different models, prompts, and parameters on canonical datasets and compare results against frozen baselines.

---

## Getting Started

**Workflow Order:** You must follow these steps in order:

1. **Prepare Source Data** (Step 0) - Generate metadata and source indexes from RSS XML files
2. **Create a Dataset** (Step 1) - Create a canonical dataset from your eval data
3. **Materialize Dataset** (Step 1a) - Validate and materialize the dataset (optional but recommended)
4. **Create a Baseline** (Step 2) - Create a baseline using that dataset
5. **Run Experiments** (Step 3) - Run experiments that compare against the baseline

**Why this order?**

- Datasets require source data with transcripts
- Baselines require a dataset to know which episodes to process
- Experiments require both a dataset (for input) and a baseline (for comparison)
- Materialization validates dataset integrity before use

---

## Overview

The experiment system consists of several components:

1. **Source Data** - Raw RSS XML files, transcripts, and metadata in `data/eval/sources/`
2. **Datasets** - Canonical, frozen sets of episodes with transcripts and golden references
3. **Materialized Datasets** - Validated, copied datasets in `data/eval/materialized/`
4. **Baselines** - Frozen reference results from a known system state
5. **Experiments** - Runs that test new configurations against datasets and compare to baselines

### Key Concepts

- **Source ID**: Identifier for a source directory (e.g., `curated_5feeds_raw_v1`)
- **Dataset ID**: Identifier for a canonical dataset (e.g., `curated_5feeds_smoke_v1`)
- **Baseline ID**: Identifier for a frozen baseline (e.g., `bart_led_baseline_v1`)
- **Experiment ID**: Unique identifier for an experiment run (e.g., `summarization_openai_long_v2`)

---

## Understanding Baselines, Experiments, and App Defaults

This section explains the correct way to think about baselines, experiments, and how they relate to your application's default behavior.

### What is a Baseline?

A baseline represents **how the default app behaves** for a given task, dataset, and provider.

**Key points:**

- ✅ **Baseline params = default app params** - The configuration used in a baseline should match what users get by default
- ✅ **Baseline pipeline = default app pipeline** - The processing steps should match the default workflow
- ✅ **Baseline output = what users should expect today** - The results represent current production behavior

**This is not just "ok" — this is the point of a baseline.**

### Critical Clarification: Baseline ≠ "Whatever the App Happens to Do"

❌ **Wrong approach:** Baseline = "whatever the app happens to do right now"

✅ **Correct approach:** Baseline = explicitly frozen snapshot of default app behavior

**Why this matters:**

Your app may evolve, but your baseline must not drift silently. The rule is:

> **The app defaults should always be derived from a baseline, not the other way around.**

### The Correct Relationship Between App and Baseline

**Ideal flow (what you should aim for):**

1. You define a config (like a YAML experiment config)
2. You run it through the evaluation system
3. You promote that run to a baseline
4. That baseline config becomes the app default
5. Future app changes are compared against that baseline

**Visually:**

```text
baseline config  ──► app default behavior
       ▲
       │
 experiments / changes
```

**Not:**

```text
app default (mutable) ──► baseline (moving target) ❌
```

### Why This Distinction is Important

**If you treat baseline as "whatever the app currently does":**

- ❌ Regressions slip in unnoticed
- ❌ Metrics history becomes meaningless
- ❌ You can't explain why quality changed
- ❌ Rollbacks become guesswork

**If you treat baseline as the authority:**

- ✅ App behavior is intentional
- ✅ Changes are deliberate
- ✅ Comparisons are meaningful
- ✅ Rollbacks are trivial

### How This Applies to Your Setup

For your case:

- `baseline_bart_small_led_long_fast` → default summarization behavior in dev

Later you'll likely have:

- `baseline_prod_authority_benchmark_v1` → default summarization behavior in prod

Those baselines should correspond 1:1 with:

- The model IDs
- Generation params
- Preprocessing logic
- Chunking strategy (once added)

### Practical Guideline

**If a user asks "what does the app do by default?", you should be able to answer: "it runs baseline X."**

If you can't answer that, the baseline isn't doing its job.

### What Baselines Should NOT Be Used For

Just to be clear:

- ❌ Baselines are not "best possible quality"
- ❌ Baselines are not "experiments"
- ❌ Baselines are not "aspirational targets"

Those are:

- **Capability baselines** - for exploring what's possible
- **Silver/gold references** - for quality targets
- **Experiments** - for testing new approaches

Different roles, different purposes.

### How This Ties Back to Configuration

Your instinct was right:

- Putting `max_length: 150` in the baseline config would literally mean: "the app default produces very short summaries"

That's why fixing the baseline params is so important.

**Once you fix and promote the baseline:**

- Those params become your app default
- Everything else is compared against them

### One-Sentence Rule to Remember

> **A baseline is the contract for default app behavior — frozen, explicit, and intentional.**

### From Baseline to App Default

The workflow for promoting a baseline to app default:

1. **Create baseline** - Run evaluation with your intended default config
2. **Validate baseline** - Ensure metrics meet acceptance criteria
3. **Promote baseline** - Mark it as the authoritative default
4. **Update app config** - Load baseline config as app defaults
5. **Verify alignment** - Confirm app behavior matches baseline

**Future changes:**

- Run experiments against the baseline
- Compare metrics and quality
- If better, create new baseline and update app defaults
- If worse, reject the change

This ensures all app behavior is intentional and traceable.

---

## Step 0: Prepare Source Data

**Prerequisites:** You need RSS XML files and transcript files in `data/eval/sources/`.

Before creating datasets, you should:

1. Generate episode metadata from RSS XML files
2. Generate source indexes for inventory management

### Generate Episode Metadata

Generate metadata JSON files from RSS XML files:

```bash
make metadata-generate INPUT_DIR=data/eval/sources
```

This will:

- Scan `data/eval/sources/` recursively for RSS XML files
- Parse each RSS feed and extract episode metadata
- Generate `*.metadata.json` files next to each XML file

**Output format:**

Each `{episode_id}.metadata.json` contains:

```json
{
  "source_episode_id": "p01_e01",
  "feed_name": "Singletrack Sessions",
  "feed_url": "http://localhost/",
  "episode_title": "Episode 1: Building Trails That Last...",
  "published_at": "2025-09-01",
  "duration_seconds": 630,
  "language": "en",
  "scraped_at": "2026-01-13T12:07:56.657450Z"
}
```

**Optional parameters:**

```bash
make metadata-generate \
  INPUT_DIR=data/eval/sources \
  OUTPUT_DIR=data/eval/metadata \
  LOG_LEVEL=DEBUG
```

### Generate Source Index

Create an inventory index for each source directory:

```bash
make source-index SOURCE_DIR=data/eval/sources/curated_5feeds_raw_v1
```

This will:

- Scan the source directory for feed subdirectories
- Find all transcript and metadata files
- Compute SHA256 hashes for transcripts
- Generate `index.json` in the source directory

**Output format:**

The `index.json` contains:

```json
{
  "source_id": "curated_5feeds_raw_v1",
  "created_at": "2026-01-13T12:11:46.314843Z",
  "episodes": [
    {
      "source_episode_id": "p01_e01",
      "feed": "feed-p01",
      "transcript_path": "feed-p01/p01_e01.txt",
      "transcript_sha256": "a650e729cc8b7379c94fd5b29c092bcd32a8c7e4c2086f1321d6ed496718b9b4",
      "meta_path": "feed-p01/p01_e01.metadata.json"
    }
  ]
}
```

**Process all sources:**

```bash
make source-index SOURCE_DIR=data/eval/sources ALL=1
```

**Benefits of source indexes:**

- Programmatic dataset generation
- Drift detection (hash changes)
- Dataset definition validation
- Avoid ad-hoc directory scanning

---

## Step 1: Create a Dataset

**Prerequisites:** You need evaluation data in `data/eval/` with transcript files (`.txt` files in any subdirectory).

Datasets are canonical, frozen sets of episodes stored as JSON files. The script recursively finds all `.txt` files in subdirectories and treats each as a transcript.

### Quick Start: Predefined Datasets

For the curated 5 feeds source, we have three predefined datasets:

**Smoke Test Dataset** (first episode per feed):

```bash
make dataset-smoke
```

Creates `data/eval/datasets/curated_5feeds_smoke_v1.json` with 5 episodes.

**Benchmark Dataset** (first 2 episodes per feed):

```bash
make dataset-benchmark
```

Creates `data/eval/datasets/curated_5feeds_benchmark_v1.json` with 10 episodes.

**Raw Dataset** (all episodes):

```bash
make dataset-raw
```

Creates `data/eval/datasets/curated_5feeds_raw_v1.json` with all episodes.

### Custom Dataset Creation

**Using the Make Command (Recommended):**

```bash
make dataset-create \
  DATASET_ID=indicator_v1 \
  EVAL_DIR=data/eval \
  DESCRIPTION="Lenny's Podcast evaluation episodes (interview style)"
```

**Default values:**

- `EVAL_DIR` defaults to `data/eval` (can be omitted)
- `OUTPUT_DIR` defaults to `benchmarks/datasets` (can be omitted)
- `DESCRIPTION` defaults to "Dataset {DATASET_ID}" (can be omitted)

**With all options:**

```bash
make dataset-create \
  DATASET_ID=indicator_v1 \
  EVAL_DIR=data/eval \
  OUTPUT_DIR=data/eval/datasets \
  DESCRIPTION="Lenny's Podcast evaluation episodes (interview style)" \
  CONTENT_REGIME=narrative \
  MAX_EPISODES_PER_FEED=2
```

**Filtering episodes:**

Use `MAX_EPISODES_PER_FEED` to limit episodes per feed:

- `MAX_EPISODES_PER_FEED=1` - First episode per feed (smoke test)
- `MAX_EPISODES_PER_FEED=2` - First 2 episodes per feed (benchmark)
- Omit parameter - All episodes (full dataset)

### Using the Script Directly

```bash
python scripts/eval/create_dataset_json.py \
  --dataset-id indicator_v1 \
  --eval-dir data/eval \
  --output-dir data/eval/datasets \
  --description "Lenny's Podcast evaluation episodes (interview style)" \
  --max-episodes-per-feed 2
```

**How it works:**

- Recursively scans `data/eval/` for all `.txt` files
- Derives episode IDs from filenames (without extension)
- Looks for associated files:
  - `{episode_id}.metadata.json` - Episode metadata (new format)
  - `metadata.json` - Episode metadata (old format)
  - `{episode_id}.raw.txt` - Raw transcript
  - `{episode_id}.summary.gold.long.txt` - Long golden summary
  - `{episode_id}.summary.gold.short.txt` - Short golden summary
- Computes SHA256 hashes for transcripts
- Creates dataset JSON with all episode information

### Dataset JSON Structure

A dataset JSON looks like this:

```json
{
  "dataset_id": "curated_5feeds_smoke_v1",
  "version": "1.0",
  "description": "Smoke test dataset: first episode per feed from curated_5feeds_raw_v1",
  "created_at": "2026-01-13T12:22:41.258855Z",
  "content_regime": "explainer",
  "num_episodes": 5,
  "episodes": [
    {
      "episode_id": "p01_e01",
      "title": "Episode 1: Building Trails That Last (with Liam Verbeek)",
      "transcript_path": "data/eval/sources/curated_5feeds_raw_v1/feed-p01/p01_e01.txt",
      "transcript_hash": "a650e729cc8b7379c94fd5b29c092bcd32a8c7e4c2086f1321d6ed496718b9b4",
      "preprocessing_profile": "cleaning_v3",
      "duration_minutes": 10.5
    }
  ]
}
```

### Manual Dataset Creation

You can also create dataset JSONs manually. Each episode must have:

- `episode_id`: Unique identifier
- `transcript_path`: Path to cleaned transcript file
- `transcript_hash`: SHA256 hash of transcript content

Optional fields:

- `title`: Episode title
- `preprocessing_profile`: Profile used for cleaning
- `transcript_raw_path`: Path to raw transcript
- `golden_summary_long_path`: Path to long golden summary
- `golden_summary_short_path`: Path to short golden summary
- `duration_minutes`: Episode duration in minutes

---

## Step 1a: Materialize Dataset (Recommended)

**Prerequisites:** You must have created a dataset first (Step 1).

Materialization validates dataset integrity and creates a clean, reproducible copy of all transcripts.

### Why Materialize?

Materialization proves:

- Dataset JSON is correct
- Paths resolve correctly
- Hashes match expected values
- Materialization is reproducible

### Materializing a Dataset

**Using the Make Command (Recommended):**

```bash
make dataset-materialize DATASET_ID=curated_5feeds_smoke_v1
```

**With custom output directory:**

```bash
make dataset-materialize \
  DATASET_ID=curated_5feeds_smoke_v1 \
  OUTPUT_DIR=data/eval/materialized
```

**Using the Script Directly:**

```bash
python scripts/eval/materialize_dataset.py \
  --dataset-id curated_5feeds_smoke_v1 \
  --output-dir data/eval/materialized
```

### What Materialization Does

1. **Validates dataset JSON** - Checks that all required fields are present
2. **Resolves paths** - Verifies all transcript files exist
3. **Validates hashes** - Computes SHA256 and compares to expected hash
4. **Copies transcripts** - Creates clean copies in materialized directory
5. **Creates metadata** - Generates episode and dataset metadata files

**Hash validation:**

If a transcript hash doesn't match, materialization fails with a clear error:

```text
ERROR: Episode p01_e01: HASH MISMATCH - transcript file has been modified!
  Expected hash: abc123...
  Actual hash:   def456...
  File:          data/eval/sources/curated_5feeds_raw_v1/feed-p01/p01_e01.txt
  This indicates the transcript file has changed since the dataset was created.
```

### Materialized Dataset Structure

```text
data/eval/materialized/curated_5feeds_smoke_v1/
├── meta.json                    # Dataset-level metadata
├── p01_e01.txt                  # Copied transcript
├── p01_e01.meta.json            # Episode metadata
├── p02_e01.txt
├── p02_e01.meta.json
└── ...
```

**Dataset metadata (`meta.json`):**

```json
{
  "dataset_id": "curated_5feeds_smoke_v1",
  "source_dataset_file": "data/eval/datasets/curated_5feeds_smoke_v1.json",
  "num_episodes": 5,
  "materialized_at": "2026-01-13T12:22:41.258855Z",
  "episodes": [
    {
      "episode_id": "p01_e01",
      "transcript_path": "p01_e01.txt",
      "meta_path": "p01_e01.meta.json"
    }
  ]
}
```

**Episode metadata (`{episode_id}.meta.json`):**

```json
{
  "episode_id": "p01_e01",
  "transcript_path": "p01_e01.txt",
  "transcript_hash": "a650e729cc8b7379c94fd5b29c092bcd32a8c7e4c2086f1321d6ed496718b9b4",
  "source_transcript_path": "/path/to/source/p01_e01.txt",
  "preprocessing_profile": "cleaning_v3",
  "title": "Episode 1: Building Trails That Last...",
  "duration_minutes": 10.5
}
```

### Reproducibility

Materialization is reproducible - you can delete the materialized directory and regenerate it byte-for-byte:

```bash
rm -rf data/eval/materialized/curated_5feeds_smoke_v1
make dataset-materialize DATASET_ID=curated_5feeds_smoke_v1
```

---

## Step 2: Create a Baseline

**Prerequisites:** You must have created a dataset first (Step 1). The baseline will use that dataset to know which episodes to process.

Baselines are frozen reference results from a known system state. They serve as comparison points for experiments.

### Creating a Baseline with Make (Recommended)

Use the make command to materialize a baseline:

```bash
make baseline-create \
  BASELINE_ID=bart_led_baseline_v1 \
  DATASET_ID=curated_5feeds_smoke_v1
```

**With optional experiment config:**

```bash
make baseline-create \
  BASELINE_ID=bart_led_baseline_v1 \
  DATASET_ID=curated_5feeds_smoke_v1 \
  EXPERIMENT_CONFIG=data/eval/configs/baseline_config.yaml \
  PREPROCESSING_PROFILE=cleaning_v3
```

### Creating a Baseline with the Script

Alternatively, you can call the script directly:

```bash
python scripts/eval/materialize_baseline.py \
  --baseline-id bart_led_baseline_v1 \
  --dataset-id curated_5feeds_smoke_v1 \
  --experiment-config data/eval/configs/baseline_config.yaml \
  --preprocessing-profile cleaning_v3
```

This will:

- Load the dataset JSON (created in Step 1)
- Process each episode using the specified configuration
- Save predictions to `benchmarks/baselines/{baseline_id}/predictions/`
- Generate metadata, fingerprints, and metrics
- **Important**: Baselines are immutable - you cannot overwrite an existing baseline

### Baseline Structure

A baseline directory contains:

```text
benchmarks/baselines/bart_led_baseline_v1/
├── metadata.json          # Baseline metadata (dataset_id, git commit, stats)
├── fingerprint.json       # System fingerprint (model, version, device)
├── metrics.json           # Aggregate metrics
├── config.yaml            # Experiment config used (if provided)
├── predictions/           # Individual episode predictions
│   ├── ep01.json
│   ├── ep02.json
│   └── ...
└── artifacts/             # Additional artifacts (if any)
```

### Baseline Metadata

The `metadata.json` includes:

- `baseline_id`: Unique identifier
- `dataset_id`: Dataset used
- `created_at`: Timestamp
- `git_commit`: Git commit SHA when baseline was created
- `git_is_dirty`: Whether repo had uncommitted changes
- `provider_type`: Provider used (e.g., "OpenAIProvider")
- `model_name`: Model name
- `preprocessing_profile`: Preprocessing profile ID
- `stats`: Processing statistics (num_episodes, avg_time, compression, etc.)

---

## Step 3: Run an Experiment

Experiments test new configurations against datasets and compare results to baselines.

### Creating an Experiment Config

Create a YAML file (e.g., `data/eval/configs/my_experiment.yaml`):

```yaml
id: "summarization_openai_long_v2"
task: "summarization"

backend:
  type: "openai"
  model: "gpt-4o-mini"

prompts:
  system: "summarization/system_v1"
  user: "summarization/long_v2_more_narrative"
  params:
    paragraphs_min: 3
    paragraphs_max: 6

data:
  dataset_id: "curated_5feeds_smoke_v1"  # Use dataset-based mode (recommended)

params:
  max_output_tokens: 900
  temperature: 0.7

# Contract fields (RFC-015)
dataset_id: "curated_5feeds_smoke_v1"
baseline_id: "bart_led_baseline_v1"
golden_required: true
golden_ref: "data/eval"  # Path to golden references
```

### Data Configuration Modes

The experiment runner supports two data configuration modes:

#### Dataset-Based Mode (Recommended)

```yaml
data:
  dataset_id: "curated_5feeds_smoke_v1"
```

This loads episode information from `data/eval/datasets/curated_5feeds_smoke_v1.json` (or `benchmarks/datasets/` if not found). Episode IDs are taken directly from the dataset JSON.

#### Glob-Based Mode (Legacy)

```yaml
data:
  episodes_glob: "data/episodes/ep*/transcript.txt"
  id_from: "parent_dir"  # or "stem"
```

This uses glob patterns to discover files. Episode IDs are derived from paths using the `id_from` rule.

**Note**: You cannot specify both `dataset_id` and `episodes_glob` in the same config.

### Running an Experiment

**Prerequisites:** You must have created both a dataset (Step 1) and a baseline (Step 2). The experiment will use the dataset for input and compare against the baseline.

#### Running an Experiment with Make (Recommended)

```bash
export OPENAI_API_KEY="your-key-here"
make experiment-run CONFIG=data/eval/configs/my_experiment.yaml
```

**With custom log level:**

```bash
make experiment-run CONFIG=data/eval/configs/my_experiment.yaml LOG_LEVEL=DEBUG
```

#### Running an Experiment with the Script

Alternatively, you can call the script directly:

```bash
export OPENAI_API_KEY="your-key-here"
python scripts/eval/run_experiment.py data/eval/configs/my_experiment.yaml
```

The experiment runner will:

1. Validate the experiment contract (dataset_id, baseline_id, etc.)
2. Load the dataset and discover input files
3. Process each episode with the specified provider
4. Save predictions to `results/{experiment_id}/predictions.jsonl`
5. Generate metadata, fingerprints, and statistics

### Experiment Results

Results are saved to `results/{experiment_id}/`:

```text
results/summarization_openai_long_v2/
├── predictions.jsonl      # One JSON object per episode (input/output/hashes/timing)
├── run_metadata.json      # Experiment metadata (config, stats, contract info)
└── fingerprint.json       # System fingerprint
```

### Understanding Predictions

Each line in `predictions.jsonl` contains:

```json
{
  "episode_id": "p01_e01",
  "input_path": "data/eval/sources/curated_5feeds_raw_v1/feed-p01/p01_e01.txt",
  "input_hash": "a650e729cc8b7379c94fd5b29c092bcd32a8c7e4c2086f1321d6ed496718b9b4",
  "output": "Summary text here...",
  "output_hash": "abc123...",
  "processing_time_seconds": 2.5,
  "input_length_chars": 50000,
  "output_length_chars": 500
}
```

---

## Step 4: Evaluate Results

Evaluation is handled automatically by the experiment runner. When you run an experiment with `--baseline` and/or `--reference` flags, the system automatically:

1. Computes intrinsic metrics (gates, length, performance, cost)
2. Computes vs_reference metrics (ROUGE, embedding similarity) if references are provided
3. Computes deltas vs baseline if baseline is provided

### Metrics Calculation Flow

```text
experiment-run → run_experiment.py → score_run() → metrics.json
```

1. **Run Experiment**: `scripts/eval/run_experiment.py` processes episodes and generates `predictions.jsonl`
2. **Compute Metrics**: `score_run()` in `src/podcast_scraper/evaluation/scorer.py` reads predictions and computes metrics
3. **Save Results**: Metrics are saved to `data/eval/runs/<run_id>/metrics.json` and `metrics_report.md`

### Running Experiments with Evaluation

To run an experiment with full evaluation, use the `--baseline` and/or `--reference` flags:

```bash
make experiment-run \
  CONFIG=experiments/my_experiment.yaml \
  BASELINE=bart_led_baseline_v1 \
  REFERENCE=silver_gpt52_v1,gold_human_v1
```

**Arguments:**

- `CONFIG` (required) - Experiment config YAML
- `BASELINE` (optional) - Baseline ID for comparison
- `REFERENCE` (optional, comma-separated) - Reference IDs for evaluation (can be silver/gold)
- `LOG_LEVEL` (optional) - Logging level

### Evaluation Architecture

The evaluation system consists of three separate roles that work together:

1. **Runner** - Produces outputs (predictions + fingerprint + run metadata)
2. **Scorer** - Computes metrics (gates, stability, cost/latency, and optionally "vs reference" metrics)
3. **Comparator** - Computes deltas vs baseline

These roles are kept separate in code, even though they can be wired together in one script.

#### Runner (Execution)

The runner executes the experiment and produces:

- `predictions.jsonl` - Model outputs for all episodes
- `fingerprint.json` - System fingerprint (reproducibility)
- `run_metadata.json` - Experiment metadata

**Location:** `scripts/eval/run_experiment.py` (runner phase)

#### Scorer (Metrics)

The scorer computes metrics from predictions. Metrics are divided into two categories:

##### Intrinsic Metrics

Intrinsic metrics are computed from predictions alone and don't require reference summaries. They include:

**1. Quality Gates**

Detect common issues in generated summaries:

- **`boilerplate_leak_rate`**: Fraction of episodes with promotional/sponsor content leaks
  - Patterns detected: "subscribe to our newsletter", "follow us on", "rate and review", etc.
- **`speaker_leak_rate`**: Fraction of episodes with speaker annotations leaking through
  - Patterns detected: "Host:", "Speaker 1:", "[laughter]", etc.
- **`truncation_rate`**: Fraction of episodes that appear truncated
  - Detected by truncation markers ("...", "[TRUNCATED]") or suspiciously short outputs
- **`failed_episodes`**: List of episode IDs that failed quality gates

**2. Length Metrics**

Token-based length statistics:

- **`avg_tokens`**: Average number of tokens per summary (estimated as chars/4)
- **`min_tokens`**: Minimum tokens across all summaries
- **`max_tokens`**: Maximum tokens across all summaries

**3. Performance Metrics**

Latency measurements:

- **`avg_latency_ms`**: Average processing time per episode in milliseconds
  - Extracted from `metadata.processing_time_seconds` in predictions

**4. Cost Metrics (OpenAI Only)**

**Note**: Cost metrics are only included for OpenAI runs. ML model runs skip this section entirely.

- **`avg_cost_usd`**: Average cost per episode in USD
- **`total_cost_usd`**: Total cost for all episodes in USD

Cost is computed from:

- `metadata.cost_usd` (if directly provided by provider)
- `metadata.usage` (token counts) with model-specific pricing:
  - GPT-4o-mini: $0.15/1M input, $0.60/1M output
  - GPT-4o: $2.50/1M input, $10.00/1M output

**Location:** `src/podcast_scraper/evaluation/scorer.py`

#### Comparator (Deltas)

The comparator computes deltas between experiment and baseline:

- Cost deltas
- Latency deltas
- Gate regressions
- ROUGE deltas (if both have same references)

**Location:** `src/podcast_scraper/evaluation/comparator.py`

### Reference Model

References are **optional evaluation targets**. You can have:

- **Baseline** (optional but usually required for experiments) - for regression detection
- **Silver references** (optional) - machine-generated, higher quality
- **Gold references** (optional) - human-verified summaries

**Key principle:** A reference is anything that looks like a run output (predictions.jsonl + fingerprint.json + baseline.json).

### vs_reference Metrics

**vs_reference** metrics compare your predictions against reference summaries (golden or silver standards). These are **optional** and only computed when references are provided.

#### When is vs_reference null?

`vs_reference` is `null` when:

- No references were provided via `--reference` CLI argument or `REFERENCE_IDS` Makefile variable
- The experiment was run without reference evaluation

This is the normal state for most runs - references are optional and only needed when you want to compare against golden/silver standards.

#### How to provide references

```bash
# Single reference via Makefile
make experiment-run CONFIG=... REFERENCE_IDS=golden_v1

# Multiple references via Makefile
make experiment-run CONFIG=... REFERENCE_IDS="golden_v1 silver_v2"

# Via CLI
python scripts/eval/run_experiment.py config.yaml --reference golden_v1 --reference silver_v2
```

#### Reference Structure

References can be:

- **Baselines**: `data/eval/baselines/<baseline_id>/`
- **References**: `data/eval/references/<dataset_id>/<reference_id>/`
- **Legacy baselines**: `benchmarks/baselines/<baseline_id>/`

Each reference must have a `predictions.jsonl` file with the same episode IDs as your run.

#### vs_reference Metrics Computed

When references are provided, the following metrics are computed:

1. **`reference_quality`**: Metadata about the reference (episode count, quality level, etc.)

2. **ROUGE Scores** (requires `rouge-score` package):
   - `rouge1_f1`: ROUGE-1 F1 score (unigram overlap) - measures coverage
   - `rouge2_f1`: ROUGE-2 F1 score (bigram overlap) - measures local coherence
   - `rougeL_f1`: ROUGE-L F1 score (longest common subsequence) - measures structural similarity

3. **BLEU Score** (requires `nltk` package):
   - `bleu`: BLEU score (n-gram precision with brevity penalty)

4. **WER (Word Error Rate)** (requires `jiwer` package):
   - `wer`: Word-level edit distance normalized by reference length

5. **Embedding Similarity** (requires `sentence-transformers` package):
   - `embedding_similarity`: Cosine similarity between embeddings of predictions and references

6. **`numbers_retained`**: TODO - Not yet implemented

#### Example vs_reference Structure

```json
{
  "vs_reference": {
    "golden_v1": {
      "reference_quality": {
        "episode_count": 5,
        "quality_level": "gold"
      },
      "rouge1_f1": 0.45,
      "rouge2_f1": 0.32,
      "rougeL_f1": 0.42,
      "bleu": 0.38,
      "wer": 0.15,
      "embedding_similarity": 0.87
    },
    "silver_v2": {
      "reference_quality": {
        "episode_count": 5,
        "quality_level": "silver"
      },
      "rouge1_f1": 0.42,
      "rouge2_f1": 0.19,
      "rougeL_f1": 0.39,
      "bleu": 0.35,
      "wer": 0.18,
      "embedding_similarity": 0.85
    }
  }
}
```

**Key points:**

- Each reference ID becomes a key in the `vs_reference` dictionary
- All metrics are computed independently for each reference
- Missing dependencies (e.g., `rouge-score` not installed) will result in `null` values for those metrics
- You can compare against multiple references in a single run

### Metrics Structure

#### metrics.json

The scorer generates a `metrics.json` file with the following structure:

```json
{
  "dataset_id": "curated_5feeds_benchmark_v1",
  "run_id": "run_2026-01-16_12-10-03",
  "episode_count": 10,

  "intrinsic": {
    "gates": {
      "speaker_leak_rate": 0.0,
      "boilerplate_leak_rate": 0.0,
      "truncation_rate": 0.0,
      "failed_episodes": []
    },
    "length": {
      "avg_tokens": 420,
      "min_tokens": 310,
      "max_tokens": 560
    },
    "performance": {
      "avg_latency_ms": 1800
    },
    "cost": {
      "total_cost_usd": 0.14,
      "avg_cost_usd": 0.014
    }
  },

  "vs_reference": null
}
```

**Key points:**

- `intrinsic` - Always present (computed from predictions alone)
- `vs_reference` - `null` when no references provided, or a dictionary with reference IDs as keys when references are provided
- Cost section is only included for OpenAI runs (ML models skip it entirely)

#### metrics_report.md

Human-readable markdown report with formatted metrics, suitable for viewing in GitHub or documentation. Includes formatted tables and summaries of all computed metrics.

#### comparisons/vs_{baseline_id}.json

The comparator generates comparison files with deltas:

```json
{
  "baseline_id": "baseline_prod_authority_v1",
  "dataset_id": "curated_5feeds_benchmark_v1",
  "experiment_run_id": "run_2026-01-16_12-10-03",
  "deltas": {
    "cost_total_usd": -0.05,
    "avg_latency_ms": 120,
    "gate_regressions": [],
    "rougeL_f1_vs_silver_gpt52_v1": 0.01
  }
}
```

**Key points:**

- Deltas are computed as: `experiment_value - baseline_value`
- `gate_regressions` is a list of gate names that regressed
- ROUGE deltas are included if both experiment and baseline have the same reference

### Reference Validation

For every reference (baseline/silver/gold), the system enforces:

1. **Dataset ID match**: `reference.dataset_id == run.dataset_id`
2. **Episode ID match**: Episode IDs match exactly (no missing/extra)
3. **Immutable**: Reference is write-once (cannot be overwritten)

If any of these fail → scoring refuses to run.

### Reference Pack Structure

A reference pack should contain at minimum:

```text
references/{dataset_id}/{reference_id}/
├── predictions.jsonl      # Reference text per episode
├── fingerprint.json        # How reference was generated
├── baseline.json           # Reference metadata (dataset_id, reference_quality)
└── config.yaml             # Config used (optional)
```

**Note:** A baseline can be promoted to a reference pack if you want. That's fine.

### Evaluation Results

When you run an experiment with evaluation, results are saved to `results/{experiment_id}/`:

```text
results/summarization_openai_long_v2/
├── predictions.jsonl      # Model outputs for all episodes
├── fingerprint.json       # System fingerprint
├── run_metadata.json      # Experiment metadata
├── metrics.json           # Intrinsic + vs_reference metrics
└── comparisons/
    └── vs_baseline_prod_authority_v1.json  # Deltas vs baseline
```

### Key Design Decisions

#### 1. Separation of Concerns

- **Runner** = execution only
- **Scorer** = metrics computation
- **Comparator** = delta computation

This allows:

- Recomputing metrics without re-running inference
- Recomputing comparisons without re-running inference
- Testing each component independently

#### 2. Optional References

References are optional because:

- You can do rigorous evaluation without goldens (Phase 1)
- You can add references incrementally (Phase 2/3)
- Different experiments may need different references

#### 3. Reference as "Anything"

A reference is anything that looks like a run output:

- Baseline can be a reference
- Silver reference can be a reference
- Gold reference can be a reference

This keeps the system flexible.

#### 4. Metrics vs Comparisons

- **Metrics** = absolute facts about this run (+ vs reference scores)
- **Comparisons** = deltas between two runs

This separation allows recomputing comparisons later without re-running inference.

---

## Complete Workflow Example

Here's a complete example workflow:

```bash
# Step 0: Prepare source data
make metadata-generate INPUT_DIR=data/eval/sources
make source-index SOURCE_DIR=data/eval/sources/curated_5feeds_raw_v1

# Step 1: Create datasets
make dataset-smoke      # Creates curated_5feeds_smoke_v1 (5 episodes)
make dataset-benchmark  # Creates curated_5feeds_benchmark_v1 (10 episodes)
make dataset-raw        # Creates curated_5feeds_raw_v1 (all episodes)

# Step 1a: Materialize dataset (validate integrity)
make dataset-materialize DATASET_ID=curated_5feeds_smoke_v1

# Step 2: Create baseline
make baseline-create \
  BASELINE_ID=bart_led_baseline_v1 \
  DATASET_ID=curated_5feeds_smoke_v1

# Step 3: Run experiment
export OPENAI_API_KEY="your-key-here"
make experiment-run CONFIG=data/eval/configs/my_experiment.yaml

# Step 4: Run experiment with evaluation
make experiment-run \
  CONFIG=experiments/my_experiment.yaml \
  BASELINE=bart_led_baseline_v1 \
  REFERENCE=silver_gpt52_v1

# Results are automatically computed:
# - results/{experiment_id}/metrics.json (intrinsic + vs_reference)
# - results/{experiment_id}/comparisons/vs_{baseline_id}.json (deltas)

# Review results:
cat results/summarization_openai_long_v2/metrics.json | jq '.intrinsic'
cat results/summarization_openai_long_v2/metrics.json | jq '.vs_reference'
cat results/summarization_openai_long_v2/comparisons/vs_bart_led_baseline_v1.json
```

---

## Best Practices

### Source Data Management

- **Generate metadata first**: Always generate metadata from RSS XML before creating datasets
- **Create source indexes**: Use indexes for inventory management and drift detection
- **Freeze source data**: Once datasets are created, avoid modifying source transcripts

### Dataset Management

- **Freeze datasets**: Once created, datasets should be immutable
- **Version datasets**: Use versioned IDs (e.g., `curated_5feeds_smoke_v1`, `curated_5feeds_smoke_v2`)
- **Document datasets**: Include clear descriptions and content regime
- **Materialize datasets**: Always materialize datasets to validate integrity before use
- **Use appropriate sizes**: Use smoke datasets for quick tests, benchmark datasets for evaluation, raw datasets for comprehensive analysis

### Baseline Management

- **Create baselines on clean commits**: Avoid creating baselines with uncommitted changes
- **Document baseline purpose**: Use descriptive baseline IDs
- **Version baselines**: Use versioned IDs (e.g., `bart_led_baseline_v1`, `bart_led_baseline_v2`)
- **Never overwrite**: Baselines are immutable - create new ones for changes

### Experiment Management

- **Use descriptive IDs**: Include model, task, and version in experiment ID
- **Always specify baseline**: Experiments must compare against a baseline
- **Validate contracts**: Ensure dataset_id matches between experiment and baseline
- **Track golden references**: Use `golden_required: true` when evaluation is needed

### Workflow

1. **Prepare source data** → `make metadata-generate` → `make source-index`
2. **Create dataset** → `make dataset-smoke` / `make dataset-benchmark` / `make dataset-raw`
3. **Materialize dataset** → `make dataset-materialize DATASET_ID=...` (recommended)
4. **Create baseline** → `make baseline-create BASELINE_ID=... DATASET_ID=...`
5. **Run experiment** → `make experiment-run CONFIG=...`
6. **Run experiment with evaluation** → `make experiment-run CONFIG=... BASELINE=... REFERENCE=...` (evaluation is automatic)

---

## Troubleshooting

### "Dataset definition not found"

- Check that `data/eval/datasets/{dataset_id}.json` or `benchmarks/datasets/{dataset_id}.json` exists
- Verify the dataset_id in your experiment config matches the JSON filename

### "Baseline not found"

- Check that `benchmarks/baselines/{baseline_id}/` exists
- Verify the baseline_id in your experiment config is correct
- Create the baseline first using `make baseline-create`

### "Dataset mismatch"

- The experiment's `dataset_id` must match the baseline's `dataset_id`
- Check `benchmarks/baselines/{baseline_id}/metadata.json` to see which dataset was used

### "No input files found"

- For dataset mode: Verify transcript paths in the dataset JSON exist
- For glob mode: Check that the glob pattern matches files in your directory

### "Episode not found in dataset"

- The transcript path in the dataset JSON must match the actual file path
- Use absolute paths or paths relative to the project root

### "Hash mismatch" (during materialization)

- The transcript file has been modified since the dataset was created
- Regenerate the dataset or restore the original transcript file
- Check `data/eval/sources/` for the original files

### "Materialized directory already exists"

- The script will automatically remove and recreate the directory
- This ensures reproducible materialization

---

## Next Steps

This guide will evolve as the experiment system matures. Planned additions:

- [ ] Automated evaluation integration
- [ ] Comparison tools (experiment vs baseline)
- [ ] Regression detection
- [ ] CI/CD integration
- [ ] Cost tracking
- [ ] Visualization tools

---

## References

- **RFC-015**: AI Experiment Pipeline
- **RFC-041**: Benchmarking Framework
- **Implementation Plan**: `docs/wip/ai-quality-implementation-plan-sync.md`
- **Dataset Format**: `data/eval/datasets/curated_5feeds_smoke_v1.json` (example)
- **Baseline Format**: `benchmarks/baselines/` (examples)
