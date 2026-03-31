# Evaluation Data Layout

This directory contains all datasets, references, baselines, and evaluation artifacts used for ML/AI quality measurement.

## Supported Tasks

The evaluation system supports these task types:

- **summarization** - Text summarization tasks (e.g., episode summaries)
- **ner_entities** - Named Entity Recognition tasks (e.g., extracting host/guest names, show titles)
- **grounded_insights** - Grounded Insight Layer (GIL) artifacts per episode (`output.gil` in
  `predictions.jsonl`). Currently **`eval_stub` backend only** (stub pipeline; no LLM).
- **knowledge_graph** - Knowledge graph artifacts per episode (`output.kg`). Currently
  **`eval_stub` backend only** (stub extraction; no LLM).

GIL and KG are **separate** experiment configs and runs (not combined in one run).

## Structure

- **sources/** - Immutable raw inputs (transcripts, RSS XML, metadata)
- **datasets/** - Dataset definitions (episode selection, canonical episode lists)
- **materialized/** - Derived inputs for runs (re-generable, byte-for-byte reproducible)
- **configs/** - Experiment configuration YAML files (inputs to experiments)
- **baselines/** - Frozen reference runs used for regression detection
- **references/** - Frozen quality targets (silver/gold references for evaluation)
- **runs/** - Ad-hoc experiments and temporary outputs

**Autoresearch (RFC-057, bullet summarization):** configs live under `configs/` (e.g.
`autoresearch_prompt_openai_smoke_bullets_v1.yaml`). The default ROUGE reference
`silver_gpt4o_smoke_bullets_v1` appears under `references/silver/` only **after** you promote a
successful run from `experiment_openai_gpt4o_smoke_bullets_v1` (see `configs/README.md`).

## Summarization metrics schema

Runs from `scripts/eval/run_experiment.py` write `metrics.json` with
`schema: metrics_summarization_v2`: nested `intrinsic` (gates, warnings, length, performance),
`episode_count`, and optional `vs_reference`. Validated against
`data/eval/schemas/metrics_summarization_v2.json`. The older flat layout is documented in
`metrics_summarization_v1.json` only (not emitted by the current scorer).


## Invariants

- **sources/**, **datasets/**, **baselines/**, **references/** are immutable once published
- **materialized/** and **runs/** can always be regenerated
- Comparisons must always use the same `dataset_id`
- This artifact is immutable once published

## What Goes Where

### Sources (`sources/`)

Raw, unmodified inputs from the ingestion pipeline. Contains:

- Original transcripts (`.txt` files)
- RSS XML feeds (`.xml` files)
- Episode metadata (`.metadata.json` files)
- Source indexes (`index.json`)

**Do not:** Edit transcripts, add derived data, store model outputs here.

### Datasets (`datasets/`)

Canonical, frozen sets of episodes defined in JSON. Each dataset:

- References episodes from `sources/`
- Defines episode selection criteria
- Is versioned (e.g., `curated_5feeds_smoke_v1`, `curated_5feeds_benchmark_v1`)

**Do not:** Modify dataset JSON files after they're used in baselines or experiments.

### Materialized (`materialized/`)

Derived, disposable state generated from datasets and sources. Contains:

- Validated, copied transcripts
- Per-episode metadata
- Hash-verified integrity

**Safe to delete** - can be regenerated from sources and dataset definitions.

### Baselines (`baselines/`)

Frozen reference runs representing known-good system behavior. Used for:

- Regression detection
- CI gating
- Cost/latency tradeoff evaluation

**Immutable** - never overwrite. Create new versioned baselines instead.

### References (`references/`)

Frozen quality targets for evaluation. Can be:

- **Silver** - LLM-generated, high-quality targets (used for summarization)
- **Gold** - Human-verified ground truth (summarization, NER, and optional **GIL** / **KG**
  under `references/gold/gil/{ref_id}/` and `references/gold/kg/{ref_id}/` as
  `{episode_id}.json` files)

**Immutable** - used for quality metrics:

- Summarization: ROUGE, embedding similarity, coverage ratio
- NER: Precision, recall, F1 (entity-set and mention-level)

### Configs (`configs/`)

Experiment configuration YAML files that define how experiments are run. Each config specifies:

- Model/provider configuration
- Prompt templates
- Dataset references
- Generation parameters

**Note:** Configs are inputs to experiments, not results. Results are stored in `runs/`.

### Runs (`runs/`)

Temporary experiment outputs. Can be:

- Promoted to baselines or references
- Deleted when no longer needed

**Disposable** - safe to clean up after promotion or review.

## Metadata Schema

Episode metadata files (`.metadata.json` and `.meta.json`) follow a JSON schema:

- **Schema file:** `schemas/episode_metadata.schema.json`
- **Current version:** 1.0
- **Key design:** Separates facts (speakers) from expectations (output quality requirements)

The schema defines:

- Required fields: `metadata_version`, `source_episode_id`
- Speaker structure: `id`, `name`, `role` (facts about who participated)
- Expectations structure: `allow_speaker_names`, `allow_speaker_labels`, `allow_sponsor_content` (output quality requirements)

See `schemas/episode_metadata.schema.json` for the complete schema definition with examples.

## Governance

Each subdirectory contains README files explaining:

- Why the artifact exists
- What it contains
- What must never change

These READMEs are part of the governance layer - they prevent accidental misuse.
