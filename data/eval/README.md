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
- **issue-477/** - Issue #477 bundled LLM eval YAMLs + README ([issue-477/README.md](issue-477/README.md))
- **baselines/** - Frozen reference runs used for regression detection
- **references/** - Frozen quality targets (silver/gold references for evaluation)
- **runs/** - Ad-hoc experiments and temporary outputs

**Autoresearch (RFC-057, summarization):** configs live under `configs/summarization/`
(paragraph) and `configs/summarization_bullets/` (bullets). Every provider has a **2×2 matrix**
of configs: smoke/benchmark × paragraph/bullets. The active silver references are:

- `silver_opus47_smoke_v1` — prose paragraph, 5 eps (smoke) — **active**, upgraded from Sonnet 4.6 per #939
- `silver_opus47_smoke_v2` — prose paragraph, 5 eps (smoke v2 dataset) — **active**
- `silver_sonnet46_smoke_v1` / `silver_sonnet46_smoke_v2` — historical paragraph smoke (kept for comparison)
- `silver_sonnet46_benchmark_v1` — prose paragraph, 10 eps (benchmark)
- `silver_sonnet46_smoke_bullets_v1` — JSON bullets, 5 eps (smoke)
- `silver_sonnet46_benchmark_bullets_v1` — JSON bullets, 10 eps (benchmark)

Paragraph smoke silvers were upgraded from Sonnet 4.6 to Opus 4.7 in June 2026 per
[#939](https://github.com/chipi/podcast_scraper/issues/939) — see
`docs/guides/eval-reports/SILVER_OPUS47_GENERATION_2026_06.md`. The original Sonnet 4.6
silvers were selected via pairwise LLM judge (won vs GPT-4o and GPT-5.4). Use
`silver_opus47_*` for new paragraph smoke experiments; bullets + benchmark tracks
still pair with the Sonnet 4.6 silvers until those quality ceilings become the
limiting factor. `silver_gpt4o_*` references are archived — retained for historical
traceability only.

See `configs/README.md` for the full eval run matrix, trigger rules, and silver selection
workflow. See `references/silver/README.md` for when to create new silver references.

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
