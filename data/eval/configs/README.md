# Experiment Configs

This directory contains experiment configuration YAML files that define how experiments are run.

## Purpose

Experiment configs are **inputs** to the experiment runner. They specify:

- Task type (`summarization` or `ner_entities`)
- Model/provider configuration (OpenAI, HuggingFace local, spaCy, etc.)
- Prompt templates to use (for OpenAI backends)
- Dataset references
- Generation parameters (temperature, max tokens, etc.)
- Scoring parameters (for NER tasks: match modes, label sets)

## Structure

Each config file is a YAML file that follows the `ExperimentConfig` schema:

```yaml
# Example: OpenAI backend (prompts required)
id: "my_experiment_v1"
task: "summarization"

backend:
  type: "openai"
  model: "gpt-4o-mini"

prompts:  # Required for OpenAI backend
  user: "summarization/long_v1"

data:
  dataset_id: "curated_5feeds_smoke_v1"

params:
  max_length: 800
  temperature: 0.0
```

```yaml
# Example: hf_local backend (prompts optional, not used)
id: "baseline_bart_small_led_long_fast"
task: "summarization"

backend:
  type: "hf_local"
  map_model: "bart-small"
  reduce_model: "long-fast"

# prompts: optional for hf_local (BART/LED models don't use prompts)

data:
  dataset_id: "curated_5feeds_smoke_v1"

params:
  max_length: 150
  min_length: 30
```

```yaml
# Example: NER task with spaCy backend
id: "ner_entities_spacy_trf_v1"
task: "ner_entities"

backend:
  type: "spacy_local"
  model: "en_core_web_trf"  # or "en_core_web_sm" for dev

data:
  dataset_id: "curated_5feeds_smoke_v1"

preprocessing_profile: "cleaning_v3"

params:
  labels: ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]
  scoring:
    mode: ["entity_set", "mention_exact", "mention_overlap"]
```

## Usage

Configs are referenced when running experiments:

```bash
make experiment-run CONFIG=data/eval/configs/my_experiment.yaml
```

## Naming Convention

**Critical Rule**: The filename **must** match the `id` field in the config file.

For example, if your config has `id: "baseline_bart_small_led_long_fast"`, the filename should be `baseline_bart_small_led_long_fast.yaml`.

This ensures:

- Consistency between filename and internal ID
- Easy discovery of configs by their ID
- No confusion when referencing configs

### Naming Guidelines

Use descriptive names that indicate:

- Task (e.g., `summarization_`, `ner_entities_`, `transcription_`)
- Model/provider (e.g., `openai_`, `bart_`, `led_`, `spacy_`)
- Version (e.g., `_v1`, `_v2`)

Examples:

- `summarization_openai_long_v1.yaml` (with `id: "summarization_openai_long_v1"`)
- `baseline_bart_small_led_long_fast.yaml` (with `id: "baseline_bart_small_led_long_fast"`)
- `ner_entities_spacy_trf_v1.yaml` (with `id: "ner_entities_spacy_trf_v1"`)
- `experiment_prompt_v4.yaml` (with `id: "experiment_prompt_v4"`)

## Backend-Specific Requirements

### OpenAI Backend

- **prompts**: **Required** - OpenAI models use prompt templates to guide generation
- Example: `prompts.user: "summarization/long_v1"`

### hf_local Backend (BART/LED)

- **prompts**: **Optional** - Local ML models (BART/LED) don't use prompts
  - They summarize based on training, not instructions
  - Prompts are ignored if provided
  - You can omit the `prompts` section entirely for cleaner configs

### spaCy Backend (NER)

- **task**: Must be `"ner_entities"`
- **model**: spaCy model name (e.g., `"en_core_web_trf"` for prod, `"en_core_web_sm"` for dev)
- **preprocessing_profile**: Preprocessing profile ID (e.g., `"cleaning_v3"`)
- **params.labels**: List of entity labels to extract (e.g., `["PERSON", "ORG"]`)
- **params.scoring.mode**: Scoring modes to compute (e.g., `["entity_set", "mention_exact", "mention_overlap"]`)
  - `entity_set`: Position-agnostic, text-based comparison (primary for KG)
  - `mention_exact`: Exact offset matching
  - `mention_overlap`: Overlapping span matching

## Notes

- Configs are versioned (via filename) to track changes over time
- Configs are immutable once used in a baseline (baseline stores a copy)
- Configs can be modified for new experiments, but old versions should be preserved for reproducibility
