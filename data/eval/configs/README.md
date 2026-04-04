# Experiment Configs

This directory contains experiment configuration YAML files that define how experiments are run.

## Purpose

Experiment configs are **inputs** to the experiment runner. They specify:

- Task type (`summarization`, `ner_entities`, `grounded_insights`, or `knowledge_graph`)
- Model/provider configuration (OpenAI, Gemini, HuggingFace local, spaCy, etc.)
- Prompt templates to use (for OpenAI, Gemini, and Ollama backends)
- Dataset references
- Generation parameters (temperature, max tokens, etc.)
- Scoring parameters (for NER tasks: match modes, label sets)

## Directory Layout

```text
configs/
├── summarization/          # Autoresearch paragraph-track configs (autoresearch_prompt_*_paragraph_v1)
├── summarization_bullets/  # Autoresearch bullets-track configs (autoresearch_prompt_*_bullets_v1)
├── silver_selection/       # Silver reference candidate configs (silver_candidate_*, silver_openai_*)
├── ml/                     # HuggingFace / hybrid ML baseline configs (baseline_ml_*, hybrid_ml_*)
├── ner/                    # NER / spaCy configs (ner_entities_*, baseline_ner_*)
├── _archive/               # Archived old configs
└── *.yaml                  # One-off experiments and legacy configs (llm_*, gil_*, kg_*, etc.)
```

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
make experiment-run CONFIG=data/eval/configs/summarization/autoresearch_prompt_openai_smoke_paragraph_v1.yaml
make experiment-run CONFIG=data/eval/configs/silver_selection/silver_candidate_anthropic_sonnet46_smoke_v1.yaml
make experiment-run CONFIG=data/eval/configs/my_experiment.yaml  # root for one-offs
```

Optional: pass a **silver reference** id (comma-separated) so `metrics.json` gets `vs_reference`
(ROUGE, embedding cosine, etc.):

```bash
make experiment-run CONFIG=data/eval/configs/summarization/autoresearch_prompt_ollama_qwen25_7b_smoke_paragraph_v1.yaml \
  REFERENCE=silver_sonnet46_smoke_v1
```

**Preprocessing:** Autoresearch and smoke configs use `preprocessing_profile: "cleaning_v4"` so
transcripts are cleaned the same way before every model sees them — comparable runs across providers.

**Metrics schema:** New summarization runs emit `schema: metrics_summarization_v2` in
`metrics.json` (nested `intrinsic`, `episode_count`, optional latency percentiles). See
`data/eval/schemas/metrics_summarization_v2.json`.

## Naming Convention

**Critical Rule**: The filename **must** match the `id` field in the config file.

For example, if your config has `id: "baseline_bart_small_led_long_fast"`, the filename should be `baseline_bart_small_led_long_fast.yaml`.

This ensures:

- Consistency between filename and internal ID
- Easy discovery of configs by their ID
- No confusion when referencing configs

### Naming Guidelines

Use descriptive names that indicate:

- Task (e.g., `summarization_`, `ner_entities_`, `gil_eval_`, `kg_eval_`, `transcription_`)
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

### eval_stub Backend (GIL / KG eval)

- **task**: Must be `"grounded_insights"` or `"knowledge_graph"` (one capability per config).
- **backend.type**: `"eval_stub"` — runs the product `build_artifact` pipeline with **stub**
  sources by default (no API keys). Tune via **`params`**:
  - GIL: `gi_insight_source`, `gi_require_grounding`, `gi_max_insights`
  - KG: `kg_extraction_source`
- Example configs: `gil_eval_stub_curated_5feeds_smoke_v1.yaml`,
  `kg_eval_stub_curated_5feeds_smoke_v1.yaml`

### Promoting a run to a silver reference (summarization)

A **silver reference** is not something you declare in YAML ahead of time. You **run an
experiment** (outputs land in `data/eval/runs/<run_id>/`), **review** the run, then
**promote** it with `make run-promote`. Only then does a frozen tree appear under
`data/eval/references/silver/<promoted_id>/`. The experiment config’s `id` is the **run id**
until promotion; the **promoted id** is chosen at promotion time.

**Silver reference selection via pairwise judge (current approach):**

Generate candidate runs for 2–3 competing frontier models using configs in
`configs/silver_selection/`, then compare them head-to-head:

```bash
# 1. Generate candidates
make experiment-run CONFIG=data/eval/configs/silver_selection/silver_candidate_openai_gpt54_smoke_v1.yaml FORCE=1
make experiment-run CONFIG=data/eval/configs/silver_selection/silver_candidate_anthropic_sonnet46_smoke_v1.yaml FORCE=1
make experiment-run CONFIG=data/eval/configs/silver_selection/silver_candidate_gemini31pro_smoke_v1.yaml FORCE=1

# 2. Run pairwise comparisons (dual OpenAI + Anthropic judges, winner on stdout)
make silver-pairwise CANDIDATE_A=silver_candidate_openai_gpt54_smoke_v1 CANDIDATE_B=silver_candidate_anthropic_sonnet46_smoke_v1 OUTPUT=results/pairwise_a_vs_b.json
make silver-pairwise CANDIDATE_A=silver_candidate_anthropic_sonnet46_smoke_v1 CANDIDATE_B=silver_candidate_gemini31pro_smoke_v1 OUTPUT=results/pairwise_b_vs_c.json

# 3. Promote the winner
make run-promote RUN_ID=silver_candidate_anthropic_sonnet46_smoke_v1 AS=reference PROMOTED_ID=silver_sonnet46_smoke_v1 REFERENCE_QUALITY=silver REASON=”...”
```

**Active silver reference:** `silver_sonnet46_smoke_v1` (Claude Sonnet 4.6, selected April 2026:
3-1-1 vs GPT-5.4, 5-0 vs Gemini 2.0 Flash). See `scripts/eval/pairwise_judge.py` for judge code.

**Notes on model compatibility:**

- `gpt-5.4` and newer OpenAI models require `max_completion_tokens` (not `max_tokens`) — handled
  automatically in the OpenAI provider.
- `gemini-2.5-pro` and `gemini-3.1-pro-preview` are thinking models that exhaust `max_output_tokens`
  on internal reasoning; use `gemini-2.0-flash` (GA, non-thinking) for Gemini candidates.

**Use** the active reference in scoring: `REFERENCE=silver_sonnet46_smoke_v1`

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
