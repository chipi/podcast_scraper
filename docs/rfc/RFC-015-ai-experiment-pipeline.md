# RFC-015: AI Experiment Pipeline

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, researchers tuning AI models/prompts, developers evaluating model performance
- **Related PRDs**: `docs/prd/PRD-006-openai-provider-integration.md`, `docs/prd/PRD-007-ai-experiment-pipeline.md`
- **Related RFCs**: `docs/rfc/RFC-012-episode-summarization.md`, `docs/rfc/RFC-013-openai-provider-implementation.md`, `docs/rfc/RFC-016-modularization-for-ai-experiments.md`, `docs/rfc/RFC-017-prompt-management.md`
- **Related Issues**: (to be created)

## Abstract

Technical design and implementation plan for a repeatable AI experiment pipeline that enables rapid iteration on model selection, prompt engineering, and parameter tuning without requiring code changes. **Think of it exactly like your unit/integration test pipeline – just for models instead of code.** This RFC focuses on the **technical implementation details** - architecture, code structure, and migration strategy. For product requirements, use cases, and functional specifications, see `docs/prd/PRD-007-ai-experiment-pipeline.md`.

**Key Concept**: The experiment pipeline wraps existing pieces (gold data, HF baseline, eval scripts) in a repeatable "AI experiment pipeline" that sits next to your normal build/CI. Treat model + prompt + params as configuration - you don't want hardcoded experiments in Python, you want config files that define an experiment, like you'd define a GitHub Actions workflow.

**Note**: This RFC describes **how** to implement the AI experiment pipeline capability. For **what** the capability is and **why** we need it, refer to PRD-007.

## Design & Implementation

### 1. Experiment Configuration Format

Experiments are defined in YAML configuration files, stored in `experiments/` directory.

**Example: Local Summarization Experiment**

````yaml

# experiments/summarization_bart_led_local.yaml

id: "summarization_bart_led_v1"
task: "summarization"
description: "Local BART + LED summarization with standard parameters"

models:
  map:
    type: "hf_local"
    name: "facebook/bart-large-cnn"
  reduce:
    type: "hf_local"
    name: "allenai/led-base-16384"

params:
  max_length: 160
  min_length: 60
  chunk_size: 2048
  word_chunk_size: 900
  word_overlap: 150
  device: "mps"  # or "cpu", "cuda"

prompts:

  # For local models, prompts are embedded in the model wrapper

  # This section can be empty or contain model-specific prompt overrides

data:
  episodes_glob: "data/eval/episodes/ep*/transcript.txt"
  gold_data_path: "data/eval/golden/summaries/"
```text

# experiments/summarization_openai_gpt4_mini_v1.yaml

id: "summarization_openai_gpt4_mini_v1"
task: "summarization"
description: "OpenAI GPT-4o-mini summarization with custom prompts"

```text

# experiments/summarization_openai_gpt4_turbo_golden.yaml

id: "summarization_openai_gpt4_turbo_golden"
task: "summarization"
description: "OpenAI GPT-4 Turbo for golden dataset creation (high quality, expensive)"

models:
  summarizer:
    type: "openai"
    name: "gpt-4o-mini"

params:
  long_summary_style: "3-6 paragraphs, high detail"
  short_summary_style: "1-2 paragraphs, concise but specific"
  max_length: 500
  temperature: 0.7

prompts:
  system: "summarization/system_v1"
  user: "summarization/long_v1"

data:
  episodes_glob: "data/eval/episodes/ep*/transcript.txt"
  gold_data_path: "data/eval/golden/summaries/"
````

```yaml

# experiments/ner_openai_gpt4_mini_v1.yaml

```

task: "ner"
description: "OpenAI GPT-4o-mini for speaker detection"

models:
detector:
type: "openai"
name: "gpt-4o-mini"

params:
max_detected_names: 4
temperature: 0.3

prompts:
system: "ner/system_ner_v1"
user: "ner/guest_host_v1"

data:
episodes_glob: "data/eval/episodes/ep\*/transcript.txt"
gold_data_path: "data/eval/golden/ner/"

````text

# experiments/transcription_openai_whisper_v1.yaml

id: "transcription_openai_whisper_v1"
task: "transcription"
description: "OpenAI Whisper API transcription"

models:
  transcriber:
    type: "openai"
    name: "whisper-1"

params:
  language: "en"
  response_format: "verbose_json"

data:
  episodes_glob: "data/eval/episodes/ep*/audio.mp3"
  gold_data_path: "data/eval/golden/transcripts/"
```python

from typing import Literal, Dict, Any, Optional
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    type: Literal["hf_local", "openai"]
    name: str

class ExperimentConfig(BaseModel):
    id: str  # Unique experiment identifier
    task: Literal["summarization", "ner", "transcription"]
    description: Optional[str] = None

    models: Dict[str, ModelConfig]  # e.g., {"map": ModelConfig(...), "reduce": ModelConfig(...)}
    params: Dict[str, Any] = Field(default_factory=dict)
    prompts: Optional[Dict[str, str]] = Field(default_factory=dict)  # system, user paths

    data: Dict[str, str]  # episodes_glob, gold_data_path

```text

## 2.1 Prompt Directory Structure

Prompts are organized in a `prompts/` directory with task-specific subdirectories:

```text

prompts/
  summarization/
    system_v1.txt
    long_v1.txt
    long_v2_more_narrative.txt
    long_v2_focus_on_frameworks.txt
    short_v1.txt
  ner/
    system_ner_v1.txt
    guest_host_v1.txt
    guest_host_v2_strict_roles.txt
    entities_generic_v1.txt

```text
You are summarizing a podcast episode.

Write a detailed, high-information narrative summary of the episode.
Guidelines:

- 3–6 paragraphs.
- Focus on key decisions, arguments, and lessons.
- Ignore sponsorships, ads, and host housekeeping.
- Do not use quotes or speaker names.
- Do not invent information that is not implied in the transcript.

```text
```text

You are summarizing a podcast episode.

Write a detailed, narrative-style summary that reads like a story.
Guidelines:

- 3–6 paragraphs with narrative flow.
- Focus on key decisions, arguments, and lessons.
- Use transitions to connect ideas naturally.
- Ignore sponsorships, ads, and host housekeeping.
- Do not use quotes or speaker names.
- Do not invent information that is not implied in the transcript.

```text

### 2.2 Experiments Reference Prompts by Path

In experiment configs, prompts are referenced by file path, not embedded as strings:

**Example: `experiments/summarization_openai_long_v1.yaml`**

```yaml

id: "summarization_openai_long_v1"
task: "summarization"

backend:
  type: "openai"
  model: "gpt-4o-mini"

prompts:
  system: "summarization/system_v1"
  user: "summarization/long_v1"

data:
  episodes_glob: "data/eval/episodes/ep*/transcript.txt"
  gold_data_path: "data/eval/golden/summaries/"

params:
  max_output_tokens: 900

```yaml

backend:
  type: "openai"

```yaml
prompts:
  system: "prompts/summarization/system_v1.txt"
  user: "prompts/summarization/long_v2_more_narrative.txt"

data:
  episodes_glob: "data/eval/episodes/ep*/transcript.txt"
  gold_data_path: "data/eval/golden/summaries/"

params:
  max_output_tokens: 900

```text

id: "ner_openai_guest_host_v1"
task: "ner_guest_host"

backend:
  type: "openai"
  model: "gpt-4o-mini"

prompts:
  system: "ner/system_ner_v1"
  user: "ner/guest_host_v1"

data:
  episodes_glob: "data/eval/episodes/ep*/meta.json"
  gold_data_path: "data/eval/golden/ner/"

```text

from podcast_scraper.prompt_store import render_prompt, get_prompt_metadata

# Usage in experiment runner:

def generate_predictions(config: ExperimentConfig) -> List[Dict[str, Any]]:
    """Generate predictions using prompts from prompt_store."""

    # Load and render prompts (with parameterization support)

    system_prompt = None
    if config.prompts["system"]:
        system_prompt = render_prompt(
            config.prompts["system"],
            **config.prompts.get("params", {}),
        )

    user_prompt = render_prompt(
        config.prompts["user"],
        **config.prompts.get("params", {}),
    )

```text
    # Get prompt metadata for tracking
```

    prompt_meta = {
        "system": (
            get_prompt_metadata(config.prompts["system"], config.prompts.get("params", {}))
            if config.prompts["system"]
            else None
        ),
        "user": get_prompt_metadata(config.prompts["user"], config.prompts.get("params", {})),
    }

```text
    # Use prompts in backend calls
```

```text
    backend = create_backend(config)
```

```text
    # ... rest of generation logic
```

```text

- LRU caching for performance
- SHA256 hashing for reproducibility
- Error handling and validation

See RFC-017 for complete implementation details.

## 2.4 Prompt Engineering Workflow

When doing prompt engineering, you never touch the runner or backends:

**Workflow:**

1. **Pick a baseline experiment config** (e.g., `summarization_openai_long_v1.yaml`)
2. **Copy the config file** → `summarization_openai_long_v2.yaml`
3. **Create a new prompt file**, e.g., `prompts/summarization/long_v2_focus_on_frameworks.j2`
4. **Update the config** to reference the new prompt:

```yaml

prompts:
  system: "summarization/system_v1"
  user: "summarization/long_v2_focus_on_frameworks"

```text

5. **Run both experiments** with the same runner:

```bash

python scripts/run_experiment.py experiments/summarization_openai_long_v1.yaml
python scripts/run_experiment.py experiments/summarization_openai_long_v2.yaml

```text

6. **Compare metrics** in `results/*/metrics.json`:
   - Did ROUGE-L improve?
   - Did compression ratio stay acceptable?
   - Did episode coverage get better?

### 2.5 Prompt Identity in Results

Prompt identity is recorded in experiment results so you can always answer: **"What exact prompt produced these metrics?"**

**Example: `results/summarization_openai_long_v2_more_narrative/metrics.json`**

```json

{
  "run_id": "summarization_openai_long_v2_more_narrative",
  "task": "summarization",
  "backend": {
    "type": "openai",
    "model": "gpt-4o-mini"
  },
  "prompts": {
    "system_path": "summarization/system_v1",
    "user_path": "summarization/long_v2_more_narrative",
    "system_sha256": "abc123def456...",
    "user_sha256": "789ghi012jkl..."
  },
  "global": {
    "rouge1_f": 0.322,
    "rougeL_f": 0.142,
    "avg_compression": 39.7
  },
  "episodes": {
    "ep01": {
      "rouge1_f": 0.315,
      "rougeL_f": 0.138,
      "compression": 42.1
    }
  }
}

```python

    metrics: Dict[str, Any],
    metrics_file: Path,
    config: ExperimentConfig,
) -> None:
    """Save metrics with prompt tracking."""

    # Load prompts to compute hashes

    system_prompt = load_prompt(config.prompts["system"])
    user_prompt = load_prompt(config.prompts["user"])

    metrics_with_prompts = {
        **metrics,
        "prompts": {
            "system_path": config.prompts["system"],
            "user_path": config.prompts["user"],
            "system_sha256": compute_prompt_hash(system_prompt),
            "user_sha256": compute_prompt_hash(user_prompt),
        }
    }

    metrics_file.write_text(
        json.dumps(metrics_with_prompts, indent=2),
        encoding="utf-8"
    )

```text

Run                                   Model          User Prompt
---------------------------------------------------------------------------
summarization_openai_long_v1          gpt-4o-mini    long_v1.txt
summarization_openai_long_v2_more…    gpt-4o-mini    long_v2_more_narrative.txt

```yaml

- **Code**: Runner + backends (HF, OpenAI, etc.) – should change rarely
- **Data**: Transcripts + gold summaries + (later) gold NER – evolves slowly
- **Experiments**: YAML configs – define what you're testing
- **Prompts**: Text files – your "hyperparameters for language behavior"

When you're "doing prompt engineering", you're really:

- Adding/editing files in `prompts/`
- Adding/changing experiment configs that reference them
- Running the same `run_experiment.py` and comparing metrics

No need to touch core logic.

### 3. Experiment Runner Implementation

The experiment runner evolves from a minimal implementation to a full-featured system. This section shows the evolution path, starting with a simple OpenAI-only runner and building up to support multiple backends, metrics, and advanced features.

#### 3.1 Minimal Implementation (Phase 1)

**Initial implementation focuses on:**

- OpenAI summarization only
- Basic prediction generation
- Integration with existing eval scripts
- Simple metadata tracking

**File: `scripts/run_experiment.py` (Minimal Version)**

```python

#!/usr/bin/env python3

"""
Minimal experiment runner for OpenAI-based summarization.

- Loads an ExperimentConfig from YAML (using experiment_config.py)
- Renders prompts via prompt_store.py (RFC-017)
- Calls OpenAI for each episode (one request per transcript)
- Writes:
    - results/<run_id>/predictions.jsonl
    - results/<run_id>/run_metadata.json

Evaluation (ROUGE, etc.) is handled by existing eval scripts that consume predictions.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from podcast_scraper.experiment_config import (
    ExperimentConfig,
    load_experiment_config,
    discover_input_files,
    episode_id_from_path,
)
from podcast_scraper.prompt_store import render_prompt, get_prompt_metadata

# OpenAI client (relies on OPENAI_API_KEY env var)

client = OpenAI()

def summarize_with_openai(
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    transcript: str,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> str:

```text
    """
    Call OpenAI to summarize a single transcript.
```

```text
    Args:
        model: OpenAI model name (e.g., "gpt-4o-mini")
        system_prompt: System prompt (optional)
        user_prompt: User prompt template (transcript will be embedded)
        transcript: Transcript text to summarize
        max_output_tokens: Maximum tokens in response
        temperature: Sampling temperature
```

```text
    Returns:
        Summary text
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
```

```text
    # Embed transcript into user message
```

    user_content = f"{user_prompt}\n\nTranscript:\n\"\"\"{transcript}\"\"\""
    messages.append({"role": "user", "content": user_content})

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

```text
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
```

```text
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()
```

def run_experiment(cfg: ExperimentConfig) -> None:

```text
    """
    Run a single experiment described by ExperimentConfig.
```

    Phase 1: Only supports OpenAI summarization.
    Future phases will add more backends and tasks.
    """
    run_id = cfg.id
    results_dir = Path("results") / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = results_dir / "predictions.jsonl"
    metadata_path = results_dir / "run_metadata.json"

```text
    # --- Prepare prompts using prompt_store (RFC-017) ---
```

    system_prompt: str | None = None
    if cfg.prompts.system:
        system_prompt = render_prompt(
            cfg.prompts.system,
            **cfg.prompts.params,
        )

    user_prompt = render_prompt(
        cfg.prompts.user,
        **cfg.prompts.params,
    )

```text
    # Collect prompt metadata for reproducibility
```

    system_meta = (
        get_prompt_metadata(cfg.prompts.system, cfg.prompts.params)
        if cfg.prompts.system
        else None
    )
    user_meta = get_prompt_metadata(cfg.prompts.user, cfg.prompts.params)

```text
    # --- Discover input files (episodes) ---
```

```text
    input_files = discover_input_files(cfg.data)
    if not input_files:
        raise RuntimeError(
            f"No input files found for glob: {cfg.data.episodes_glob}"
        )
```

```text
    print(f"[{run_id}] Found {len(input_files)} episode file(s).")
```

```text
    # --- Validate backend (Phase 1: OpenAI only) ---
```

```text
    if cfg.backend.type != "openai":
        raise NotImplementedError(
            f"Phase 1 only supports OpenAI backend. Got: {cfg.backend.type}"
        )
    if cfg.task != "summarization":
        raise NotImplementedError(
            f"Phase 1 only supports summarization task. Got: {cfg.task}"
        )
```

    openai_model = cfg.backend.model
    max_output_tokens = cfg.params.max_output_tokens
    temperature = cfg.params.temperature

```text
    # --- Run per-episode summarization ---
```

```text
    start_time = time.time()
    num_episodes = 0
    total_chars_in = 0
    total_chars_out = 0
```

```text
    with predictions_path.open("w", encoding="utf-8") as pred_f:
        for path in input_files:
            episode_id = episode_id_from_path(path, cfg.data)
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                print(f"[{run_id}] Skipping empty transcript: {path}")
                continue
```

            num_episodes += 1
            total_chars_in += len(text)

```text
            print(f"[{run_id}] Summarizing {episode_id} ({len(text)} chars)...")
            t0 = time.time()
```

            summary = summarize_with_openai(
                model=openai_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                transcript=text,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )

```text
            dt = time.time() - t0
            total_chars_out += len(summary)
            print(f"[{run_id}]   Done in {dt:.1f}s, summary length={len(summary)} chars")
```

```text
            # Write one JSON object per line (JSONL format)
```

            record = {
                "episode_id": episode_id,
                "input_file": str(path),
                "summary": summary,
            }
            pred_f.write(json.dumps(record, ensure_ascii=False) + "\n")

```text
    total_time = time.time() - start_time
```

```text
    # --- Save run-level metadata ---
```

    avg_time = total_time / num_episodes if num_episodes > 0 else 0.0
    avg_compression = (
        (total_chars_in / total_chars_out) if total_chars_out > 0 else None
    )

    metadata = {
        "run_id": run_id,
        "task": cfg.task,
        "backend": {
            "type": cfg.backend.type,
            "model": openai_model,
        },
        "prompts": {
            "system": system_meta,
            "user": user_meta,
        },
        "params": cfg.params.model_dump(),
        "data": {
            "episodes_glob": cfg.data.episodes_glob,
            "id_from": cfg.data.id_from,
            "num_episodes": num_episodes,
        },
        "stats": {
            "total_time_seconds": total_time,
            "avg_time_seconds": avg_time,
            "total_chars_in": total_chars_in,
            "total_chars_out": total_chars_out,
            "avg_compression": avg_compression,
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

```text
    print(f"[{run_id}] Done. Predictions: {predictions_path}")
    print(f"[{run_id}] Metadata: {metadata_path}")
    print(
        f"[{run_id}] Episodes={num_episodes}, "
        f"avg_time={avg_time:.1f}s, "
        f"avg_compression={avg_compression:.1f}x"
        if avg_compression
        else f"[{run_id}] Episodes={num_episodes}, avg_time={avg_time:.1f}s"
    )
```

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an LLM experiment (OpenAI summarization)."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment YAML config.",
    )
    args = parser.parse_args()

```text
    cfg = load_experiment_config(args.config)
```

```text
    # Validate API key
```

```text
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Export it before running this script."
        )
```

```text
    run_experiment(cfg)
```

if __name__ == "__main__":

```text
    main()
```

```yaml

backend:
  type: "openai"

````

prompts:
system: "summarization/system_v1"
user: "summarization/long_v1"
params:
paragraphs_min: 3
paragraphs_max: 6

data:
episodes_glob: "data/eval/episodes/ep\*/transcript.txt"
id_from: "parent_dir"

params:
max_output_tokens: 900
temperature: 0.7

````text
```text

  summarization_openai_long_v1/
    predictions.jsonl        # One prediction per line
    run_metadata.json        # Backend + prompts + stats

```text
```text

```python

# scripts/eval_experiment.py

"""
Evaluate experiment predictions using existing eval scripts.

This script bridges the experiment pipeline with existing evaluation logic.
"""

from pathlib import Path
import json
from typing import Dict, Any

from scripts.eval_summaries import (
    load_golden_summaries,
    compute_rouge_scores,
    compute_compression_ratio,
)

def evaluate_experiment_predictions(
    predictions_file: Path,
    gold_data_path: Path,
) -> Dict[str, Any]:

```text
    """
    Evaluate experiment predictions against golden dataset.
```

    Reuses existing eval_summaries.py logic.
    """

```python
    # Load predictions from JSONL
```

    predictions = {}
    with predictions_file.open("r", encoding="utf-8") as f:

```text
        for line in f:
            record = json.loads(line)
            predictions[record["episode_id"]] = record["summary"]
```

```text
    # Load golden summaries (existing function)
```

```text
    gold_summaries = load_golden_summaries(gold_data_path)
```

```text
    # Compute metrics (existing functions)
```

```text
    rouge_scores = compute_rouge_scores(predictions, gold_summaries)
    compression_ratios = {
        ep_id: compute_compression_ratio(pred, gold_summaries.get(ep_id, ""))
        for ep_id, pred in predictions.items()
    }
```

```text
    # Aggregate metrics
```

    global_metrics = {
        "rouge1_f": sum(s["rouge1_f"] for s in rouge_scores.values()) / len(rouge_scores),
        "rougeL_f": sum(s["rougeL_f"] for s in rouge_scores.values()) / len(rouge_scores),
        "avg_compression": sum(compression_ratios.values()) / len(compression_ratios),
        "num_episodes": len(predictions),
    }

    episode_metrics = {
        ep_id: {
            **rouge_scores.get(ep_id, {}),
            "compression": compression_ratios.get(ep_id),
        }
        for ep_id in predictions.keys()
    }

```text
    return {
        "global": global_metrics,
        "episodes": episode_metrics,
    }
```

```text

# Generate predictions

python scripts/run_experiment.py experiments/summarization_openai_long_v1.yaml

# Evaluate predictions

python scripts/eval_experiment.py \
  results/summarization_openai_long_v1/predictions.jsonl \
  data/eval/golden/summaries/

```text
```python

def run_experiment(cfg: ExperimentConfig) -> None:
    """Run experiment using provider system."""

    # Create provider using factory (RFC-016)

    if cfg.task == "summarization":
        from podcast_scraper.summarization import SummarizationProviderFactory
        provider = SummarizationProviderFactory.create(cfg)
        resource = provider.initialize(cfg) if provider else None
    elif cfg.task.startswith("ner_"):
        from podcast_scraper.speaker_detectors import SpeakerDetectorFactory
        provider = SpeakerDetectorFactory.create(cfg)
        resource = provider.initialize(cfg) if provider else None
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    # Process episodes using provider protocol

    predictions = []
    for path in discover_input_files(cfg.data):

```text
        episode_id = episode_id_from_path(path, cfg.data)
        text = path.read_text(encoding="utf-8").strip()
```

```text
        if cfg.task == "summarization" and provider:
            result = provider.summarize(
                text=text,
                cfg=cfg,
                resource=resource,
                max_length=cfg.params.max_length,
                min_length=cfg.params.min_length,
            )
            predictions.append({
                "episode_id": episode_id,
                "summary": result["summary"],
            })
```

```text
        # ... handle other tasks
```

```text
    # Cleanup
```

```text
    if provider and resource:
        provider.cleanup(resource)
```

```text
    # Save predictions and metadata...
```

```yaml

- ✅ **Multiple Backends**: Easy to add HF local, etc.
- ✅ **Protocol Compliance**: All providers implement same interface
- ✅ **No Code Duplication**: Reuse production provider implementations

## 3.4 Evolution Path: Integrated Evaluation

**Phase 3: Built-in Evaluation**

The runner can optionally compute metrics directly:

```python

def run_experiment(
    cfg: ExperimentConfig,
    evaluate: bool = False,
    gold_data_path: Path | None = None,
) -> None:
    """Run experiment with optional evaluation."""

    # ... generate predictions ...

    if evaluate and gold_data_path:
        from scripts.eval_experiment import evaluate_experiment_predictions

        metrics = evaluate_experiment_predictions(
            predictions_path,
            gold_data_path,
        )

        # Add prompt metadata to metrics

```python
        from podcast_scraper.prompt_store import get_prompt_metadata
```

        metrics["prompts"] = {
            "system": (
                get_prompt_metadata(cfg.prompts.system, cfg.prompts.params)
                if cfg.prompts.system
                else None
            ),
            "user": get_prompt_metadata(cfg.prompts.user, cfg.prompts.params),
        }

```text
        # Save metrics
```

        metrics_path = results_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

```text

# Generate + evaluate in one command

python scripts/run_experiment.py \
  experiments/summarization_openai_long_v1.yaml \
  --evaluate \
  --gold-data data/eval/golden/summaries/

```text

```python

def run_experiment(cfg: ExperimentConfig) -> None:
    """Run experiment supporting all tasks and backends."""

    # Create provider based on config

    provider = create_provider_for_experiment(cfg)
    resource = provider.initialize(cfg) if provider else None

    # Process episodes based on task

    predictions = []
    for path in discover_input_files(cfg.data):
        episode_id = episode_id_from_path(path, cfg.data)
        episode_data = load_episode_data(path, cfg.task)

        if cfg.task == "summarization":
            prediction = generate_summarization_prediction(
                provider, resource, episode_data, cfg
            )
        elif cfg.task.startswith("ner_"):
            prediction = generate_ner_prediction(
                provider, resource, episode_data, cfg
            )
        elif cfg.task == "transcription":
            prediction = generate_transcription_prediction(
                provider, resource, episode_data, cfg
            )

        predictions.append({
            "episode_id": episode_id,
            "prediction": prediction,
        })

```text
    # Save predictions...
```

```text

# scripts/run_experiment.py

import argparse
import json
from pathlib import Path
from typing import List, Optional

def run_experiment(
    config_file: str,
    mode: Literal["gen", "eval", "gen+eval"] = "gen+eval",
    output_dir: str = "results",
    force_regenerate: bool = False,
) -> None:
    """Run an AI experiment from a configuration file.

    Args:
        config_file: Path to experiment YAML config
        mode: "gen" (generate only), "eval" (evaluate only), "gen+eval" (both)
        output_dir: Base directory for results
        force_regenerate: If True, regenerate predictions even if they exist
    """

```text
    # Load experiment config
```

```text
    config = load_experiment_config(config_file)
```

```text
    # Create output directory
```

```text
    experiment_output_dir = Path(output_dir) / config.id
    experiment_output_dir.mkdir(parents=True, exist_ok=True)
```

    predictions_file = experiment_output_dir / "predictions.jsonl"
    metrics_file = experiment_output_dir / "metrics.json"

```text
    # Generation phase
```

```text
    if mode in ("gen", "gen+eval"):
        if predictions_file.exists() and not force_regenerate:
            print(f"Predictions already exist: {predictions_file}")
            print("Use --force-regenerate to regenerate")
        else:
            print(f"Generating predictions for experiment: {config.id}")
            predictions = generate_predictions(config)
            save_predictions(predictions, predictions_file)
```

```text
    # Evaluation phase
```

```text
    if mode in ("eval", "gen+eval"):
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
```

```text
        print(f"Evaluating predictions for experiment: {config.id}")
        metrics = evaluate_predictions(config, predictions_file)
        save_metrics(metrics, metrics_file)
```

```text
        # Print summary
```

```text
        print_metrics_summary(metrics)
```

def run_experiments(
    config_files: List[str],
    config_dir: Optional[str] = None,
    task_filter: Optional[str] = None,
    mode: Literal["gen", "eval", "gen+eval"] = "gen+eval",
    output_dir: str = "results",
    force_regenerate: bool = False,
    parallel: bool = False,
    max_workers: int = 4,
    include_golden: bool = False,
) -> None:

```text
    """Run multiple AI experiments.
```

```text
    Args:
        config_files: List of config file paths (can be empty if config_dir provided)
        config_dir: Optional directory containing config files (glob pattern)
        task_filter: Optional task type filter ("summarization", "ner", "transcription")
        mode: "gen" (generate only), "eval" (evaluate only), "gen+eval" (both)
        output_dir: Base directory for results
        force_regenerate: If True, regenerate predictions even if they exist
        parallel: If True, run experiments in parallel
        max_workers: Maximum number of parallel workers
        include_golden: If True, include golden experiments (default: False, excludes golden)
    """
```

```text
    # Collect all config files
```

    all_config_files = []

```text
    # Add explicit config files
```

```text
    all_config_files.extend(config_files)
```

```python
    # Add config files from directory/glob
```

```text
    if config_dir:
        config_path = Path(config_dir)
        if config_path.is_dir():
```

```text
            # Find all YAML files in directory
```

```text
            all_config_files.extend(config_path.glob("*.yaml"))
            all_config_files.extend(config_path.glob("*.yml"))
        else:
```

```text
            # Treat as glob pattern
```

```python
            from glob import glob
            all_config_files.extend(glob(str(config_dir)))
```

```text
    # Filter by task type if specified
```

```text
    if task_filter:
        filtered_configs = []
        for config_file in all_config_files:
            config = load_experiment_config(str(config_file))
            if config.task == task_filter:
                filtered_configs.append(config_file)
        all_config_files = filtered_configs
```

```text
    # Exclude golden experiments unless explicitly included
```

```text
    if not include_golden:
        filtered_configs = []
        for config_file in all_config_files:
```

```text
            # Check filename for golden naming convention
```

```text
            config_file_str = str(config_file)
            is_golden = (
                "_golden" in config_file_str.lower() or
                "_gold" in config_file_str.lower() or
                config_file_str.endswith("_golden.yaml") or
                config_file_str.endswith("_golden.yml") or
                config_file_str.endswith("_gold.yaml") or
                config_file_str.endswith("_gold.yml")
            )
```

```text
            # Also check config file content if is_golden flag is set
```

```text
            if not is_golden:
                try:
                    config = load_experiment_config(config_file_str)
                    is_golden = getattr(config, "is_golden", False)
                except Exception:
                    pass  # If we can't load config, rely on filename
```

```text
            if not is_golden:
                filtered_configs.append(config_file)
            else:
                print(f"Skipping golden experiment: {config_file}")
```

        all_config_files = filtered_configs

```text
    if not all_config_files:
        print("No experiment configs found (excluding golden experiments)")
        print("Use --include-golden to include golden experiments")
        return
```

```text
    print(f"Running {len(all_config_files)} experiment(s)...")
```

```text
    # Run experiments
```

```python
    if parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
```

```text
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_experiment,
                    config_file=str(config_file),
                    mode=mode,
                    output_dir=output_dir,
                    force_regenerate=force_regenerate,
                ): config_file
                for config_file in all_config_files
            }
```

```text
            for future in as_completed(futures):
                config_file = futures[future]
                try:
                    future.result()
                    print(f"✓ Completed: {config_file}")
                except Exception as e:
                    print(f"✗ Failed: {config_file} - {e}")
    else:
```

```text
        # Sequential execution
```

```text
        for config_file in all_config_files:
            try:
                print(f"\n{'='*60}")
                print(f"Running experiment: {config_file}")
                print(f"{'='*60}")
                run_experiment(
                    config_file=str(config_file),
                    mode=mode,
                    output_dir=output_dir,
                    force_regenerate=force_regenerate,
                )
                print(f"✓ Completed: {config_file}")
            except Exception as e:
                print(f"✗ Failed: {config_file} - {e}")
                if not parallel:
```

```text
                    # In sequential mode, optionally continue or stop
```

                    continue

```text
    print(f"\n{'='*60}")
    print(f"Completed {len(all_config_files)} experiment(s)")
    print(f"{'='*60}")
```

```python

    """Generate predictions for all episodes in the experiment.

    Returns:
        List of prediction dictionaries, one per episode
    """

    # Load episodes

    episodes = load_episodes(config.data["episodes_glob"])

    # Load prompts from files

    system_prompt = load_prompt(config.prompts["system"])
    user_prompt_template = load_prompt(config.prompts["user"])

    # Initialize model backend based on config

    backend = create_backend(config)

    predictions = []
    for episode in episodes:
        episode_id = episode["id"]
        print(f"Processing episode: {episode_id}")

```text
        # Apply preprocessing (provider-agnostic, RFC-012)
```

```text
        if config.task == "summarization":
```

```text
            # Preprocess transcript before summarization
```

            cleaned_transcript = clean_transcript(
                episode["transcript"],
                remove_timestamps=True,
                normalize_speakers=True,
                collapse_blank_lines=True,
            )
            cleaned_transcript = remove_sponsor_blocks(cleaned_transcript)

```text
        # Generate prediction based on task type
```

```text
        if config.task == "summarization":
```

```text
            # Format user prompt with episode data
```

            user_prompt = user_prompt_template.format(
                transcript=cleaned_transcript,
                title=episode.get("title", ""),
                description=episode.get("description", ""),
            )

            prediction = backend.summarize(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                params=config.params,
            )
        elif config.task == "ner":

```text
            # Format user prompt with episode data
```

            user_prompt = user_prompt_template.format(
                transcript=episode["transcript"],
                feed_title=episode.get("feed_title", ""),
                episode_title=episode.get("title", ""),
            )

            prediction = backend.detect_speakers(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                params=config.params,
            )
        elif config.task == "transcription":
            prediction = backend.transcribe(
                audio_file=episode["audio_file"],
                params=config.params,
            )

        predictions.append({
            "episode_id": episode_id,
            "experiment_id": config.id,
            "task": config.task,
            "prediction": prediction,
            "metadata": {
                "model": config.models,
                "params": config.params,
                "prompts": {
                    "system_path": config.prompts["system"],
                    "user_path": config.prompts["user"],
                }
            }
        })

```text
    return predictions
```

```python

def create_backend(config: ExperimentConfig):
    """Create appropriate backend based on experiment config.

    Backends are implemented using the provider pattern (RFC-016).
    This function maps experiment configs to providers.
    """
    if config.task == "summarization":
        if config.models.get("summarizer", {}).get("type") == "openai":
            from .backends.openai_summarization import OpenAISummarizationBackend
            return OpenAISummarizationBackend(config)
        else:
            from .backends.local_summarization import LocalSummarizationBackend
            return LocalSummarizationBackend(config)
    elif config.task == "ner":
        if config.models.get("detector", {}).get("type") == "openai":
            from .backends.openai_ner import OpenAINERBackend
            return OpenAINERBackend(config)
        else:

```python
            from .backends.local_ner import LocalNERBackend
            return LocalNERBackend(config)
    elif config.task == "transcription":
        if config.models.get("transcriber", {}).get("type") == "openai":
            from .backends.openai_transcription import OpenAITranscriptionBackend
            return OpenAITranscriptionBackend(config)
        else:
            from .backends.local_transcription import LocalTranscriptionBackend
            return LocalTranscriptionBackend(config)
```

```python

    config: ExperimentConfig,
    predictions_file: Path,
) -> Dict[str, Any]:
    """Evaluate predictions against golden dataset.

    Reuses existing eval scripts (eval_summaries.py, eval_ner.py, etc.)
    """

    # Load predictions

    predictions = load_predictions(predictions_file)

    # Load golden data

    gold_data = load_golden_data(config.data["gold_data_path"])

    # Route to appropriate eval script based on task

```python
    if config.task == "summarization":
        from scripts.eval_summaries import evaluate_summaries
        metrics = evaluate_summaries(predictions, gold_data)
    elif config.task == "ner":
        from scripts.eval_ner import evaluate_ner
        metrics = evaluate_ner(predictions, gold_data)
    elif config.task == "transcription":
        from scripts.eval_transcription import evaluate_transcription
        metrics = evaluate_transcription(predictions, gold_data)
```

```python
    # Get prompt metadata for tracking (using prompt_store from RFC-017)
```

```python
    from podcast_scraper.prompt_store import get_prompt_metadata
```

    prompt_meta = {
        "system": (
            get_prompt_metadata(config.prompts["system"], config.prompts.get("params", {}))
            if config.prompts.get("system")
            else None
        ),
        "user": get_prompt_metadata(
            config.prompts["user"],
            config.prompts.get("params", {}),
        ),
    }

```text
    return {
        "experiment_id": config.id,
        "task": config.task,
        "backend": {
            "type": list(config.models.values())[0].get("type"),
            "model": list(config.models.values())[0].get("name"),
        },
        "prompts": prompt_meta,
        "global": metrics.get("global", {}),
        "episodes": metrics.get("episodes", {}),
        "metadata": {
            "model": config.models,
            "params": config.params,
            "gold_data_path": config.data["gold_data_path"],
        }
    }
```

```text
```text

results/
├── summarization_bart_led_v1/
│   ├── predictions.jsonl       # One line per episode
│   ├── metrics.json            # Aggregated metrics
│   └── metadata.json           # Experiment metadata (optional)
├── summarization_openai_gpt4_mini_v1/
│   ├── predictions.jsonl
│   ├── metrics.json
│   └── metadata.json
└── ner_openai_gpt4_mini_v1/
    ├── predictions.jsonl
    ├── metrics.json
    └── metadata.json

```json

{"episode_id": "ep01", "experiment_id": "summarization_openai_gpt4_mini_v1", "task": "summarization", "prediction": {"summary_long": "...", "summary_short": "..."}, "metadata": {"model": {"summarizer": {"type": "openai", "name": "gpt-4o-mini"}}, "params": {"max_length": 500}}}
{"episode_id": "ep02", "experiment_id": "summarization_openai_gpt4_mini_v1", "task": "summarization", "prediction": {"summary_long": "...", "summary_short": "..."}, "metadata": {...}}

```json

  "backend": {
    "type": "openai",
    "model": "gpt-4o-mini"
  },
  "prompts": {
    "system_path": "summarization/system_v1",
    "user_path": "summarization/long_v2_more_narrative",
    "system_sha256": "abc123def4567890...",
    "user_sha256": "789ghi012jkl3456..."
  },
  "global": {
    "rouge1_f": 0.322,
    "rouge1_p": 0.315,
    "rouge1_r": 0.330,
    "rouge2_f": 0.145,
    "rouge2_p": 0.140,
    "rouge2_r": 0.150,
    "rougeL_f": 0.142,
    "rougeL_p": 0.138,
    "rougeL_r": 0.145,
    "avg_compression": 39.7,
    "num_episodes": 10
  },
  "episodes": {
    "ep01": {
      "rouge1_f": 0.35,
      "rouge1_p": 0.34,
      "rouge1_r": 0.36,
      "rougeL_f": 0.16,
      "compression": 38.5
    },
    "ep02": {
      "rouge1_f": 0.31,
      "rouge1_p": 0.30,
      "rouge1_r": 0.32,
      "rougeL_f": 0.13,
      "compression": 42.1
    }
  },
  "metadata": {
    "model": {
      "summarizer": {
        "type": "openai",
        "name": "gpt-4o-mini"
      }
    },
    "params": {
      "max_length": 500,
      "temperature": 0.7
    },
    "gold_data_path": "data/eval/golden/summaries/"
  }
}

```text

# Generate predictions only

python scripts/run_experiment.py \
  --config experiments/summarization_openai_gpt4_mini_v1.yaml \
  --mode gen \
  --output-dir results

# Evaluate existing predictions

python scripts/run_experiment.py \
  --config experiments/summarization_openai_gpt4_mini_v1.yaml \
  --mode eval \
  --output-dir results

# Generate and evaluate (default)

python scripts/run_experiment.py \
  --config experiments/summarization_openai_gpt4_mini_v1.yaml \
  --output-dir results

# Run multiple experiments

python scripts/run_experiment.py \
  --config experiments/summarization_bart_led_local.yaml \
  --config experiments/summarization_openai_gpt4_mini_v1.yaml \
  --output-dir results

# Force regenerate predictions

python scripts/run_experiment.py \
  --config experiments/summarization_openai_gpt4_mini_v1.yaml \
  --force-regenerate

# Compare experiments (table format)

python scripts/compare_experiments.py \
  --experiments results/summarization_bart_led_v1 results/summarization_openai_gpt4_mini_v1 \
  --task summarization \
  --format table

# Compare experiments (markdown format)

python scripts/compare_experiments.py \
  --experiments results/summarization_* \
  --format markdown \
  --output comparison_report.md

# Compare with baseline (detect regressions)

python scripts/compare_experiments.py \
  --baseline results/summarization_bart_led_v1 \
  --experiments results/summarization_* \
  --format table

```text

## 5.1 Phase 1: Direct Integration (Minimal Changes)

**Current State:**

Existing `eval_summaries.py` expects:

- Input: Transcript files + config for model selection
- Output: JSON with ROUGE scores, compression ratios, etc.

**Integration Approach:**

The experiment runner writes `predictions.jsonl` that can be consumed by existing eval scripts with minimal changes:

```python

# scripts/eval_experiment.py (New wrapper script)

"""
Bridge between experiment pipeline and existing eval scripts.
"""

from pathlib import Path
import json
from typing import Dict, Any

from scripts.eval_summaries import (
    compute_rouge_scores,
    compute_compression_ratio,
    load_golden_summaries,
)

def load_predictions_from_jsonl(predictions_file: Path) -> Dict[str, str]:

```python
    """Load predictions from experiment JSONL output.
```

```text
    Args:
        predictions_file: Path to predictions.jsonl
```

```text
    Returns:
        Dict mapping episode_id to summary text
    """
    predictions = {}
    with predictions_file.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            predictions[record["episode_id"]] = record["summary"]
    return predictions
```

def evaluate_experiment_predictions(
    predictions_file: Path,
    gold_data_path: Path,
) -> Dict[str, Any]:

```text
    """
    Evaluate experiment predictions using existing eval logic.
```

    Reuses functions from eval_summaries.py without modifying them.
    """

```python
    # Load predictions from experiment output
```

```text
    predictions = load_predictions_from_jsonl(predictions_file)
```

```text
    # Load golden summaries (existing function)
```

```text
    gold_summaries = load_golden_summaries(gold_data_path)
```

```text
    # Compute metrics using existing functions
```

    episode_metrics = {}
    for ep_id, pred_summary in predictions.items():

```text
        gold_summary = gold_summaries.get(ep_id)
        if not gold_summary:
            continue
```

```text
        # Use existing ROUGE computation
```

```text
        rouge_scores = compute_rouge_scores(pred_summary, gold_summary)
```

```text
        # Use existing compression computation
```

```text
        compression = compute_compression_ratio(pred_summary, gold_summary)
```

        episode_metrics[ep_id] = {
            "rouge1_f": rouge_scores["rouge1"].fmeasure,
            "rouge1_p": rouge_scores["rouge1"].precision,
            "rouge1_r": rouge_scores["rouge1"].recall,
            "rouge2_f": rouge_scores["rouge2"].fmeasure,
            "rouge2_p": rouge_scores["rouge2"].precision,
            "rouge2_r": rouge_scores["rouge2"].recall,
            "rougeL_f": rouge_scores["rougeL"].fmeasure,
            "rougeL_p": rouge_scores["rougeL"].precision,
            "rougeL_r": rouge_scores["rougeL"].recall,
            "compression": compression,
        }

```text
    # Aggregate global metrics
```

```text
    if episode_metrics:
        global_metrics = {
            "rouge1_f": sum(m["rouge1_f"] for m in episode_metrics.values()) / len(episode_metrics),
            "rouge1_p": sum(m["rouge1_p"] for m in episode_metrics.values()) / len(episode_metrics),
            "rouge1_r": sum(m["rouge1_r"] for m in episode_metrics.values()) / len(episode_metrics),
            "rouge2_f": sum(m["rouge2_f"] for m in episode_metrics.values()) / len(episode_metrics),
            "rouge2_p": sum(m["rouge2_p"] for m in episode_metrics.values()) / len(episode_metrics),
            "rouge2_r": sum(m["rouge2_r"] for m in episode_metrics.values()) / len(episode_metrics),
            "rougeL_f": sum(m["rougeL_f"] for m in episode_metrics.values()) / len(episode_metrics),
            "rougeL_p": sum(m["rougeL_p"] for m in episode_metrics.values()) / len(episode_metrics),
            "rougeL_r": sum(m["rougeL_r"] for m in episode_metrics.values()) / len(episode_metrics),
            "avg_compression": sum(m["compression"] for m in episode_metrics.values()) / len(episode_metrics),
            "num_episodes": len(episode_metrics),
        }
    else:
        global_metrics = {}
```

```text
    return {
        "global": global_metrics,
        "episodes": episode_metrics,
    }
```

```text

# Step 1: Run experiment (generates predictions.jsonl)

python scripts/run_experiment.py experiments/summarization_openai_long_v1.yaml

# Step 2: Evaluate predictions (uses existing eval logic)

python scripts/eval_experiment.py \
  results/summarization_openai_long_v1/predictions.jsonl \
  data/eval/golden/summaries/ \
  --output results/summarization_openai_long_v1/metrics.json

```text

Refactor `eval_summaries.py` to expose reusable functions:

```python

# scripts/eval_summaries.py (refactored)

"""
Evaluate summarization quality using ROUGE metrics.

Can be used standalone (CLI) or imported by experiment pipeline.
"""

def evaluate_summaries(
    predictions: Dict[str, str],  # episode_id -> summary
    gold_data: Dict[str, str],   # episode_id -> gold summary
) -> Dict[str, Any]:
    """
    Evaluate summaries against golden dataset.

    Args:
        predictions: Dict mapping episode_id to predicted summary
        gold_data: Dict mapping episode_id to gold summary

```text
    Returns:
        Metrics dictionary with global and per-episode metrics
    """
    episode_metrics = {}
    for ep_id, pred_summary in predictions.items():
        gold_summary = gold_data.get(ep_id)
        if not gold_summary:
            continue
```

```text
        rouge_scores = compute_rouge_scores(pred_summary, gold_summary)
        compression = compute_compression_ratio(pred_summary, gold_summary)
```

        episode_metrics[ep_id] = {
            "rouge1_f": rouge_scores["rouge1"].fmeasure,
            "rouge1_p": rouge_scores["rouge1"].precision,
            "rouge1_r": rouge_scores["rouge1"].recall,
            "rouge2_f": rouge_scores["rouge2"].fmeasure,
            "rouge2_p": rouge_scores["rouge2"].precision,
            "rouge2_r": rouge_scores["rouge2"].recall,
            "rougeL_f": rouge_scores["rougeL"].fmeasure,
            "rougeL_p": rouge_scores["rougeL"].precision,
            "rougeL_r": rouge_scores["rougeL"].recall,
            "compression": compression,
        }

```text
    # Aggregate global metrics
```

```text
    if episode_metrics:
        global_metrics = {
            "rouge1_f": sum(m["rouge1_f"] for m in episode_metrics.values()) / len(episode_metrics),
            "rougeL_f": sum(m["rougeL_f"] for m in episode_metrics.values()) / len(episode_metrics),
            "avg_compression": sum(m["compression"] for m in episode_metrics.values()) / len(episode_metrics),
            "num_episodes": len(episode_metrics),
        }
    else:
        global_metrics = {}
```

```text
    return {
        "global": global_metrics,
        "episodes": episode_metrics,
    }
```

# CLI entry point (preserved for backward compatibility)

def main():

```text
    """CLI entry point for standalone evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_file", type=Path)
    parser.add_argument("gold_data_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
```

```text
    # Load predictions
```

```text
    predictions = load_predictions_from_jsonl(args.predictions_file)
```

```text
    # Load gold data
```

```text
    gold_data = load_golden_summaries(args.gold_data_path)
```

```text
    # Evaluate
```

```text
    metrics = evaluate_summaries(predictions, gold_data)
```

```text
    # Output
```

```text
    if args.output:
        args.output.write_text(json.dumps(metrics, indent=2))
    else:
        print(json.dumps(metrics, indent=2))
```

```python

- ✅ **Backward Compatible**: CLI still works as before
- ✅ **Reusable**: Can be imported by experiment pipeline
- ✅ **Testable**: Core logic separated from CLI parsing
- ✅ **Extensible**: Easy to add new metrics

## 5.3 Phase 3: Integrated Evaluation in Runner

**Evolution: Built-in Evaluation**

The experiment runner can optionally compute metrics directly:

```python

def run_experiment(
    cfg: ExperimentConfig,
    mode: Literal["gen", "eval", "gen+eval"] = "gen+eval",
    gold_data_path: Path | None = None,
) -> None:
    """
    Run experiment with optional integrated evaluation.

    Args:
        cfg: Experiment configuration
        mode: "gen" (generate only), "eval" (evaluate only), "gen+eval" (both)
        gold_data_path: Path to golden dataset (required if mode includes "eval")
    """
    results_dir = Path("results") / cfg.id
    predictions_path = results_dir / "predictions.jsonl"
    metrics_path = results_dir / "metrics.json"

    # Generation phase

```text
    if mode in ("gen", "gen+eval"):
```

```text
        # ... generate predictions ...
```

```text
        save_predictions(predictions, predictions_path)
```

```text
    # Evaluation phase
```

```text
    if mode in ("eval", "gen+eval"):
        if not gold_data_path:
            gold_data_path = Path(cfg.data.gold_data_path)
```

```text
        # Use refactored eval function
```

```python
        from scripts.eval_summaries import evaluate_summaries
```

```text
        predictions = load_predictions_from_jsonl(predictions_path)
        gold_data = load_golden_summaries(gold_data_path)
```

```text
        metrics = evaluate_summaries(predictions, gold_data)
```

```text
        # Add prompt metadata
```

```python
        from podcast_scraper.prompt_store import get_prompt_metadata
```

        metrics["prompts"] = {
            "system": (
                get_prompt_metadata(cfg.prompts.system, cfg.prompts.params)
                if cfg.prompts.system
                else None
            ),
            "user": get_prompt_metadata(cfg.prompts.user, cfg.prompts.params),
        }

```text
        # Save metrics
```

        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

```text
        # Print summary
```

```text
        print_metrics_summary(metrics)
```

```text

# Generate + evaluate in one command

python scripts/run_experiment.py \
  experiments/summarization_openai_long_v1.yaml \
  --mode gen+eval

# Or evaluate existing predictions

python scripts/run_experiment.py \
  experiments/summarization_openai_long_v1.yaml \
  --mode eval

```text

| **Phase 2** | Multiple tasks, provider pattern | Refactored eval functions | OpenAI + HF local |
| **Phase 3** | Integrated evaluation | Built-in metrics | All providers |
| **Phase 4** | Full feature set | Advanced metrics, comparison | All providers + custom |

**Migration Path:**

1. **Start Simple**: Minimal runner with OpenAI only
2. **Add Provider Support**: Integrate with provider system (RFC-016)
3. **Refactor Eval Scripts**: Extract reusable functions
4. **Add Integrated Eval**: Optional built-in evaluation
5. **Extend Features**: Add comparison, reporting, CI/CD integration

## 6. Golden Dataset Structure

**Current Structure:**

```text

data/eval/
├── episodes/
│   ├── ep01/
│   │   ├── transcript.txt
│   │   └── audio.mp3
│   ├── ep02/
│   │   ├── transcript.txt
│   │   └── audio.mp3
│   └── ...
├── golden/
│   ├── summaries/
│   │   ├── ep01.summary.txt
│   │   ├── ep02.summary.txt
│   │   └── ...
│   ├── ner/
│   │   ├── ep01.ner.json
│   │   ├── ep02.ner.json
│   │   └── ...
│   └── transcripts/
│       ├── ep01.transcript.txt
│       ├── ep02.transcript.txt
│       └── ...
└── MANUAL_EVAL_CHECKLIST.md

```json

  "hosts": ["Host Name 1", "Host Name 2"],
  "guests": ["Guest Name 1"],
  "all_speakers": ["Host Name 1", "Host Name 2", "Guest Name 1"]
}

```text

It was created using expensive OpenAI models and manually reviewed.

```yaml

### 7.1 Layer A: CI Smoke Tests

**Purpose**: Fast sanity check on every push/PR to catch breakages quickly.

**GitHub Actions Workflow:**

```yaml

# .github/workflows/ai-experiments-smoke.yml

name: AI Experiments Smoke Tests

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  smoke-tests:
    runs-on: ubuntu-latest
    steps:

      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4

```text
        with:
          python-version: '3.10'
```

      - name: Install dependencies
        run: |

          pip install -e ".[ml]"

```text
      - name: Run smoke test (single episode, baseline config)
        env:
```

          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python scripts/run_experiment.py \
            --config experiments/summarization_bart_led_v1.yaml \
            --episodes ep01 \
            --mode gen+eval

      - name: Assert quality thresholds
        run: |

          python scripts/check_metrics.py \
            --metrics results/summarization_bart_led_v1/metrics.json \
            --assert rougeL_f ">=" 0.10 \
            --assert no_errors \
            --assert no_nans \
            --assert no_missing_fields

```text

- Use single baseline config (HF baseline or OpenAI baseline)
- Assert quality thresholds (e.g., `rougeL_f >= threshold`)
- Assert no runtime errors, no NaNs, no missing fields
- Quick sanity check that pipeline wasn't broken (like unit tests)

## 7.2 Layer B: Full AI Eval Pipeline

**Purpose**: Comprehensive evaluation of all experiments, like integration/regression testing for models.

**GitHub Actions Workflow:**

```yaml

# .github/workflows/ai-experiments-full.yml

name: AI Experiments Full Pipeline

on:
  schedule:

    - cron: '0 0 * * 0'  # Weekly runs
  workflow_dispatch:  # Manual trigger

  push:
    paths:

      - 'experiments/**'
      - 'prompts/**'
    branches: [main]

jobs:
  full-eval:
    runs-on: ubuntu-latest
    steps:

      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4

```text
        with:
          python-version: '3.10'
```

      - name: Install dependencies
        run: |

          pip install -e ".[ml]"

      - name: Run all experiments
        env:

          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python scripts/run_all_experiments.py \
            --config-dir experiments/ \
            --output-dir .build/experiment-results

      - name: Generate summary report
        run: |

          python scripts/compare_experiments.py \
            --results-dir .build/experiment-results \
            --output .build/summary_report.md \
            --format markdown

      - name: Compare with baseline
        run: |

          python scripts/compare_experiments.py \
            --baseline .build/baseline-results \
            --current .build/experiment-results \
            --output .build/comparison.json \
            --detect-regressions

      - name: Upload results
        uses: actions/upload-artifact@v3

```text
        with:
          name: experiment-results
          path: .build/experiment-results
```

```text
      - name: Comment PR with results (if PR)
        if: github.event_name == 'pull_request'
```

        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('.build/summary_report.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## AI Experiment Results\n\n${report}`
            });

```text

- Produces metrics.json per experiment
- Generates combined summary_report.md with table of ROUGE/precision
- Like integration/regression testing for models
- Can compare current results with baseline
- Can detect regressions automatically

## 8. Comparison and Reporting

**Compare Experiments:**

```python

# scripts/compare_experiments.py

from typing import List, Optional, Dict, Any
from pathlib import Path
import json

def compare_experiments(
    experiment_dirs: List[Path],
    task: Optional[str] = None,
    output_file: Optional[Path] = None,
    format: Literal["table", "json", "markdown"] = "table",
) -> Dict[str, Any]:
    """Compare multiple experiments and generate report.

    Args:
        experiment_dirs: List of experiment result directories to compare
        task: Filter by task type (summarization, ner, transcription)
        output_file: Optional file to save comparison report
        format: Output format ("table", "json", "markdown")

```text
    Returns:
        Comparison report dictionary
    """
    comparisons = {}
```

```python
    # Load metrics from all experiments
```

```text
    for exp_dir in experiment_dirs:
        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue
```

```text
        metrics = json.loads(metrics_file.read_text())
```

```text
        # Filter by task if specified
```

```text
        if task and metrics.get("task") != task:
            continue
```

        comparisons[exp_dir.name] = {
            "experiment_id": metrics.get("experiment_id"),
            "task": metrics.get("task"),
            "metrics": metrics.get("global", {}),
            "metadata": metrics.get("metadata", {}),
        }

```text
    # Generate comparison report
```

    comparison_report = {
        "experiments": comparisons,
        "best_performing": find_best_experiment(comparisons),
        "regressions": find_regressions(comparisons),
    }

```text
    # Format output
```

```text
    if format == "table":
        output = format_comparison_table(comparisons, task)
        print(output)
    elif format == "markdown":
        output = format_comparison_markdown(comparisons, task)
        print(output)
    else:  # json
        output = json.dumps(comparison_report, indent=2)
        print(output)
```

```text
    if output_file:
        output_file.write_text(output)
```

```text
    return comparison_report
```

def format_comparison_table(
    comparisons: Dict[str, Dict[str, Any]],
    task: Optional[str] = None,
) -> str:

```text
    """Format comparison results as a simple table.
```

```text
    Example output:
        Run                         ROUGE-L   Avg Compression
        -----------------------------------------------------
        bart_led_v1                 0.120     43.4×
        gpt4_1_mini_v1              0.145     38.2×
        gpt4_1_mini_promptB         0.152     37.5×   <-- best
    """
```

```python
    # Determine task type from first experiment
```

```text
    if not task:
        first_exp = next(iter(comparisons.values()))
        task = first_exp.get("task", "summarization")
```

```text
    # Select metrics based on task type
```

```text
    if task == "summarization":
        metric_columns = [
            ("rougeL_f", "ROUGE-L"),
            ("avg_compression", "Avg Compression"),
        ]
    elif task == "ner":
        metric_columns = [
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
        ]
    elif task == "transcription":
        metric_columns = [
            ("wer", "Word Error Rate"),
            ("cer", "Char Error Rate"),
        ]
    else:
        metric_columns = []
```

```text
    # Build table
```

    lines = []

```text
    # Header
```

```text
    header = "Run".ljust(30)
    for _, label in metric_columns:
        header += label.rjust(15)
    lines.append(header)
    lines.append("-" * len(header))
```

```text
    # Find best performing experiment for each metric
```

    best_experiments = {}
    for metric_key, _ in metric_columns:
        best_value = None
        best_exp = None
        for exp_name, exp_data in comparisons.items():

```text
            value = exp_data["metrics"].get(metric_key)
            if value is not None:
```

```text
                # For error rates (WER, CER), lower is better
```

```text
                # For other metrics (ROUGE, F1), higher is better
```

```text
                is_error_rate = metric_key in ("wer", "cer")
                if best_value is None:
                    best_value = value
                    best_exp = exp_name
                elif is_error_rate:
                    if value < best_value:
                        best_value = value
                        best_exp = exp_name
                else:
                    if value > best_value:
                        best_value = value
                        best_exp = exp_name
        if best_exp:
            best_experiments[metric_key] = best_exp
```

```text
    # Overall best (based on primary metric)
```

    primary_metric = metric_columns[0][0] if metric_columns else None
    overall_best = best_experiments.get(primary_metric) if primary_metric else None

```text
    # Rows
```

```text
    for exp_name, exp_data in sorted(comparisons.items()):
        row = exp_name.ljust(30)
        for metric_key, _ in metric_columns:
            value = exp_data["metrics"].get(metric_key)
            if value is not None:
```

```text
                # Format value
```

```text
                if isinstance(value, float):
                    formatted = f"{value:.3f}"
                else:
                    formatted = str(value)
```

```text
                # Add compression multiplier format
```

```text
                if metric_key == "avg_compression":
                    formatted = f"{value:.1f}×"
```

```text
                row += formatted.rjust(15)
            else:
                row += "N/A".rjust(15)
```

```text
        # Mark best performing
```

```text
        if exp_name == overall_best:
            row += "   <-- best"
```

```text
        lines.append(row)
```

```text
    return "\n".join(lines)
```

def format_comparison_markdown(
    comparisons: Dict[str, Dict[str, Any]],
    task: Optional[str] = None,
) -> str:

```text
    """Format comparison results as Markdown table."""
```

```text
    # Similar to format_comparison_table but with Markdown syntax
```

```text
    # Implementation similar to above but with Markdown table formatting
```

    pass

def find_best_experiment(
    comparisons: Dict[str, Dict[str, Any]],
) -> Optional[str]:

```text
    """Find best performing experiment based on primary metrics."""
```

```text
    # Implementation to determine best experiment
```

```text
    # Can be based on ROUGE-L for summarization, F1 for NER, etc.
```

    pass

def find_regressions(
    comparisons: Dict[str, Dict[str, Any]],
    baseline: Optional[str] = None,
) -> List[Dict[str, Any]]:

```text
    """Find experiments that regress compared to baseline."""
```

```text
    # Implementation to detect regressions
```

    pass

```text

# Compare all summarization experiments

python scripts/compare_experiments.py \
  --experiments results/summarization_bart_led_v1 \
                results/summarization_openai_gpt4_mini_v1 \
                results/summarization_openai_gpt4_mini_promptB \
  --task summarization \
  --format table

# Compare all experiments (auto-detect task)

python scripts/compare_experiments.py \
  --experiments results/* \
  --format table

# Compare and save to file

python scripts/compare_experiments.py \
  --experiments results/summarization_* \
  --format markdown \
  --output comparison_report.md

# Compare with baseline (detect regressions)

python scripts/compare_experiments.py \
  --baseline results/summarization_bart_led_v1 \
  --experiments results/summarization_* \
  --format table

```text

Run                         ROUGE-L   Avg Compression
-----------------------------------------------------
summarization_bart_led_v1   0.120     43.4×
summarization_openai_gpt4_mini_v1   0.145     38.2×
summarization_openai_gpt4_mini_promptB   0.152     37.5×   <-- best

```text

| summarization_openai_gpt4_mini_v1 | 0.145 | 38.2× |
| summarization_openai_gpt4_mini_promptB | **0.152** | 37.5× ⭐ |

```json

{
  "experiments": {
    "summarization_bart_led_v1": {
      "rougeL_f": 0.120,
      "avg_compression": 43.4
    },
    "summarization_openai_gpt4_mini_v1": {
      "rougeL_f": 0.145,
      "avg_compression": 38.2
    },
    "summarization_openai_gpt4_mini_promptB": {
      "rougeL_f": 0.152,
      "avg_compression": 37.5
    }
  },
  "best_performing": "summarization_openai_gpt4_mini_promptB",
  "regressions": []
}

```text

- ROUGE-1, ROUGE-2 (optional)

**NER:**

- Precision
- Recall
- F1 Score

**Transcription:**

- Word Error Rate (WER)
- Character Error Rate (CER)

## Benefits

**Note**: For product benefits and value proposition, see `docs/prd/PRD-007-ai-experiment-pipeline.md`. This section focuses on technical benefits.

1. **Code Reuse**: Experiment pipeline reuses production providers, no duplication
2. **Separation of Concerns**: Generation separate from evaluation enables recomputation
3. **Modularity**: Clean provider interfaces enable independent testing
4. **Extensibility**: Easy to add new providers or experiment types
5. **Maintainability**: Single source of truth for provider implementations

## Result Summary and Tracking

### Excel-Based Result Aggregation

**Problem**: With multiple experiment types (NER, summarization, transcription) and many experiments per type, tracking and comparing results becomes challenging. Individual JSON files are hard to compare across experiments and task types.

**Solution**: Maintain a single Excel workbook (`results/experiment_results.xlsx`) with one tab per evaluation type. This provides:

- ✅ **Centralized Tracking**: All experiment results in one place
- ✅ **Easy Comparison**: Side-by-side comparison across experiments
- ✅ **Visual Analysis**: Excel charts and pivot tables for trend analysis
- ✅ **Version Control Friendly**: Can track changes over time
- ✅ **Human Readable**: Easy to review and share with stakeholders

**Excel Structure:**

```text

results/
└── experiment_results.xlsx
    ├── Tab: "Summarization"
    │   Columns: Experiment ID | ROUGE-L | ROUGE-1 | ROUGE-2 | Avg Compression | Date | Notes
    │   Rows: One per experiment
    │
    ├── Tab: "NER"
    │   Columns: Experiment ID | Precision | Recall | F1 | Date | Notes
    │   Rows: One per experiment
    │
    └── Tab: "Transcription"
        Columns: Experiment ID | WER | CER | Date | Notes
        Rows: One per experiment

```text

| summarization_bart_led_v1 | 0.120 | 0.315 | 0.145 | 43.4× | 2024-01-15 | Baseline local model |
| summarization_openai_gpt4_mini_v1 | 0.145 | 0.330 | 0.150 | 38.2× | 2024-01-16 | OpenAI GPT-4o-mini |
| summarization_openai_gpt4_mini_promptB | 0.152 | 0.335 | 0.155 | 37.5× | 2024-01-17 | Improved prompt ⭐ |

**NER Tab:**

| Experiment ID | Precision | Recall | F1 | Date | Notes |
| -------------- | --------- | ------ | -- | ---- | ----- |
| ner_openai_gpt4_mini_v1 | 0.85 | 0.82 | 0.835 | 2024-01-15 | OpenAI GPT-4o-mini |
| ner_spacy_en_core_web_sm | 0.78 | 0.75 | 0.765 | 2024-01-15 | Baseline spaCy |

**Transcription Tab:**

| Experiment ID | WER | CER | Date | Notes |
| -------------- | --- | --- | ---- | ----- |
| transcription_openai_whisper_v1 | 0.12 | 0.08 | 2024-01-15 | OpenAI Whisper API |
| transcription_whisper_large_v3 | 0.15 | 0.10 | 2024-01-15 | Local Whisper large-v3 |

**Excel Update Workflow:**

```python

# scripts/update_experiment_results.py

import pandas as pd
from pathlib import Path
from typing import Dict, Any

def update_experiment_results(
    experiment_id: str,
    task: str,
    metrics: Dict[str, Any],
    excel_file: Path = Path("results/experiment_results.xlsx"),
) -> None:
    """Update Excel workbook with experiment results.

    Args:
        experiment_id: Unique experiment identifier
        task: Task type ("summarization", "ner", "transcription")
        metrics: Metrics dictionary from metrics.json
        excel_file: Path to Excel workbook
    """

```text
    # Load or create Excel workbook
```

```text
    if excel_file.exists():
        excel = pd.ExcelFile(excel_file)
        sheets = {sheet: pd.read_excel(excel, sheet_name=sheet) for sheet in excel.sheet_names}
    else:
        sheets = {
            "Summarization": pd.DataFrame(columns=["Experiment ID", "ROUGE-L", "ROUGE-1", "ROUGE-2", "Avg Compression", "Date", "Notes"]),
            "NER": pd.DataFrame(columns=["Experiment ID", "Precision", "Recall", "F1", "Date", "Notes"]),
            "Transcription": pd.DataFrame(columns=["Experiment ID", "WER", "CER", "Date", "Notes"]),
        }
```

```text
    # Select appropriate sheet
```

```text
    sheet_name = task.capitalize()
    df = sheets[sheet_name]
```

```text
    # Extract metrics based on task type
```

```text
    if task == "summarization":
        row = {
            "Experiment ID": experiment_id,
            "ROUGE-L": metrics["global"].get("rougeL_f", None),
            "ROUGE-1": metrics["global"].get("rouge1_f", None),
            "ROUGE-2": metrics["global"].get("rouge2_f", None),
            "Avg Compression": metrics["global"].get("avg_compression", None),
            "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "Notes": "",
        }
    elif task == "ner":
        row = {
            "Experiment ID": experiment_id,
            "Precision": metrics["global"].get("precision", None),
            "Recall": metrics["global"].get("recall", None),
            "F1": metrics["global"].get("f1", None),
            "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "Notes": "",
        }
    elif task == "transcription":
        row = {
            "Experiment ID": experiment_id,
            "WER": metrics["global"].get("wer", None),
            "CER": metrics["global"].get("cer", None),
            "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "Notes": "",
        }
```

```text
    # Update or append row
```

```text
    if experiment_id in df["Experiment ID"].values:
```

```text
        # Update existing row
```

        idx = df[df["Experiment ID"] == experiment_id].index[0]
        for col, val in row.items():
            df.at[idx, col] = val
    else:

```text
        # Append new row
```

```text
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
```

    sheets[sheet_name] = df

```text
    # Write back to Excel
```

```text
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
```

```text

# In scripts/run_experiment.py, after evaluation phase:

if mode in ("eval", "gen+eval"):
    metrics = evaluate_predictions(config, predictions_file)
    save_metrics(metrics, metrics_file)

    # Update Excel summary

    from .update_experiment_results import update_experiment_results
    update_experiment_results(
        experiment_id=config.id,
        task=config.task,
        metrics=metrics,
    )

```text

# Run experiment and automatically update Excel

python scripts/run_experiment.py \
  --config experiments/summarization_openai_gpt4_mini_v1.yaml \
  --update-excel

# Manually update Excel from existing results

python scripts/update_experiment_results.py \
  --experiment-id summarization_openai_gpt4_mini_v1 \
  --task summarization \
  --metrics-file results/summarization_openai_gpt4_mini_v1/metrics.json

```text

3. **Trend Analysis**: Track improvements over time
4. **Visualization**: Excel charts show performance trends
5. **Collaboration**: Easy to share and review with team
6. **Version Control**: Excel files can be tracked in git (with care)
7. **Export**: Easy to export to CSV or other formats for analysis

**Best Practices:**

- **One Row Per Experiment**: Each experiment gets one row per task type
- **Consistent Naming**: Use consistent experiment ID naming conventions
- **Date Tracking**: Always include date to track when experiment was run
- **Notes Column**: Use notes to document model changes, prompt improvements, etc.
- **Regular Updates**: Update Excel after each experiment run
- **Backup**: Keep Excel file in version control or backup regularly

## Success Criteria

**Note**: For product success criteria, see `docs/prd/PRD-007-ai-experiment-pipeline.md`. This section focuses on technical success criteria.

- ✅ Experiment pipeline reuses production providers (no code duplication)
- ✅ Production workflow continues to work unchanged
- ✅ Provider interfaces are well-defined and testable
- ✅ Evaluation scripts work with both experiment and production outputs
- ✅ Clear separation between experiment and production pipelines
- ✅ Easy to add new providers or experiment types

## Implementation Plan

The implementation follows a phased approach that wraps existing pieces (gold data, HF baseline, eval scripts) into a repeatable "AI experiment pipeline" that sits next to your normal build/CI. This aligns with PRD-007's phased approach and the modularization plan (RFC-016) and prompt management (RFC-017).

**Key Principle**: Normalize what you have now, then build a generic runner that wraps existing pieces. Once that's in place, adding new providers is just "add config + small backend class".

### Phase 1: Normalize Existing Structure

**Goal**: Establish baseline structure and normalize existing data

**Deliverables:**

1. **Normalize data structure**:
   - Move gold data under `data/eval/episodes/*`
   - Ensure consistent episode structure
   - Document golden dataset format

2. **Establish baseline**:
   - Keep existing baseline as `results/summarization_bart_led_v1/metrics.json`
   - Document baseline experiment config
   - Create baseline config file

3. **Document current state**:
   - Document what we already have (gold data, HF baseline, eval scripts)
   - Identify gaps and what needs to be wrapped

**Timeline**: 1-2 days

**Success Criteria:**

- ✅ Gold data normalized under `data/eval/episodes/*`
- ✅ Baseline experiment documented and config file created
- ✅ Current state documented

### Phase 2: Generic Runner

**Goal**: Build a generic runner that wraps existing pieces

**Deliverables:**

1. **Create `experiment_config.py`**:
   - Pydantic models for experiment configs (using RFC-017 patterns)
   - YAML loader
   - Data discovery helpers

2. **Create generic `run_experiment.py`**:
   - Takes config path as input
   - Loads episodes listed in config
   - Calls appropriate backend (local HF or OpenAI API)
   - Writes predictions + metrics separately
   - Support episode filtering (e.g., `--episodes ep01`)

3. **Create `eval_experiment.py` wrapper**:
   - Bridges experiment output to existing eval scripts
   - Reuses `eval_summaries.py` logic
   - No changes to existing eval scripts required

4. **Test with real data**:
   - Run experiments on golden dataset
   - Verify predictions.jsonl format
   - Verify metrics computation

**Timeline**: 2-3 days

**Success Criteria:**

- ✅ Generic runner takes config and produces predictions + metrics
- ✅ Can run experiments with existing HF baseline
- ✅ Generates predictions.jsonl
- ✅ Can evaluate predictions using existing eval logic
- ✅ Prompt metadata tracked in results

### Phase 3: CI Smoke Tests (Layer A)

**Goal**: Add fast smoke tests that run on every push/PR

**Deliverables:**

1. **Add smoke test workflow**:
   - GitHub Actions workflow for smoke tests
   - Runs on every push/PR
   - Uses tiny subset (e.g., `ep01` only)
   - Uses single baseline config

2. **Add quality assertions**:
   - Script to check metrics thresholds
   - Assert no errors, no NaNs, no missing fields
   - Fail fast if pipeline broken

3. **Integrate with CI**:
   - Add workflow to `.github/workflows/`
   - Configure to run on push/PR
   - Quick sanity check (like unit tests)

**Timeline**: 1-2 days

**Success Criteria:**

- ✅ Smoke tests run on every push/PR
- ✅ Asserts quality thresholds
- ✅ Catches breakages quickly

### Phase 4: Full Eval Pipeline (Layer B)

**Goal**: Comprehensive evaluation pipeline for all experiments

**Deliverables:**

1. **Add full eval script**:
   - Script that loops over all YAMLs in `experiments/*.yaml`
   - Runs them, writes results
   - Prints summary table

2. **Add full eval workflow**:
   - GitHub Actions workflow for full evaluation
   - Runs nightly or on-demand
   - Runs all experiment configs
   - Generates summary report

3. **Add comparison tooling**:
   - Compare multiple experiment results
   - Generate comparison reports (markdown, JSON)
   - Detect regressions

**Timeline**: 2-3 days

**Success Criteria:**

- ✅ Full eval script runs all experiments
- ✅ Generates summary report with metrics table
- ✅ Can compare experiments and detect regressions
- ✅ Works in CI/CD (nightly/on-demand)

### Phase 5: Comparison Tooling

**Goal**: Build comparison tool that creates Excel with all experiments and key metrics

**Deliverables:**

1. **Build comparison tool**:
   - Reads all experiment results
   - Creates Excel workbook with all experiments
   - One tab per task type
   - Key metrics columns (ROUGE, precision, F1, etc.)

2. **Enable data-driven decisions**:
   - Answer "which model + prompt is best?" becomes a data question
   - Visual comparison tables
   - Trend analysis

**Timeline**: 2-3 days

**Success Criteria:**

- ✅ Comparison tool creates Excel workbook
- ✅ All experiments and key metrics visible
- ✅ Easy to compare and make data-driven decisions

### Phase 6: Extend to New Providers

**Goal**: Add support for new providers (OpenAI, etc.)

**Deliverables:**

1. **Integrate with provider system**:
   - Use `SummarizationProviderFactory` (RFC-016)
   - Support both OpenAI and HF local backends
   - Use provider protocol interfaces

2. **Add support for multiple tasks**:
   - NER/speaker detection experiments
   - Transcription experiments

3. **Refactor eval scripts**:
   - Extract core evaluation functions
   - Make them reusable by experiment pipeline
   - Preserve CLI backward compatibility

**Timeline**: 3-4 days

**Success Criteria:**

- ✅ Runner uses provider system
- ✅ Supports multiple backends (OpenAI, HF local)
- ✅ Supports multiple tasks (summarization, NER)
- ✅ Eval scripts refactored and reusable

**Note**: Once structure is in place (Phases 1-5), adding new providers is just "add config + small backend class" (if needed).

## Evolution Path Summary

The experiment runner evolves through four phases:

| Phase | Capabilities | Eval Integration | Backends | Timeline |
| ----- | ------------ | ---------------- | -------- | -------- |
| **Phase 1** | OpenAI summarization only | Separate eval script | OpenAI | 2-3 days |
| **Phase 2** | Multiple tasks, provider pattern | Refactored eval functions | OpenAI + HF local | 3-4 days |
| **Phase 3** | Integrated evaluation | Built-in metrics | All providers | 2-3 days |
| **Phase 4** | Full feature set | Advanced metrics, comparison | All providers + custom | 3-5 days |

**Key Principles:**

1. **Start Simple**: Begin with minimal MVP (OpenAI only)
2. **Incremental Evolution**: Add features incrementally
3. **Reuse Existing**: Leverage existing eval scripts, don't duplicate
4. **Provider Integration**: Evolve to use provider system (RFC-016)
5. **Prompt Management**: Use `prompt_store` (RFC-017) from the start

**Migration Strategy:**

- **Phase 1**: Get basic runner working, prove concept
- **Phase 2**: Integrate with provider system as it's implemented
- **Phase 3**: Add integrated evaluation once eval scripts are refactored
- **Phase 4**: Add advanced features as needed

## Related Documents

- `docs/rfc/RFC-012-episode-summarization.md`: Summarization design
- `docs/prd/PRD-007-ai-experiment-pipeline.md`: Product requirements, use cases, and functional specifications
- `docs/rfc/RFC-013-openai-provider-implementation.md`: OpenAI provider design
- `docs/rfc/RFC-016-modularization-for-ai-experiments.md`: Provider system architecture
- `docs/rfc/RFC-017-prompt-management.md`: Prompt management and loading implementation
- `docs/EVALUATION_STRATEGY.md`: Evaluation strategy (to be created)
- `scripts/eval_summaries.py`: Existing summarization evaluation script
- `scripts/eval_cleaning.py`: Existing cleaning evaluation script

````
