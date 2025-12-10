# RFC-015: AI Experiment Pipeline

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, researchers tuning AI models/prompts, developers evaluating model performance
- **Related PRDs**: `docs/prd/PRD-006-openai-provider-integration.md`
- **Related RFCs**: `docs/rfc/RFC-012-episode-summarization.md`, `docs/rfc/RFC-013-openai-provider-implementation.md`
- **Related Issues**: (to be created)

## Abstract

Design and implement a repeatable AI experiment pipeline that enables rapid iteration on model selection, prompt engineering, and parameter tuning without requiring code changes. This pipeline separates generation (model inference) from evaluation (metrics computation), allowing for efficient experimentation, comparison, and integration with CI/CD workflows.

## Problem Statement

Currently, evaluating different AI models, prompts, and parameters requires:

- **Code Changes**: Modifying Python code to test different configurations
- **Slow Iteration**: Code changes require redeployment and full pipeline runs
- **Tight Coupling**: Generation and evaluation are intertwined, making it hard to recompute metrics
- **Manual Comparison**: Comparing results across experiments is manual and error-prone
- **No Reproducibility**: Experiments are not easily reproducible or shareable
- **CI/CD Integration**: Hard to integrate AI experiments into automated testing workflows

**Use Cases:**

1. **Prompt Engineering**: Tune OpenAI prompts without code changes
2. **Model Comparison**: Compare different models (local vs OpenAI, different OpenAI models)
3. **Parameter Tuning**: Test different chunk sizes, overlap, max lengths
4. **Regression Testing**: Ensure model changes don't degrade performance
5. **A/B Testing**: Compare multiple configurations side-by-side

## Goals

1. **Configuration-Driven**: Model + prompt + params defined in YAML config files (like GitHub Actions workflows)
2. **No Code Changes**: Change model/prompt/params by editing config files only
3. **Reuse Existing Eval Scripts**: Leverage `eval_summaries.py`, `eval_cleaning.py`, `eval_ner.py` (planned)
4. **Separation of Concerns**: Generation (inference) separate from evaluation (metrics)
5. **Reproducibility**: Experiments are fully reproducible from config files
6. **CI/CD Integration**: Can run experiments in CI/CD pipelines
7. **Multiple Task Types**: Support NER, transcription, and summarization experiments
8. **Golden Dataset**: Use existing `data/eval/` folder as ground truth test set

## Design & Implementation

### 1. Experiment Configuration Format

Experiments are defined in YAML configuration files, stored in `experiments/` directory.

**Example: Local Summarization Experiment**

```yaml
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
```

**Example: OpenAI Summarization Experiment**

```yaml
# experiments/summarization_openai_gpt4_mini_v1.yaml
id: "summarization_openai_gpt4_mini_v1"
task: "summarization"
description: "OpenAI GPT-4o-mini summarization with custom prompts"
```

**Example: Golden Dataset Creation Experiment**

```yaml
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
  prompts_file: "prompts/openai_prompts_v2.yaml"
  # Or inline prompts:
  # system_prompt: "You are an expert at creating concise summaries..."
  # user_prompt_template: "Create a summary of: {transcript}"

data:
  episodes_glob: "data/eval/episodes/ep*/transcript.txt"
  gold_data_path: "data/eval/golden/summaries/"
```

**Example: NER/Speaker Detection Experiment**

```yaml
# experiments/ner_openai_gpt4_mini_v1.yaml
id: "ner_openai_gpt4_mini_v1"
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
  prompts_file: "prompts/openai_prompts_v2.yaml"

data:
  episodes_glob: "data/eval/episodes/ep*/transcript.txt"
  gold_data_path: "data/eval/golden/ner/"
```

**Example: Transcription Experiment**

```yaml
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
```

**Configuration Schema:**

```python
from typing import Literal, Dict, Any, Optional
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    type: Literal["hf_local", "openai", "anthropic"]
    name: str

class ExperimentConfig(BaseModel):
    id: str  # Unique experiment identifier
    task: Literal["summarization", "ner", "transcription"]
    description: Optional[str] = None
    
    models: Dict[str, ModelConfig]  # e.g., {"map": ModelConfig(...), "reduce": ModelConfig(...)}
    params: Dict[str, Any] = Field(default_factory=dict)
    prompts: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    data: Dict[str, str]  # episodes_glob, gold_data_path
```

### 2. Generic Runner Architecture

**Main Entry Point:**

```python
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
    # Load experiment config
    config = load_experiment_config(config_file)
    
    # Create output directory
    experiment_output_dir = Path(output_dir) / config.id
    experiment_output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_file = experiment_output_dir / "predictions.jsonl"
    metrics_file = experiment_output_dir / "metrics.json"
    
    # Generation phase
    if mode in ("gen", "gen+eval"):
        if predictions_file.exists() and not force_regenerate:
            print(f"Predictions already exist: {predictions_file}")
            print("Use --force-regenerate to regenerate")
        else:
            print(f"Generating predictions for experiment: {config.id}")
            predictions = generate_predictions(config)
            save_predictions(predictions, predictions_file)
    
    # Evaluation phase
    if mode in ("eval", "gen+eval"):
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        
        print(f"Evaluating predictions for experiment: {config.id}")
        metrics = evaluate_predictions(config, predictions_file)
        save_metrics(metrics, metrics_file)
        
        # Print summary
        print_metrics_summary(metrics)

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
    """Run multiple AI experiments.
    
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
    # Collect all config files
    all_config_files = []
    
    # Add explicit config files
    all_config_files.extend(config_files)
    
    # Add config files from directory/glob
    if config_dir:
        config_path = Path(config_dir)
        if config_path.is_dir():
            # Find all YAML files in directory
            all_config_files.extend(config_path.glob("*.yaml"))
            all_config_files.extend(config_path.glob("*.yml"))
        else:
            # Treat as glob pattern
            from glob import glob
            all_config_files.extend(glob(str(config_dir)))
    
    # Filter by task type if specified
    if task_filter:
        filtered_configs = []
        for config_file in all_config_files:
            config = load_experiment_config(str(config_file))
            if config.task == task_filter:
                filtered_configs.append(config_file)
        all_config_files = filtered_configs
    
    # Exclude golden experiments unless explicitly included
    if not include_golden:
        filtered_configs = []
        for config_file in all_config_files:
            # Check filename for golden naming convention
            config_file_str = str(config_file)
            is_golden = (
                "_golden" in config_file_str.lower() or
                "_gold" in config_file_str.lower() or
                config_file_str.endswith("_golden.yaml") or
                config_file_str.endswith("_golden.yml") or
                config_file_str.endswith("_gold.yaml") or
                config_file_str.endswith("_gold.yml")
            )
            
            # Also check config file content if is_golden flag is set
            if not is_golden:
                try:
                    config = load_experiment_config(config_file_str)
                    is_golden = getattr(config, "is_golden", False)
                except Exception:
                    pass  # If we can't load config, rely on filename
            
            if not is_golden:
                filtered_configs.append(config_file)
            else:
                print(f"Skipping golden experiment: {config_file}")
        
        all_config_files = filtered_configs
    
    if not all_config_files:
        print("No experiment configs found (excluding golden experiments)")
        print("Use --include-golden to include golden experiments")
        return
    
    print(f"Running {len(all_config_files)} experiment(s)...")
    
    # Run experiments
    if parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
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
            
            for future in as_completed(futures):
                config_file = futures[future]
                try:
                    future.result()
                    print(f"✓ Completed: {config_file}")
                except Exception as e:
                    print(f"✗ Failed: {config_file} - {e}")
    else:
        # Sequential execution
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
                    # In sequential mode, optionally continue or stop
                    continue
    
    print(f"\n{'='*60}")
    print(f"Completed {len(all_config_files)} experiment(s)")
    print(f"{'='*60}")
```

**Generation Phase:**

```python
def generate_predictions(config: ExperimentConfig) -> List[Dict[str, Any]]:
    """Generate predictions for all episodes in the experiment.
    
    Returns:
        List of prediction dictionaries, one per episode
    """
    # Load episodes
    episodes = load_episodes(config.data["episodes_glob"])
    
    # Initialize model backend based on config
    backend = create_backend(config)
    
    predictions = []
    for episode in episodes:
        episode_id = episode["id"]
        print(f"Processing episode: {episode_id}")
        
        # Apply preprocessing (provider-agnostic, RFC-012)
        if config.task == "summarization":
            # Preprocess transcript before summarization
            cleaned_transcript = clean_transcript(
                episode["transcript"],
                remove_timestamps=True,
                normalize_speakers=True,
                collapse_blank_lines=True,
            )
            cleaned_transcript = remove_sponsor_blocks(cleaned_transcript)
        
        # Generate prediction based on task type
        if config.task == "summarization":
            prediction = backend.summarize(
                transcript=cleaned_transcript,  # Use preprocessed transcript
                episode_title=episode.get("title"),
                episode_description=episode.get("description"),
                params=config.params,
            )
        elif config.task == "ner":
            prediction = backend.detect_speakers(
                transcript=episode["transcript"],
                feed_title=episode.get("feed_title"),
                episode_title=episode.get("title"),
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
            }
        })
    
    return predictions
```

**Backend Factory:**

**Note**: Backends use the provider pattern (see RFC-016). The experiment pipeline uses "backend" terminology for clarity, but these are implemented as providers internally.

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

**Evaluation Phase:**

```python
def evaluate_predictions(
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
    if config.task == "summarization":
        from scripts.eval_summaries import evaluate_summaries
        metrics = evaluate_summaries(predictions, gold_data)
    elif config.task == "ner":
        from scripts.eval_ner import evaluate_ner
        metrics = evaluate_ner(predictions, gold_data)
    elif config.task == "transcription":
        from scripts.eval_transcription import evaluate_transcription
        metrics = evaluate_transcription(predictions, gold_data)
    
    return {
        "experiment_id": config.id,
        "task": config.task,
        "global": metrics.get("global", {}),
        "episodes": metrics.get("episodes", {}),
        "metadata": {
            "model": config.models,
            "params": config.params,
            "gold_data_path": config.data["gold_data_path"],
        }
    }
```

### 3. Output Structure

**Directory Layout:**

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
```

**Predictions Format (JSONL):**

```json
{"episode_id": "ep01", "experiment_id": "summarization_openai_gpt4_mini_v1", "task": "summarization", "prediction": {"summary_long": "...", "summary_short": "..."}, "metadata": {"model": {"summarizer": {"type": "openai", "name": "gpt-4o-mini"}}, "params": {"max_length": 500}}}
{"episode_id": "ep02", "experiment_id": "summarization_openai_gpt4_mini_v1", "task": "summarization", "prediction": {"summary_long": "...", "summary_short": "..."}, "metadata": {...}}
```

**Metrics Format (JSON):**

```json
{
  "experiment_id": "summarization_openai_gpt4_mini_v1",
  "task": "summarization",
  "global": {
    "rouge1_f": 0.322,
    "rouge1_p": 0.315,
    "rouge1_r": 0.330,
    "rouge2_f": 0.145,
    "rouge2_p": 0.140,
    "rouge2_r": 0.150,
    "rougeL_f": 0.135,
    "rougeL_p": 0.132,
    "rougeL_r": 0.138,
    "avg_compression": 40.1,
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
```

### 4. CLI Interface

```bash
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
```

### 5. Integration with Existing Eval Scripts

**Refactoring Strategy:**

1. **Extract Core Logic**: Extract metric computation logic from `eval_summaries.py` into reusable functions
2. **Standardize Input/Output**: Ensure eval functions accept predictions + gold data, return metrics dict
3. **Create Wrapper**: `run_experiment.py` calls existing eval functions with standardized format

**Example Refactoring:**

```python
# scripts/eval_summaries.py (refactored)
def evaluate_summaries(
    predictions: List[Dict[str, Any]],
    gold_data: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """Evaluate summaries against golden dataset.
    
    Args:
        predictions: List of prediction dicts with episode_id and prediction
        gold_data: Dict mapping episode_id to gold summary dict
        
    Returns:
        Metrics dictionary with global and per-episode metrics
    """
    # Existing evaluation logic
    # ...
    return {
        "global": {
            "rouge1_f": ...,
            "rougeL_f": ...,
            "avg_compression": ...,
        },
        "episodes": {
            "ep01": {...},
            "ep02": {...},
        }
    }

# CLI entry point (preserved for backward compatibility)
def main():
    # Parse CLI args
    # Load predictions and gold data
    # Call evaluate_summaries()
    # Print results
```

### 6. Golden Dataset Structure

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
```

**Golden Data Format:**

```json
// data/eval/golden/ner/ep01.ner.json
{
  "episode_id": "ep01",
  "hosts": ["Host Name 1", "Host Name 2"],
  "guests": ["Guest Name 1"],
  "all_speakers": ["Host Name 1", "Host Name 2", "Guest Name 1"]
}
```

```text
# data/eval/golden/summaries/ep01.summary.txt
This is the golden summary for episode 01.
It was created using expensive OpenAI models and manually reviewed.
```

### 7. CI/CD Integration

**GitHub Actions Workflow:**

```yaml
# .github/workflows/ai-experiments.yml
name: AI Experiments

on:
  push:
    paths:
      - 'experiments/**'
      - 'prompts/**'
  pull_request:
    paths:
      - 'experiments/**'
      - 'prompts/**'
  schedule:
    - cron: '0 0 * * 0'  # Weekly runs

jobs:
  run-experiments:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e ".[ml]"
      
      - name: Run experiments
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python scripts/run_experiment.py \
            --config experiments/summarization_openai_gpt4_mini_v1.yaml \
            --output-dir .build/experiment-results
      
      - name: Compare with baseline
        run: |
          python scripts/compare_experiments.py \
            --baseline .build/baseline-results \
            --current .build/experiment-results \
            --output .build/comparison.json
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: experiment-results
          path: .build/experiment-results
```

### 8. Comparison and Reporting

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
    
    Returns:
        Comparison report dictionary
    """
    comparisons = {}
    
    # Load metrics from all experiments
    for exp_dir in experiment_dirs:
        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        metrics = json.loads(metrics_file.read_text())
        
        # Filter by task if specified
        if task and metrics.get("task") != task:
            continue
        
        comparisons[exp_dir.name] = {
            "experiment_id": metrics.get("experiment_id"),
            "task": metrics.get("task"),
            "metrics": metrics.get("global", {}),
            "metadata": metrics.get("metadata", {}),
        }
    
    # Generate comparison report
    comparison_report = {
        "experiments": comparisons,
        "best_performing": find_best_experiment(comparisons),
        "regressions": find_regressions(comparisons),
    }
    
    # Format output
    if format == "table":
        output = format_comparison_table(comparisons, task)
        print(output)
    elif format == "markdown":
        output = format_comparison_markdown(comparisons, task)
        print(output)
    else:  # json
        output = json.dumps(comparison_report, indent=2)
        print(output)
    
    if output_file:
        output_file.write_text(output)
    
    return comparison_report

def format_comparison_table(
    comparisons: Dict[str, Dict[str, Any]],
    task: Optional[str] = None,
) -> str:
    """Format comparison results as a simple table.
    
    Example output:
        Run                         ROUGE-L   Avg Compression
        -----------------------------------------------------
        bart_led_v1                 0.120     43.4×
        gpt4_1_mini_v1              0.145     38.2×
        gpt4_1_mini_promptB         0.152     37.5×   <-- best
    """
    # Determine task type from first experiment
    if not task:
        first_exp = next(iter(comparisons.values()))
        task = first_exp.get("task", "summarization")
    
    # Select metrics based on task type
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
    
    # Build table
    lines = []
    
    # Header
    header = "Run".ljust(30)
    for _, label in metric_columns:
        header += label.rjust(15)
    lines.append(header)
    lines.append("-" * len(header))
    
    # Find best performing experiment for each metric
    best_experiments = {}
    for metric_key, _ in metric_columns:
        best_value = None
        best_exp = None
        for exp_name, exp_data in comparisons.items():
            value = exp_data["metrics"].get(metric_key)
            if value is not None:
                # For error rates (WER, CER), lower is better
                # For other metrics (ROUGE, F1), higher is better
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
    
    # Overall best (based on primary metric)
    primary_metric = metric_columns[0][0] if metric_columns else None
    overall_best = best_experiments.get(primary_metric) if primary_metric else None
    
    # Rows
    for exp_name, exp_data in sorted(comparisons.items()):
        row = exp_name.ljust(30)
        for metric_key, _ in metric_columns:
            value = exp_data["metrics"].get(metric_key)
            if value is not None:
                # Format value
                if isinstance(value, float):
                    formatted = f"{value:.3f}"
                else:
                    formatted = str(value)
                
                # Add compression multiplier format
                if metric_key == "avg_compression":
                    formatted = f"{value:.1f}×"
                
                row += formatted.rjust(15)
            else:
                row += "N/A".rjust(15)
        
        # Mark best performing
        if exp_name == overall_best:
            row += "   <-- best"
        
        lines.append(row)
    
    return "\n".join(lines)

def format_comparison_markdown(
    comparisons: Dict[str, Dict[str, Any]],
    task: Optional[str] = None,
) -> str:
    """Format comparison results as Markdown table."""
    # Similar to format_comparison_table but with Markdown syntax
    # Implementation similar to above but with Markdown table formatting
    pass

def find_best_experiment(
    comparisons: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Find best performing experiment based on primary metrics."""
    # Implementation to determine best experiment
    # Can be based on ROUGE-L for summarization, F1 for NER, etc.
    pass

def find_regressions(
    comparisons: Dict[str, Dict[str, Any]],
    baseline: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Find experiments that regress compared to baseline."""
    # Implementation to detect regressions
    pass
```

**CLI Usage:**

```bash
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
```

**Example Output:**

```text
Run                         ROUGE-L   Avg Compression
-----------------------------------------------------
summarization_bart_led_v1   0.120     43.4×
summarization_openai_gpt4_mini_v1   0.145     38.2×
summarization_openai_gpt4_mini_promptB   0.152     37.5×   <-- best
```

**Markdown Table Output:**

```markdown
| Run | ROUGE-L | Avg Compression |
|-----|---------|-----------------|
| summarization_bart_led_v1 | 0.120 | 43.4× |
| summarization_openai_gpt4_mini_v1 | 0.145 | 38.2× |
| summarization_openai_gpt4_mini_promptB | **0.152** | 37.5× ⭐ |
```

**JSON Output:**

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
```

**Task-Specific Comparison:**

**Summarization:**

- ROUGE-L (primary)
- Average Compression
- ROUGE-1, ROUGE-2 (optional)

**NER:**

- Precision
- Recall
- F1 Score

**Transcription:**

- Word Error Rate (WER)
- Character Error Rate (CER)

## Benefits

1. **Rapid Iteration**: Change model/prompt/params by editing YAML, no code changes
2. **Reproducibility**: Experiments are fully reproducible from config files
3. **Separation of Concerns**: Generation separate from evaluation enables recomputation
4. **Reusability**: Reuses existing eval scripts, no duplication
5. **CI/CD Integration**: Can run experiments in automated workflows
6. **Comparison**: Easy to compare multiple experiments side-by-side
7. **Version Control**: Config files are version controlled, experiments are tracked
8. **Scalability**: Easy to add new experiments by creating new config files
9. **Flexible Execution**: Run single experiment for golden dataset creation, or batch multiple experiments for model comparison
10. **Parallel Processing**: Run multiple experiments in parallel for faster batch execution
11. **Task Filtering**: Run all experiments of a specific type (e.g., all summarization experiments)

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
```

**Example Excel Content:**

**Summarization Tab:**

| Experiment ID | ROUGE-L | ROUGE-1 | ROUGE-2 | Avg Compression | Date | Notes |
| -------------- | ------- | ------- | ------- | --------------- | ---- | ----- |
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
    # Load or create Excel workbook
    if excel_file.exists():
        excel = pd.ExcelFile(excel_file)
        sheets = {sheet: pd.read_excel(excel, sheet_name=sheet) for sheet in excel.sheet_names}
    else:
        sheets = {
            "Summarization": pd.DataFrame(columns=["Experiment ID", "ROUGE-L", "ROUGE-1", "ROUGE-2", "Avg Compression", "Date", "Notes"]),
            "NER": pd.DataFrame(columns=["Experiment ID", "Precision", "Recall", "F1", "Date", "Notes"]),
            "Transcription": pd.DataFrame(columns=["Experiment ID", "WER", "CER", "Date", "Notes"]),
        }
    
    # Select appropriate sheet
    sheet_name = task.capitalize()
    df = sheets[sheet_name]
    
    # Extract metrics based on task type
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
    
    # Update or append row
    if experiment_id in df["Experiment ID"].values:
        # Update existing row
        idx = df[df["Experiment ID"] == experiment_id].index[0]
        for col, val in row.items():
            df.at[idx, col] = val
    else:
        # Append new row
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    sheets[sheet_name] = df
    
    # Write back to Excel
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
```

**Integration with Experiment Pipeline:**

```python
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
```

**CLI Usage:**

```bash
# Run experiment and automatically update Excel
python scripts/run_experiment.py \
  --config experiments/summarization_openai_gpt4_mini_v1.yaml \
  --update-excel

# Manually update Excel from existing results
python scripts/update_experiment_results.py \
  --experiment-id summarization_openai_gpt4_mini_v1 \
  --task summarization \
  --metrics-file results/summarization_openai_gpt4_mini_v1/metrics.json
```

**Benefits:**

1. **Single Source of Truth**: One Excel file tracks all experiments
2. **Easy Comparison**: Sort/filter by metrics to find best experiments
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

- ✅ Can run experiments by specifying only a config file
- ✅ Can regenerate predictions without recomputing metrics
- ✅ Can recompute metrics without regenerating predictions
- ✅ Can compare multiple experiments easily
- ✅ Integrates with existing eval scripts
- ✅ Works with CI/CD pipelines
- ✅ Supports all three task types (NER, transcription, summarization)
- ✅ Config files are human-readable and version controlled
- ✅ Results are tracked in centralized Excel workbook (one tab per task type)

## Implementation Plan

### Phase 1: Core Infrastructure

1. Create experiment config schema (Pydantic models)
2. Implement config loading and validation
3. Create generic runner framework
4. Implement backend factory

### Phase 2: Generation Phase

1. Implement local summarization backend
2. Implement OpenAI summarization backend
3. Implement NER backends (local + OpenAI)
4. Implement transcription backends (local + OpenAI)
5. Implement prediction saving (JSONL format)

### Phase 3: Evaluation Phase

1. Refactor existing eval scripts to accept standardized input
2. Implement metrics computation wrapper
3. Implement metrics saving (JSON format)
4. Create comparison utilities with table/markdown/json output formats
5. Implement regression detection logic

### Phase 4: CLI and Integration

1. Implement CLI interface
2. Create example experiment configs
3. Document golden dataset structure
4. Create CI/CD workflow example

### Phase 5: Documentation and Examples

1. Create comprehensive documentation
2. Add example experiments for each task type
3. Create comparison examples
4. Document best practices

## Related Documents

- `docs/rfc/RFC-012-episode-summarization.md`: Summarization design
- `docs/rfc/RFC-013-openai-provider-implementation.md`: OpenAI provider design
- `docs/EVALUATION_STRATEGY.md`: Evaluation strategy (to be created)
- `scripts/eval_summaries.py`: Existing summarization evaluation script
- `scripts/eval_cleaning.py`: Existing cleaning evaluation script
