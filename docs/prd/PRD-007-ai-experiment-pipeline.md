# PRD-007: AI Experiment Pipeline

- **Status**: 📋 Draft
- **Related RFCs**: RFC-015, RFC-016

## Summary

Add a configuration-driven AI experiment pipeline capability that enables rapid iteration on model selection, prompt engineering, and parameter tuning without requiring code changes. **Think of it exactly like your unit/integration test pipeline – just for models instead of code.** This capability separates generation (model inference) from evaluation (metrics computation), enabling efficient experimentation, comparison, and integration with CI/CD workflows.

**Key Concept**: Treat model + prompt + params as configuration. You don't want hardcoded experiments in Python. You want config files that define an experiment, like you'd define a GitHub Actions workflow. Now "trying a different model or prompt" is just adding another config file.

## Background & Context

**Analogy**: Just as you have unit tests and integration tests for code, you need a test pipeline for models. The AI experiment pipeline is your "test suite" for AI models, prompts, and parameters.

Currently, evaluating different AI models, prompts, and parameters requires:

- **Code Changes**: Modifying Python code to test different configurations
- **Slow Iteration**: Code changes require redeployment and full pipeline runs
- **Tight Coupling**: Generation and evaluation are intertwined, making it hard to recompute metrics
- **Manual Comparison**: Comparing results across experiments is manual and error-prone
- **No Reproducibility**: Experiments are not easily reproducible or shareable
- **CI/CD Integration**: Hard to integrate AI experiments into automated testing workflows

**What We Already Have**:

- Golden dataset (`data/eval/`)
- Hugging Face baseline models
- Evaluation scripts (`eval_summaries.py`, etc.)

**What We Need**:

- A generic runner that wraps existing pieces in a repeatable "AI experiment pipeline"
- Configuration-driven experiments (like GitHub Actions workflows)
- Separation of generation from scoring
- CI/CD integration with two layers: fast smoke tests + full evaluation pipeline
- Comparison tooling to answer "which model + prompt is best?" with data

This PRD addresses the need for a systematic, repeatable way to experiment with AI models and prompts while maintaining separation between experimentation and production workflows.

## Goals

1. **Configuration-Driven**: Model + prompt + params defined in YAML config files (like GitHub Actions workflows)
2. **No Code Changes**: Change model/prompt/params by editing config files only - "trying a different model or prompt" is just adding another config file
3. **Reuse Existing Eval Scripts**: Leverage `eval_summaries.py`, `eval_cleaning.py`, `eval_ner.py` (planned)
4. **Separation of Generation from Scoring**: Generation (inference) separate from evaluation (metrics) - enables recomputing metrics without regenerating predictions
5. **Reproducibility**: Experiments are fully reproducible from config files
6. **Two-Layer CI/CD Integration**:
   - **Layer A**: Fast smoke tests on every push/PR (small subset, quick sanity check)
   - **Layer B**: Full evaluation pipeline (nightly or on-demand, all experiments)
7. **Multiple Task Types**: Support NER, transcription, and summarization experiments
8. **Golden Dataset**: Use existing `data/eval/` folder as ground truth test set
9. **Comparison Tooling**: Excel-based comparison to answer "which model + prompt is best?" with data, not feelings

## Use Cases

### UC1: Prompt Engineering

**Actor**: Researcher tuning AI prompts

**Goal**: Optimize prompt wording and structure for better results

**Scenario:**

1. Create new prompt file: `prompts/summarization/long_v2_focus_on_frameworks.j2`
2. Copy baseline experiment config: `experiments/summarization_openai_long_v1.yaml` → `summarization_openai_long_v2.yaml`
3. Update config to reference new prompt
4. Run experiment: `python scripts/run_experiment.py experiments/summarization_openai_long_v2.yaml`
5. Compare metrics with baseline to see if ROUGE scores improved

**Success**: Can iterate on prompts without touching code, compare results easily

### UC2: Model Comparison

**Actor**: Developer evaluating model options

**Goal**: Compare different models (local vs OpenAI, different OpenAI models)

**Scenario:**

1. Create experiment configs for each model variant:
   - `summarization_bart_led_local.yaml` (local transformers)
   - `summarization_openai_gpt4_mini.yaml` (OpenAI GPT-4o-mini)
   - `summarization_openai_gpt4_turbo.yaml` (OpenAI GPT-4 Turbo)
2. Run all experiments: `python scripts/run_experiment.py experiments/summarization_*.yaml`
3. Compare metrics side-by-side in Excel workbook
4. Make informed decision based on quality vs cost trade-offs

**Success**: Can compare multiple models systematically, make data-driven decisions

### UC3: Parameter Tuning

**Actor**: ML engineer optimizing model parameters

**Goal**: Find optimal chunk sizes, overlap, max lengths for best performance

**Scenario:**

1. Create experiment configs with different parameter values:
   - `summarization_openai_chunk_900.yaml` (chunk_size: 900)
   - `summarization_openai_chunk_1200.yaml` (chunk_size: 1200)
   - `summarization_openai_chunk_1500.yaml` (chunk_size: 1500)
2. Run experiments in parallel
3. Analyze which parameters yield best ROUGE scores
4. Document findings in experiment notes

**Success**: Can systematically test parameter combinations, identify optimal settings

### UC4: Regression Testing

**Actor**: Developer ensuring model changes don't degrade performance

**Goal**: Ensure model/prompt changes maintain or improve quality

**Scenario:**

1. Establish baseline experiment: `summarization_openai_baseline_v1.yaml`
2. Run baseline, record metrics in Excel workbook
3. Make changes to model or prompt
4. Run same experiment config again
5. Compare metrics - ensure no degradation
6. If improved, update baseline

**Success**: Can detect regressions automatically, maintain quality standards

### UC5: A/B Testing

**Actor**: Product manager comparing multiple configurations

**Goal**: Compare multiple configurations side-by-side

**Scenario:**

1. Create multiple experiment configs with different approaches:
   - `summarization_openai_narrative_style.yaml`
   - `summarization_openai_bullet_points.yaml`
   - `summarization_openai_executive_summary.yaml`
2. Run all experiments on same dataset
3. Compare results in Excel workbook
4. Select best approach based on metrics and use case

**Success**: Can compare multiple approaches systematically, choose best option

### UC6: Golden Dataset Creation

**Actor**: Researcher creating high-quality reference data

**Goal**: Generate high-quality summaries/annotations for evaluation

**Scenario:**

1. Create experiment config using best model (e.g., GPT-4 Turbo)
2. Run on evaluation episodes: `python scripts/run_experiment.py experiments/summarization_golden_dataset.yaml`
3. Review outputs manually
4. Use as golden dataset for future experiments

**Success**: Can create high-quality reference data systematically

## Personas

### Researcher Rachel

- **Role**: AI/ML researcher tuning models and prompts
- **Goals**: Optimize model performance, improve quality metrics
- **Pain Points**: Slow iteration cycles, hard to compare experiments
- **Needs**: Fast iteration, easy comparison, reproducible experiments

**User Story**: _As Researcher Rachel, I can create a new prompt variant, run an experiment, and compare results with previous experiments in under 5 minutes without writing any code._

### Developer Devin

- **Role**: Developer evaluating model options and ensuring quality
- **Goals**: Choose best model, prevent regressions, maintain quality
- **Pain Points**: Hard to compare models, no systematic testing
- **Needs**: Easy model comparison, regression testing, CI/CD integration

**User Story**: _As Developer Devin, I can compare three different models side-by-side and integrate experiment runs into CI/CD to catch regressions automatically._

### ML Engineer Max

- **Role**: ML engineer optimizing parameters and performance
- **Goals**: Find optimal parameters, tune performance
- **Pain Points**: Manual parameter testing, hard to track results
- **Needs**: Systematic parameter testing, result tracking, analysis tools

**User Story**: _As ML Engineer Max, I can test 10 different parameter combinations in parallel and analyze results to find optimal settings._

## Functional Requirements

### FR1: Experiment Configuration

- **FR1.1**: Experiments defined in YAML configuration files
- **FR1.2**: Config files specify model, prompts, parameters, and data sources
- **FR1.3**: Config files are human-readable and version-controlled
- **FR1.4**: Support for multiple task types (summarization, NER, transcription)
- **FR1.5**: Support for multiple backends (local HF, OpenAI, future providers)
- **FR1.6**: Config files reference prompts by logical name (e.g., `"summarization/long_v1"`)
- **FR1.7**: Config files support parameter overrides (temperature, max_tokens, etc.)

### FR2: Experiment Execution

**Generic Runner Requirements:**

- **FR2.1**: Generic runner takes experiment config as input
- **FR2.2**: Runner loads episodes listed in config
- **FR2.3**: Runner calls appropriate backend (local HF or OpenAI API)
- **FR2.4**: Runner saves predictions and metrics separately
- **FR2.5**: Run single experiment from config file
- **FR2.6**: Run multiple experiments in batch (glob pattern)
- **FR2.7**: Run experiments filtered by task type
- **FR2.8**: Generate predictions without evaluation (Step 1: generation)
- **FR2.9**: Evaluate existing predictions without regeneration (Step 2: evaluation)
- **FR2.10**: Generate + evaluate in single command (optional convenience)
- **FR2.11**: Parallel execution of multiple experiments (respect rate limits)
- **FR2.12**: Progress tracking and logging during execution
- **FR2.13**: Support episode filtering (e.g., `--episodes ep01` for smoke tests)

### FR3: Prompt Management

- **FR3.1**: Prompts stored as versioned files (e.g., `prompts/summarization/long_v1.j2`)
- **FR3.2**: Prompts support templating (Jinja2) for parameterization
- **FR3.3**: Experiments reference prompts by logical name
- **FR3.4**: Prompt metadata (name, path, SHA256) tracked in results
- **FR3.5**: Easy to create new prompt variants without code changes

### FR4: Evaluation & Metrics

**Separation of Generation from Scoring:**

- **FR4.1**: Reuse existing evaluation scripts (`eval_summaries.py`, etc.)
- **FR4.2**: Compute standard metrics (ROUGE, compression, etc.)
- **FR4.3**: Generate per-episode and global metrics
- **FR4.4**: Track prompt metadata in metrics output
- **FR4.5**: Support for multiple evaluation types (summarization, NER, transcription)
- **FR4.6**: Can recompute metrics without regenerating predictions (enables new ROUGE variants, custom metrics, LLM-as-judge)
- **FR4.7**: Can compare any two runs by comparing their metrics.json files

### FR5: Result Tracking & Comparison

**Comparing Runs Like You Compare Builds:**

- **FR5.1**: Results stored in structured format (JSONL for predictions, JSON for metrics)
- **FR5.2**: Centralized Excel workbook for result aggregation
- **FR5.3**: One tab per task type in Excel workbook
- **FR5.4**: Easy side-by-side comparison of experiments
- **FR5.5**: Track experiment metadata (date, config, prompts, backend)
- **FR5.6**: Support for experiment notes and annotations
- **FR5.7**: Comparison tool that creates Excel with all experiments and key metrics
- **FR5.8**: Answer "which model + prompt is best?" becomes a data question, not a feeling
- **FR5.9**: Visual comparison tables showing ROUGE/precision/F1 across experiments

### FR6: CI/CD Integration (Two-Layer Approach)

**Layer A: CI Smoke Tests (Fast, Small Subset)**

- **FR6.1**: Run on every push/PR
- **FR6.2**: Use tiny subset of episodes (e.g., ep01 only)
- **FR6.3**: Use single baseline config (HF baseline or OpenAI baseline)
- **FR6.4**: Assert quality thresholds (e.g., `rougeL_f >= threshold`)
- **FR6.5**: Assert no runtime errors, no NaNs, no missing fields
- **FR6.6**: Quick sanity check that pipeline wasn't broken (like unit tests)

**Layer B: Full AI Eval Pipeline (Nightly/On-Demand)**

- **FR6.7**: Triggered on schedule (nightly) or manually
- **FR6.8**: Runs all experiment configs (summarization_bart_led_v1, summarization_openai_gpt4_mini_v1, etc.)
- **FR6.9**: Produces metrics.json per experiment
- **FR6.10**: Generates combined summary_report.md with table of ROUGE/precision
- **FR6.11**: Like integration/regression testing for models
- **FR6.12**: Can compare current results with baseline
- **FR6.13**: Can detect regressions automatically
- **FR6.14**: Can trigger experiments on config/prompt changes

### FR7: Separation of Concerns

- **FR7.1**: Generation (inference) separate from evaluation (metrics)
- **FR7.2**: Can regenerate predictions without recomputing metrics
- **FR7.3**: Can recompute metrics without regenerating predictions
- **FR7.4**: Experiment pipeline independent from production pipeline
- **FR7.5**: Reuse production providers, no code duplication

## Non-Functional Requirements

### NFR1: Performance

- **NFR1.1**: Experiments should run efficiently (parallel execution where possible)
- **NFR1.2**: Respect API rate limits (OpenAI, etc.)
- **NFR1.3**: Support batch execution for multiple experiments

### NFR2: Usability

- **NFR2.1**: Config files should be intuitive and well-documented
- **NFR2.2**: Error messages should be clear and actionable
- **NFR2.3**: Results should be easy to interpret and compare

### NFR3: Maintainability

- **NFR3.1**: Experiment pipeline should reuse production code
- **NFR3.2**: Easy to add new experiment types
- **NFR3.3**: Easy to add new backends/providers

### NFR4: Reproducibility

- **NFR4.1**: Experiments fully reproducible from config files
- **NFR4.2**: Prompt versions tracked in results
- **NFR4.3**: All parameters tracked in metadata

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
- ✅ Prompt engineering workflow is fast and iterative (< 5 minutes per iteration)
- ✅ No code changes required for new experiments

## Benefits

1. **Rapid Iteration**: Change model/prompt/params by editing YAML, no code changes - "trying a different model or prompt" is just adding another config file
2. **Reproducibility**: Experiments are fully reproducible from config files
3. **Separation of Generation from Scoring**:
   - Can recompute metrics without regenerating predictions (new ROUGE variant, custom metric, LLM-as-judge)
   - Can compare any two runs by comparing their metrics.json files
4. **Reusability**: Reuses existing eval scripts, no duplication - wraps existing pieces in repeatable pipeline
5. **Two-Layer CI/CD**: Fast smoke tests catch breakages quickly, full pipeline provides comprehensive evaluation
6. **Comparison**: Easy to compare multiple experiments side-by-side - "which model + prompt is best?" becomes a data question
7. **Version Control**: Config files are version controlled, experiments are tracked (like test files)
8. **Scalability**: Easy to add new experiments by creating new config files
9. **Flexible Execution**: Run single experiment for golden dataset creation, or batch multiple experiments for model comparison
10. **Parallel Processing**: Run multiple experiments in parallel for faster batch execution
11. **Task Filtering**: Run all experiments of a specific type (e.g., all summarization experiments)
12. **Test Pipeline Analogy**: Treats model evaluation like code testing - systematic, repeatable, automated

## Out of Scope

- Interactive UI for experiment management
- Real-time experiment monitoring dashboard
- Automatic hyperparameter optimization
- Experiment result visualization (beyond Excel)
- Cost tracking or usage monitoring
- Experiment result sharing/collaboration features
- Model training capabilities

## Dependencies

- **Internal**: Provider system refactoring (RFC-016)
- **Internal**: Prompt management system (RFC-017)
- **Internal**: OpenAI provider implementation (RFC-013, PRD-006)
- **External**: Python packages: `python-dotenv`, `pydantic`, `pyyaml`, `jinja2`

## Risks & Mitigations

**Risk**: Experiment pipeline duplicates production code

**Mitigation**: Reuse production providers via provider system (RFC-016), no duplication

**Risk**: Breaking existing production workflow

**Mitigation**: Experiment pipeline is independent, production workflow unchanged

**Risk**: Config files become complex and hard to maintain

**Mitigation**: Well-documented config format, examples, validation

**Risk**: Results tracking becomes unwieldy with many experiments

**Mitigation**: Excel workbook with organized tabs, clear naming conventions

## Related Documents

- `docs/rfc/RFC-015-ai-experiment-pipeline.md`: Technical design and implementation details
- `docs/rfc/RFC-016-modularization-for-ai-experiments.md`: Provider system architecture
- `docs/rfc/RFC-017-prompt-management.md`: Prompt management implementation
- `docs/rfc/RFC-013-openai-provider-implementation.md`: OpenAI provider technical design
- `docs/prd/PRD-006-openai-provider-integration.md`: OpenAI provider product requirements

## Implementation Phases

**Phase 1: Normalize Existing Structure**

- Move gold data under `data/eval/episodes/*`
- Keep existing baseline as `results/summarization_bart_led_v1/metrics.json`
- Establish baseline structure

**Phase 2: Generic Runner**

- Write single generic runner (`run_experiment.py`) that:
  - Takes config path as input
  - Loads episodes listed in config
  - Calls appropriate backend (local HF or OpenAI API)
  - Writes predictions + metrics separately
- Support episode filtering (e.g., `--episodes ep01`)

**Phase 3: CI Smoke Tests (Layer A)**

- Add tiny CI job that:
  - Runs `run_experiment.py configs/experiments/summarization_bart_led_v1.yaml --episodes ep01`
  - Asserts ROUGE-L > threshold
  - Asserts no errors, no NaNs, no missing fields
- Runs on every push/PR

**Phase 4: Full Eval Pipeline (Layer B)**

- Add "full eval" script that:
  - Loops over all YAMLs in `configs/experiments/*.yaml`
  - Runs them, writes results
  - Prints summary table
- Can run nightly or on-demand

**Phase 5: Comparison Tooling**

- Build comparison tool that:
  - Reads all experiment results
  - Creates Excel workbook with all experiments and key metrics
  - Enables data-driven decisions on "which model + prompt is best?"

**Phase 6: Extend to New Providers**

- Once structure is in place, adding OpenAI summarization, OpenAI NER, etc. is just:
  - Add config file
  - Add small backend class (if needed)

**Timeline**: See RFC-015 for detailed technical implementation plan.
