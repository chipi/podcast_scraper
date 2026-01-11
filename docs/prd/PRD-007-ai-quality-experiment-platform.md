# PRD-007: AI Quality & Experimentation Platform

- **Status**: 📋 Draft
- **Authors**: Podcast Scraper Team
- **Related RFCs**: RFC-015 (Runner), RFC-016 (Modularization), RFC-041 (Benchmarking), RFC-043 (Alerts)

## Summary

The **AI Quality & Experimentation Platform** is a unified system for managing, measuring, and driving the evolution of the podcast processing pipeline. It transforms the project from a set of script-driven tasks into a data-first product where quality is objective, progress is measurable, and the management of AI models and data is a first-class capability.

By combining an **Experiment Runner** (RFC-015), **Modular Providers** (RFC-016), a **Benchmarking Framework** (RFC-041), and **Proactive Alerting** (RFC-043), we establish the management layer needed to govern the evolution of the AI pipeline. The platform serves as the **universal evaluation harness** for all evaluable AI changes, ensuring that every model, prompt, or preprocessing update is driven by objective data.

## Background & Context

As the project scales from simple transcript downloads to complex summarization and speaker detection, we face three primary challenges:

1. **Management Complexity**: As we add more models and task types, managing the matrix of configurations becomes impossible without a unified platform.
2. **Lack of Direction**: Without objective metrics (ROUGE, WER, Quality Gates), we cannot "drive" the product toward specific goals (e.g., "reduce cost by 50% without losing accuracy").
3. **Silent Drift**: Without proactive monitoring and alerting, model updates or upstream changes can silently degrade the product value.

**The Solution**: An integrated management platform that treats models, prompts, and datasets as codified assets. This platform provides the "steering wheel" and "dashboard" needed to drive AI development with engineering rigor.

## Goals

1. **Objective Gold Standard**: Establish frozen, versioned "Golden Datasets" (The Indicator, The Journal) that define what success looks like.
2. **Measurable Progress**: Quantify the trade-offs between cost, latency, and accuracy across all supported providers.
3. **Rapid Iteration**: Enable "Prompt Engineering" and "Model Swaps" via YAML configuration with zero code changes.
4. **Automated Quality Gates**: Detect and block common failure modes (boilerplate leaks, speaker attribution errors, truncation) in CI.
5. **Data-Driven Decisions**: Use side-by-side Excel/Markdown comparisons to answer "Which model + prompt is best?" with evidence.

## Core Pillars

### 1. The Experiment Runner (The Lab)
- **RFC Reference**: RFC-015
- **Capability**: A repeatable pipeline that executes model configurations against datasets.
- **Driving Value**: The experiment runner is the **single execution path** for all evaluable AI changes (models, prompts, preprocessing, chunking, ASR). It enables data-driven experimentation by decoupling execution from scoring.

### 2. Modular Provider System (The Engine)
- **RFC Reference**: RFC-016
- **Capability**: A unified, pluggable interface for all AI backends.
- **Driving Value**: Provides the management flexibility to swap and compare providers without code changes.

### 3. Benchmarking Framework (The Judge)
- **RFC Reference**: RFC-041
- **Capability**: Automated scoring against "Golden" standards using defined metrics and quality gates.
- **Driving Value**: Establishes the objective "Scoreboard" that drives decision-making.

### 4. Proactive Alerting (The Watchman)
- **RFC Reference**: RFC-043
- **Capability**: Automated notifications and PR feedback on metric deviations.
- **Driving Value**: Closes the feedback loop, ensuring management is aware of quality drift immediately.

## Personas & Use Cases

### UC1: Strategic Model Steering (The "Drive" Case)

**Persona**: Product Lead
**Scenario**: "We need to reduce our reliance on expensive cloud providers while maintaining quality."
**Action**: The Lead uses the platform to compare current production metrics against experimental local configurations.
**Result**: The platform provides the comparative data needed to drive the migration strategy with objective evidence.

### UC2: Configuration-Based Iteration (The "Manage" Case)

**Persona**: Developer Devin
**Scenario**: "I need to test a new version of our summarization prompt."
**Action**: Devin adds a new versioned prompt file and a corresponding experiment config.
**Result**: The system manages the execution and comparison, allowing Devin to manage multiple variants simultaneously without code drift.

### UC3: Quality Governance (The "Guardrail" Case)

**Persona**: Maintainer Mike
**Scenario**: "A contributor submitted a change that might affect our speaker detection accuracy."
**Action**: The CI system automatically drives the benchmarking suite and alerts Mike of any regressions.
**Result**: Mike can manage the quality of contributions objectively, ensuring the product's quality standard is maintained.

## Management & Quality Contracts

To ensure objective driving of the platform, the following "Product Contracts" are enforced:

### 1. The Experiment Contract

Every experiment MUST declare:

- **`dataset_id`**: The immutable identity of the data being processed.
- **`baseline_id`**: The reference point for comparison (the "current best" or "production").
- **`golden_required`**: Whether a human-verified reference is mandatory for this run.

*Experiments failing to declare these are considered invalid and blocked from execution. If `golden_required = true` and no golden artifact exists, the experiment fails at validation time, not runtime.*

### 2. The Comparison Constraint

Metrics and regressions are ONLY valid when:
`experiment.dataset_id == baseline.dataset_id`

This prevents the "Invalid Comparison" anti-pattern (e.g., comparing The Indicator results against The Journal baseline).

## Product Requirements

### PR1: Canonical Benchmarking Datasets
- **FR1.1**: Support content-regime buckets (Explainer, Science, Narrative).
- **FR1.2**: Frozen datasets committed to the repo or stable storage (JSON metadata + audio/transcripts).
- **FR1.3**: Versioned datasets (e.g., `indicator_v1`) to prevent historical drift.

### PR2: Quality Gates (Non-Negotiable Defaults)
- **FR2.1**: **Boilerplate Detection**: Fail if "credits," "timestamps," or "newsletter" text leaks into summaries.
- **FR2.2**: **Speaker Leak Detection**: Fail if speaker labels (e.g., "Host 1:") appear in the final summary.
- **FR2.3**: **Content Integrity**: Measure "Numbers Retained" to ensure quantitative data isn't lost.
- **FR2.4**: **Stability Check**: Measure variance across 3 runs of the same input to detect hallucination/instability.
- **FR2.5**: **Stability Thresholds**: Thresholds (cosine similarity, variance) are defined and versioned by the benchmarking framework (RFC-041), not per experiment.

### PR3: Platform Reproducibility (The Product Contract)
- **FR3.1**: **Full Provider Fingerprinting**: Every experiment result MUST record a full provider fingerprint: `model`, `version`, `device`, `precision`, `preprocessing_profile`, and `git_commit`.
- **FR3.2**: **Artifact Locking**: Content-hash (`SHA256`) all inputs and prompt templates to detect silent data or configuration changes.
- **FR3.3**: **Reproducible Defaults**: A configuration must produce the same result (within stability thresholds) regardless of which developer runs it.

### PR4: Management Reporting
- **FR4.1**: Side-by-side comparison tables showing: `Quality Delta | Latency Delta | Cost Delta`.
- **FR4.2**: Long-term tracking for trends across months of experimentation.

## Implementation Roadmap & Dependencies

### Phase 0: Prerequisite Infrastructure

- **RFC-016 Phase 2 (Typed Params & Factory)**: This is a **hard prerequisite**. The experiment runner cannot be built until the provider system supports typed configurations and fingerprinting.

### Phase 1: Core Execution & Data

- **RFC-015 Phase 1 (Runner)**: Implementation of the runner with `baseline_id` and `dataset_id` enforcement.
- **RFC-041 Phase 0 (Datasets)**: Freezing the first "Golden" datasets and establishing initial baselines.

### Phase 2: Full Benchmarking & Automation

- **RFC-041 (Full Suite)**: Integration of quality gates and ROUGE/WER metrics.
- **RFC-043 (Alerts)**: CI integration for proactive alerting.

## Success Criteria

- ✅ **Management Visibility**: 100% of AI models and prompts are managed as versioned configurations, not code.
- ✅ **Goal-Driven Development**: Every model update or swap is justified by a benchmarking report showing the impact on key metrics.
- ✅ **Zero Silent Regressions**: All critical failure modes are caught and alerted on automatically in CI.
- ✅ **Operational Rigor**: New experiments or benchmarks can be configured and executed in < 10 minutes.

## Future Vision

- **LLM-as-a-Judge**: Use high-capability models to grade the quality of low-cost models on nuance and tone.
- **Human-in-the-loop**: A lightweight UI for manually approving "Golden Reference" data.
- **Automatic Hyperparameter Tuning**: Automatically find the best chunk size and overlap for a given model.
