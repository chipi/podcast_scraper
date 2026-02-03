# AI Quality & Experimentation Platform - Implementation Plan

**Date:** 2026-01-16
**Status:** Phase 1 Complete ✅ | Phase 2 Mostly Complete ✅ (CI integration remaining)
**Objective:** Move from manual "eyeball" comparisons to a data-first product driven by objective, reproducible benchmarking.

---

## Executive Summary

This document tracks the complete implementation plan for the **AI Quality & Experimentation Platform (PRD-007)**, covering three RFCs:

- **RFC-016**: Provider Modularization
- **RFC-015**: AI Experiment Pipeline
- **RFC-041**: Podcast ML Benchmarking Framework

**Current Status:**

- ✅ **Phase 1 Complete** (~5 weeks): Foundation work establishing reproducible artifacts and experiment infrastructure
- ✅ **Phase 2 Mostly Complete** (~10 weeks done, ~2 weeks remaining): Evaluation, storage, and benchmarking complete; CI integration pending
- ⏳ **Phase 3 Planned** (~1 week): Proactive alerting and monitoring (RFC-043)

**GitHub Issues (Phase-Based Organization):**

- [#303](https://github.com/chipi/podcast_scraper/issues/303) - Phase 1: Foundation (COMPLETED ✅)
- [#304](https://github.com/chipi/podcast_scraper/issues/304) - Phase 2: Evaluation, Storage & CI (IN PROGRESS ⏳)
- [#333](https://github.com/chipi/podcast_scraper/issues/333) - Phase 3: Proactive Alerting & Monitoring (PLANNED ⏳)

---

## Phase 1: Foundation (COMPLETED ✅)

**Timeline:** ~5 weeks
**Status:** All deliverables complete and validated

### Step 1: RFC-016 Phase 2 - Provider Modularization Foundation

**Objective:** Establish typed params, fingerprinting, and preprocessing profiles for reproducible provider runs.

#### ✅ RFC-016 Deliverables Completed

- ✅ **`src/podcast_scraper/providers/params.py`**: Typed Pydantic models for `SummarizationParams`, `TranscriptionParams`, `SpeakerDetectionParams`
- ✅ **`src/podcast_scraper/providers/fingerprint.py`**: `ProviderFingerprint` model with comprehensive environment capture (model name, version, device, precision, git commit)
- ✅ **`src/podcast_scraper/preprocessing/profiles.py`**: `PreprocessingProfile` registry with versioned cleaning logic
- ✅ **Factory functions enhanced**: Accept both `Config` objects (backward compat) and typed `ProviderParams` models
- ✅ **Provider fingerprinting integrated**: All provider outputs include full fingerprint metadata
- ✅ **Preprocessing profile tracking**: Profile IDs tracked in provider metadata

#### ✅ Validation Gates Passed

| Gate | Status |
| --- | --- |
| **Malformed Params** | ✅ Immediate failure at config load with clear type errors |
| **Explicit Defaults** | ✅ Fingerprint contains concrete values (no nulls or "default") |
| **Fingerprint Sensitivity** | ✅ Different runs produce different fingerprint hashes |
| **Preprocessing Visibility** | ✅ Preprocessing profile visible in output artifacts |

**Core Invariant:** ✅ "Every provider run produces a fully fingerprinted, typed, and reproducible artifact."

---

### Step 2: RFC-015 Phase 1 - Experiment Runner Foundation

**Objective:** Minimal experiment runner with contract enforcement and structured outputs.

#### ✅ RFC-015 Deliverables Completed

- ✅ **`src/podcast_scraper/evaluation/config.py`**: Experiment configuration schema with Pydantic models
- ✅ **`scripts/eval/run_experiment.py`**: Minimal experiment runner with contract enforcement
- ✅ **Contract enforcement**: Mandatory `dataset_id`, `baseline_id`, `golden_required` fields
- ✅ **Structured outputs**: Deterministic directory structure (`results/<run_id>/`)
- ✅ **Provider fingerprinting integration**: Experiments capture full provider fingerprints
- ✅ **Content hashing**: Input transcripts and generated outputs are hashed for verification
- ✅ **Initial OpenAI support**: Experiments can run with OpenAI summarization provider

#### ✅ RFC-015 Validation Gates Passed

| Gate | Status |
| --- | --- |
| **Contract Enforcement** | ✅ Run blocked before execution if required fields missing |
| **Dataset Identity** | ✅ Cross-dataset comparisons rejected with explicit error |
| **Structured Outputs** | ✅ Outputs include fingerprints and structured metadata |
| **Determinism Check** | ✅ Identical configs produce identical artifacts |

**Core Invariant:** ✅ "An experiment cannot run unless it is objectively comparable."

---

### Step 3: RFC-041 Phase 0 - Dataset Freezing & Baselines

**Objective:** Freeze datasets and establish first baselines as "Sacred Baseline" artifacts.

#### ✅ RFC-041 Deliverables Completed

- ✅ **`benchmarks/datasets/indicator_v1.json`**: Canonical dataset definition
- ✅ **`scripts/eval/create_dataset_json.py`**: Dataset JSON creation script
- ✅ **`scripts/eval/materialize_baseline.py`**: Baseline materialization script
- ✅ **`benchmarks/CHANGELOG.md`**: Baseline changelog documenting baseline creation
- ✅ **`benchmarks/README.md`**: Benchmarking framework documentation
- ✅ **Baseline immutability**: Hard failure on overwrite attempts (requires new versioned ID)
- ✅ **Comprehensive metadata**: Baselines include git commit, fingerprint, stats

#### ✅ RFC-041 Validation Gates Passed

| Gate | Status |
| --- | --- |
| **Single Source of Truth** | ✅ Runner only accepts valid Dataset IDs from registry |
| **Baseline Immutability** | ✅ Hard failure on overwrite; requires new versioned ID |
| **Explainability** | ✅ Metadata contains all answers about baseline origin |

**Core Invariant:** ✅ "Baselines are frozen, reviewable, and impossible to mutate accidentally."

---

## Phase 2: Evaluation, Storage & CI (MOSTLY COMPLETE ✅)

**Timeline:** ~12 weeks estimated, ~10 weeks completed
**Status:** Work tracked in issue #304
**Completion:** ~85% complete - Only CI integration remaining

### RFC-016 Phase 2 (Remaining Work)

**Estimate:** ~2 weeks

#### Phase 1: Legacy Module Cleanup (Optional - Low Priority)

- [x] Deprecate legacy modules (`summarizer.py`, `speaker_detection.py`, `whisper_integration.py`)
  - ✅ `summarizer.py` moved to `ml/summarizer.py` (properly encapsulated)
  - ✅ `speaker_detection.py` moved to `ml/speaker_detection.py` (properly encapsulated)
  - ✅ `whisper_integration.py` deprecated (unused, all functionality in `ml_provider.py`)
  - ✅ All references updated to use provider APIs (workflow uses factories)
  - ✅ Backward compatibility maintained via deprecation stubs
- [x] Update all references to use new provider APIs
  - ✅ Workflow uses `create_speaker_detector()`, `create_transcription_provider()`, `create_summarization_provider()`
  - ✅ No direct imports of legacy modules from workflow code
  - ✅ `ml_provider.py` imports from `ml.speaker_detection` (not root-level)
- [x] Move remaining legacy modules to `ml/` directory
  - ✅ `speaker_detection.py` moved to `ml/speaker_detection.py`
  - ✅ Root-level modules have deprecation stubs that re-export from new location
- **Status:** ✅ COMPLETE (100%)
  - All legacy modules moved to `ml/` directory
  - Root-level modules are deprecation stubs only
  - All external code uses provider factories
  - Backward compatibility maintained for tests
- **Priority:** Low - doesn't block other work
- **Estimate:** 1 week (COMPLETED)
- **Can be done in parallel with other work**

#### Phase 3: Extract Evaluation Infrastructure (Important)

- [x] Create `src/podcast_scraper/evaluation/scorer.py` (evaluation module exists)
- [x] Extract ROUGE calculation from `scripts/eval/eval_summaries.py` (replaced by new evaluation loop)
- [x] Extract BLEU calculation
- [x] Add WER calculation (using `jiwer`)
- [x] Add semantic similarity (using `sentence-transformers`)
- [x] Create unified evaluation API (`score_run()` function)
- **Status:** ✅ COMPLETE
- **Actual Location:** `src/podcast_scraper/evaluation/scorer.py` (not `experiments/evaluation.py`)

---

### RFC-015 Phase 2 (Remaining Work)

**Estimate:** ~4 weeks

#### Phase 2: Evaluation Metrics Integration (Weeks 3-4)

- [x] Integrate RFC-016 Phase 3 evaluation module
- [x] Add automated metric calculation (WER, ROUGE, BLEU, semantic similarity)
- [x] Generate human-readable evaluation reports (`reporter.py`)
- [x] Add metric comparison against baselines (`comparator.py`)
- **Status:** ✅ COMPLETE
- **Deliverables:** ✅ Experiments generate evaluation metrics automatically
- **Actual Implementation:** `run_experiment.py` integrates scorer, comparator, and reporter

#### Phase 3: Storage & Comparison (Week 5)

- [x] Experiment results storage in structured format (`data/eval/runs/<run_id>/`)
- [x] Historical tracking of experiment runs (`history.py`)
- [x] Comparison tools (experiment vs baseline, experiment vs experiment) (`comparator.py`, `compare_runs.py`)
- [ ] Visualization of metrics over time (not yet implemented)
- **Status:** ✅ MOSTLY COMPLETE (visualization pending)
- **Deliverables:** ✅ Can compare any two experiments
- **Actual Implementation:** `make runs-compare` command available

#### Phase 4: CI Integration (Week 6)

- [ ] Smoke tests on PRs (fast subset of experiments)
- [ ] Nightly comprehensive experiments
- [ ] Regression detection and reporting
- [ ] PR comments with experiment results
- **Deliverables:** PRs automatically run smoke tests
- **Estimate:** 1 week

---

### RFC-041 Phase 2 (Remaining Work)

**Estimate:** ~6 weeks

#### Phase 1: Regression Rules & Quality Gates (Week 3)

- [x] Define regression rules for summarization (ROUGE thresholds, boilerplate detection)
- [x] Define regression rules for transcription (WER thresholds)
- [x] Define regression rules for speaker detection (accuracy thresholds)
- [x] Implement regression checker (`regression.py` with `RegressionChecker` class)
- [ ] Create regression rule configuration format (YAML) (default rules exist, config format pending)
- **Status:** ✅ MOSTLY COMPLETE (10 default rules implemented, YAML config format pending)
- **Deliverables:** ✅ Regression rules defined and checker working
- **Actual Implementation:** `RegressionChecker` with default rules for ROUGE, BLEU, embedding similarity, latency, cost

#### Phase 2: Benchmark Runner (Weeks 4-5)

- [x] Create benchmark orchestrator (`run_benchmark.py`)
- [x] Implement benchmark execution (run experiments on all datasets)
- [x] Generate comparison reports (experiment vs baseline)
- [x] Add `make benchmark` command
- [x] Support for multiple baseline comparisons
- **Status:** ✅ COMPLETE
- **Deliverables:** ✅ `make benchmark` command working
- **Actual Implementation:** `make benchmark CONFIG=... BASELINE=...` with smoke/ALL/DATASETS options

#### Phase 3: CI Integration (Week 6) - ⏳ REMAINING WORK

- [ ] Smoke benchmarks on PRs (fast subset)
- [ ] Nightly full benchmarks
- [ ] PR comments with regression status
- [ ] Block PRs on critical regressions
- **Status:** ❌ NOT STARTED
- **Deliverables:** Automated benchmarking in CI
- **Estimate:** 1 week
- **Next Steps:** See "Remaining Work" section below

#### Phase 4: Documentation & Polish (Week 7)

- [ ] Complete benchmarking framework documentation
- [ ] Examples and tutorials
- [ ] User guide for creating new datasets
- [ ] User guide for creating new baselines
- **Deliverables:** Documentation complete
- **Estimate:** 1 week

#### Phase 5: Enhancements & Iteration (Week 8)

- [ ] Cost tracking and visualization
- [ ] Optimize execution time (parallelization, caching)
- [ ] Additional metrics (perplexity, coherence scores)
- [ ] Baseline comparison dashboards
- **Deliverables:** Enhanced metrics and optimizations
- **Estimate:** 1 week

---

## Phase 2 Remaining Work Summary

### ✅ What's Complete (85% of Phase 2)

1. **Evaluation Infrastructure** - ✅ 100% Complete
   - ✅ ROUGE, BLEU, WER, semantic similarity all implemented
   - ✅ Unified `score_run()` API in `scorer.py`
   - ✅ Intrinsic and extrinsic metrics

2. **Storage & Comparison** - ✅ 95% Complete
   - ✅ Experiment results stored in `data/eval/runs/<run_id>/`
   - ✅ Historical tracking via `history.py`
   - ✅ Comparison tools (`comparator.py`, `compare_runs.py`)
   - ✅ Report generation (`reporter.py`)
   - ⏳ Missing: Visualization dashboards (low priority)

3. **Regression Detection** - ✅ 90% Complete
   - ✅ `RegressionChecker` with 10 default rules
   - ✅ Rules for ROUGE, BLEU, embedding similarity, latency, cost
   - ⏳ Missing: YAML config format for custom rules (low priority)

4. **Benchmark Runner** - ✅ 100% Complete
   - ✅ `make benchmark` command working
   - ✅ Supports smoke mode, all datasets, or specific datasets
   - ✅ Generates comprehensive comparison reports

### ⏳ What's Remaining (15% of Phase 2)

**Only CI Integration is missing:**

1. **RFC-015 Phase 4: CI Integration** (~1 week)
   - [ ] Add smoke test job to `.github/workflows/python-app.yml`
   - [ ] Run fast experiment subset on PRs (use `SMOKE_INFERENCE_ONLY=1` or `SCORE_ONLY=1`)
   - [ ] Add nightly comprehensive experiment job to `.github/workflows/nightly.yml`
   - [ ] Generate PR comments with experiment results (use `reporter.py`)
   - [ ] Block PRs on critical regressions (use `RegressionChecker`)

2. **RFC-041 Phase 3: CI Integration** (~1 week)
   - [ ] Add smoke benchmark job to PR workflow (`make benchmark SMOKE=1`)
   - [ ] Add full benchmark job to nightly workflow (`make benchmark ALL=1`)
   - [ ] Generate PR comments with benchmark results
   - [ ] Block PRs on critical benchmark regressions

**Total Remaining:** ~2 weeks of work

**Next Steps:**

1. Create smoke test experiment config (fast subset)
2. Add experiment job to PR workflow
3. Add benchmark job to PR workflow
4. Add comprehensive jobs to nightly workflow
5. Create PR comment generation script
6. Integrate regression checking into CI

---

## Phase 3: Proactive Alerting & Monitoring (PLANNED ⏳)

**Timeline:** ~1 week
**Status:** Not started - depends on Phase 2 completion
**Issue:** [#333](https://github.com/chipi/podcast_scraper/issues/333)

### RFC-043: Automated Metrics Alerts

**Objective:** Close the feedback loop with proactive notifications on metric deviations and quality regressions.

#### Overview

RFC-043 provides automated alerting for experiment metrics and quality gates, ensuring developers are immediately aware of regressions without manual checking. This is the "Watchman" pillar of PRD-007.

**Key Principle:** Metrics should alert developers proactively, not require manual checking.

#### Phase 1: PR Comments (Priority 1)

- [ ] Create `scripts/generate_pr_comment.py` - Generate comparison markdown
- [ ] Update `.github/workflows/python-app.yml` - Add PR comment steps
- [ ] Post comparison comment on every PR showing metric changes vs baseline
- [ ] Show comparison table (main vs PR) with deltas
- [ ] Update existing comments (not duplicate) when PR is updated
- **Deliverables:** PR comments automatically posted with metric comparisons
- **Estimate:** 2-3 hours

#### Phase 2: Webhook Alerts (Priority 2)

- [ ] Create `scripts/send_webhook_alert.py` - Send alerts to webhooks
- [ ] Support Slack and Discord webhook formats
- [ ] Update workflows (python-app.yml, nightly.yml) - Add webhook steps
- [ ] Send alerts for critical regressions on main branch only
- [ ] Documentation for secret configuration (METRICS_WEBHOOK_URL, METRICS_WEBHOOK_TYPE)
- **Deliverables:** Webhook notifications for critical regressions (optional)
- **Estimate:** 1-2 hours

#### Phase 3: Testing & Refinement

- [ ] Test PR comments don't duplicate
- [ ] Test webhook alerts (if configured)
- [ ] Refine alert thresholds based on feedback
- [ ] Update documentation (docs/ci/METRICS.md, docs/ci/WORKFLOWS.md)
- **Deliverables:** Fully tested and documented alerting system
- **Estimate:** 1-2 hours

#### Alert Thresholds

| Metric | Threshold | Severity | Action |
| --- | --- | --- | --- |
| Runtime increase | > 10% | Warning | PR comment |
| Runtime increase | > 20% | Error | PR comment + webhook (main only) |
| Coverage drop | > 1% | Error | PR comment + webhook (main only) |
| Test failures | > 0 | Error | Already in job summary |
| Flaky tests | > 0 | Warning | PR comment |
| Flaky tests | > 5 | Error | PR comment + webhook (main only) |

**Total Effort:** ~1 day

**Dependencies:**

- ⏳ Phase 2 completion (experiment metrics must be available)
- ⏳ RFC-015 Phase 4 (CI integration) - for experiment metrics in PRs
- ⏳ RFC-041 Phase 3 (CI integration) - for benchmark metrics in PRs

**Core Invariant:** "Known regressions cannot reach main silently - developers are alerted proactively."

---

## Validation Principles

These principles guide all implementation work:

### Principle A — Validation ≠ Tests

Unit tests are necessary but not sufficient. Validation means answering: *"Can this step be misused, bypassed, or misunderstood—and does the system stop that?"*

### Principle B — Every Phase Has a Hard Invariant

Each phase introduces one non-negotiable architectural or product invariant. If the invariant can be violated (e.g., running an experiment without a baseline), the phase is not done.

### Principle C — Validation is Negative Testing

We validate by attempting to:

- Omit required fields
- Compare incompatible artifacts (cross-dataset)
- Run things in the wrong order or with "fuzzy" parameters

### The Ultimate Sanity Check

After each phase, ask: *"Can a careless but well-meaning engineer accidentally invalidate our results?"*

- **If YES** → The phase is not done. Add more validation logic.
- **If NO** → Move forward to the next step.

---

## Dependencies & Timeline

### Completed Dependencies

- ✅ RFC-016 Phase 2 (Provider params & fingerprinting) - **COMPLETE**
- ✅ RFC-015 Phase 1 (Experiment runner foundation) - **COMPLETE**
- ✅ RFC-041 Phase 0 (Dataset freezing & baselines) - **COMPLETE**

### Current Dependencies

- ⏳ RFC-016 Phase 3 (Evaluation extraction) - **BLOCKS** RFC-015 Phase 2
- ⏳ RFC-015 Phase 2 (Evaluation metrics) - **BLOCKS** RFC-041 Phase 1
- ⏳ RFC-015 Phase 4 (CI integration) - **BLOCKS** RFC-043 Phase 1
- ⏳ RFC-041 Phase 3 (CI integration) - **BLOCKS** RFC-043 Phase 1

### Timeline Summary

| Phase | Status | Timeline |
| --- | --- | --- |
| **Phase 1** | ✅ Complete | ~5 weeks |
| **Phase 2** | ✅ Mostly Complete | ~10 weeks done, ~2 weeks remaining (CI integration) |
| **Phase 3** | ⏳ Planned | ~1 week |
| **Total** | | **~18 weeks** (16 weeks done, 3 weeks remaining) |

---

## References

- **PRD-007:** AI Quality & Experimentation Platform
- **RFC-016:** Provider Modularization (`docs/rfc/RFC-016-modularization-for-ai-experiments.md`)
- **RFC-015:** AI Experiment Pipeline (`docs/rfc/RFC-015-ai-experiment-pipeline.md`)
- **RFC-041:** Benchmarking Framework (`docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md`)
- **RFC-043:** Automated Metrics Alerts (`docs/rfc/RFC-043-automated-metrics-alerts.md`)
- **GitHub Issues:**
  - [#303](https://github.com/chipi/podcast_scraper/issues/303) - Phase 1: Foundation (COMPLETED ✅)
  - [#304](https://github.com/chipi/podcast_scraper/issues/304) - Phase 2: Evaluation, Storage & CI (IN PROGRESS ⏳)
  - [#333](https://github.com/chipi/podcast_scraper/issues/333) - Phase 3: Proactive Alerting & Monitoring (PLANNED ⏳)
