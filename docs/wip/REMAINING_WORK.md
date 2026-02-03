# Remaining Implementation Work

**Date:** 2026-01-16
**Status:** Phase 2 In Progress - Core infrastructure complete, CI integration pending

---

## ✅ Completed (Phase 1-2)

### RFC-015: AI Experiment Pipeline
- ✅ Phase 1: Experiment Runner
- ✅ Phase 2: Evaluation Metrics Integration
- ✅ Phase 3: Storage & Comparison (visualization pending)

### RFC-041: Podcast ML Benchmarking Framework
- ✅ Phase 0: Dataset Freezing & Baselines
- ✅ Phase 1: Integration with RFC-015
- ✅ Phase 2: Benchmark Runner

### RFC-016: Provider Modularization
- ✅ Phase 2: Provider params & fingerprinting
- ✅ Phase 3: Evaluation infrastructure (ROUGE, BLEU, WER, semantic similarity implemented)

---

## ⏳ Remaining Work

### Priority 1: CI Integration (Critical Path)

#### RFC-015 Phase 4: CI Integration
**Estimate:** 1 week
**Status:** Pending

- [ ] **Smoke tests on PRs**
  - Add GitHub Actions workflow for PR smoke tests
  - Run fast subset of experiments (smoke datasets)
  - Fail PR if smoke tests fail

- [ ] **Nightly comprehensive experiments**
  - Add scheduled GitHub Actions workflow
  - Run full benchmark suite on `main` branch
  - Store results for historical tracking

- [ ] **Regression detection automation**
  - Integrate `RegressionChecker` into CI workflows
  - Block PRs on critical regressions
  - Generate regression reports

- [ ] **PR comments with experiment results**
  - Create `scripts/generate_pr_comment.py`
  - Post comparison table (PR vs baseline) as PR comment
  - Update existing comments (don't duplicate)

**Deliverables:**
- `.github/workflows/experiment-smoke.yml` - PR smoke tests
- `.github/workflows/experiment-nightly.yml` - Nightly full runs
- PR comments automatically posted with metric comparisons

---

#### RFC-041 Phase 3: CI Integration
**Estimate:** 1 week
**Status:** Pending

- [ ] **Smoke benchmarks on PRs**
  - Use `make benchmark SMOKE=1` in PR workflow
  - Fast subset (smoke datasets only)
  - Fail PR on regression

- [ ] **Nightly full benchmarks**
  - Use `make benchmark ALL=1` in scheduled workflow
  - Run on all datasets
  - Store results in `data/eval/benchmarks/`

- [ ] **PR comments with regression status**
  - Post benchmark summary as PR comment
  - Show per-dataset results
  - Highlight regressions

- [ ] **Block PRs on critical regressions**
  - Use `RegressionChecker` to detect violations
  - Fail CI job on error-level regressions
  - Allow warning-level regressions (non-blocking)

**Deliverables:**
- `.github/workflows/benchmark-smoke.yml` - PR smoke benchmarks
- `.github/workflows/benchmark-nightly.yml` - Nightly full benchmarks
- Automated regression blocking

---

### Priority 2: Documentation & Polish

#### RFC-041 Phase 4: Documentation & Polish
**Estimate:** 1 week
**Status:** Pending

- [ ] **Complete benchmarking framework documentation**
  - Update `docs/guides/EXPERIMENT_GUIDE.md` with CI integration
  - Document benchmark workflow
  - Add troubleshooting section

- [ ] **Examples and tutorials**
  - Create example experiment configs
  - Create example benchmark configs
  - Add step-by-step tutorials

- [ ] **User guide for creating new datasets**
  - Document dataset creation workflow
  - Best practices for dataset selection
  - Dataset versioning guidelines

- [ ] **User guide for creating new baselines**
  - Document baseline creation workflow
  - When to update baselines
  - Baseline promotion guidelines

**Deliverables:**
- Updated documentation
- Example configs in `data/eval/configs/examples/`
- User guides in `docs/guides/`

---

### Priority 3: Enhancements (Nice-to-Have)

#### RFC-015 Phase 3: Visualization (Pending)
**Estimate:** 1-2 days
**Status:** Low priority

- [ ] **Visualization of metrics over time**
  - Generate charts from historical data
  - Track metric trends
  - Compare multiple runs visually

**Deliverables:**
- `scripts/eval/generate_metrics_chart.py`
- HTML dashboard for metrics visualization

---

#### RFC-041 Phase 5: Enhancements & Iteration
**Estimate:** 1 week
**Status:** Low priority

- [ ] **Cost tracking and visualization**
  - Enhanced cost reporting
  - Cost trend analysis
  - Budget alerts

- [ ] **Optimize execution time**
  - Parallelize experiment runs
  - Cache model outputs
  - Optimize dataset loading

- [ ] **Additional metrics**
  - Perplexity scores
  - Coherence scores
  - Custom metric plugins

- [ ] **Baseline comparison dashboards**
  - Web-based dashboard
  - Interactive comparisons
  - Historical trend visualization

**Deliverables:**
- Performance optimizations
- Enhanced metrics
- Dashboard UI

---

### Priority 4: Phase 3 - Proactive Alerting

#### RFC-043: Automated Metrics Alerts
**Estimate:** 1 day
**Status:** Depends on CI integration
**Issue:** [#333](https://github.com/chipi/podcast_scraper/issues/333)

- [ ] **PR Comments (Priority 1)**
  - Generate comparison markdown
  - Post on every PR
  - Update existing comments

- [ ] **Webhook Alerts (Priority 2)**
  - Slack/Discord webhook support
  - Critical regression alerts
  - Main branch only

- [ ] **Testing & Refinement**
  - Test PR comments don't duplicate
  - Test webhook alerts
  - Refine thresholds

**Deliverables:**
- `scripts/generate_pr_comment.py`
- Webhook alerting (optional)
- Fully tested alerting system

**Dependencies:**
- ⏳ RFC-015 Phase 4 (CI integration)
- ⏳ RFC-041 Phase 3 (CI integration)

---

## Summary

### Critical Path (Must Have)
1. **RFC-015 Phase 4: CI Integration** (1 week)
2. **RFC-041 Phase 3: CI Integration** (1 week)

**Total:** ~2 weeks

### Important (Should Have)
3. **RFC-041 Phase 4: Documentation & Polish** (1 week)

**Total:** ~1 week

### Nice-to-Have (Can Wait)
4. **RFC-015 Phase 3: Visualization** (1-2 days)
5. **RFC-041 Phase 5: Enhancements** (1 week)
6. **RFC-043: Automated Metrics Alerts** (1 day, depends on CI)

**Total:** ~2 weeks

---

## Next Steps

1. **Start with CI Integration** (Priority 1)
   - Implement smoke tests for PRs
   - Add nightly comprehensive runs
   - Integrate regression detection
   - Add PR comment automation

2. **Then Documentation** (Priority 2)
   - Complete user guides
   - Add examples and tutorials
   - Document workflows

3. **Finally Enhancements** (Priority 3)
   - Visualization
   - Performance optimizations
   - Additional metrics
   - Alerting system

---

## Current Status

**Phase 1:** ✅ Complete (Foundation)
**Phase 2:** ⏳ In Progress (~70% complete)
- ✅ Evaluation metrics
- ✅ Storage & comparison
- ✅ Benchmark runner
- ⏳ CI integration (next)
- ⏳ Documentation

**Phase 3:** ⏳ Planned (depends on Phase 2)
