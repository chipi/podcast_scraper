# Optimization Workflow Guide

How to investigate and solve performance or cost problems in the pipeline.
This process is data-driven: every decision is backed by measured numbers,
and all artifacts live in dedicated folders so you can always trace back
what was measured, when, and why.

**Related guides** (read first if unfamiliar with the tooling):

- [Experiment Guide](EXPERIMENT_GUIDE.md) -- eval system: datasets,
  baselines, configs, scoring, references, silver/gold promotion
- [Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md) -- frozen
  profiles (`make profile-freeze`), psutil sampling, stage attribution,
  `stage_truth.json` (RFC-064)
- [Experiment Guide](EXPERIMENT_GUIDE.md) -- what each metric means
  (ROUGE, gates, latency, cost), `metrics.json` schema, vs_reference
  scoring (see Step 4: Evaluate Results)
- [Live Pipeline Monitor](LIVE_PIPELINE_MONITOR.md) -- real-time stage /
  RSS / CPU view during a dev run (`--monitor`, RFC-065)
- [Performance Guide](PERFORMANCE.md) -- general performance
  considerations, cache behavior, audio preprocessing costs
- [AI Provider Comparison Guide](AI_PROVIDER_COMPARISON_GUIDE.md) --
  provider decision matrices, cost analysis, eval reports
- [Experiment Guide](EXPERIMENT_GUIDE.md) -- how runs become baselines,
  silvers, or app defaults (see "Step 5: Promote a run")

---

## The process

### 1. Start from a GitHub issue

The GitHub issue is the **source of truth** for the optimization. It
captures the objective, constraints, product decisions, and all results.

- Open (or find) an issue that describes the problem or opportunity.
- Use the issue body for scope, constraints, and success criteria.
- Post updates as comments: hypothesis results, baseline numbers,
  experiment outcomes, and the final decision.

### 2. Analyze and make hypotheses

Before writing any code, understand where the time and money go.

- Read existing profiles (`data/profiles/`) for wall time per stage --
  see [Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md) for how
  to interpret them.
- Read existing eval runs (`data/eval/runs/`) for quality and cost
  numbers -- see [Experiment Guide](EXPERIMENT_GUIDE.md) (Step 4) for
  what each metric means.
- Optionally use [Live Pipeline Monitor](LIVE_PIPELINE_MONITOR.md)
  (`--monitor`) to watch a single run in real time and spot where the
  pipeline stalls.
- Check [AI Provider Comparison Guide](AI_PROVIDER_COMPARISON_GUIDE.md)
  for existing cross-provider benchmarks that may already answer your
  question.
- Write down testable hypotheses (e.g. "cleaning + summarization
  dominate sequential cost per episode").
- Each hypothesis needs a **how to test** (which metric, which tool)
  and a **what it means if true** (which lever to pull).
- Post hypotheses on the GitHub issue.

### 3. Measure the baseline (eval + profile)

Create a **dedicated folder** inside `data/eval/` and `data/profiles/`
for this work (e.g. `data/eval/issue-NNN/`, `data/profiles/issue-NNN/`).
Everything for this optimization lives there: configs, outputs, READMEs.

**Quality + cost baseline (eval):**

Use the experiment system described in [Experiment Guide](EXPERIMENT_GUIDE.md).
Create an experiment config YAML in your dedicated folder and run it with
`--cost-report` to capture token counts:

```bash
.venv/bin/python3 scripts/eval/experiment/run_experiment.py \
  data/eval/issue-NNN/my_config.yaml \
  --reference silver_sonnet46_smoke_v1 --cost-report
```

This produces `metrics.json` (ROUGE, gates, latency -- see
[Experiment Guide](EXPERIMENT_GUIDE.md) Step 4) and `eval_pipeline_metrics.json`
(tokens, calls, estimated USD per stage).

**Performance baseline (profile):**

Use the profiling system described in
[Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md). Create a
capture config in your dedicated folder and run a freeze:

```bash
.venv/bin/python3 scripts/eval/profile/freeze_profile.py \
  --version issue-NNN-staged \
  --pipeline-config data/profiles/issue-NNN/capture_config.yaml \
  --dataset-id e2e_podcast1_mtb_n2 \
  --output data/profiles/issue-NNN/issue-NNN-staged.yaml
```

This produces a frozen YAML profile (per-stage wall time, CPU%, RSS) and
a companion `stage_truth.json` (token counts, per-episode breakdowns,
parallelism ratio).

**What to capture in the baseline:**

| Dimension | Source | Key metrics |
| --- | --- | --- |
| Quality | `metrics.json` | ROUGE-L, gates, embedding cosine |
| Cost | `eval_pipeline_metrics.json` | Tokens per stage, calls, est. USD |
| Time | Profile YAML + stage_truth | Wall time per stage, CPU%, parallelism |

### 4. Validate hypotheses with numbers

Compare the measured data against each hypothesis. For every hypothesis,
state: confirmed or rejected, with the specific numbers.

- Post the validation results on the GitHub issue.
- Update the experiment plan (`docs/wip/`) with a Problem Definition
  section that combines quality, cost, and time baselines.
- If a hypothesis is rejected, re-analyze before proceeding.

### 5. Implement the solution

Now that the problem is defined with data, implement the fix behind an
opt-in flag or config (so the existing path remains the default).

- Keep the implementation focused on the lever identified in step 4.
- Do not change multiple things at once -- one lever per experiment so
  diffs stay interpretable.

### 6. Validate against the baseline

Re-run the same eval and profile from step 3, but with the new code
path enabled. Compare against the baseline:

```bash
# Eval (same reference, same dataset -- see Experiment Guide)
.venv/bin/python3 scripts/eval/experiment/run_experiment.py \
  data/eval/issue-NNN/my_optimized_config.yaml \
  --reference silver_sonnet46_smoke_v1 --cost-report

# Profile (same capture setup -- see Performance Profile Guide)
.venv/bin/python3 scripts/eval/profile/freeze_profile.py \
  --version issue-NNN-optimized \
  --pipeline-config data/profiles/issue-NNN/capture_optimized.yaml \
  --dataset-id e2e_podcast1_mtb_n2 \
  --output data/profiles/issue-NNN/issue-NNN-optimized.yaml

# Diff the two profiles
.venv/bin/python3 scripts/eval/profile/diff_profiles.py \
  data/profiles/issue-NNN/issue-NNN-staged.yaml \
  data/profiles/issue-NNN/issue-NNN-optimized.yaml
```

**Decision gate:** Compare quality, cost, and time against the baseline
thresholds defined in step 1. Post the results on the GitHub issue and
make the ship/iterate/reject decision.

If the optimization passes, consider whether the new config should be
promoted to a baseline or app default -- see
[Experiment Guide](EXPERIMENT_GUIDE.md) (Step 5: Promote a run).

---

## Folder layout

```text
data/eval/issue-NNN/
  README.md                    # What each config measures, commands
  baseline_config.yaml         # Eval config for the current path
  optimized_config.yaml        # Eval config for the new path

data/profiles/issue-NNN/
  README.md                    # What each profile measures, commands
  capture_baseline.yaml        # Pipeline config for baseline profile
  capture_optimized.yaml       # Pipeline config for optimized profile
  issue-NNN-staged.yaml        # Frozen baseline profile
  issue-NNN-staged.stage_truth.json
  issue-NNN-optimized.yaml     # Frozen optimized profile
  issue-NNN-optimized.stage_truth.json
```

---

## Checklist

- [ ] GitHub issue exists with objective and constraints
- [ ] Hypotheses written and posted on the issue
- [ ] Dedicated folders created in `data/eval/` and `data/profiles/`
- [ ] Quality baseline measured (eval with `--cost-report`)
- [ ] Performance baseline measured (profile freeze)
- [ ] Hypotheses validated with numbers, posted on the issue
- [ ] Problem definition documented (quality + cost + time)
- [ ] Solution implemented behind opt-in
- [ ] Solution validated against baseline (eval + profile + diff)
- [ ] Decision posted on the GitHub issue

---

## Tooling quick reference

| What you need | Tool / guide |
| --- | --- |
| Run an experiment (quality + cost) | [Experiment Guide](EXPERIMENT_GUIDE.md) |
| Understand metrics (ROUGE, gates, cost) | [Experiment Guide](EXPERIMENT_GUIDE.md) (Step 4) |
| Capture a frozen profile (wall time, CPU%, RSS) | [Performance Profile Guide](PERFORMANCE_PROFILE_GUIDE.md) |
| Watch a run in real time (stage, RSS, CPU) | [Live Pipeline Monitor](LIVE_PIPELINE_MONITOR.md) |
| Compare providers (cost, quality, speed) | [AI Provider Comparison Guide](AI_PROVIDER_COMPARISON_GUIDE.md) |
| General performance tips (caching, audio) | [Performance Guide](PERFORMANCE.md) |
| Promote a run to baseline or silver | [Experiment Guide](EXPERIMENT_GUIDE.md) (Step 5) |
| Eval system mechanics (datasets, refs) | [Experiment Guide](EXPERIMENT_GUIDE.md) |
