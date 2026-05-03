# GIL Evidence Bundling — Operator Runbook (#698)

This directory contains the autoresearch scaffold for #698. The goal is to
validate that bundled ``extract_quotes`` (Layer A) and bundled
``score_entailment`` (Layer B) cut LLM calls per episode without dropping
GI insight coverage below the published Gemini baseline (80% on
``curated_5feeds_benchmark_v2``).

See [`program.md`](program.md) for the full hypothesis, gates, and
references. This file is the **how-to-run** runbook.

## Contents

```text
gil_evidence_bundling/
├── README.md                                         # this file
├── program.md                                        # hypothesis + gates + silver refs
├── eval/score.py                                     # diff-mode scorer (internal)
├── experiments/
│   ├── baseline_staged.yaml                          # reproduces published 80% Gemini baseline
│   ├── bundled_a_only.yaml
│   ├── bundled_b_only.yaml
│   └── bundled_ab.yaml                               # headline cell
└── results.tsv                                       # rolling result log (empty)
```

## Pre-flight (one-time)

1. Confirm Gemini API key is in ``.env`` (``GEMINI_API_KEY=...``).
2. Confirm dataset + silver references exist:

   ```bash
   ls data/eval/materialized/curated_5feeds_benchmark_v2/ | head
   ls data/eval/references/silver/silver_sonnet46_gi_multiquote_benchmark_v2/predictions.jsonl
   ```

3. Estimated cost: 4 cells × 5 episodes × ``gemini-2.5-flash-lite`` ≈
   **$1-3 total**.

## Run a cell

Each cell runs once via ``scripts/eval/experiment/run_experiment.py``:

```bash
.venv/bin/python scripts/eval/experiment/run_experiment.py \
    autoresearch/gil_evidence_bundling/experiments/baseline_staged.yaml \
    --reference silver_sonnet46_gi_multiquote_benchmark_v2 \
    --force --log-level INFO
```

Repeat for ``bundled_a_only.yaml`` / ``bundled_b_only.yaml`` /
``bundled_ab.yaml``. Each run lands artifacts at
``data/eval/runs/<run_id>/`` (run_id matches the YAML's ``id``).

## Score a cell

Two scorers, both required.

**Primary quality gate — vs silver:**

```bash
.venv/bin/python scripts/eval/score/score_gi_insight_coverage.py \
    --run-id <run_id> \
    --silver silver_sonnet46_gi_multiquote_benchmark_v2 \
    --dataset curated_5feeds_benchmark_v2
```

This is the canonical comparison that produced the published 80%
baseline. Champion gate: GI insight coverage ≥75% (within 5pp of the
80% staged target).

**Internal cost / latency scalar — variant vs staged:**

```bash
.venv/bin/python autoresearch/gil_evidence_bundling/eval/score.py \
    --baseline data/eval/runs/gil_bundling_baseline_staged_v1 \
    --variant  data/eval/runs/gil_bundling_bundled_ab_v1
```

This emits a single scalar (higher = better; champion ≥0.30) and
exits non-zero on any internal-quality gate violation:

- Internal grounding rate regression > 5pp absolute
- Bundled fallback rate > 20%
- Input tokens per episode > 50k

Append a row to ``results.tsv`` after each successful cell.

## Recommended sequence

```bash
# 1. Reproduce the published baseline (must hit ~80% GI coverage)
./<run_experiment.py> baseline_staged.yaml --reference silver_sonnet46_gi_multiquote_benchmark_v2 --force
./<score_gi_insight_coverage.py> --run-id gil_bundling_baseline_staged_v1 \
    --silver silver_sonnet46_gi_multiquote_benchmark_v2 \
    --dataset curated_5feeds_benchmark_v2

# 2. Run the 3 bundled variants (only if step 1 reproduced ≥77% baseline)
./<run_experiment.py> bundled_a_only.yaml ...
./<run_experiment.py> bundled_b_only.yaml ...
./<run_experiment.py> bundled_ab.yaml ...

# 3. Score each variant against silver + custom score.py
# 4. Append to results.tsv
# 5. If a champion qualifies (GI cov ≥75%, cost ↓≥30%, fallback ≤20%):
#    inspect 1 artifact by eye, then update PR #711 with results.
```

## What this does NOT do

- **No paid LLM calls in this scaffold itself.** Each cell run + scoring
  is a manual operator action (with budget approval). The scaffold only
  defines configs, helpers, and quality gates.
- **No CI integration.** Matrix is operator-driven, not a CI job.
- **No automatic rollout.** Even if a champion qualifies, flipping a
  profile default (e.g. ``cloud_thin: gil_evidence_quote_mode: bundled``)
  is a separate decision in a follow-up PR — not in #711.
