# GIL Evidence Bundling — Operator Runbook (#698)

This directory contains the autoresearch scaffold for #698. The goal is to
validate that bundled ``extract_quotes`` (Layer A) and bundled
``score_entailment`` (Layer B) cut LLM calls per episode without dropping
grounding quality below operator-acceptable levels.

See [`program.md`](program.md) for the full hypothesis, gates, and score
formula. This file is the **how-to-run** runbook.

## Contents

```text
gil_evidence_bundling/
├── README.md                                         # this file
├── program.md                                        # hypothesis + gates
├── eval/score.py                                     # diff-mode scorer
├── experiments/
│   ├── baseline_staged.yaml                          # reference cell
│   ├── bundled_a_only.yaml
│   ├── bundled_b_only.yaml
│   └── bundled_ab.yaml                               # headline win
└── results.tsv                                       # rolling result log (empty)
```

## Pre-flight (one-time)

1. Confirm Gemini API key is in ``.env`` (``GEMINI_API_KEY=...``).
2. Confirm the dev dataset exists:

   ```bash
   ls data/eval/materialized/curated_5feeds_dev_v1/ | head
   ```

3. Estimate cost: 4 cells × 10 episodes × Gemini Flash Lite ≈ **$2-5 total**.

## Run a cell

The actual experiment runner is the existing
``scripts/eval/run_experiment.py``. Each cell runs once:

```bash
.venv/bin/python scripts/eval/run_experiment.py \
    autoresearch/gil_evidence_bundling/experiments/baseline_staged.yaml \
    --reference silver_gpt4o_smoke_bullets_v1 \
    --force --log-level INFO

# Repeat for bundled_a_only.yaml / bundled_b_only.yaml / bundled_ab.yaml.
```

Each run lands artifacts at ``data/eval/runs/<run_id>/`` — note the
``predictions.jsonl`` and ``metrics.json`` paths printed at the end.

## Score a variant against the baseline

```bash
.venv/bin/python autoresearch/gil_evidence_bundling/eval/score.py \
    --baseline data/eval/runs/<baseline_run_id> \
    --variant  data/eval/runs/<variant_run_id>
```

The script prints a single scalar (higher = better; champion threshold
≥ 0.30) and exits non-zero on any quality-gate violation:

- Grounding regression > 5pp absolute
- Bundled fallback rate > 20%
- Input tokens per episode > 50k

Append the result to ``results.tsv`` manually after a successful run.

## Held-out validation (only after dev champion picked)

```yaml
# Switch the dataset_id in the experiment YAML for the chosen champion:
data:
  dataset_id: curated_5feeds_benchmark_v2
  max_episodes: 5
```

Held-out ≥ dev × 0.95 confirms the champion generalises (RFC-073 dev /
held-out rule).

## What this does NOT do

- **No paid LLM calls in this scaffold itself.** ``score.py`` only reads
  existing ``metrics.json`` outputs. The pipeline is run separately.
- **No CI integration.** This runs on the operator's machine with budget
  approval; it's not a recurring CI job.
- **No automatic rollout.** Even if a champion emerges, flipping a profile
  default (e.g. setting ``gil_evidence_quote_mode: bundled`` in
  ``cloud_thin.yaml``) is a separate decision tracked in the PR review.
