# Reference: silver_sonnet46_smoke_v2

**Promoted from:** `silver_candidate_anthropic_claudesonnet46_smoke_v2_paragraph`
**Promoted at:** 2026-06-07T12:01:50.802329Z
**Role:** Reference (used as truth for evaluation)
**Dataset:** curated_5feeds_smoke_v2
**Reference Quality:** silver

## Purpose

v2 silver re-selection (#903) — smoke × paragraph cell of the v1↔v2 winner
re-check. Sonnet 4.6 wins both head-to-heads on the regenerated v2 smoke
content:

- vs OpenAI GPT-4o: **5–0–0** (sweep)
- vs OpenAI GPT-5.4: **3–0–2** (Sonnet wins, GPT-5.4 never beats)

Same provider as the v1 silver (`silver_sonnet46_smoke_v1`) — v2's content
shape (sponsor blocks, recurring guests, position arcs) did not shift judge
preference. See `docs/guides/eval-reports/EVAL_FIXTURES_V2_PIPELINE_2026_06_07.md`
for the full 2×2 matrix (smoke/benchmark × paragraph/bullets) and the silver
naming convention (`_kg_v2` suffix for v2-content benchmark cells to avoid
collision with the autoresearch-v2-framework silvers).

## Reproduction

```bash
PYTHONPATH=. make experiment-run \
  CONFIG=data/eval/configs/silver_selection/silver_candidate_anthropic_claudesonnet46_smoke_v2_paragraph.yaml
# Pairwise judge: scripts/eval/score/pairwise_judge.py
# (judge keys from .env.autoresearch)
```

## Usage

This reference is used as "truth" for evaluation metrics (e.g., ROUGE).
It is:

- Not required for experiments (experiments can run without references)
- Cannot block CI (references are informational)
- Rarely updated (only when truth changes)
- Used for absolute quality assessment

## Artifacts

- `predictions.jsonl` - Reference outputs for all episodes
- `metrics.json` - Aggregate metrics
- `fingerprint.json` - System fingerprint (reproducibility)
- `baseline.json` - Metadata (kept for compatibility)
- `config.yaml` - Experiment config used (if provided)
