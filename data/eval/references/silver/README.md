# Silver references (eval)

Silver references are **promoted experiment runs**: a directory named by `reference_id`
containing at least `predictions.jsonl` (and often `metrics.json`, `baseline.json`).

## GI and KG vs-reference scoring

For `grounded_insights` and `knowledge_graph`, `compute_*_vs_reference_metrics` accepts:

- **Per-episode JSON files** (`{episode_id}.json`) with the same shape as `output.gil` / `output.kg`
  (gold-style layout), or
- **Silver layout**: `predictions.jsonl` where each line has `episode_id` and
  `output.gil` / `output.kg` — the same format as a normal eval run.

Point `references:` in your experiment YAML at the silver run id; `find_reference_path`
resolves `data/eval/references/silver/<reference_id>/`.

## Promoting a run to silver

Copy or symlink a completed run directory under `data/eval/references/silver/<new_id>/`,
or document your promotion command in the project eval guide. Keep dataset coverage
aligned with the experiment `dataset_id` when comparing counts.
