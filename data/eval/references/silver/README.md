# Silver references (eval)

Silver references are **promoted experiment runs**: high-quality outputs from a flagship
model (currently Claude Sonnet 4.6) used as the measuring stick for all other providers.

## Active silver references

| Reference ID | Model | Dataset | Episodes | Format | Status |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `silver_sonnet46_smoke_v1` | Claude Sonnet 4.6 | smoke | 5 | prose paragraphs | **Active — paragraph smoke** |
| `silver_sonnet46_smoke_bullets_v1` | Claude Sonnet 4.6 | smoke | 5 | JSON bullets | **Active — bullets smoke** |
| `silver_sonnet46_benchmark_v1` | Claude Sonnet 4.6 | benchmark | 10 | prose paragraphs | **Active — paragraph benchmark** |
| `silver_sonnet46_benchmark_bullets_v1` | Claude Sonnet 4.6 | benchmark | 10 | JSON bullets | **Active — bullets benchmark** |
| `silver_gpt4o_smoke_v1` | GPT-4o | smoke | 5 | prose paragraphs | Archived (superseded by sonnet46) |
| `silver_gpt4o_smoke_bullets_v1` | GPT-4o | smoke | 5 | JSON bullets | Archived (superseded by sonnet46) |
| `silver_gpt4o_benchmark_v1` | GPT-4o | benchmark | 10 | prose paragraphs | Archived (superseded by sonnet46) |

**Rule:** Always use `silver_sonnet46_*` for new experiments and comparisons. The `silver_gpt4o_*`
references are retained for historical traceability only — do not use them for new runs.

## The two dimensions

Every silver reference lives at the intersection of two dimensions:

```text
           smoke (5 eps)         benchmark (10 eps)
           ─────────────────     ──────────────────
paragraph  silver_sonnet46_      silver_sonnet46_
           smoke_v1              benchmark_v1

bullets    silver_sonnet46_      silver_sonnet46_
           smoke_bullets_v1      benchmark_bullets_v1
```

- **smoke**: used in the prompt tuning loop (fast, cheap, ~5 min/run)
- **benchmark**: used for production-quality numbers in ADRs and eval reports (10 eps)
- **paragraph**: prose summary via `long_v1.j2` template
- **bullets**: JSON bullet array via `bullets_json_v1.j2` template

## When to create a new silver reference

Create a new silver reference when:

1. **A better frontier model becomes available** — run pairwise judge to confirm quality
   improvement, then promote (see `configs/README.md` § Silver reference selection).
2. **A new output format is introduced** — e.g. a new template style requires its own reference.
3. **A new dataset is created** — each dataset needs its own silver coverage.

**Do not** create a new silver reference just to change the reference model within the same
output format. The reference model should be the best available at selection time and should
not change frequently.

## When to re-run all provider experiments

Re-run all providers against the current silver references when:

1. **A new silver reference replaces an old one** — all numbers from the old reference are
   incomparable to new numbers; a full re-run is required to restore comparability.
2. **A shared prompt template changes** (e.g. `bullets_json_v1.j2`) — re-run all providers
   using that template to get updated numbers with the new template.
3. **A new provider/model is added** — run all 4 config types (smoke paragraph, smoke bullets,
   benchmark paragraph, benchmark bullets) before publishing numbers.

After a full re-run, generate an updated eval report:

```bash
# Run all providers (see data/eval/configs/README.md § Eval run matrix)
make experiment-run CONFIG=... REFERENCE=silver_sonnet46_smoke_v1

# Generate comparison report across all runs
make runs-compare RUN_IDS=<comma-separated> REFERENCE_ID=silver_sonnet46_smoke_v1
```

## Promoting a run to silver

```bash
python scripts/eval/promote_run.py \
  --run-id <run_id> \
  --as reference \
  --promoted-id silver_sonnet46_<scope>_v1 \
  --reason "<why this model, why now>"
```

Keep dataset coverage aligned — a smoke reference must cover exactly the smoke dataset;
a benchmark reference must cover exactly the benchmark dataset.

## GI and KG vs-reference scoring

For `grounded_insights` and `knowledge_graph`, `compute_*_vs_reference_metrics` accepts:

- **Per-episode JSON files** (`{episode_id}.json`) with the same shape as `output.gil` / `output.kg`
  (gold-style layout), or
- **Silver layout**: `predictions.jsonl` where each line has `episode_id` and
  `output.gil` / `output.kg` — the same format as a normal eval run.

Point `references:` in your experiment YAML at the silver run id; `find_reference_path`
resolves `data/eval/references/silver/<reference_id>/`.
