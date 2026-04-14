# Autoresearch

Thin automation layer on top of `scripts/eval/run_experiment.py` and
`src/podcast_scraper/evaluation/`.

**Current framework: v2** — see [RFC-073](../docs/rfc/RFC-073-autoresearch-v2-framework.md).
**Original framework: v1** — see [RFC-057](../docs/rfc/RFC-057-autoresearch-optimization-loop.md)
(closed via [ADR-073](../docs/adr/ADR-073-rfc057-autoresearch-closure.md); Track B still v1).

## v2 headline: dev + held-out, not smoke + benchmark

| Tier | Dataset | Size | Role | Touch during iteration? |
| ---- | ------- | ---- | ---- | ----------------------- |
| **Dev** | `curated_5feeds_dev_v1` | 10 ep (e01+e02) | Ratchet iteration | Yes (every experiment) |
| **Held-out** | `curated_5feeds_benchmark_v2` | 5 ep (e03, ~32 min) | Champion validation | **Never** — once per committed champion |

`curated_5feeds_smoke_v1` and `curated_5feeds_benchmark_v1` are preserved **as-is** for reproducibility of
v1 runs, baselines, and silvers. Do not modify. New work uses the v2 datasets.

See [JUDGING.md](JUDGING.md) for the dual-judge system (rubric, fraction-based contestation, prose
extraction, seed plumbing).

---

## Current champions (OpenAI v2, held-out scores)

| Track | Approach | Config | Silver | Held-out final | ROUGE-L |
| ----- | -------- | ------ | ------ | -------------- | ------- |
| Bullets | **Non-bundled** (winner) | `autoresearch_prompt_openai_benchmark_bullets_v2.yaml` | `silver_sonnet46_benchmark_v2_bullets` | **0.566** | 39.6% |
| Bullets | Bundled | `autoresearch_prompt_openai_bundled_benchmark_bullets_v2.yaml` | `silver_sonnet46_benchmark_v2_bullets` | 0.505 | 33.2% |
| Paragraph | **Non-bundled** (winner) | `autoresearch_prompt_openai_benchmark_paragraph_v2.yaml` | `silver_sonnet46_benchmark_v2_paragraph` | **0.481** | 31.7% |
| Paragraph | Bundled | `autoresearch_prompt_openai_bundled_benchmark_paragraph_v2.yaml` | `silver_sonnet46_benchmark_v2_paragraph` | 0.469 | 29.5% |

Full comparison: [`openai_v2_comparison_2026-04-14.md`](openai_v2_comparison_2026-04-14.md).

---

## Running the ratchet (v2)

### Dev iteration (the normal loop)

Run the orchestrator directly with explicit `CONFIG` and `REFERENCE`:

```bash
# Bundled bullets ratchet (gpt-4o, 10 ep dev, scoring_output_field: bullets)
make autoresearch-score-bundled \
  CONFIG=data/eval/configs/summarization_bullets/autoresearch_prompt_openai_bundled_dev_bullets_v2.yaml \
  REFERENCE=silver_sonnet46_dev_v1_bullets

# Non-bundled bullets ratchet (gpt-4o, 10 ep dev)
AUTORESEARCH_EVAL_N=10 .venv/bin/python autoresearch/bundled_prompt_tuning/eval/score.py \
  --config data/eval/configs/summarization_bullets/autoresearch_prompt_openai_dev_bullets_v2.yaml \
  --reference silver_sonnet46_dev_v1_bullets

# Bundled paragraph ratchet
AUTORESEARCH_EVAL_N=10 .venv/bin/python autoresearch/bundled_prompt_tuning/eval/score.py \
  --config data/eval/configs/summarization/autoresearch_prompt_openai_bundled_dev_paragraph_v2.yaml \
  --reference silver_sonnet46_dev_v1_paragraph

# Non-bundled paragraph ratchet
AUTORESEARCH_EVAL_N=10 .venv/bin/python autoresearch/bundled_prompt_tuning/eval/score.py \
  --config data/eval/configs/summarization/autoresearch_prompt_openai_dev_paragraph_v2.yaml \
  --reference silver_sonnet46_dev_v1_paragraph
```

Output: one `final` scalar on stdout (higher is better); `rougeL_f1`, `judge_mean`,
`contested`, per-episode contestation count on stderr.

**Accept rule** (dev): `+1%` delta vs the last committed champion. For prompts shared between
tracks (bundled system+user cover both bullets and paragraph), apply dual-metric:
target track `+1%` AND the other track `−1%` or better.

### Held-out validation (after accepting a champion)

Run **once** per committed champion — never iterate against held-out:

```bash
# Bundled bullets held-out
AUTORESEARCH_EVAL_N=5 .venv/bin/python autoresearch/bundled_prompt_tuning/eval/score.py \
  --config data/eval/configs/summarization_bullets/autoresearch_prompt_openai_bundled_benchmark_bullets_v2.yaml \
  --reference silver_sonnet46_benchmark_v2_bullets

# Non-bundled paragraph held-out (example)
AUTORESEARCH_EVAL_N=5 .venv/bin/python autoresearch/bundled_prompt_tuning/eval/score.py \
  --config data/eval/configs/summarization/autoresearch_prompt_openai_benchmark_paragraph_v2.yaml \
  --reference silver_sonnet46_benchmark_v2_paragraph
```

If held-out score is within `dev ± noise floor` (~±3-5%), the champion generalises — commit it.
If held-out collapses, the change was overfitting dev — revert.

### Dry re-score (ROUGE only, no judge API calls)

Useful to re-check ROUGE from existing `predictions.jsonl` without re-running summarization:

```bash
make autoresearch-score-bundled \
  CONFIG=... REFERENCE=... DRY_RUN=1
```

Not sufficient to evaluate a *new* prompt — judges don't run. Run the full harness after template changes.

---

## Allowlisted files for prompt tuning

### Bundled mode

| File | Role |
| ---- | ---- |
| `src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_system_v1.j2` | Bundled system prompt (OpenAI-specific) |
| `src/podcast_scraper/prompts/openai/summarization/bundled_clean_summary_user_v1.j2` | Bundled user prompt (OpenAI-specific) |

### Non-bundled mode

| File | Role |
| ---- | ---- |
| `src/podcast_scraper/prompts/shared/summarization/system_bullets_v1.j2` | Shared bullets system prompt |
| `src/podcast_scraper/prompts/shared/summarization/bullets_json_v1.j2` | Shared bullets user prompt |
| `src/podcast_scraper/prompts/openai/summarization/system_v1.j2` | OpenAI paragraph system prompt |
| `src/podcast_scraper/prompts/openai/summarization/long_v1.j2` | OpenAI paragraph user prompt |

## Files you must NOT edit during a ratchet

| Path | Reason |
| ---- | ------ |
| `autoresearch/bundled_prompt_tuning/eval/score.py` | Immutable harness |
| `autoresearch/bundled_prompt_tuning/eval/rubric.md` | Immutable during a run |
| `autoresearch/bundled_prompt_tuning/eval/judge_config.yaml` | Pinned judge models (human-only between rounds) |
| `data/eval/**` | Gold inputs, references, datasets — read-only |
| Benchmark v2 configs (`*_benchmark_bullets_v2.yaml`, `*_benchmark_paragraph_v2.yaml`) | Held-out — run once per champion only |

---

## Environment

The score CLI loads `.env` then `.env.autoresearch` (optional overrides) from the repo root. Required
for a full run:

- `AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY` — OpenAI for `run_experiment` summarization
- `AUTORESEARCH_JUDGE_OPENAI_API_KEY` — judge A
- `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY` — judge B
- Optional: `AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1` — fall back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`

All v2 ratchet configs set `params.seed: 42` — OpenAI's approximately-deterministic sampling. See
[JUDGING.md](JUDGING.md) §Seed & determinism for the empirical characterisation.

---

## Prerequisites

- Materialized transcripts (both datasets):
  - `data/eval/materialized/curated_5feeds_dev_v1/` (10 ep) — `make dataset-materialize DATASET_ID=curated_5feeds_dev_v1`
  - `data/eval/materialized/curated_5feeds_benchmark_v2/` (5 ep) — `make dataset-materialize DATASET_ID=curated_5feeds_benchmark_v2`
- Silver references:
  - `silver_sonnet46_dev_v1_bullets`, `silver_sonnet46_dev_v1_paragraph`
  - `silver_sonnet46_benchmark_v2_bullets`, `silver_sonnet46_benchmark_v2_paragraph`
- All seeded automatically by `silver_candidate_sonnet46_<dataset>_<track>.yaml` + `promote_run.py`.

---

## Git workflow

1. Use a dedicated branch, e.g. `autoresearch/<short-tag>` — not `main`.
2. Before each experiment: state a one-sentence hypothesis (in chat or results TSV notes).
3. After each ratchet run:
   - **Improved** vs last committed champion: `git add` only allowlisted files + results TSV →
     commit with `[autoresearch-<track>] <exp-id>: <hypothesis> (+X%)`.
   - **Not improved**: `git checkout HEAD -- <template paths>`.
4. Append one row per experiment to the relevant results TSV.
5. After accepting a champion, run held-out validation and record the held-out score in the TSV
   as a separate `held-out` row.

### Results TSVs

| Track | File |
| ----- | ---- |
| Bundled bullets (dev v2) | `autoresearch/bundled_prompt_tuning/results/results_openai_r1.tsv` (rolling) |
| Bundled paragraph (dev v2) | `autoresearch/bundled_prompt_tuning/results/results_openai_paragraph_r1.tsv` |
| Non-bundled bullets (dev v2) | `autoresearch/prompt_tuning/results/results_openai_nonbundled_bullets_dev_v2.tsv` |
| Non-bundled paragraph (dev v2) | `autoresearch/prompt_tuning/results/results_openai_nonbundled_paragraph_dev_v2.tsv` |

TSV columns: `experiment_id  score  delta  status  notes  judge_a_model  judge_b_model  rubric_hash  eval_dataset_ref`.

---

## Next work (deferred)

1. **Multi-run averaging** (N=3) to eliminate OpenAI API non-determinism as a signal noise source. See [RFC-073](../docs/rfc/RFC-073-autoresearch-v2-framework.md) §Future Work.
2. **Other providers** — Anthropic, Gemini, Mistral under v2. Port OpenAI champion prompts as starting template, re-use Sonnet 4.6 silvers.
3. **Grow held-out dataset** from 5 episodes if champion discrimination becomes the limiting factor.

For any of the above, start by reading [RFC-073](../docs/rfc/RFC-073-autoresearch-v2-framework.md) §Replication for other providers.
