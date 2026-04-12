# Issue #477 — LLM bundle experiment (clean + summary + bullets)

**Status:** WIP — implementation landed; ship only after measured smoke/benchmark runs.

**GitHub:** [Issue #477](https://github.com/chipi/podcast_scraper/issues/477)

## What shipped in code

- `Config.llm_pipeline_mode`: `staged` (default) | `bundled`
- `Config.llm_bundled_max_output_tokens` (default `16384`)
- `summarize_bundled()` on OpenAI, Anthropic, Gemini providers
- Workflow: pattern pre-clean, then one structured JSON completion; fallback to staged on failure
- `Metrics`: `llm_bundled_*`, `llm_bundled_fallback_to_staged_count`,
  `total_episode_estimated_cost_usd`, `total_episode_prompt_tokens`,
  `total_episode_completion_tokens`, `llm_token_totals_by_stage`
- Eval: `--cost-report` on `scripts/eval/run_experiment.py` writes `eval_pipeline_metrics.json`
- Eval configs (canonical tree: `data/eval/issue-477/`, see `README.md` there):
  - `data/eval/issue-477/experiment_openai_gpt4o_smoke_bullets_v1.yaml`
  - `data/eval/issue-477/experiment_openai_gpt4o_smoke_bundled_v1.yaml`
  - `data/eval/issue-477/experiment_openai_gpt4o_benchmark_bundled_v1.yaml`
  - `data/eval/issue-477/autoresearch_prompt_openai_smoke_bundled_v1.yaml`
  - Track B (paragraph smoke, same model tier as bullets): `data/eval/issue-477/experiment_openai_gpt4o_smoke_paragraph_v1.yaml`

## How to run (OpenAI smoke)

Staged baseline (existing):

```bash
make experiment-run \
  CONFIG=data/eval/issue-477/experiment_openai_gpt4o_smoke_bullets_v1.yaml \
  REFERENCE=silver_gpt4o_smoke_bullets_v1
```

Bundled candidate:

```bash
make experiment-run \
  CONFIG=data/eval/issue-477/experiment_openai_gpt4o_smoke_bundled_v1.yaml \
  REFERENCE=silver_gpt4o_smoke_bullets_v1
```

With token snapshot:

```bash
python scripts/eval/run_experiment.py \
  data/eval/issue-477/experiment_openai_gpt4o_smoke_bundled_v1.yaml \
  --reference silver_gpt4o_smoke_bullets_v1 \
  --cost-report
```

## Two-track process (bullets + paragraph, no extra eval machinery)

Issue #477’s **bundled** implementation today produces **JSON** (`cleaned_text`, `title`, `bullets`). It does **not** emit a separate long **paragraph** summary, and the scorer compares each run to **one** reference’s `summary_final` string. So you do **not** need one mega-experiment or new prediction fields.

**Treat the two concerns as separate runs**, then fold both into one **process / decision**:

### Track A — Bullets (primary for bundle cost/quality)

This is where **staged vs bundled** matters. Same dataset, same bullet silver, two configs (commands above).

### Track B — Paragraph (regression guard, staged only)

Use **`experiment_openai_gpt4o_smoke_paragraph_v1`** so paragraph smoke matches bullet smoke (**`gpt-4o`**, same dataset and preprocessing). Do not use `llm_pipeline_mode: bundled` here; bundled is not wired for paragraph prose.

```bash
make experiment-run \
  CONFIG=data/eval/issue-477/experiment_openai_gpt4o_smoke_paragraph_v1.yaml \
  REFERENCE=silver_gpt4o_smoke_v1
```

**Intent:** Shared code paths (`Config`, provider wiring, `run_experiment`, preprocessing) should not silently hurt **paragraph** quality. Compare ROUGE / gates to a **known baseline** (e.g. last green run on `main`, or a saved `metrics.json`), not to the bullet bundled run.

**Autoresearch:** `autoresearch_prompt_openai_smoke_paragraph_v1` remains on **`gpt-4o-mini`** for cheaper prompt-tuning; it is not required for Issue #477 Track B.

**Future (optional product work):** If you want **staged vs bundled** on **paragraph** too, the bundle would need to ask the model for a **paragraph field** in the same JSON (or a second bundled path). That is separate from this two-run workflow.

## Decision gate (fill in after runs)

### Track A — Bullets (staged vs bundled)

| Criterion | Threshold | Staged result | Bundled result | Pass? |
| --- | --- | --- | --- | --- |
| ROUGE-L F1 vs silver | within -0.02 of staged | | | |
| Eval gates | all pass | | | |
| Est. cost / tokens | meaningful reduction | | | |
| Fallback rate | under 5% | n/a | | |

### Track B — Paragraph (staged vs your baseline)

| Criterion | Threshold | Baseline / main | This branch | Pass? |
| --- | --- | --- | --- | --- |
| ROUGE-L F1 vs `silver_gpt4o_smoke_v1` | no large regression (set delta) | | | |
| Eval gates | all pass | | | |

**Outcome:** Prefer shipping bundled bullets only if **Track A** passes and **Track B** is acceptable (no accidental regression). Otherwise iterate or keep bundled off.

**Outcome (one line):** (keep bundled opt-in / reject / iterate prompts)

## References

- [openai-pipeline-performance-hypotheses.md](openai-pipeline-performance-hypotheses.md)
- [PERFORMANCE_PROFILE_GUIDE.md](../guides/PERFORMANCE_PROFILE_GUIDE.md)
- RFC-065 / RFC-066 monitoring and run-compare Performance tab
