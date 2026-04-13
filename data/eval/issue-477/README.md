# Issue #477 — eval configs

**GitHub:** [Issue #477](https://github.com/chipi/podcast_scraper/issues/477)

**Process / thresholds:** [docs/wip/issue-477-llm-bundle-experiment-plan.md](../../../docs/wip/issue-477-llm-bundle-experiment-plan.md)

This folder holds **experiment YAMLs** for measuring summarization quality (paragraph and
bullets) on `gpt-4o` smoke. All references use the **active Sonnet 4.6 silvers**.

**Run IDs** (`id` in each file) are unchanged; outputs land under `data/eval/runs/<id>/`.

## Configs

| File | What it measures | `REFERENCE` |
| --- | --- | --- |
| `experiment_openai_gpt4o_smoke_bullets_v1.yaml` | Bullets baseline (staged) | `silver_sonnet46_smoke_bullets_v1` |
| `experiment_openai_gpt4o_smoke_paragraph_v1.yaml` | Paragraph baseline (staged) | `silver_sonnet46_smoke_v1` |
| `experiment_openai_gpt4o_smoke_bundled_v1.yaml` | Bundled candidate (Step 3, after baseline) | `silver_sonnet46_smoke_bullets_v1` |
| `experiment_openai_gpt4o_benchmark_bundled_v1.yaml` | Bundled benchmark (10 eps) | `silver_sonnet46_benchmark_bullets_v1` |
| `autoresearch_prompt_openai_smoke_bundled_v1.yaml` | Cheaper bundled prompt tuning (`gpt-4o-mini`) | `silver_sonnet46_smoke_bullets_v1` |

## Step 1 — Baseline (paragraph + bullets, staged)

```bash
make experiment-run CONFIG=data/eval/issue-477/experiment_openai_gpt4o_smoke_bullets_v1.yaml \
  REFERENCE=silver_sonnet46_smoke_bullets_v1

make experiment-run CONFIG=data/eval/issue-477/experiment_openai_gpt4o_smoke_paragraph_v1.yaml \
  REFERENCE=silver_sonnet46_smoke_v1
```

With cost report:

```bash
.venv/bin/python3 scripts/eval/run_experiment.py \
  data/eval/issue-477/experiment_openai_gpt4o_smoke_bullets_v1.yaml \
  --reference silver_sonnet46_smoke_bullets_v1 --cost-report

.venv/bin/python3 scripts/eval/run_experiment.py \
  data/eval/issue-477/experiment_openai_gpt4o_smoke_paragraph_v1.yaml \
  --reference silver_sonnet46_smoke_v1 --cost-report
```

## General eval layout

The rest of the eval config tree lives under `data/eval/configs/` (see `data/eval/configs/README.md`).
