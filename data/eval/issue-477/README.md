# Issue #477 — bundled LLM eval configs

**GitHub:** [Issue #477](https://github.com/chipi/podcast_scraper/issues/477)

**Process / thresholds:** [docs/wip/issue-477-llm-bundle-experiment-plan.md](../../../docs/wip/issue-477-llm-bundle-experiment-plan.md)

This folder holds **experiment YAMLs** used to measure **staged vs bundled** summarization (bullets)
and a **paragraph regression** line on the same model tier (`gpt-4o` smoke). Keeping them here
makes the bundle work easy to find later without scanning `configs/summarization*`.

**Run IDs** (`id` in each file) are unchanged; outputs still land under
`data/eval/runs/<id>/`.

## Configs

| File | Role | Typical `REFERENCE` |
| --- | --- | --- |
| `experiment_openai_gpt4o_smoke_bullets_v1.yaml` | Track A staged baseline (bullets) | `silver_gpt4o_smoke_bullets_v1` |
| `experiment_openai_gpt4o_smoke_bundled_v1.yaml` | Track A bundled candidate (smoke) | `silver_gpt4o_smoke_bullets_v1` |
| `experiment_openai_gpt4o_benchmark_bundled_v1.yaml` | Bundled benchmark (10 eps) | e.g. `silver_sonnet46_benchmark_bullets_v1` (see file header) |
| `experiment_openai_gpt4o_smoke_paragraph_v1.yaml` | Track B paragraph guard (staged, `gpt-4o`) | `silver_gpt4o_smoke_v1` |
| `autoresearch_prompt_openai_smoke_bundled_v1.yaml` | Cheaper bundled prompt tuning (`gpt-4o-mini`) | `silver_gpt4o_smoke_bullets_v1` |

## Commands

```bash
make experiment-run CONFIG=data/eval/issue-477/experiment_openai_gpt4o_smoke_bullets_v1.yaml \
  REFERENCE=silver_gpt4o_smoke_bullets_v1

make experiment-run CONFIG=data/eval/issue-477/experiment_openai_gpt4o_smoke_bundled_v1.yaml \
  REFERENCE=silver_gpt4o_smoke_bullets_v1

make experiment-run CONFIG=data/eval/issue-477/experiment_openai_gpt4o_smoke_paragraph_v1.yaml \
  REFERENCE=silver_gpt4o_smoke_v1
```

Token / rough USD snapshot (not wired through `make experiment-run`; use project Python):

```bash
.venv/bin/python3 scripts/eval/run_experiment.py \
  data/eval/issue-477/experiment_openai_gpt4o_smoke_bundled_v1.yaml \
  --reference silver_gpt4o_smoke_bullets_v1 \
  --cost-report
```

## General eval layout

The rest of the eval config tree lives under `data/eval/configs/` (see `data/eval/configs/README.md`).
