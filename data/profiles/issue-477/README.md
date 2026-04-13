# Issue #477 -- profile captures

**GitHub:** [Issue #477](https://github.com/chipi/podcast_scraper/issues/477)

**Eval configs:** `data/eval/issue-477/`
| **Experiment plan:** `docs/wip/issue-477-llm-bundle-experiment-plan.md`

This folder holds **capture configs** and **frozen profile outputs** for measuring
per-stage pipeline wall times with `gpt-4o`, matching the eval baseline model.

## Configs

| File | Model | Purpose |
| --- | --- | --- |
| `capture_e2e_openai_gpt4o.yaml` | `gpt-4o` (summary), `gpt-4o-mini` (speaker) | Staged baseline profile |

## Profiles

| File | Description |
| --- | --- |
| `issue477-staged-gpt4o.yaml` | Frozen profile: staged pipeline, gpt-4o, 2 episodes |
| `issue477-staged-gpt4o.stage_truth.json` | Audit companion (per-stage walls, psutil samples) |

## Commands

Capture the staged baseline:

```bash
.venv/bin/python3 scripts/eval/freeze_profile.py \
  --version issue477-staged-gpt4o \
  --pipeline-config data/profiles/issue-477/capture_e2e_openai_gpt4o.yaml \
  --dataset-id e2e_podcast1_mtb_n2 \
  --output data/profiles/issue-477/issue477-staged-gpt4o.yaml
```

Compare against the existing gpt-4o-mini profile:

```bash
.venv/bin/python3 scripts/eval/diff_profiles.py \
  data/profiles/v2.6-wip-openai.yaml \
  data/profiles/issue-477/issue477-staged-gpt4o.yaml
```
