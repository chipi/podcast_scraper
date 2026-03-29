# Manual validation configs (GI + KG)

YAML files here are for **human manual testing** (see
[`docs/wip/manual-test-plan-gi-kg.md`](../../docs/wip/manual-test-plan-gi-kg.md)), not CI.

They use the same **NPR Planet Money** RSS as most GI/KG acceptance configs:
`https://feeds.npr.org/510289/podcast.xml`.

## Files

| File | Stack | GI / KG | Notes |
| --- | --- | --- | --- |
| `manual_planet_money_openai_summaries_only.yaml` | OpenAI (transcribe + speaker + summary) | Off | Baseline summaries/metadata only. |
| `manual_planet_money_openai_gi_kg_summary_bullets.yaml` | OpenAI | On; bullets from OpenAI summaries | **Start here** for cheap, coherent GI+KG (no extra LLM insight/graph calls). |
| `manual_planet_money_openai_gi_kg_provider.yaml` | OpenAI | On; `provider` extraction | Stress-tests `generate_insights` + `extract_kg_graph`. |
| `manual_planet_money_ml_gi_kg_summary_bullets.yaml` | Whisper + spaCy + transformers | On; `summary_bullets` | Local ML; set `whisper_device` / `summary_device` to `cpu` if not on Apple Silicon. |

## How to run

**Direct CLI** (writes to the `output_dir` in each YAML):

```bash
python -m podcast_scraper.cli --config config/manual/manual_planet_money_openai_gi_kg_summary_bullets.yaml
```

**Acceptance runner** (session under `.test_outputs/acceptance/…` unless you set `OUTPUT_DIR`):

```bash
make test-acceptance CONFIGS="config/manual/manual_planet_money_openai_gi_kg_summary_bullets.yaml"
```

OpenAI configs require `OPENAI_API_KEY` (see [`examples/.env.example`](../../examples/.env.example)).

## Related acceptance configs

For provider matrices and CI-style runs, see [`config/acceptance/README.md`](../acceptance/README.md).
Examples: `config/acceptance/gi/acceptance_planet_money_gi_openai.yaml` (GI only, stub insights),
`config/acceptance/kg/acceptance_planet_money_kg_openai_provider.yaml` (KG `provider` with mixed
local Whisper + OpenAI summary).
