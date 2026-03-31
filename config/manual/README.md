# Manual validation configs (GI + KG)

YAML files here are for **human manual testing** (see
[`docs/wip/manual-test-plan-gi-kg.md`](../../docs/wip/manual-test-plan-gi-kg.md)), not CI.

They use the same **NPR Planet Money** RSS as most GI/KG acceptance configs:
`https://feeds.npr.org/510289/podcast.xml`.

## Files

| File | Stack | GI / KG | Notes |
| --- | --- | --- | --- |
| `manual_planet_money_openai_summaries_only.yaml` | OpenAI (transcribe + speaker + summary) | Off | Baseline summaries/metadata only. |
| `manual_planet_money_openai_gi_kg_summary_bullets.yaml` | OpenAI | On; `summary_bullets` from summaries | **Start here** for cheap, coherent GI+KG (no extra LLM insight/graph calls). All API providers default to shared JSON bullet prompts. |
| `manual_planet_money_openai_gi_kg_provider.yaml` | OpenAI | On; `provider` extraction | Stress-tests `generate_insights` + `extract_kg_graph`. |
| `manual_planet_money_ml_gi_kg_summary_bullets.yaml` | Whisper + spaCy + transformers | On; `summary_bullets` | Local ML; set `whisper_device` / `summary_device` to `cpu` if not on Apple Silicon. |

### Provider cost comparison (same RSS + `max_episodes: 3`)

Use these in parallel to compare spend (see each run’s `metrics.json` and your provider bills).

| Scenario | OpenAI | Gemini | Anthropic | DeepSeek | Mistral | Grok |
| --- | --- | --- | --- | --- | --- | --- |
| Summaries only | `manual_planet_money_openai_summaries_only.yaml` | `manual_planet_money_gemini_summaries_only.yaml` | `manual_planet_money_anthropic_summaries_only.yaml` | `manual_planet_money_deepseek_summaries_only.yaml` | `manual_planet_money_mistral_summaries_only.yaml` | `manual_planet_money_grok_summaries_only.yaml` |
| GI+KG bullets | `..._openai_gi_kg_summary_bullets.yaml` | `..._gemini_gi_kg_summary_bullets.yaml` | `..._anthropic_gi_kg_summary_bullets.yaml` | `..._deepseek_gi_kg_summary_bullets.yaml` | `..._mistral_gi_kg_summary_bullets.yaml` | `..._grok_gi_kg_summary_bullets.yaml` |
| GI+KG provider | `..._openai_gi_kg_provider.yaml` | `..._gemini_gi_kg_provider.yaml` | `..._anthropic_gi_kg_provider.yaml` | `..._deepseek_gi_kg_provider.yaml` | `..._mistral_gi_kg_provider.yaml` | `..._grok_gi_kg_provider.yaml` |

- **OpenAI:** manual YAMLs set `openai_transcription_model`, `openai_speaker_model`, `openai_summary_model`, and `openai_cleaning_model` (hybrid/LLM transcript cleaning before summarization; matches `gpt-4o-mini` unless you change it). Default summaries are JSON bullets unless you override `openai_summary_*_prompt`.
- **Gemini:** `gemini-2.0-flash` for transcribe + speaker + summary (`GEMINI_API_KEY`).
- **Mistral:** `voxtral-mini-latest` + `mistral-small-latest` for transcribe + speaker + summary (`MISTRAL_API_KEY`).
- **Anthropic / DeepSeek / Grok (audio):** no transcription in our stack — configs use **OpenAI `whisper-1`** for audio, then **Claude Haiku 4.5**, **deepseek-chat**, or **grok-beta** for speaker + summary (`OPENAI_API_KEY` plus `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, or `GROK_API_KEY`). Use **`grok-2`** in YAML if your API no longer serves `grok-beta`.

## How to run

**Direct CLI** (writes to the `output_dir` in each YAML):

```bash
python -m podcast_scraper.cli --config config/manual/manual_planet_money_openai_gi_kg_summary_bullets.yaml
```

**Acceptance runner** (session under `.test_outputs/acceptance/…` unless you set `OUTPUT_DIR`):

```bash
make test-acceptance CONFIGS="config/manual/manual_planet_money_openai_gi_kg_summary_bullets.yaml"
```

Provider keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `GROK_API_KEY` (see [`config/examples/.env.example`](../examples/.env.example)).

## Related acceptance configs

For provider matrices and CI-style runs, see [`config/acceptance/README.md`](../acceptance/README.md).
Examples: `config/acceptance/gi/acceptance_planet_money_gi_openai.yaml` (GI only, stub insights),
`config/acceptance/kg/acceptance_planet_money_kg_openai_provider.yaml` (KG `provider` with mixed
local Whisper + OpenAI summary).
