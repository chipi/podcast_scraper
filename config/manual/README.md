# Manual validation configs (GI + KG)

YAML files here are for **human manual testing** (see
[`docs/wip/manual-test-plan-gi-kg.md`](../../docs/wip/manual-test-plan-gi-kg.md)), not CI.

Most presets use the same **NPR Planet Money** RSS as most GI/KG acceptance configs:
`https://feeds.npr.org/510289/podcast.xml`. **Multi-feed** presets add **The Journal** (WSJ) in one corpus; see the table below.

## Files

| File | Stack | GI / KG | Notes |
| --- | --- | --- | --- |
| `manual_multi_feed_planet_money_journal_openai.yaml` | OpenAI | On (`summary_bullets`) | **Multi-feed (GitHub #440):** Planet Money + The Journal under one `output_dir` (`feeds/rss_*/*` per feed). With `vector_search`, parent **`search/`** + **`corpus_manifest.json`** / **`corpus_run_summary.json`** (#505/#506). Requires `OPENAI_API_KEY`; use **`corpus-status`** for offline inspection. |
| `manual_multi_feed_planet_money_journal_openai_append.yaml` | OpenAI | On (`summary_bullets`) | Same feeds as above with **`append: true`** (GitHub #444): stable `run_append_*` per feed; re-run the same CLI to skip complete episodes. Header includes **handoff text** for OpenAI / LLM ops channels. |
| `manual_planet_money_openai_summaries_only.yaml` | OpenAI (transcribe + speaker + summary) | Off | Baseline summaries/metadata only. |
| `manual_planet_money_openai_gi_kg_summary_bullets.yaml` | OpenAI | On; `summary_bullets` from summaries | **Start here** for cheap, coherent GI+KG (no extra LLM insight/graph calls). With default **`gil_evidence_match_summary_provider: true`**, GIL grounding uses **OpenAI** (not local HF) — no **`.[ml]`** for GIL on this preset. |
| `manual_planet_money_openai_gi_kg_provider.yaml` | OpenAI | On; `provider` extraction | Stress-tests `generate_insights` + `extract_kg_graph`. |
| `manual_planet_money_ml_gi_kg_summary_bullets.yaml` | Whisper + spaCy + transformers | On; `summary_bullets` | **Local GI preset:** summaries + GIL evidence on **transformers** / **`.[ml]`** (NLI). Contrast with OpenAI row above. |

### Preset intent (GIL dependency profile)

| Preset | Summaries | GIL evidence (default config) | Typical install |
| --- | --- | --- | --- |
| **B1 — API-first GI** | LLM API (`*_gi_kg_summary_bullets.yaml` with `summary_provider: openai` etc.) | Same API as summary (**`gil_evidence_match_summary_provider`** aligns quote + entail) | Provider API keys only |
| **B2 — Local GI** | `manual_planet_money_ml_gi_kg_summary_bullets.yaml` | **`transformers`** QA + NLI | **`pip install -e ".[ml]"`** |
| **Hybrid (advanced)** | API | Set **`gil_evidence_match_summary_provider: false`** and choose **`quote_extraction_provider`** / **`entailment_provider`** explicitly | Keys + **`.[ml]`** if any evidence backend is local |

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
Examples: `config/acceptance/full/acceptance_planet_money_openai.yaml` (full pipeline: summaries,
GI, KG from summary bullets, semantic index). For mixed stacks or `provider`-mode GI/KG, use
presets under `config/manual/` (see table above).
