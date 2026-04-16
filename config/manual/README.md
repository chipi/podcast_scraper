# Manual validation configs (GI + KG)

YAML files here are for **human manual testing** (see
[`docs/wip/manual-test-plan-gi-kg.md`](../../docs/wip/manual-test-plan-gi-kg.md)), not CI.

**Git:** `*.yaml` under this folder is **gitignored** (like `config/acceptance/`). Only this
`README.md` is versioned—keep your own copies or backups of presets you rely on.

Most presets use the same **NPR Planet Money** RSS as most GI/KG acceptance configs:
`https://feeds.npr.org/510289/podcast.xml`. **Multi-feed** presets add **The Journal** (WSJ) in one corpus; see the table below.

## Files

| File | Stack | GI / KG | Notes |
| --- | --- | --- | --- |
| `manual_e2e_mock_five_podcasts.yaml` (tracked: `../examples/manual_e2e_mock_five_podcasts.yaml`) | Local Whisper + spaCy + transformers | Off | **Multi-feed (GitHub #440):** five primary E2E mock podcasts (`podcast1`–`podcast5`) **plus** long-form **`podcast7_sustainability`**, **`podcast8_solar`**, **`podcast9_solo`** (p07–p09; **p06** edge-case feed omitted). **Start the fixture server first:** `make serve-e2e-mock` (default port **18765**, override with **`E2E_MOCK_PORT`**). Then run CLI with **`--config config/examples/manual_e2e_mock_five_podcasts.yaml`** or a copy under this folder. No API keys required for transcription; uses **`http://127.0.0.1:<port>/feeds/<slug>/feed.xml`** URLs. |
| `manual_multi_feed_planet_money_journal_openai_gemini.yaml` | OpenAI Whisper + Gemini | On (`summary_bullets`) | **Multi-feed (GitHub #440):** Planet Money + The Journal under one `output_dir` (`feeds/rss_*/*` per feed). **`openai` + `whisper-1`** for transcription; **`gemini`** + **`gemini-2.5-flash-lite`** for speaker + summary. With `vector_search`, parent **`search/`** + **`corpus_manifest.json`** / **`corpus_run_summary.json`** (#505/#506). Requires **`OPENAI_API_KEY`** and **`GEMINI_API_KEY`**; use **`corpus-status`** for offline inspection. **GitHub #562:** **`screenplay: true`** with non-Whisper **`transcription_provider`** is coerced to **`false`** at validation (one INFO); keep YAML **`screenplay: false`** for clarity when using OpenAI audio transcription. |
| `manual_multi_feed_planet_money_journal_openai_gemini_append.yaml` | OpenAI Whisper + Gemini | On (`summary_bullets`) | Same feeds as above with **`append: true`** (GitHub #444): stable `run_append_*` per feed; re-run the same CLI to skip complete episodes. Header includes **handoff text** for LLM ops channels. |
| `manual_multi_feed_planet_money_journal_openai_gemini_monitor.yaml` | OpenAI Whisper + Gemini | On (`summary_bullets`) | Same multi-feed hybrid stack as the first row with **`monitor: true`** (RFC-065: live RSS/CPU/stage + **`.pipeline_status.json`** per feed; **`log_file`** keeps stderr clean). Optional commented **`memray`** lines for **`.[monitor]`** installs. |
| `manual_multi_feed_corpus_rss_registry_openai_gemini.yaml` | OpenAI Whisper + Gemini | On (`summary_bullets`) | **11 feeds** from [`docs/wip/CORPUS-RSS-FEEDS.md`](../../docs/wip/CORPUS-RSS-FEEDS.md) (ingest-priority order): Hard Fork, No Priors, The Journal (WSJ), Invest Like the Best, Odd Lots, Planet Money, Latent Space, The Daily, NVIDIA AI Podcast, FT Unhedged, Capital Allocators. GI/KG + **`vector_search`**; **`output_dir`:** `.test_outputs/manual/corpus_rss_registry_openai_gemini` (override with **`--output-dir`**). Lower **`max_episodes`** before wide runs; see registry notes (NPR geo, URL verification). |
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
- **Gemini:** defaults use **`gemini-2.5-flash-lite`** for transcribe + speaker + summary when all three are Gemini (`GEMINI_API_KEY`). Manual **OpenAI + Gemini** presets use Whisper for audio and Gemini for speaker + summary only.
- **Mistral:** `voxtral-mini-latest` + `mistral-small-latest` for transcribe + speaker + summary (`MISTRAL_API_KEY`).
- **Anthropic / DeepSeek / Grok (audio):** no transcription in our stack — configs use **OpenAI `whisper-1`** for audio, then **Claude Haiku 4.5**, **deepseek-chat**, or **grok-beta** for speaker + summary (`OPENAI_API_KEY` plus `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, or `GROK_API_KEY`). Use **`grok-2`** in YAML if your API no longer serves `grok-beta`.

## How to run

**Direct CLI** (writes to the `output_dir` in each YAML):

```bash
python -m podcast_scraper.cli --config config/manual/manual_planet_money_openai_gi_kg_summary_bullets.yaml
```

**E2E mock fixture multi-feed** (local RSS fixtures, no real podcasts; primary five + long-form p07–p09):

```bash
make serve-e2e-mock
python -m podcast_scraper.cli --config config/examples/manual_e2e_mock_five_podcasts.yaml
```

**Acceptance runner** (session under `.test_outputs/acceptance/…` unless you set `OUTPUT_DIR`):

```bash
make test-acceptance CONFIGS="config/manual/manual_planet_money_openai_gi_kg_summary_bullets.yaml"
```

Provider keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `GROK_API_KEY` (see [`config/examples/.env.example`](../examples/.env.example)).

## Related acceptance configs

For provider matrices and CI-style runs, see [`config/acceptance/README.md`](../acceptance/README.md).
Examples: `config/acceptance/acceptance_planet_money_openai.yaml` (full pipeline: summaries,
GI, KG from summary bullets, semantic index). For mixed stacks or `provider`-mode GI/KG, use
presets under `config/manual/` (see table above).
