# Acceptance Test Configs

Configs in this directory are used by the **acceptance test runner** to run
full pipeline runs (RSS → transcript → summary → GI → KG → semantic index)
with different providers.

For what acceptance tests are, how the runner works, and how to run them,
see **`scripts/acceptance/README.md`**.

## Layout

**Preset YAMLs** live in this directory next to **`README.md`** and **`FAST_CONFIGS.txt`**.
Most `*.yaml` files here are **gitignored** (local-only, often with real RSS URLs you do not
want in Git). Exceptions: this **`README.md`**, **`FAST_CONFIGS.txt`**, and the tracked
**`sample_acceptance_e2e_fixture_*.yaml`** files (placeholders + E2E mock feeds — see below),
including variants that mirror **`acceptance_planet_money_*`** (and multi-feed OpenAI /
DeepSeek journal presets where applicable): OpenAI, DeepSeek, Anthropic, Mistral
(`mistral-small-latest`), and Ollama **`qwen3.5:35b`** / **`mistral-small3.2`**.

Each preset runs the **full pipeline**: summaries, grounded insights (GI), knowledge graph
(KG), and a FAISS vector index. Mostly **one feed per YAML** × provider; plus **multi-feed**
YAMLs (GitHub #440).

Local presets use the naming pattern `acceptance_<feed>_<provider_or_detail>.yaml`.

### Sample configs (E2E mock RSS, safe to commit)

These files use **non-localhost placeholder URLs** on purpose. They are **not** real feeds.

| File | Feeds / providers |
| ---- | ----------------- |
| **`sample_acceptance_e2e_fixture_single.yaml`** | Single `rss:` — local ML (no API keys) |
| **`sample_acceptance_e2e_fixture_single_openai.yaml`** | Single `rss:` — OpenAI (Whisper + gpt-4o-mini); needs `OPENAI_API_KEY` |
| **`sample_acceptance_e2e_fixture_single_deepseek.yaml`** | Single `rss:` — Whisper + DeepSeek; needs `DEEPSEEK_API_KEY` |
| **`sample_acceptance_e2e_fixture_single_anthropic.yaml`** | Single `rss:` — Whisper + Anthropic (claude-haiku-4-5); needs `ANTHROPIC_API_KEY` |
| **`sample_acceptance_e2e_fixture_single_mistral.yaml`** | Single `rss:` — Mistral (voxtral + mistral-small-latest); needs `MISTRAL_API_KEY` |
| **`sample_acceptance_e2e_fixture_single_ollama_qwen3_5_35b.yaml`** | Single `rss:` — Whisper + Ollama `qwen3.5:35b`; needs local Ollama |
| **`sample_acceptance_e2e_fixture_multi_ml.yaml`** | Five `feeds:` → `podcast1`..`podcast5`; **local ML**; `max_episodes: 5` per feed (safe if copied to real RSS) |
| **`sample_acceptance_e2e_fixture_multi_openai.yaml`** | Same five placeholders — OpenAI; needs `OPENAI_API_KEY`; `max_episodes: 5` per feed |
| **`sample_acceptance_e2e_fixture_multi_deepseek.yaml`** | Same five placeholders — OpenAI Whisper + DeepSeek; both keys; `max_episodes: 5` per feed |
| **`sample_acceptance_e2e_fixture_multi_anthropic.yaml`** | Same five placeholders — Anthropic; needs `ANTHROPIC_API_KEY`; `max_episodes: 5` per feed |
| **`sample_acceptance_e2e_fixture_multi_mistral.yaml`** | Same five placeholders — Mistral; needs `MISTRAL_API_KEY`; `max_episodes: 5` per feed |
| **`sample_acceptance_e2e_fixture_multi_ollama_mistral_small3_2.yaml`** | Same five placeholders — Ollama Mistral Small 3.2; local Ollama; `max_episodes: 5` per feed |
| **`sample_acceptance_e2e_fixture_multi_ollama_qwen3_5_35b.yaml`** | Same five placeholders — Ollama `qwen3.5:35b`; local Ollama; `max_episodes: 5` per feed |

**You must run them with `make test-acceptance` and `USE_FIXTURES=1`.** The acceptance
runner starts the E2E HTTP server and **rewrites** those URLs to in-repo mock RSS feeds
(and points provider APIs at the mock). Multi-feed samples stay **five generic
placeholders** you can copy: the runner maps them to full **p01–p05** fixtures (three
episodes each), using an internal E2E alias for the first slot so default fast E2E mode
still serves **p01_mtb**, not **p01_fast**. Without **`USE_FIXTURES=1`**, the run would try to
fetch the placeholder hosts and fail.

**Your private presets** with **real** `https://…` RSS URLs should be run **without**
`USE_FIXTURES=1` (or `USE_FIXTURES=0`): the runner then uses your YAML as-is and hits the
real network (and real APIs if configured).

```bash
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_single.yaml" USE_FIXTURES=1
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_multi_ml.yaml" USE_FIXTURES=1
# API / Ollama samples (export keys or start Ollama as needed):
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_single_openai.yaml" USE_FIXTURES=1
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_single_deepseek.yaml" USE_FIXTURES=1
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_single_anthropic.yaml" USE_FIXTURES=1
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_single_mistral.yaml" USE_FIXTURES=1
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_single_ollama_qwen3_5_35b.yaml" USE_FIXTURES=1
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_multi_openai.yaml" USE_FIXTURES=1
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_multi_deepseek.yaml" USE_FIXTURES=1
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_multi_anthropic.yaml" USE_FIXTURES=1
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_multi_mistral.yaml" USE_FIXTURES=1
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_multi_ollama_mistral_small3_2.yaml" USE_FIXTURES=1
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_multi_ollama_qwen3_5_35b.yaml" USE_FIXTURES=1
```

### What each config produces

Every preset YAML sets:

- `generate_summaries: true`
- `generate_gi: true` with `gi_insight_source: summary_bullets`
- `generate_kg: true` with `kg_extraction_source: summary_bullets`
- `vector_search: true` with `vector_backend: faiss`

GI and KG use the summary bullets generated by the summarization step
(no stubs).

### Optional corpus topic clustering (RFC-075)

The acceptance runner does **not** invoke **`topic-clusters`** automatically. After a successful
run with **`vector_search: true`** (FAISS under **`<corpus_parent>/search/`**), you can build
**`search/topic_clusters.json`** and optionally merge **`topic_id_aliases`**:

```bash
python -m podcast_scraper.cli topic-clusters --output-dir <corpus_parent>
# optional:
python -m podcast_scraper.cli topic-clusters --output-dir <corpus_parent> --merge-cil-overrides
```

See [RFC-075](../../docs/rfc/RFC-075-corpus-topic-clustering.md) and [Semantic Search Guide](../../docs/guides/SEMANTIC_SEARCH_GUIDE.md).

### Provider matrix

| Provider | Transcription | Speaker | Summary | API key |
| -------- | ------------- | ------- | ------- | ------- |
| **ML dev** | whisper | spacy | transformers (bart-small) | none |
| **ML prod** | whisper | spacy | transformers (pegasus-cnn) | none |
| **OpenAI** | openai (whisper-1) | openai | openai (gpt-4o-mini) | `OPENAI_API_KEY` |
| **Anthropic** | whisper | anthropic | anthropic (claude-haiku-4-5) | `ANTHROPIC_API_KEY` |
| **Gemini** | gemini | gemini | gemini (gemini-2.0-flash) | `GEMINI_API_KEY` |
| **DeepSeek** | whisper | deepseek | deepseek (deepseek-chat) | `DEEPSEEK_API_KEY` |
| **Mistral** | mistral (voxtral) | mistral | mistral (mistral-small-latest) | `MISTRAL_API_KEY` |
| **Grok** | whisper | grok | grok (grok-3-mini) | `GROK_API_KEY` |
| **Ollama** | whisper | ollama | ollama | none (local) |

Ollama variants: `llama3.1:8b` (default), `gemma2:9b`, `mistral:7b`,
`phi3:mini`, `qwen2.5:7b`, `qwen2.5:32b`, `qwen3.5:9b`, `qwen3.5:27b`,
`qwen3.5:35b`. Each requires `ollama pull <model>`.

### Feeds

| Feed | RSS | Configs |
| ---- | --- | ------- |
| **Planet Money** | `https://feeds.npr.org/510289/podcast.xml` | 18 single-feed configs |
| **The Journal** | `https://video-api.wsj.com/podcast/rss/wsj/the-journal` | 18 single-feed configs |
| **Planet Money + The Journal (one corpus)** | Both URLs in YAML `feeds:` | Presets: `acceptance_multi_feed_planet_money_journal_openai.yaml`, `acceptance_multi_feed_planet_money_journal_deepseek.yaml` under `config/acceptance/` (local); generic shape: `config/examples/config.example.multi-feed.yaml` |
| **Same two feeds + OpenAI + append / resume (#444)** | Same as above with `append: true` | Preset: `config/acceptance/acceptance_multi_feed_planet_money_journal_openai_append.yaml` |
| **Same two feeds + DeepSeek + append / resume (#444)** | Same with `append: true` | Preset: `config/acceptance/acceptance_multi_feed_planet_money_journal_deepseek_append.yaml` |

Single-feed rows have identical provider coverage across the two shows. **Multi-feed** configs use `feeds:` (alias of `rss_urls`); corpus layout is `<output_dir>/feeds/rss_<host>_<hash>/` per feed. With **`USE_FIXTURES=1`**, the acceptance runner replaces each external feed URL with a **distinct local E2E fixture feed** so multi-feed runs stay offline.

**Example — multi-feed acceptance (fixtures or real RSS):**

```bash
make test-acceptance CONFIGS="config/acceptance/acceptance_multi_feed_planet_money_journal_openai.yaml" USE_FIXTURES=1
```

**Example — multi-feed + append (second identical run should skip complete episodes):**

```bash
make test-acceptance CONFIGS="config/acceptance/acceptance_multi_feed_planet_money_journal_openai_append.yaml" USE_FIXTURES=1
```

**Example — multi-feed + DeepSeek + append:**

```bash
make test-acceptance CONFIGS="config/acceptance/acceptance_multi_feed_planet_money_journal_deepseek_append.yaml" USE_FIXTURES=1
```

For real NPR + WSJ + OpenAI, set **`USE_FIXTURES=0`** and export **`OPENAI_API_KEY`**. For DeepSeek multi-feed (append or not), also export **`DEEPSEEK_API_KEY`** (Whisper still uses **`OPENAI_API_KEY`**).

### Fast configs (for CI)

**`FAST_CONFIGS.txt`** (in this directory) lists config stems for a fast subset:
one stem per line (filename without `.yaml`). Use **`FAST_ONLY=1`** together
with a **`CONFIGS`** glob so the runner filters to those stems only.

**`FAST_CONFIGS.txt`** is committed next to this README. Stems must match **tracked**
`config/acceptance/<stem>.yaml` files (the repo ships **`sample_acceptance_e2e_fixture_*`** only;
private `acceptance_*` presets are usually gitignored — see **Layout** above). If this file is
missing or empty, the runner may read optional local **`config/ci/acceptance_fast_stems.txt`**
(gitignored; see **`config/ci/README.md`**). Keep any local CI copy aligned when you change the
fast matrix.

**Full fast matrix + E2E fixtures** (offline RSS and mock APIs for every fast
preset, including multi-feed — see **`scripts/acceptance/README.md`**):

```bash
make test-acceptance-fixtures-fast
```

Equivalent Make flags:

```bash
make test-acceptance FROM_FAST_STEMS=1 USE_FIXTURES=1 \
  NO_AUTO_ANALYZE=1 NO_AUTO_BENCHMARK=1 TIMEOUT=900
```

On **main / release** branches, CI runs `make test-acceptance-fixtures-fast`
(job `test-acceptance-fixtures` in `.github/workflows/python-app.yml`). Those runs use
**`USE_FIXTURES=1`**: OpenAI / Anthropic / Mistral / DeepSeek (and similar) traffic goes to the
**E2E fixture HTTP server** that simulates APIs — not to real billed endpoints, and CI does not
need provider API keys for that job.

## Running acceptance tests

Run all configs:

```bash
make test-acceptance CONFIGS="config/acceptance/*.yaml"
```

Run only Planet Money:

```bash
make test-acceptance CONFIGS="config/acceptance/acceptance_planet_money_*.yaml"
```

Run only The Journal:

```bash
make test-acceptance CONFIGS="config/acceptance/acceptance_the_journal_*.yaml"
```

Run a single config:

```bash
make test-acceptance CONFIGS="config/acceptance/acceptance_planet_money_ml_dev.yaml"
```

Run only ML configs (no API keys needed):

```bash
make test-acceptance CONFIGS="config/acceptance/acceptance_*_ml_*.yaml"
```

Run only Ollama configs:

```bash
make test-acceptance CONFIGS="config/acceptance/acceptance_*_ollama*.yaml"
```

Run fast subset after expanding a glob (filter to stems in `FAST_CONFIGS.txt`):

```bash
make test-acceptance CONFIGS="config/acceptance/*.yaml" USE_FIXTURES=1 FAST_ONLY=1
```

## CI: GIL vs FAISS offset check after the fast matrix

On **push** to `main` / release branches, `.github/workflows/python-app.yml` runs
`make test-acceptance-fixtures-fast`, then **`make verify-gil-offsets-after-acceptance`**.
That walks the latest acceptance **`session_*/runs/run_*`** trees and runs
`verify-gil-chunk-offsets --strict` on every run that has **`search/metadata.json`**
(RFC-072 / issue #528). **Nightly** (`.github/workflows/nightly.yml`) does not run the
acceptance matrix today; it runs **`make test-nightly`** instead. To re-check offsets locally
after a fixture session:

```bash
make verify-gil-offsets-after-acceptance
```
