# Acceptance Test Configs

Configs in this directory are used by the **acceptance test runner** to run
full pipeline runs (RSS → transcript → summary → GI → KG → semantic index)
with different providers.

For what acceptance tests are, how the runner works, and how to run them,
see **`scripts/acceptance/README.md`**.

## Directory layout

| Path | Role |
| ---- | ---- |
| **`README.md`** | This file |
| **`FAST_CONFIGS.txt`** | Stems for `FAST_ONLY=1` / `make test-acceptance-fixtures-fast` (each stem must match a `config/acceptance/<stem>.yaml` file you pass to the runner) |
| **`sample_acceptance_e2e_fixture_*.yaml`** | Sample presets using **placeholder RSS URLs** only (`https://example.invalid/...`) — not real podcast feeds |

## Sample configs (E2E mock RSS)

These YAMLs use **made-up hostnames** on purpose (`example.invalid`). They are **not** fetched as
written. **You must run them with `USE_FIXTURES=1`.** The acceptance runner starts the E2E HTTP
server, **rewrites** those URLs to in-repo mock RSS feeds, and (for cloud-provider samples)
points HTTP clients at the mock APIs. Without **`USE_FIXTURES=1`**, the run would try to resolve
the placeholder hosts and fail.

| File | Feeds / providers |
| ---- | ----------------- |
| **`sample_acceptance_e2e_fixture_single.yaml`** | Single placeholder `rss:` — local ML (no API keys) |
| **`sample_acceptance_e2e_fixture_single_openai.yaml`** | Single placeholder `rss:` — OpenAI (Whisper + gpt-4o-mini); needs `OPENAI_API_KEY` |
| **`sample_acceptance_e2e_fixture_single_deepseek.yaml`** | Single placeholder `rss:` — Whisper + DeepSeek; needs `DEEPSEEK_API_KEY` |
| **`sample_acceptance_e2e_fixture_single_anthropic.yaml`** | Single placeholder `rss:` — Whisper + Anthropic (claude-haiku-4-5); needs `ANTHROPIC_API_KEY` |
| **`sample_acceptance_e2e_fixture_single_mistral.yaml`** | Single placeholder `rss:` — Mistral (voxtral + mistral-small-latest); needs `MISTRAL_API_KEY` |
| **`sample_acceptance_e2e_fixture_single_ollama_qwen3_5_35b.yaml`** | Single placeholder `rss:` — Whisper + Ollama `qwen3.5:35b`; needs local Ollama |
| **`sample_acceptance_e2e_fixture_multi_ml.yaml`** | Five placeholder `feeds:` URLs — **local ML**; `max_episodes: 5` per feed |
| **`sample_acceptance_e2e_fixture_multi_openai.yaml`** | Same five placeholders — OpenAI; needs `OPENAI_API_KEY`; `max_episodes: 5` per feed |
| **`sample_acceptance_e2e_fixture_multi_deepseek.yaml`** | Same five placeholders — OpenAI Whisper + DeepSeek; both keys; `max_episodes: 5` per feed |
| **`sample_acceptance_e2e_fixture_multi_anthropic.yaml`** | Same five placeholders — Anthropic; needs `ANTHROPIC_API_KEY`; `max_episodes: 5` per feed |
| **`sample_acceptance_e2e_fixture_multi_mistral.yaml`** | Same five placeholders — Mistral; needs `MISTRAL_API_KEY`; `max_episodes: 5` per feed |
| **`sample_acceptance_e2e_fixture_multi_ollama_mistral_small3_2.yaml`** | Same five placeholders — Ollama Mistral Small 3.2; local Ollama; `max_episodes: 5` per feed |
| **`sample_acceptance_e2e_fixture_multi_ollama_qwen3_5_35b.yaml`** | Same five placeholders — Ollama `qwen3.5:35b`; local Ollama; `max_episodes: 5` per feed |

Multi-feed samples use **five distinct placeholder URLs** so each slot maps to a separate mock
feed (fixtures **p01–p05**; three items each in the default bundle).

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

### What each sample produces

Every sample YAML sets:

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

### Provider matrix (samples + common stacks)

| Provider | Transcription | Speaker | Summary | API key |
| -------- | ------------- | ------- | ------- | ------- |
| **ML dev** (single + multi `_ml`) | whisper | spacy | transformers (bart-small) | none |
| **OpenAI** (`*_openai`) | openai (whisper-1) | openai | openai (gpt-4o-mini) | `OPENAI_API_KEY` |
| **Anthropic** (`*_anthropic`) | whisper | anthropic | anthropic (claude-haiku-4-5) | `ANTHROPIC_API_KEY` |
| **DeepSeek** (`*_deepseek`) | whisper | deepseek | deepseek (deepseek-chat) | `DEEPSEEK_API_KEY` |
| **Mistral** (`*_mistral`) | mistral (voxtral) | mistral | mistral (mistral-small-latest) | `MISTRAL_API_KEY` |
| **Ollama** (`*_ollama_*`) | whisper | ollama | ollama | none (local) |

Other stacks (for example Gemini, Grok) are supported by the product, but the sample YAMLs
in this directory do not include them.

Ollama model names used in samples: **`qwen3.5:35b`**, **`mistral-small3.2`**. Each requires
`ollama pull <model>`.

### Multi-feed layout (all multi samples)

Multi-feed configs use `feeds:` (alias of `rss_urls`). Corpus layout is
`<output_dir>/feeds/rss_<host>_<hash>/` per feed. With **`USE_FIXTURES=1`**, the runner replaces
each placeholder URL with a **distinct local E2E fixture feed** so multi-feed runs stay offline.

For **multi-feed templates** with placeholder feeds outside this directory, see under
**`config/examples/`**: `config.example.multi-feed.cloud-llm.yaml`, `config.example.multi-feed.ollama.yaml`,
`config.example.multi-feed.ml-dev.yaml`, and `config.example.multi-feed.ml-prod.yaml` (each with a JSON twin).

### Fast configs (for CI)

**`FAST_CONFIGS.txt`** lists config stems for a fast subset: one stem per line (filename without
`.yaml`). Use **`FAST_ONLY=1`** together with a **`CONFIGS`** glob so the runner filters to those
stems only.

Each stem must name a real file **`config/acceptance/<stem>.yaml`** next to this README (in a
default checkout, those are the **`sample_acceptance_e2e_fixture_*`** files). If
**`FAST_CONFIGS.txt`** is missing or empty, the runner may read an optional fallback list; see
**`config/ci/README.md`**.

**Full fast matrix + E2E fixtures** (offline RSS and mock APIs for every fast stem that lists
OpenAI / Anthropic / Mistral / DeepSeek samples — see **`scripts/acceptance/README.md`**):

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
**`USE_FIXTURES=1`**: OpenAI / Anthropic / Mistral / DeepSeek traffic goes to the **E2E fixture
HTTP server** that simulates APIs — not to real billed endpoints, and CI does not need provider
API keys for that job.

## Running acceptance tests

Run every sample preset:

```bash
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_*.yaml" USE_FIXTURES=1
```

Run only local-ML samples (no API keys):

```bash
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_single.yaml" USE_FIXTURES=1
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_multi_ml.yaml" USE_FIXTURES=1
```

Run only Ollama-backed samples (local daemon):

```bash
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_*ollama*.yaml" USE_FIXTURES=1
```

Run fast subset after expanding a glob (filter to stems in **`FAST_CONFIGS.txt`**):

```bash
make test-acceptance CONFIGS="config/acceptance/sample_acceptance_e2e_fixture_*.yaml" USE_FIXTURES=1 FAST_ONLY=1
```

You can point **`CONFIGS`** at any other YAML path on your machine the same way; use
**`USE_FIXTURES=0`** only when you intend real network and real provider endpoints. Details stay
in **`scripts/acceptance/README.md`**.

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
