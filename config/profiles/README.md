# Profile capture configs (RFC-064)

YAML files here are **pipeline configs** for `make profile-freeze`: standard
`podcast_scraper` `Config` fields (RSS URL, `output_dir`, `max_episodes`, flags).

This directory is intentionally **not** under `config/eval/`: profiling is a
separate track from quality evaluation (`data/eval/`), even though both may use
similar episode counts or feeds.

**Operator manual:** [Performance Profile Guide](../../docs/guides/PERFORMANCE_PROFILE_GUIDE.md).

**Outputs** of a freeze go under [`data/profiles/`](../data/profiles/) (frozen
snapshots per release tag).

You can keep **multiple** configs here (for example `profile_freeze_laptop.yaml`
vs `profile_freeze_ci_like.yaml`) and pass the right one as `PIPELINE_CONFIG=`.

Start from [`profile_freeze.example.yaml`](profile_freeze.example.yaml): copy to
a new name, edit, and do not commit secrets or private feed URLs if the repo is
public.

## Recommended capture set for a release (e.g. v2.6.0)

Use **one frozen file per pipeline variant** so you can compare cost shape across
ML dev/prod, cloud LLMs, and Ollama. Naming: `VERSION=v2.6.0-<variant>` produces
`data/profiles/v2.6.0-<variant>.yaml`.

### E2E mock RSS (no real feeds)

The `capture_e2e_*.yaml` presets match **[`sample_acceptance_e2e_fixture_single.yaml`](../acceptance/sample_acceptance_e2e_fixture_single.yaml)**:
placeholder `rss` (`example.invalid` / `e2e-placeholder`), **`max_episodes: 2`**,
**`cpu`** for Whisper/summary (same as the sample; switch to `mps` in a local copy
if you only care about Apple Silicon profiles).

`freeze_profile.py` detects that placeholder, starts **`E2EHTTPServer`**, and uses
fixture feed **`podcast1_mtb`** (same family as `USE_FIXTURES=1` slot 0 in the
acceptance runner). Override the fixture name with **`E2E_FEED=podcast2`** or
`--e2e-feed podcast1_multi_episode` if you pass it through `make` (see Makefile).

| Variant | Config | Needs |
| --- | --- | --- |
| ML dev | [`capture_e2e_ml_dev.yaml`](capture_e2e_ml_dev.yaml) | Local ML only |
| ML prod | [`capture_e2e_ml_prod.yaml`](capture_e2e_ml_prod.yaml) | Same |
| OpenAI | [`capture_e2e_openai.yaml`](capture_e2e_openai.yaml) | `OPENAI_API_KEY` (RSS still mock) |
| Anthropic | [`capture_e2e_anthropic.yaml`](capture_e2e_anthropic.yaml) | `ANTHROPIC_API_KEY` |
| DeepSeek | [`capture_e2e_deepseek.yaml`](capture_e2e_deepseek.yaml) | `DEEPSEEK_API_KEY` (Whisper + spaCy + `deepseek-chat` summary; AI guide Config 1) |
| Gemini | [`capture_e2e_gemini.yaml`](capture_e2e_gemini.yaml) | `GEMINI_API_KEY` (Whisper + spaCy + Gemini summary) |
| Grok | [`capture_e2e_grok.yaml`](capture_e2e_grok.yaml) | `GROK_API_KEY` (Whisper + spaCy + Grok `grok-3-mini` summary/cleaning; x.ai IDs change) |
| Mistral | [`capture_e2e_mistral.yaml`](capture_e2e_mistral.yaml) | `MISTRAL_API_KEY` (Whisper + spaCy + `mistral-large-latest` summary) |
| Ollama fast | [`capture_e2e_ollama_llama32.yaml`](capture_e2e_ollama_llama32.yaml) | `ollama pull llama3.2:3b` |
| Ollama privacy default | [`capture_e2e_ollama_llama31_8b.yaml`](capture_e2e_ollama_llama31_8b.yaml) | `ollama pull llama3.1:8b` ([AI Provider Guide](../../docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md) Config 3) |
| Ollama quality | [`capture_e2e_ollama_qwen35.yaml`](capture_e2e_ollama_qwen35.yaml) | `ollama pull qwen3.5:35b` |

Shapes for DeepSeek / Gemini / Grok / Mistral follow **Recommended Configurations** in
[AI Provider Comparison Guide](../../docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md) (local
transcribe + local spaCy NER + cloud summarization where applicable).

Example (dataset label should match what you run: 2 episodes of `podcast1_mtb`):

```bash
make profile-freeze VERSION=v2.6.0-ml-dev \
  PIPELINE_CONFIG=config/profiles/capture_e2e_ml_dev.yaml \
  DATASET_ID=e2e_podcast1_mtb_n2
```

Repeat for each row. **Order:** ML first, then cloud, then Ollama. **Compare**
profiles on the **same hostname**.

**Minimal subset** if time is tight: **ml_dev + ml_prod + openai**.

Capture outputs under `.tmp/profile_capture/` (gitignored); **commit** only
`data/profiles/*.yaml`.
