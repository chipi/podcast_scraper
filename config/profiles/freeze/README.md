# Performance capture profiles (RFC-064)

Used by `make profile-freeze` to capture per-provider timing and cost profiles
under a fixed E2E fixture. Frozen outputs land in
[`data/profiles/`](../../../data/profiles/) per release tag.

## How it works

`scripts/eval/freeze_profile.py` loads two YAMLs in order:

1. [`_defaults.yaml`](_defaults.yaml) — operational fields (RSS placeholder,
   `output_dir` root, `max_episodes`, `whisper_device`, `transcribe_missing`).
2. One of the per-provider profiles below — provider/research settings only.

The provider profile wins on any key overlap. The `output_dir` root is
suffixed with the profile filename (e.g. `gemini.yaml` →
`.tmp/profile_capture/e2e/gemini/`). The placeholder RSS triggers the E2E
fixture server (`podcast1_mtb` by default; override with `--e2e-feed`).

## Profiles

Each profile is **maximally oriented toward its provider**: where the provider
offers its own transcription / NER, the profile uses it; otherwise it falls
back to local Whisper `small.en` (production WER sweet spot) and spaCy trf.

| Profile | Transcription | NER | Summary | Prereq |
| ------- | ------------- | --- | ------- | ------ |
| [`openai.yaml`](openai.yaml) | OpenAI `whisper-1` | OpenAI `gpt-4o-mini` | OpenAI `gpt-4o-mini` | `OPENAI_API_KEY` |
| [`anthropic.yaml`](anthropic.yaml) | Whisper `small.en` | Anthropic `claude-haiku-4-5` | Anthropic `claude-haiku-4-5` | `ANTHROPIC_API_KEY` |
| [`deepseek.yaml`](deepseek.yaml) | Whisper `small.en` | DeepSeek `deepseek-chat` | DeepSeek `deepseek-chat` | `DEEPSEEK_API_KEY` |
| [`gemini.yaml`](gemini.yaml) | Gemini audio | Gemini `gemini-2.5-flash-lite` | Gemini `gemini-2.5-flash-lite` | `GEMINI_API_KEY` |
| [`mistral.yaml`](mistral.yaml) | Mistral `voxtral-mini-latest` | Mistral `mistral-large-latest` | Mistral `mistral-large-latest` | `MISTRAL_API_KEY` |
| [`grok.yaml`](grok.yaml) | Whisper `small.en` | Grok `grok-3-mini` | Grok `grok-3-mini` | `GROK_API_KEY` |
| [`ml_dev.yaml`](ml_dev.yaml) | Whisper `base.en` (dev) | spaCy sm | transformers `bart-small` | Local ML only |
| [`ml_prod.yaml`](ml_prod.yaml) | Whisper `small.en` | spaCy trf | transformers `pegasus-cnn` | Local ML only |
| [`ollama_llama31_8b.yaml`](ollama_llama31_8b.yaml) | Whisper `small.en` | Ollama `llama3.1:8b` | Ollama `llama3.1:8b` | `ollama pull llama3.1:8b` |
| [`ollama_llama32.yaml`](ollama_llama32.yaml) | Whisper `small.en` | Ollama `llama3.2:3b` | Ollama `llama3.2:3b` | `ollama pull llama3.2:3b` |
| [`ollama_qwen35.yaml`](ollama_qwen35.yaml) | Whisper `small.en` | Ollama `qwen3.5:35b` | Ollama `qwen3.5:35b` bundled | `ollama pull qwen3.5:35b` |

## Usage

```bash
make profile-freeze VERSION=v2.6.0-gemini \
  PIPELINE_CONFIG=config/profiles/freeze/gemini.yaml \
  DATASET_ID=e2e_podcast1_mtb_n2
```

**Order for a release:** ML first, then cloud, then Ollama. **Compare
profiles on the same hostname.** **Minimal subset** if time is tight:
`ml_dev` + `ml_prod` + `openai`.

Capture outputs land under `.tmp/profile_capture/e2e/<profile>/`
(gitignored). **Commit** only the frozen YAMLs in `data/profiles/`.

## Operator notes

- **Override the RSS placeholder** with `--rss <url>` on the freeze_profile
  CLI (or `--e2e-feed <fixture_name>` to pick a different fixture).
- **Override the output root** with `--output <path>` on freeze_profile (this
  writes the frozen YAML; the intermediate capture output is still under
  `_defaults.yaml`'s `output_dir`).
- **Edit [`_defaults.yaml`](_defaults.yaml)** only when the change applies to
  all freeze captures (e.g. switching capture device to `mps` for Apple
  Silicon profiling).
