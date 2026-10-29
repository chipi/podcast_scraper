# Performance capture profiles (RFC-064)

Per-provider timing and cost capture profiles under a fixed E2E fixture
(`make profile-freeze` is retired from this repository (#2134)). Frozen outputs land in
`data/profiles/` per release tag — that tree moved to `chipi/podcast-scraper-eval-data`
in arc 2, since a frozen profile is a materialised research artifact.

## How it works

A capture merges two YAMLs in order:

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

## Operator notes

- **Edit [`_defaults.yaml`](_defaults.yaml)** only when the change applies to
  all freeze captures (e.g. switching capture device to `mps` for Apple
  Silicon profiling).
