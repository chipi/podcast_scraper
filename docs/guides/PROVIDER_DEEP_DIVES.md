# Provider Deep Dives

Per-provider reference cards for the providers this repository ships: what each one does in the
pipeline, how it is selected, which credential it reads, and its built-in default models.

This page deliberately carries **no prices, speed figures or quality scores**. Prices change,
and quality and cost measurements (benchmarks, model comparisons, evaluation data) live in the
private eval repository (see the README), not here. For current prices use the vendor's pricing
page; for what a run actually cost use the run's own cost metrics.

---

## How providers are selected

Each pipeline capability is chosen independently in config (`config.py`):

| Capability | Config key | Values |
| ---------- | ---------- | ------ |
| Transcription | `transcription_provider` | `whisper` (default), `openai`, `gemini`, `mistral`, `deepgram`, `groq`, `tailnet_dgx_whisper`, `moss` |
| Transcription fallback | `transcription_fallback_provider` | `whisper`, `openai`, `gemini`, `mistral`, `deepgram` |
| Summarization | `summary_provider` | `transformers` (default), `summllama`, `openai`, `gemini`, `grok`, `mistral`, `deepseek`, `anthropic`, `ollama`, `vllm`, `litellm`, `qwen`, `groq` |
| Speaker detection | `speaker_detector_provider` | `spacy` (default), `openai`, `gemini`, `mistral`, `grok`, `deepseek`, `anthropic`, `ollama`, `vllm`, `litellm`, `qwen`, `groq` |

A provider class keeps the generic calls this repository uses (summary, translation, the basic
knowledge-graph pass, cleaning). The methods that speaker detection, host detection and grounded
insights add to a provider are supplied by an installed extension
(`src/podcast_scraper/providers/extensible.py`, ADR-176); a public-only install does not detect
speakers (see the [Audio Pipeline Guide](AUDIO_PIPELINE_GUIDE.md)).

The code under `src/podcast_scraper/providers/` has one package per provider: `ml` (local),
`openai`, `anthropic`, `mistral`, `gemini`, `grok`, `deepseek`, `groq`, `ollama`, `qwen`, `vllm`,
`litellm`, `deepgram`, `moss` and `tailnet_dgx`. `deepseek`, `groq`, `qwen`, `vllm` and `litellm`
are siblings of `openai`: they share the OpenAI-compatible transport
(`OpenAICompatibleProvider`) and change only identity, auth and defaults.

---

## Capability matrix

What each provider's code implements (`providers/*/`, `get_capabilities()` where declared).

| Provider | Transcription | Summarization | Credential |
| -------- | :-----------: | :-----------: | ---------- |
| Local ML (`whisper`, `transformers`, `summllama`) | yes (Whisper) | yes | none |
| OpenAI | yes | yes | `OPENAI_API_KEY` |
| Anthropic | no | yes | `ANTHROPIC_API_KEY` |
| Mistral | yes | yes | `MISTRAL_API_KEY` |
| Gemini | yes | yes | `GEMINI_API_KEY` |
| Grok | no | yes | `GROK_API_KEY` |
| DeepSeek | n/a (config: summary and speaker models only) | yes | `DEEPSEEK_API_KEY` |
| Groq | yes (Whisper models) | yes | `GROQ_API_KEY` |
| Ollama | no | yes | none (local server) |
| Qwen | n/a (OpenAI-compatible endpoint you provide) | yes | `QWEN_API_KEY`, optional |
| vLLM | n/a (OpenAI-compatible endpoint you provide) | yes | `VLLM_API_KEY`, optional |
| LiteLLM gateway | n/a (gateway aliases) | yes | `LITELLM_API_KEY` |
| Deepgram | yes (native speaker labels) | no | `DEEPGRAM_API_KEY` |
| Moss | yes (transcript and speaker labels in one pass) | no | none (self-hosted service) |
| Self-hosted GPU Whisper (`tailnet_dgx_whisper`) | yes | no | none (private network) |

"n/a" means the provider's config namespace has no transcription model field, or the provider
is not selectable for that capability.

---

## Provider reference cards

### 1. Local ML (default)

- **Transcription:** local Whisper (`transcription_provider: whisper`, `whisper_model`, default
  `base.en`).
- **Summarization:** `summary_provider: transformers` (Hugging Face seq2seq models, BART/LED
  families, MAP/REDUCE) or `summary_provider: summllama` (single-pass SummLlama3.2-3B, no
  MAP/REDUCE).
- **Speaker detection:** `speaker_detector_provider: spacy` (spaCy NER), run by the speaker stage
  an installed extension provides.
- **Privacy:** nothing leaves the machine.
- **Install:** `pip install -e ".[ml]"`. Hardware depends on the model size you pick; GPU is
  optional (CPU, CUDA and MPS are supported).

For model comparisons see the [ML Model Comparison Guide](ML_MODEL_COMPARISON_GUIDE.md).

### 2. OpenAI

- **Capabilities:** transcription (`openai_transcription_model`, default `whisper-1`),
  summarization (`openai_summary_model`, default `gpt-4o`).
- **Credential:** `OPENAI_API_KEY`; `openai_api_base` points the same class at another
  OpenAI-compatible endpoint.
- **Audio size:** the cloud transcription upload limit applies; oversized files are split by
  `AudioChunker` (see the [Audio Pipeline Guide](AUDIO_PIPELINE_GUIDE.md)).
- **Install:** `pip install -e ".[llm]"`.

### 3. Anthropic (Claude)

- **Capabilities:** summarization and text work only. Anthropic has no audio API, so it cannot
  transcribe (`get_capabilities()` returns `supports_transcription=False`).
- **Credential:** `ANTHROPIC_API_KEY`.
- **Models:** `anthropic_summary_model` defaults to `claude-3-5-sonnet-20241022`;
  `anthropic_cleaning_model` defaults to `claude-haiku-4-5`. Profiles under `config/profiles/`
  pin the models they use; check the models available to your account.
- **Context:** `anthropic_max_context_tokens`, default 200000.

### 4. Mistral

- **Capabilities:** transcription (`mistral_transcription_model`, default `voxtral-mini-latest`,
  plain text output) and summarization (`mistral_summary_model`, default
  `mistral-large-latest`).
- **Credential:** `MISTRAL_API_KEY`.
- **Context:** `mistral_max_context_tokens`, default 256000.
- **Diarization:** cannot diarize; `diarize: true` with `transcription_provider: mistral` is a
  config error.

### 5. DeepSeek

- **Capabilities:** summarization through DeepSeek's OpenAI-compatible API
  (`deepseek_summary_model`, default `deepseek-chat`; `deepseek_cleaning_model`, default
  `deepseek-chat`).
- **Credential:** `DEEPSEEK_API_KEY`. Default endpoint `api.deepseek.com`; `deepseek_api_base`
  can point the class at a LiteLLM gateway instead.

### 6. Google Gemini

- **Capabilities:** transcription (`gemini_transcription_model`, default `gemini-2.5-flash-lite`,
  plain text output) and summarization (`gemini_summary_model`, default `gemini-2.5-flash-lite`).
- **Credential:** `GEMINI_API_KEY`.
- **Context:** `gemini_max_context_tokens`, default 2000000.
- **Diarization:** `gemini` is selectable as `diarization_provider`, but cannot diarize as the
  transcription provider.
- **Install:** `pip install -e ".[llm]"` (`google-genai`).

### 7. Grok (xAI)

- **Capabilities:** summarization only; no audio API.
- **Credential:** `GROK_API_KEY`.
- **Models:** `grok_summary_model` and `grok_speaker_model` default to `grok-4.3`.
- **Context:** `grok_max_context_tokens`, default 128000.

### 8. Groq

- **Capabilities:** low-latency hosted open models for summarization
  (`groq_summary_model`, default `llama-3.3-70b-versatile`) and Whisper transcription
  (`groq_transcription_model`, default `whisper-large-v3-turbo`).
- **Credential:** `GROQ_API_KEY` (required; Groq is cloud-only). Default endpoint
  `api.groq.com/openai/v1`; `groq_api_base` can route the same class through a LiteLLM gateway.

### 9. Ollama (local LLMs)

- **Capabilities:** summarization (`ollama_summary_model`, default `llama3.1:8b`) against a local
  or reachable Ollama server (default port 11434). No audio API, so transcription stays with
  Whisper or a cloud provider.
- **Credential:** none.
- **Privacy:** requests stay on the machine (or the host you point `ollama_api_base` at).

Setup, model choice and troubleshooting: [Ollama Provider Guide](OLLAMA_PROVIDER_GUIDE.md).

### 10. vLLM, Qwen and the LiteLLM gateway

Three OpenAI-compatible providers for models you serve or route yourself:

- **`vllm`:** open models on a vLLM server. The model fields name the real Hugging Face model id;
  the served model is verified against `GET /v1/models`. A bearer is optional (`VLLM_API_KEY`).
- **`qwen`:** a Qwen3 endpoint (cloud host or local vLLM). No vendor default endpoint: the
  profile always sets `qwen_api_base`. Bearer optional (`QWEN_API_KEY`).
- **`litellm`:** a LiteLLM gateway. The model fields name gateway aliases, not vendor ids; the
  gateway decides the route. Credential `LITELLM_API_KEY`.

Model fields default to empty for all three: set the `*_summary_model` (and any other model you
use) explicitly.

### 11. Deepgram

- **Capabilities:** transcription only (`deepgram_model`, default `nova-3`), with utterance-level
  speaker labels in the API response. Selectable as `diarization_provider` too.
- **Credential:** `DEEPGRAM_API_KEY`. SDK ships in the `[llm]` extra (`deepgram-sdk`).

### 12. Moss and self-hosted GPU Whisper

- **`moss`:** a self-hosted joint model that returns transcript and speaker labels in one pass
  (segment-level timestamps only, no word-level output). Config: `moss_port`, `moss_model`,
  `moss_request_timeout_sec`.
- **`tailnet_dgx_whisper`:** faster-whisper on a GPU host on a private network, with a fallback
  chain behind it (`transcription_fallback_provider`). ADR-096 and RFC-089 are not in this
  repository.

---

## Choosing, by situation

| If you need... | Reach for |
| -------------- | --------- |
| Nothing leaves your machine | Local ML, Ollama, or a self-hosted vLLM |
| Transcription without a GPU | A cloud transcription provider (`openai`, `groq`, `deepgram`, `gemini`, `mistral`) |
| Speaker labels from the transcription call itself | `deepgram` or `moss` |
| A very long context in one call | Gemini (2000000 tokens by default) |
| One endpoint in front of several vendors | `litellm` |
| EU-hosted summarization | Mistral (check the vendor's current data-residency terms) |

Measured quality, cost and speed comparisons are not part of this repository.

---

## Related Documentation

- [Ollama Provider Guide](OLLAMA_PROVIDER_GUIDE.md) — complete Ollama setup and
  troubleshooting
- [Provider Configuration Quick Reference](PROVIDER_CONFIGURATION_QUICK_REFERENCE.md)
- [Provider Implementation Guide](PROVIDER_IMPLEMENTATION_GUIDE.md)
- [ML Provider Reference](ML_PROVIDER_REFERENCE.md)
- [Audio Pipeline Guide](AUDIO_PIPELINE_GUIDE.md) — transcription providers, diarization and
  chunking
