# Configuration API

The `Config` class is the central configuration model for podcast_scraper, built on Pydantic for validation and type safety.

## Overview

All runtime options flow through the `Config` model:

```python
from podcast_scraper import Config

cfg = Config(
    rss="https://example.com/feed.xml",
    output_dir="./transcripts",
    max_episodes=50,
    transcribe_missing=True,  # Default: True (automatically transcribe missing transcripts)
    whisper_model="base.en",
    workers=8
)
```

## Helper Functions

::: podcast_scraper.config.load_config_file
    options:
      show_root_heading: true

## Environment Variables

Many configuration options can be set via environment variables for flexible deployment. Environment variables can be set:

1. **System environment variables** (highest priority among env vars)
2. **`.env` file** (loaded automatically from project root or current directory, lower priority than system env)

Environment variables are automatically loaded when the `podcast_scraper.config` module is imported using `python-dotenv`.

### Priority Order

**General Rule** (for each configuration field):

1. **Config file field** (highest priority) - if the field is set in the config file and not `null`/empty, it takes precedence
2. **Environment variable** - only used if the config file field is `null`, not set, or empty
3. **Default value** - used if neither config file nor environment variable is set

**Important**: You can define the same field in both the config file and as an environment variable. The config file value will be used if it's set (even if an environment variable is also set). This allows you to:

- Use config files for project-specific defaults (committed to repo)
- Use environment variables for deployment-specific overrides (secrets, per-environment settings)
- Override config file values by removing them from the config file (set to `null` or omit the field)

**Exception**: `LOG_LEVEL` environment variable takes precedence over config file (allows easy runtime log level control without modifying config files).

**Example**:

```yaml
# config.yaml
workers: 8
```

```bash
# .env
WORKERS=4
```

**Result**: `workers = 8` (config file wins)

```yaml
# config.yaml
# workers: (not set or null)
```

```bash
# .env
WORKERS=4
```

**Result**: `workers = 4` (env var used because config file is not set)

### Supported Environment Variables

#### RSS feed cache (optional)

**`PODCAST_SCRAPER_RSS_CACHE_DIR`**

- **Description**: If set to a writable directory, the app reads cached RSS XML from disk when present and writes the feed body after a successful HTTP fetch and parse. Reduces repeated downloads for the same feed URL (for example, many acceptance configs in one session).
- **Default**: Unset (no caching; each pipeline run fetches the feed over HTTP).
- **Example**: `export PODCAST_SCRAPER_RSS_CACHE_DIR=/tmp/podcast_rss_cache`
- **Note**: Does not cache episode media; use `reuse_media` in config for that. The acceptance test runner sets this variable per session (`sessions/session_*/rss_cache`).
- **HTTP retries**: Feed downloads use `fetch_rss_feed_url` in code, which applies urllib3 retries with exponential backoff (stronger defaults than transcript/media `fetch_url`). Not separately configurable via environment variables.

#### LLM cost estimate assumptions (optional YAML)

Pipeline **estimated USD spend** (end-of-run LLM summary, per-call estimates, and `--dry-run` projections) can use rates from a **YAML file** merged on top of built-in provider constants. This lets you refresh vendor prices **without reinstalling** the package.

| Mechanism | Description |
| --------- | ----------- |
| **Config field** | `pricing_assumptions_file` — path to the YAML (empty string = disabled; built-in rates only). |
| **Environment** | `PRICING_ASSUMPTIONS_FILE` — same as the field; follows normal [priority](#priority-order) (config file beats env when set). |
| **Default template** | Repository file `config/pricing_assumptions.yaml` (not loaded until you set the path or env). |
| **Path resolution** | Relative paths are tried against the **current working directory**, then against **ancestor directories** up to the filesystem root, so `config/pricing_assumptions.yaml` usually resolves from the repo root even if the process CWD is a subfolder. |

**YAML shape (summary)**:

- `schema_version` — integer for future migrations.
- `metadata` — human workflow only (see **Staleness** below).
- `providers.<provider>.transcription` — per-model or `default` entries with `cost_per_minute` and/or `cost_per_second` (Gemini-style audio).
- `providers.<provider>.text` — rates for **both** speaker detection and summarization: `input_cost_per_1m_tokens`, `output_cost_per_1m_tokens` (optional: `cache_hit_input_cost_per_1m_tokens` for DeepSeek). Model keys match by **exact id** or **longest substring** (e.g. `gpt-4o-mini` wins over `gpt-4o`).

**Merge rule**: For each lookup, built-in `*Provider.get_pricing()` values are copied, then **any key present in YAML overwrites** the built-in value. Missing YAML entries keep code defaults.

##### Staleness: when estimates may be out of date

Staleness is **advisory**: it does **not** change runtime math. It exists so teams remember to re-check vendor pages.

| YAML key | Role |
| -------- | ---- |
| `metadata.last_reviewed` | **ISO date** (`YYYY-MM-DD`) of the last time someone compared this file to vendor pricing pages. |
| `metadata.stale_review_after_days` | Positive integer. If **calendar age** of `last_reviewed` exceeds this many **whole days**, the file is considered stale. |
| `metadata.pricing_effective_date` | Optional note of which price sheet / period the numbers reflect (documentation only). |
| `metadata.source_urls` | Map of provider → official pricing URL (open in browser when refreshing). |

**Stale condition** (all must hold):

1. `metadata` is a mapping.
2. `last_reviewed` parses as a date.
3. `stale_review_after_days` is an integer **greater than zero**.
4. `(today_utc_date - last_reviewed).days > stale_review_after_days`

If `last_reviewed` or `stale_review_after_days` is missing or invalid, **no staleness message** is emitted (not treated as stale).

**Operational workflow**

1. Run [`pricing-assumptions` CLI](CLI.md#pricing-assumptions-command): `python -m podcast_scraper.cli pricing-assumptions` (or `make check-pricing-assumptions`).
2. If the report prints a **Staleness** section, open each URL under `source_urls`, update numeric rates in the YAML, then set **`last_reviewed`** to today and optionally **`pricing_effective_date`**.
3. For automation, pass **`--strict`**: exit code **1** when the stale condition holds (e.g. CI or pre-release). Example: `make check-pricing-assumptions ARGS='--strict'`.

**Related**: [Cost projection / dry-run](CLI.md#cost-projection-in-dry-run-mode) uses the same pricing resolution when the YAML is enabled.

#### OpenAI API Configuration

**`OPENAI_API_KEY`**

- **Description**: OpenAI API key for OpenAI-based providers (transcription, speaker detection, summarization)
- **Required**: Yes, when using OpenAI providers (`transcription_provider=openai`, `speaker_detector_provider=openai`, or `summary_provider=openai`)
- **Example**: `export OPENAI_API_KEY=sk-your-actual-api-key-here`
- **Security**: Never commit `.env` files containing API keys. API keys are never logged or exposed in error messages.

#### OpenAI Model Configuration

OpenAI providers support configurable model selection for dev/test vs production environments.

| Field | CLI Flag | Default | Description |
| ------- | ---------- | --------- | ------------- |
| `openai_api_key` | `--openai-api-key` | `None` (from env var) | OpenAI API key (or set `OPENAI_API_KEY` env var) |
| `openai_transcription_model` | `--openai-transcription-model` | `whisper-1` | OpenAI model for transcription |
| `openai_speaker_model` | `--openai-speaker-model` | `gpt-4o-mini` | OpenAI model for speaker detection |
| `openai_summary_model` | `--openai-summary-model` | `gpt-4o-mini` | OpenAI model for summarization |
| `openai_insight_model` | `--openai-insight-model` | `None` (uses `openai_summary_model`) | OpenAI model for GIL `generate_insights` when `gi_insight_source` is `provider` |
| `openai_temperature` | `--openai-temperature` | `0.3` | Temperature for generation (0.0-2.0) |
| `openai_max_tokens` | `--openai-max-tokens` | `None` | Max tokens for generation (None = model default) |

**Summary bullets (all API `summary_provider` values):** Default user templates expect JSON bullet output. Global `summary_prompt_params` supplies `bullet_min` (soft minimum), `max_words_per_bullet`, and optional `bullet_max` (hard cap). Config points at logical names under `prompts/<provider>/summarization/`; if that file is missing, the loader uses `prompts/shared/summarization/` with the **same filename** (see [RFC-017 — Shared summarization templates vs per-provider overrides](../rfc/RFC-017-prompt-management.md) and `src/podcast_scraper/prompts/shared/README.md` in the repository). Parsed JSON with a **title** (or episode title fallback) and **bullets** is stored as **`schema_status: valid`** even when `key_quotes` / `named_entities` are omitted (bullet-only contract).

**Note on Transcription Limits**: The OpenAI Whisper API has a **25 MB file size limit**. The system proactively checks file sizes via HTTP HEAD requests before downloading or attempting transcription with the OpenAI provider. Episodes exceeding this limit will be skipped to avoid API errors and unnecessary bandwidth usage.

**Recommended Models by Environment**:

| Purpose | Test/Dev | Production | Notes |
| --------- | ---------- | ------------ | ------- |
| Transcription | `whisper-1` | `whisper-1` | Only OpenAI option |
| Speaker Detection | `gpt-4o-mini` | `gpt-4o` | Mini is fast/cheap; 4o is more accurate |
| Summarization | `gpt-4o-mini` | `gpt-4o` | Mini is fast/cheap; 4o produces better summaries |

#### Gemini API Configuration

**`GEMINI_API_KEY`**

- **Description**: Google AI (Gemini) API key for Gemini-based providers (transcription, speaker detection, summarization)
- **Required**: Yes, when using Gemini providers (`transcription_provider=gemini`, `speaker_detector_provider=gemini`, or `summary_provider=gemini`)
- **Example**: `export GEMINI_API_KEY=your-actual-api-key-here`
- **Security**: Never commit `.env` files containing API keys. API keys are never logged or exposed in error messages.
- **Getting an API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to generate a Gemini API key.

#### Mistral API Configuration

**`MISTRAL_API_KEY`**

- **Description**: Mistral AI API key for Mistral-based providers (transcription, speaker detection, summarization)
- **Required**: Yes, when using Mistral providers (`transcription_provider=mistral`, `speaker_detector_provider=mistral`, or `summary_provider=mistral`)
- **Example**: `export MISTRAL_API_KEY=your-actual-api-key-here`
- **Security**: Never commit `.env` files containing API keys. API keys are never logged or exposed in error messages.
- **Getting an API Key**: Visit [Mistral AI Platform](https://console.mistral.ai/) to generate a Mistral API key.
- **Note**: Mistral is a full-stack provider (all three capabilities) with EU data residency support.

#### Anthropic API Configuration

**`ANTHROPIC_API_KEY`**

- **Description**: Anthropic API key for Anthropic-based providers (speaker detection, summarization)
- **Required**: Yes, when using Anthropic providers (`speaker_detector_provider=anthropic` or `summary_provider=anthropic`)
- **Example**: `export ANTHROPIC_API_KEY=your-actual-api-key-here`
- **Security**: Never commit `.env` files containing API keys. API keys are never logged or exposed in error messages.
- **Getting an API Key**: Visit [Anthropic Console](https://console.anthropic.com/) to generate an Anthropic API key.
- **Note**: Anthropic does NOT support native audio transcription. Use `whisper` (local) or `openai` for transcription.

#### Anthropic Model Configuration

Anthropic providers support configurable model selection for dev/test vs production environments.

| Field | CLI Flag | Default | Description |
| ------- | ---------- | --------- | ------------- |
| `anthropic_speaker_model` | `--anthropic-speaker-model` | `claude-haiku-4-5` (test) / `claude-3-5-sonnet-20241022` (prod) | Anthropic model for speaker detection |
| `anthropic_summary_model` | `--anthropic-summary-model` | `claude-haiku-4-5` (test) / `claude-3-5-sonnet-20241022` (prod) | Anthropic model for summarization |
| `anthropic_temperature` | `--anthropic-temperature` | `0.3` | Temperature for generation (0.0-1.0) |
| `anthropic_max_tokens` | `--anthropic-max-tokens` | `None` | Max tokens for generation (None = model default) |

**Recommended Models by Environment**:

| Purpose | Test/Dev | Production | Notes |
| --------- | ---------- | ------------ | ------- |
| Speaker Detection | `claude-haiku-4-5` | `claude-3-5-sonnet-20241022` | Haiku is fast/cheap; Sonnet is more accurate |
| Summarization | `claude-haiku-4-5` | `claude-3-5-sonnet-20241022` | Haiku is fast/cheap; Sonnet produces better summaries |

**Note on Transcription**: Anthropic does NOT support native audio transcription. If you set `transcription_provider=anthropic`, you will get a clear error message suggesting alternatives (`whisper` or `openai`).

#### Gemini Model Configuration

Gemini providers support configurable model selection for dev/test vs production environments.

| Field | CLI Flag | Default | Description |
| ------- | ---------- | --------- | ------------- |
| `gemini_transcription_model` | `--gemini-transcription-model` | `gemini-1.5-flash` (test) / `gemini-1.5-pro` (prod) | Gemini model for transcription |
| `gemini_speaker_model` | `--gemini-speaker-model` | `gemini-1.5-flash` (test) / `gemini-1.5-pro` (prod) | Gemini model for speaker detection |
| `gemini_summary_model` | `--gemini-summary-model` | `gemini-1.5-flash` (test) / `gemini-1.5-pro` (prod) | Gemini model for summarization |
| `gemini_temperature` | `--gemini-temperature` | `0.3` | Temperature for generation (0.0-2.0) |
| `gemini_max_tokens` | `--gemini-max-tokens` | `None` | Max tokens for generation (None = model default) |

**Note on Transcription Limits**: The Gemini API has file size limits that vary by model. The system proactively checks file sizes via HTTP HEAD requests before downloading or attempting transcription with the Gemini provider. Episodes exceeding limits will be skipped to avoid API errors and unnecessary bandwidth usage.

**Recommended Models by Environment**:

| Purpose | Test/Dev | Production | Notes |
| --------- | ---------- | ------------ | ------- |
| Transcription | `gemini-1.5-flash` | `gemini-1.5-pro` | Flash is fast/cheap; Pro is higher quality |
| Speaker Detection | `gemini-1.5-flash` | `gemini-1.5-pro` | Flash is fast/cheap; Pro is more accurate |
| Summarization | `gemini-1.5-flash` | `gemini-1.5-pro` | Flash is fast/cheap; Pro produces better summaries |

**Example** (OpenAI config file):

```yaml
transcription_provider: openai
speaker_detector_provider: openai
summary_provider: openai
openai_transcription_model: whisper-1
openai_speaker_model: gpt-4o      # Production: higher quality
openai_summary_model: gpt-4o      # Production: better summaries
openai_temperature: 0.3           # Lower = more deterministic
```

**Example** (OpenAI CLI):

```bash
podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider openai \
  --speaker-detector-provider openai \
  --summary-provider openai \
  --openai-speaker-model gpt-4o \
  --openai-summary-model gpt-4o
```

**Example** (Gemini config file):

```yaml
transcription_provider: gemini
speaker_detector_provider: gemini
summary_provider: gemini
gemini_transcription_model: gemini-1.5-pro      # Production: higher quality
gemini_speaker_model: gemini-1.5-pro           # Production: more accurate
gemini_summary_model: gemini-1.5-pro           # Production: better summaries
gemini_temperature: 0.3                        # Lower = more deterministic
gemini_max_tokens: null                        # Use model default
```

**Example** (Mistral config file):

```yaml
transcription_provider: mistral
speaker_detector_provider: mistral
summary_provider: mistral
mistral_transcription_model: voxtral-mini-latest      # Default: voxtral-mini-latest
mistral_speaker_model: mistral-large-latest           # Default: environment-based
mistral_summary_model: mistral-large-latest           # Default: environment-based
mistral_temperature: 0.3                              # Lower = more deterministic
mistral_max_tokens: null                              # Use model default
```

**Example** (Mistral CLI):

```bash
podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider mistral \
  --speaker-detector-provider mistral \
  --summary-provider mistral \
  --mistral-speaker-model mistral-large-latest \
  --mistral-summary-model mistral-large-latest
```

**Example** (Gemini CLI):

```bash
podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider gemini \
  --speaker-detector-provider gemini \
  --summary-provider gemini \
  --gemini-transcription-model gemini-1.5-pro \
  --gemini-speaker-model gemini-1.5-pro \
  --gemini-summary-model gemini-1.5-pro \
  --gemini-temperature 0.3
```

#### Mistral Model Configuration

Mistral providers support configurable model selection for dev/test vs production environments.

| Field | CLI Flag | Default | Description |
| ------- | ---------- | --------- | ------------- |
| `mistral_transcription_model` | `--mistral-transcription-model` | `voxtral-mini-latest` | Mistral Voxtral model for transcription |
| `mistral_speaker_model` | `--mistral-speaker-model` | `mistral-small-latest` (test) / `mistral-large-latest` (prod) | Mistral model for speaker detection |
| `mistral_summary_model` | `--mistral-summary-model` | `mistral-small-latest` (test) / `mistral-large-latest` (prod) | Mistral model for summarization |
| `mistral_temperature` | `--mistral-temperature` | `0.3` | Temperature for generation (0.0-1.0) |
| `mistral_max_tokens` | `--mistral-max-tokens` | `None` | Max tokens for generation (None = model default) |

**Recommended Models by Environment**:

| Purpose | Test/Dev | Production | Notes |
| --------- | ---------- | ------------ | ------- |
| Transcription | `voxtral-mini-latest` | `voxtral-mini-latest` | Single model for all environments |
| Speaker Detection | `mistral-small-latest` | `mistral-large-latest` | Small is fast/cheap; Large is more accurate |
| Summarization | `mistral-small-latest` | `mistral-large-latest` | Small is fast/cheap; Large produces better summaries |

#### DeepSeek API Configuration

**`DEEPSEEK_API_KEY`**

- **Description**: DeepSeek API key for DeepSeek-based providers (speaker detection, summarization)
- **Required**: Yes, when using DeepSeek providers (`speaker_detector_provider=deepseek` or `summary_provider=deepseek`)
- **Example**: `export DEEPSEEK_API_KEY=your-actual-api-key-here`
- **Security**: Never commit `.env` files containing API keys. API keys are never logged or exposed in error messages.
- **Getting an API Key**: Visit [DeepSeek Platform](https://platform.deepseek.com/) to generate a DeepSeek API key.
- **Note**: DeepSeek does NOT support native audio transcription. Use `whisper` (local), `openai`, `gemini`, or `mistral` for transcription. DeepSeek is 95% cheaper than OpenAI for text processing.

#### DeepSeek Model Configuration

DeepSeek providers support configurable model selection.

| Field | CLI Flag | Default | Description |
| ------- | ---------- | --------- | ------------- |
| `deepseek_speaker_model` | `--deepseek-speaker-model` | `deepseek-chat` | DeepSeek model for speaker detection |
| `deepseek_summary_model` | `--deepseek-summary-model` | `deepseek-chat` | DeepSeek model for summarization |
| `deepseek_temperature` | `--deepseek-temperature` | `0.3` | Temperature for generation (0.0-2.0) |
| `deepseek_max_tokens` | `--deepseek-max-tokens` | `None` | Max tokens for generation (None = model default) |

**Recommended Models by Environment**:

| Purpose | Test/Dev | Production | Notes |
| --------- | ---------- | ------------ | ------- |
| Speaker Detection | `deepseek-chat` | `deepseek-chat` | Same model for all environments (extremely cheap) |
| Summarization | `deepseek-chat` | `deepseek-chat` | Same model for all environments (extremely cheap) |

**Example** (DeepSeek config file):

```yaml
speaker_detector_provider: deepseek
summary_provider: deepseek
deepseek_speaker_model: deepseek-chat
deepseek_summary_model: deepseek-chat
deepseek_temperature: 0.3
```

**Example** (DeepSeek CLI):

```bash
podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider deepseek \
  --summary-provider deepseek \
  --deepseek-speaker-model deepseek-chat \
  --deepseek-summary-model deepseek-chat \
  --deepseek-temperature 0.3
```

#### Grok API Configuration

**`GROK_API_KEY`**

- **Description**: Grok API key for Grok-based providers (speaker detection, summarization)
- **Required**: Yes, when using Grok providers (`speaker_detector_provider=grok` or `summary_provider=grok`)
- **Example**: `export GROK_API_KEY=your-actual-api-key-here`
- **Security**: Never commit `.env` files containing API keys. API keys are never logged or exposed in error messages.
- **Getting an API Key**: Visit [xAI Platform](https://console.x.ai/) to generate a Grok API key.
- **Note**: Grok does NOT support native audio transcription. Use `whisper` (local), `openai`, `gemini`, or `mistral` for transcription. Grok provides real-time information access.

#### Grok Model Configuration

Grok providers support configurable model selection.

| Field | CLI Flag | Default | Description |
| ------- | ---------- | --------- | ------------- |
| `grok_speaker_model` | `--grok-speaker-model` | `grok-2` | Grok model for speaker detection |
| `grok_summary_model` | `--grok-summary-model` | `grok-2` | Grok model for summarization |
| `grok_temperature` | `--grok-temperature` | `0.3` | Temperature for generation (0.0-2.0) |
| `grok_max_tokens` | `--grok-max-tokens` | `None` | Max tokens for generation (None = model default) |

**Recommended Models by Environment**:

| Purpose | Test/Dev | Production | Notes |
| --------- | ---------- | ------------ | ------- |
| Speaker Detection | `grok-2` | `grok-2` | Same model for all environments |
| Summarization | `grok-2` | `grok-2` | Same model for all environments |

**Example** (Grok config file):

```yaml
speaker_detector_provider: grok
summary_provider: grok
grok_speaker_model: grok-2
grok_summary_model: grok-2
grok_temperature: 0.3
```

**Example** (Grok CLI):

```bash
podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider grok \
  --summary-provider grok \
  --grok-speaker-model grok-2 \
  --grok-summary-model grok-2 \
  --grok-temperature 0.3
```

#### Ollama API Configuration

**`OLLAMA_API_BASE`**

- **Description**: Ollama API base URL (for local Ollama server or remote instance)
- **Required**: No (defaults to `http://localhost:11434/v1` if not specified)
- **Example**: `export OLLAMA_API_BASE=http://localhost:11434/v1`
- **Note**: Ollama is a local, self-hosted solution. No API key is required. The server must be running locally or accessible at the specified URL.

#### Ollama Model Configuration

Ollama providers support configurable model selection for dev/test vs production environments.

| Field | CLI Flag | Default | Description |
| ------- | ---------- | --------- | ------------- |
| `ollama_speaker_model` | `--ollama-speaker-model` | `llama3.2:latest` (test) / `llama3.3:latest` (prod) | Ollama model for speaker detection |
| `ollama_summary_model` | `--ollama-summary-model` | `llama3.1:8b` | Ollama model for summarization |
| `ollama_temperature` | `--ollama-temperature` | `0.3` | Temperature for generation (0.0-2.0) |
| `ollama_max_tokens` | `--ollama-max-tokens` | `None` | Max tokens for generation (None = model default) |
| `ollama_timeout` | `--ollama-timeout` | `120` | Timeout in seconds for API calls (local inference can be slow) |

**Important Notes**:

- **No Transcription Support**: Ollama does NOT support transcription (no audio API). Use `whisper` (local) or `openai` for transcription.
- **Zero Cost**: Ollama is completely free - all processing happens on your local hardware with no per-token pricing.
- **Prerequisites**: Ollama must be installed and running locally. Models must be pulled before use (e.g., `ollama pull llama3.3`).
- **Privacy**: All data stays on your local machine - perfect for sensitive content or air-gapped environments.

**Recommended Models by Environment**:

| Purpose | Test/Dev | Production | Notes |
| --------- | ---------- | ------------ | ------- |
| Speaker Detection | `llama3.2:latest` | `llama3.3:latest` | 3.2 is smaller/faster; 3.3 is higher quality |
| Summarization | `llama3.2:latest` | `llama3.3:latest` | 3.2 is smaller/faster; 3.3 produces better summaries |

**Example** (Ollama config file):

```yaml
speaker_detector_provider: ollama
summary_provider: ollama
ollama_api_base: http://localhost:11434/v1  # Default, can be omitted
ollama_speaker_model: llama3.3:latest       # Production: higher quality
ollama_summary_model: llama3.3:latest       # Production: better summaries
ollama_temperature: 0.3                      # Lower = more deterministic
ollama_timeout: 300                          # 5 minutes for slow inference
```

**Example** (Ollama CLI):

```bash
podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider ollama \
  --summary-provider ollama \
  --ollama-speaker-model llama3.3:latest \
  --ollama-summary-model llama3.3:latest \
  --ollama-temperature 0.3 \
  --ollama-timeout 300
```

#### Hybrid ML (MAP-REDUCE) Configuration

When `summary_provider: hybrid_ml`, summarization uses a local MAP model (e.g. LongT5) for chunk summarization and a configurable REDUCE backend for the final synthesis.

| Field | CLI Flag | Default | Description |
| ------- | ---------- | --------- | ------------- |
| `hybrid_map_model` | `--hybrid-map-model` | `longt5-base` | HuggingFace model for MAP phase (chunk summarization). |
| `hybrid_map_device` | `--hybrid-map-device` | (auto) | Device for MAP (e.g. `mps`, `cuda`, `cpu`, `auto`). |
| `hybrid_reduce_model` | `--hybrid-reduce-model` | `google/flan-t5-base` | REDUCE model: HuggingFace ID (transformers), Ollama tag (e.g. `llama3.1:8b`) for ollama, or path to GGUF file for llama_cpp. |
| `hybrid_reduce_backend` | `--hybrid-reduce-backend` | `transformers` | REDUCE backend: `transformers`, `ollama`, or `llama_cpp`. |
| `hybrid_reduce_device` | `--hybrid-reduce-device` | (auto) | Device for REDUCE when backend is transformers (e.g. `mps`, `cuda`, `cpu`). |
| `hybrid_reduce_n_ctx` | N/A | (optional) | Context size for llama_cpp REDUCE (default 4096). Config file only. |
| `hybrid_internal_preprocessing_after_pattern` | `--hybrid-internal-preprocessing-after-pattern` | `cleaning_hybrid_after_pattern` | When `transcript_cleaning_strategy` is `pattern`, preprocessing profile applied **inside** `HybridMLProvider.summarize` after workflow pattern cleaning (avoids repeating full `cleaning_v4` sponsor/outro work). Registered profile id from `preprocessing.profiles`. |

**Layered cleaning (Issue #419):** The pipeline runs `transcript_cleaning_strategy` (and `HybridMLProvider.cleaning_processor`) **before** `summarize()`. For `pattern` + `hybrid_ml`, the workflow injects `preprocessing_profile: hybrid_internal_preprocessing_after_pattern` so MAP sees v4-only delta steps without duplicate sponsor passes. For `llm` / `hybrid` strategies, internal preprocessing stays **`cleaning_v4`**. Details: [RFC-042 § Layered transcript cleaning](../rfc/RFC-042-hybrid-summarization-pipeline.md#layered-transcript-cleaning-issue-419).

**Example** (Hybrid ML with Ollama REDUCE):

```yaml
summary_provider: hybrid_ml
hybrid_map_model: longt5-base
hybrid_reduce_backend: ollama
hybrid_reduce_model: llama3.1:8b
```

**Example** (Hybrid ML with transformers REDUCE):

```yaml
summary_provider: hybrid_ml
hybrid_map_model: longt5-base
hybrid_reduce_backend: transformers
hybrid_reduce_model: google/flan-t5-base
hybrid_reduce_device: mps
```

**Example** (Hybrid ML + pattern strategy + layered internal profile, Issue #419):

```yaml
summary_provider: hybrid_ml
transcript_cleaning_strategy: pattern
hybrid_internal_preprocessing_after_pattern: cleaning_hybrid_after_pattern
hybrid_map_model: longt5-base
hybrid_reduce_backend: transformers
hybrid_reduce_model: google/flan-t5-base
```

See [ML Provider Reference](../guides/ML_PROVIDER_REFERENCE.md#hybrid-ml-provider-summary_provider-hybrid_ml) and [Ollama Provider Guide](../guides/OLLAMA_PROVIDER_GUIDE.md) for details.

#### Logging Configuration

**`LOG_LEVEL`**

- **Description**: Default logging level for the application
- **Required**: No (defaults to "INFO" if not specified)
- **Valid Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Priority**: Takes precedence over config file `log_level` field (unlike other variables)
- **Behavior notes**:
  - At **`DEBUG`**, the CLI also logs the full resolved configuration (field-by-field detail).
  - At **`INFO`** and above, HTTP client libraries (`httpx`, `httpcore`) are capped at **WARNING** so routine requests do not flood the log.
  - End-of-run artifacts (metrics, run index, run summary) log per-file paths at **DEBUG**; a single **INFO** line lists all successfully written artifact paths.

#### Performance Configuration

**`WORKERS`**

- **Description**: Number of parallel download workers
- **Required**: No (defaults to CPU count bounded between 1 and 8)
- **Priority**: Config file → Environment variable → Default
- **Use Cases**: Docker containers (`WORKERS=2`), High-performance servers (`WORKERS=8`), CI/CD (`WORKERS=1`)

**`TRANSCRIPTION_PARALLELISM`**

- **Description**: Number of episodes to transcribe in parallel (episode-level parallelism)
- **Required**: No (defaults to 1 for sequential processing)
- **Priority**: Config file → Environment variable → Default
- **Note**:
  - **Local Whisper**: Now respects configured parallelism for experimentation. Values > 1 are **EXPERIMENTAL** and not production-ready. May cause memory/GPU contention. Use with caution.
  - **OpenAI provider**: Uses configured parallelism for parallel API calls.
- **Use Cases**:
  - OpenAI provider (`TRANSCRIPTION_PARALLELISM=3` for parallel API calls)
  - Whisper experimentation (`TRANSCRIPTION_PARALLELISM=2` for testing parallel transcription - experimental)
  - Rate limit tuning
- **Logging**: When using Whisper with parallelism > 1, logs will show a warning indicating experimental status.

**`PROCESSING_PARALLELISM`**

- **Description**: Number of episodes to process (metadata/summarization) in parallel
- **Required**: No (defaults to 2)
- **Priority**: Config file → Environment variable → Default
- **Use Cases**: Memory-constrained environments (`PROCESSING_PARALLELISM=1`), High-memory servers (`PROCESSING_PARALLELISM=4`)

**`SUMMARY_BATCH_SIZE`**

- **Description**: Number of episodes to summarize in parallel (episode-level parallelism)
- **Required**: No (defaults to 2)
- **Priority**: Config file → Environment variable → Default
- **Use Cases**: Memory-bound for local providers, Rate-limited for API providers (OpenAI)

**`SUMMARY_CHUNK_PARALLELISM`**

- **Description**: Number of chunks to process in parallel within a single episode (CPU-bound, local providers only)
- **Required**: No (defaults to 1)
- **Priority**: Config file → Environment variable → Default
- **Note**: API providers handle parallelism internally via rate limiting
- **Use Cases**: Multi-core CPUs (`SUMMARY_CHUNK_PARALLELISM=2`), Single-core or memory-limited (`SUMMARY_CHUNK_PARALLELISM=1`)

**`TIMEOUT`**

- **Description**: Request timeout in seconds for HTTP requests
- **Required**: No (defaults to 20 seconds)
- **Minimum Value**: 1 second
- **Priority**: Config file → Environment variable → Default
- **Use Cases**: Slow networks (`TIMEOUT=60`), CI/CD (`TIMEOUT=30`), Fast networks (`TIMEOUT=10`)

**`SUMMARY_DEVICE`**

- **Description**: Device for summarization model execution (CPU, CUDA, MPS, or None for auto-detection)
- **Required**: No (defaults to None for auto-detection)
- **Valid Values**: `cpu`, `cuda`, `mps`, or empty string (for None/auto-detect)
- **Priority**: Config file → Environment variable → Default
- **Use Cases**: Docker containers (`SUMMARY_DEVICE=cpu`), CI/CD (`SUMMARY_DEVICE=cpu`), NVIDIA GPU (`SUMMARY_DEVICE=cuda` or auto-detect), Apple Silicon (`SUMMARY_DEVICE=mps` or auto-detect)

**`WHISPER_DEVICE`**

- **Description**: Device for Whisper transcription (CPU, CUDA, MPS, or None for auto-detection)
- **Required**: No (defaults to None for auto-detection)
- **Valid Values**: `cpu`, `cuda`, `mps`, or empty string (for None/auto-detect)
- **Priority**: Config file → Environment variable → Default
- **Use Cases**: Docker containers (`WHISPER_DEVICE=cpu`), CI/CD (`WHISPER_DEVICE=cpu`), NVIDIA GPU (`WHISPER_DEVICE=cuda` or auto-detect), Apple Silicon (`WHISPER_DEVICE=mps` or auto-detect)

**`TRANSCRIPTION_DEVICE`** (Issue #387)

- **Description**: Device for transcription stage (overrides provider-specific device like `WHISPER_DEVICE`)
- **Required**: No (defaults to None for auto-detection)
- **Valid Values**: `cpu`, `cuda`, `mps`, or empty string (for None/auto-detect)
- **Priority**: Config file → Environment variable → Default
- **Use Cases**:
  - CPU/GPU mix to regain overlap: `TRANSCRIPTION_DEVICE=cpu` with `SUMMARIZATION_DEVICE=mps` allows concurrent processing
  - Force CPU for transcription: `TRANSCRIPTION_DEVICE=cpu` (useful when GPU memory is limited)
- **Behavior**: Stage-level device config takes precedence over provider-specific device config. Allows independent device selection per stage.

**`SUMMARIZATION_DEVICE`** (Issue #387)

- **Description**: Device for summarization stage (overrides provider-specific device like `SUMMARY_DEVICE`)
- **Required**: No (defaults to None for auto-detection)
- **Valid Values**: `cpu`, `cuda`, `mps`, or empty string (for None/auto-detect)
- **Priority**: Config file → Environment variable → Default
- **Use Cases**:
  - CPU/GPU mix to regain overlap: `SUMMARIZATION_DEVICE=mps` with `TRANSCRIPTION_DEVICE=cpu` allows concurrent processing
  - Force CPU for summarization: `SUMMARIZATION_DEVICE=cpu` (useful when GPU memory is limited)
- **Behavior**: Stage-level device config takes precedence over provider-specific device config. Allows independent device selection per stage.

**`MPS_EXCLUSIVE`**

- **Description**: Serialize GPU work on MPS to prevent memory contention between Whisper transcription and summarization
- **Required**: No (defaults to `true` for safer behavior)
- **Valid Values**: `1`, `true`, `yes`, `on` (enable) or `0`, `false`, `no`, `off` (disable)
- **Priority**: Config file → Environment variable → Default
- **Use Cases**:
  - Apple Silicon with limited GPU memory: `MPS_EXCLUSIVE=1` (default) prevents both models from competing for GPU memory
  - Systems with sufficient GPU memory: `MPS_EXCLUSIVE=0` allows concurrent GPU operations for better throughput
- **Behavior**: When enabled and both transcription and summarization stages use MPS (checked via stage-level device config), transcription completes before summarization starts. I/O operations (downloads, parsing) remain parallel. With CPU/GPU mix (e.g., `TRANSCRIPTION_DEVICE=cpu`, `SUMMARIZATION_DEVICE=mps`), serialization is not needed and overlap is regained (Issue #387).
- **Related**: See [Segfault Mitigation Guide](../guides/SEGFAULT_MITIGATION.md) for MPS stability issues

#### Audio Preprocessing Configuration (RFC-040)

Audio preprocessing optimizes audio files before transcription, reducing file size and improving transcription quality. This is especially useful for API-based transcription providers with file size limits (e.g., OpenAI 25 MB limit).

| Field | CLI Flag | Default | Description |
| ------- | ---------- | --------- | ------------- |
| `preprocessing_enabled` | `--enable-preprocessing` | `false` | Enable audio preprocessing before transcription |
| `preprocessing_cache_dir` | `--preprocessing-cache-dir` | `.cache/preprocessing` | Custom cache directory for preprocessed audio |
| `preprocessing_sample_rate` | `--preprocessing-sample-rate` | `16000` | Target sample rate in Hz (must be Opus-supported: 8000, 12000, 16000, 24000, 48000) |
| `preprocessing_silence_threshold` | `--preprocessing-silence-threshold` | `-50dB` | Silence detection threshold for VAD |
| `preprocessing_silence_duration` | `--preprocessing-silence-duration` | `2.0` | Minimum silence duration to remove in seconds |
| `preprocessing_target_loudness` | `--preprocessing-target-loudness` | `-16` | Target loudness in LUFS for normalization |

**Preprocessing Pipeline**:

1. Converts audio to mono
2. Resamples to configured sample rate (default: 16 kHz)
3. Removes silence using Voice Activity Detection (VAD)
4. Normalizes loudness to configured target (default: -16 LUFS)
5. Compresses using Opus codec at 24 kbps

**Benefits**:

- **File Size Reduction**: Typically 10-25× smaller (50 MB → 2-5 MB)
- **API Compatibility**: Ensures files fit within 25 MB limit for OpenAI/Grok
- **Cost Savings**: Reduces API costs by processing less audio (30-60% reduction)
- **Performance**: Faster transcription for both local and API providers
- **Caching**: Preprocessed audio is cached to avoid reprocessing

**Requirements**:

- `ffmpeg` must be installed and available in PATH
- If `ffmpeg` is not available, preprocessing is automatically disabled with a warning

**Example** (config file):

```yaml
preprocessing_enabled: true
preprocessing_sample_rate: 16000
preprocessing_silence_threshold: -50dB
preprocessing_silence_duration: 2.0
preprocessing_target_loudness: -16
```

**Example** (CLI):

```bash
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --enable-preprocessing \
  --preprocessing-sample-rate 16000
```

#### Transcript Cleaning Configuration (Issue #418)

Transcript cleaning removes unwanted content (sponsor blocks, ads, timestamps) from transcripts before summarization. This improves summary quality by ensuring models focus on the core content.

**Cleaning Strategies**:

| Strategy | Description | Use Case |
| -------- | ----------- | -------- |
| `pattern` | Pattern-based cleaning using regex rules (default for ML providers) | Fast, deterministic, no API costs |
| `llm` | LLM-based semantic cleaning using language models | Highest quality, removes semantic noise |
| `hybrid` | Pattern-based first, then conditional LLM cleaning (default for LLM providers) | Best balance of quality and cost |

**Default Behavior**:

- **LLM Providers** (OpenAI, Gemini, Anthropic, etc.): `hybrid` (pattern + conditional LLM)
- **ML Providers** (`transformers`): typically `pattern` (pattern-based only; Config default is still `hybrid`, but hybrid LLM cleaning is not used the same way without an API cleaner)
- **Hybrid ML** (`hybrid_ml`): same `transcript_cleaning_strategy` / `cleaning_processor` behavior as API providers; combine with `hybrid_internal_preprocessing_after_pattern` when using `pattern` to layer internal MAP preprocessing (see Hybrid ML table above and RFC-042)

**Hybrid Strategy Details**:

The hybrid strategy applies pattern-based cleaning first, then conditionally uses LLM cleaning when:

- Pattern-based cleaning reduces text by less than the threshold (default: 10%)
- Heuristics detect sponsor keywords or high promotional density

This reduces LLM API calls by 70-90% while maintaining high quality.

**Configuration Fields**:

| Field | CLI Flag | Default | Description |
| ------- | ---------- | --------- | ------------- |
| `transcript_cleaning_strategy` | `--transcript-cleaning-strategy` | `hybrid` | Cleaning strategy: `pattern`, `llm`, or `hybrid` (applies to LLM + `hybrid_ml` summarization providers). Pair with `hybrid_internal_preprocessing_after_pattern` under [Hybrid ML](#hybrid-ml-map-reduce-configuration) when using **`hybrid_ml`** + **`pattern`**. |
| `llm_pipeline_mode` | N/A | `staged` | Issue #477: `staged` = separate semantic clean + summarize; `bundled` = one structured completion when the provider implements `summarize_bundled` (OpenAI, Anthropic, Gemini), with automatic fallback to staged on failure. Config file only unless wired via CLI later. |
| `llm_bundled_max_output_tokens` | N/A | `16384` | Max completion tokens for bundled clean+summary+bullets (large default because JSON includes full `cleaned_text`). Config file only. |
| `openai_cleaning_model` | `--openai-cleaning-model` | `gpt-4o-mini` | OpenAI model for cleaning (cheaper than summary model) |
| `openai_cleaning_temperature` | `--openai-cleaning-temperature` | `0.2` | Temperature for OpenAI cleaning (lower = more deterministic) |
| `gemini_cleaning_model` | `--gemini-cleaning-model` | `gemini-1.5-flash` | Gemini model for cleaning (cheaper than summary model) |
| `gemini_cleaning_temperature` | `--gemini-cleaning-temperature` | `0.2` | Temperature for Gemini cleaning (lower = more deterministic) |
| `anthropic_cleaning_model` | `--anthropic-cleaning-model` | `claude-haiku-4-5` | Anthropic model for cleaning (cheaper than summary model) |
| `anthropic_cleaning_temperature` | `--anthropic-cleaning-temperature` | `0.2` | Temperature for Anthropic cleaning (lower = more deterministic) |
| `mistral_cleaning_model` | `--mistral-cleaning-model` | `mistral-small-latest` | Mistral model for cleaning (cheaper than summary model) |
| `mistral_cleaning_temperature` | `--mistral-cleaning-temperature` | `0.2` | Temperature for Mistral cleaning (lower = more deterministic) |
| `deepseek_cleaning_model` | `--deepseek-cleaning-model` | `deepseek-chat` | DeepSeek model for cleaning (cheaper than summary model) |
| `deepseek_cleaning_temperature` | `--deepseek-cleaning-temperature` | `0.2` | Temperature for DeepSeek cleaning (lower = more deterministic) |
| `grok_cleaning_model` | `--grok-cleaning-model` | `grok-3-mini` | Grok model for cleaning (cheaper than summary model) |
| `grok_cleaning_temperature` | `--grok-cleaning-temperature` | `0.2` | Temperature for Grok cleaning (lower = more deterministic) |
| `ollama_cleaning_model` | `--ollama-cleaning-model` | `llama3.1:8b` | Ollama model for cleaning (smaller/faster than summary model) |
| `ollama_cleaning_temperature` | `--ollama-cleaning-temperature` | `0.2` | Temperature for Ollama cleaning (lower = more deterministic) |
| `{provider}_cleaning_max_tokens` | N/A | `None` (80-90% of input) | Max tokens for cleaning output (config file only) |
| `{provider}_cleaning_llm_threshold` | N/A | `0.10` | Reduction ratio threshold for hybrid cleaning (0.0-1.0, config file only) |

**Provider-Specific Cleaning Models**:

Each LLM provider can use a different model for cleaning (typically cheaper/faster than summarization):

- **OpenAI**: `openai_cleaning_model` (defaults to `gpt-4o-mini` via `TEST_DEFAULT_OPENAI_CLEANING_MODEL` / `PROD_DEFAULT_OPENAI_CLEANING_MODEL` in `config_constants.py`; override with e.g. `gpt-3.5-turbo` for lower cost)
- **Gemini**: `gemini_cleaning_model` (defaults to `gemini-1.5-flash`)
- **Anthropic**: `anthropic_cleaning_model` (defaults to `claude-haiku-4-5`)
- **Mistral**: `mistral_cleaning_model` (defaults to `mistral-small-latest`)
- **DeepSeek**: `deepseek_cleaning_model` (defaults to `deepseek-chat`)
- **Grok**: `grok_cleaning_model` (defaults to `grok-3-mini`)
- **Ollama**: `ollama_cleaning_model` (defaults to `llama3.1:8b`)

**Example** (config file with hybrid cleaning):

```yaml
summary_provider: openai
transcript_cleaning_strategy: hybrid  # Pattern + conditional LLM
openai_cleaning_model: gpt-4o-mini    # Cheaper model for cleaning
openai_cleaning_temperature: 0.2      # Lower temp for deterministic cleaning
openai_cleaning_llm_threshold: 0.10   # Use LLM if pattern reduces < 10%
```

**Example** (CLI with OpenAI):

```bash
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --summary-provider openai \
  --transcript-cleaning-strategy hybrid \
  --openai-cleaning-model gpt-4o-mini \
  --openai-cleaning-temperature 0.2
```

**Example** (CLI with Gemini):

```bash
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --summary-provider gemini \
  --transcript-cleaning-strategy hybrid \
  --gemini-cleaning-model gemini-1.5-flash \
  --gemini-cleaning-temperature 0.2
```

**Example** (CLI with Anthropic):

```bash
python3 -m podcast_scraper.cli https://example.com/feed.xml \
  --summary-provider anthropic \
  --transcript-cleaning-strategy hybrid \
  --anthropic-cleaning-model claude-haiku-4-5 \
  --anthropic-cleaning-temperature 0.2
```

**Example** (pattern-only for ML provider):

```yaml
summary_provider: transformers
transcript_cleaning_strategy: pattern  # Pattern-based only (ML doesn't support LLM cleaning)
```

**Note**: LLM-based cleaning runs when the selected `cleaning_processor` uses LLM calls (`llm` or `hybrid` strategy with an LLM provider). **`transformers`** summarization uses pattern-oriented cleaning in typical setups. **`hybrid_ml`** uses the same strategy-driven cleaners as API providers; internal MAP preprocessing defaults to **`cleaning_v4`** except for the layered **`pattern`** path (Issue #419).

**Note**: Preprocessing happens at the pipeline level before any transcription provider receives the audio. All providers (Whisper, OpenAI, future providers) benefit from optimized audio.

#### Knowledge Graph (KG)

| Field | CLI Flag | Default | Description |
| --- | --- | --- | --- |
| `generate_kg` | `--generate-kg` | `false` | Write per-episode `*.kg.json` during metadata generation. Requires `generate_metadata=true`. Separate from GIL. See [Knowledge Graph Guide](../guides/KNOWLEDGE_GRAPH_GUIDE.md). |
| `kg_extraction_source` | `--kg-extraction-source` | `summary_bullets` | `stub`, `summary_bullets`, or `provider` (LLM `extract_kg_graph`; see `kg_extraction_provider`; default uses `summary_provider`; ML backends no-op). |
| `kg_extraction_provider` | `--kg-extraction-provider` | (none) | Which summarization backend runs KG LLM calls (`extract_kg_graph`, and KG-from-bullets when that path is used). Unset means same as `summary_provider`. Ignored for `kg_extraction_source: stub`. See [KG guide § KG LLM provider vs summary](../guides/KNOWLEDGE_GRAPH_GUIDE.md#kg-llm-provider-vs-summary-provider). |
| `kg_max_topics` | `--kg-max-topics` | `20` | Max topic nodes (bullets or provider). Hard cap `20` in pipeline. |
| `kg_max_entities` | `--kg-max-entities` | `15` | Max entity nodes from provider extraction. |
| `kg_extraction_model` | `--kg-extraction-model` | (none) | Optional model override for KG LLM calls; else summary model. |
| `kg_merge_pipeline_entities` | `--no-kg-merge-pipeline-entities` | `true` | Merge hosts/guests after provider extraction (dedup vs existing entities by **entity_kind + name**); disable with flag. |

**Shallow v1 decisions (KG):** Extraction vs ML, entity roll-up limits, CLI surface (no `kg query` IR), and Postgres deferral are summarized in [KNOWLEDGE_GRAPH_GUIDE § Recorded product decisions (v1, KG shallow)](../guides/KNOWLEDGE_GRAPH_GUIDE.md#recorded-product-decisions-v1-kg). For GIL, see [GROUNDED_INSIGHTS_GUIDE § Recorded product decisions (v1, issue 460)](../guides/GROUNDED_INSIGHTS_GUIDE.md#recorded-product-decisions-v1-issue-460).

#### Grounded Insights (GIL)

| Field | CLI Flag | Default | Description |
| --- | --- | --- | --- |
| `generate_gi` | `--generate-gi` | `false` | Write per-episode `*.gi.json` during metadata generation. Requires `generate_metadata=true`. |
| `gi_insight_source` | `--gi-insight-source` | `stub` | Insight text source: `stub` (placeholder), `summary_bullets` (needs summaries + bullets), or `provider` (LLM `generate_insights`; ML providers do not implement it). See [GROUNDED_INSIGHTS_GUIDE](../guides/GROUNDED_INSIGHTS_GUIDE.md#ml-summarization-and-gil-insight-wording). |
| `gi_max_insights` | `--gi-max-insights` | `20` | Max insights when using `provider` or `summary_bullets` (`1`–`50`). |
| `gi_require_grounding` | (config) | `true` | When true, run QA/NLI (or provider evidence) to attach quotes; when false, insights without evidence stack. |
| `gi_fail_on_missing_grounding` | (config) | `false` | When true with `gi_require_grounding`, raise `GILGroundingUnsatisfiedError` if an episode ends with zero grounded quotes (strict CI). |
| `gi_evidence_extract_retries` | (config) | `1` | Provider `extract_quotes` only: extra attempts when the first returns no candidates (hint appended to insight text). NLI still uses the original insight. Range `0`–`5`. |
| `gi_qa_score_min` | (config) | `0.3` | Minimum extractive QA confidence to keep a quote candidate (local QA or `extract_quotes` `qa_score`). Range `0`–`1`. |
| `gi_nli_entailment_min` | (config) | `0.5` | Minimum NLI entailment probability to attach a `SUPPORTED_BY` quote. Range `0`–`1`. |
| `gi_qa_window_chars` | (config) | `1800` | Local GIL QA: when &gt; `0` and transcript is longer, scan overlapping windows of this size (characters) and take the best QA span. `0` = one QA call on the full transcript. |
| `gi_qa_window_overlap_chars` | (config) | `300` | Overlap between QA windows; must be **&lt;** `gi_qa_window_chars` when windowing is enabled. |
| `gi_qa_model` | (config) | (see `config_constants`) | HuggingFace / registry id for extractive QA (`transformers` / `hybrid_ml` evidence). |
| `gi_nli_model` | (config) | (see `config_constants`) | CrossEncoder NLI model id for local entailment. |
| `gi_embedding_model` | (config) | (see `config_constants`) | Sentence-transformers embedding model used when GIL evidence stack preloads embeddings. |

**Config file and the CLI:** For `python -m podcast_scraper.cli --config <file.yaml>`, validated YAML is merged into argparse defaults, then `cli._build_config()` rebuilds `Config`. GIL tuning keys (thresholds, `gi_qa_model`, `gi_nli_model`, `gi_embedding_model`, `extractive_qa_device`, `nli_device`, window sizes, retries, `gi_require_grounding`, etc.) must be **copied in `_build_config`**; otherwise file values are dropped and model defaults apply. If you add new `Config` fields for GIL, extend that forward list in `cli.py`.

`gi.json` **model_version** is not a separate config field: it is derived from `gi_insight_source` and the summarization / insight model in use (see `podcast_scraper.gi.provenance`).

When `summary_provider` is `transformers` or `hybrid_ml`, use **`gi_insight_source: summary_bullets`** (with bullets) or an LLM provider for real insight wording; **`stub`** is for tests/smoke only (CLI warns outside pytest when `generate_gi` + `stub`).

Local entailment (`entailment_provider`: `transformers` or `hybrid_ml`) requires **`sentence-transformers`** (install **`pip install -e ".[ml]"`**). The pipeline validates this at startup when `generate_gi` and `gi_require_grounding` are true.

**Shallow v1 decisions (issue #460):** ML vs `stub` vs `summary_bullets`, topic explore semantics, speaker best-effort behavior, and Postgres deferral are summarized in [GROUNDED_INSIGHTS_GUIDE § Recorded product decisions (v1, issue 460)](../guides/GROUNDED_INSIGHTS_GUIDE.md#recorded-product-decisions-v1-issue-460).

#### Grounded Insights (GIL) evidence providers

When `generate_gi` is true and `gi_require_grounding` is true, GIL uses a configurable evidence stack for quote extraction (QA) and entailment (NLI). Same backends as `summary_provider`.

| Field | CLI Flag | Default | Description |
| ------- | ---------- | --------- | ------------- |
| `quote_extraction_provider` | `--quote-extraction-provider` | `transformers` | Provider for GIL quote extraction (QA). Options: transformers, hybrid_ml, openai, gemini, grok, mistral, deepseek, anthropic, ollama. |
| `entailment_provider` | `--entailment-provider` | `transformers` | Provider for GIL entailment (NLI). Same options as quote_extraction_provider. |
| `gil_evidence_match_summary_provider` | N/A | `true` | When `generate_gi` is true: if `summary_provider` is an API LLM or `hybrid_ml` and both evidence fields are still default `transformers`, align them to `summary_provider` so grounding matches the summary backend. Set `false` to keep local QA + NLI with API summaries. |

**GIL evidence × provider (parity):** All backends in the `quote_extraction_provider` / `entailment_provider` enums implement `extract_quotes` and `score_entailment`. ML backends use local QA + NLI; LLM backends use chat-style calls. Full table: [GROUNDED_INSIGHTS_GUIDE — GIL evidence provider matrix](../guides/GROUNDED_INSIGHTS_GUIDE.md#gil-evidence-provider-matrix).

When `summary_provider` is an API but either evidence field remains `transformers` or `hybrid_ml` (after your overrides), the CLI emits a **WARNING** at config log time about **`.[ml]`** / **sentence-transformers** so hybrid setups are visible.

If either is set to an LLM (e.g. openai, anthropic), the corresponding API key must be set. See [GROUNDED_INSIGHTS_GUIDE](../guides/GROUNDED_INSIGHTS_GUIDE.md) and RFC-049.

<a id="live-pipeline-monitor-rfc-065-512"></a>

#### Live pipeline monitor (RFC-065, #512)

| Field | CLI Flag | Default | Description |
| ----- | -------- | ------- | ----------- |
| `monitor` | `--monitor` | `false` | Spawn a subprocess that shows live **RSS**, **CPU%**, and pipeline **stage** (reads/writes **`.pipeline_status.json`** under the effective output directory). Uses **psutil** + **rich** (core deps). See [Live Pipeline Monitor Guide](../guides/LIVE_PIPELINE_MONITOR.md) and [RFC-065](../rfc/RFC-065-live-pipeline-monitor.md). |
| `memray` | `--memray` | `false` | Re-exec the CLI or service under **memray** for heap profiling (optional extra **`.[monitor]`**). Sets **`PODCAST_SCRAPER_MEMRAY_ACTIVE=1`** in the child to avoid re-exec loops. |
| `memray_output` | `--memray-output` | *(derived)* | Memray capture **`.bin`** path. Default: **`<output_dir>/debug/memray_<timestamp>.bin`**, or **`./debug/...`** when `output_dir` is unset (service: cwd-based default). |

#### Logging & Operational Configuration (Issue #379)

**`json_logs`**

- **Description**: Output structured JSON logs for monitoring/alerting systems (ELK, Splunk, CloudWatch)
- **CLI Flag**: `--json-logs`
- **Default**: `false`
- **Type**: `bool`
- **Use Cases**: Production logging with log aggregation, Monitoring and alerting systems, Structured log analysis

- **Example**:

  ```yaml
  json_logs: true
  ```

  ```bash
  python3 -m podcast_scraper.cli https://example.com/feed.xml --json-logs
  ```

**`fail_fast`**

- **Description**: Stop pipeline on first episode failure (Issue #379)
- **CLI Flag**: `--fail-fast`
- **Default**: `false`
- **Type**: `bool`
- **Use Cases**: Development/debugging, CI/CD pipelines where early failure is preferred, Testing failure scenarios

- **Example**:

  ```yaml
  fail_fast: true
  ```

  ```bash
  python3 -m podcast_scraper.cli https://example.com/feed.xml --fail-fast
  ```

**`max_failures`**

- **Description**: Stop pipeline after N episode failures (Issue #379). `None` means no limit.
- **CLI Flag**: `--max-failures N`
- **Default**: `None` (no limit)
- **Type**: `Optional[int]`
- **Use Cases**: Production runs with failure tolerance, Batch processing with quality gates, Preventing cascading failures

- **Example**:

  ```yaml
  max_failures: 5
  ```

  ```bash
  python3 -m podcast_scraper.cli https://example.com/feed.xml --max-failures 5
  ```

**`transcription_timeout`**

- **Description**: Timeout in seconds for transcription operations (Issue #379). `None` means no timeout.
- **CLI Flag**: Not available (config file only)
- **Default**: `None` (no timeout)
- **Type**: `Optional[int]`
- **Use Cases**: Preventing hung transcription operations, Production reliability, Long-running episode handling

- **Example**:

  ```yaml
  transcription_timeout: 3600  # 1 hour
  ```

**`summarization_timeout`**

- **Description**: Timeout in seconds for summarization operations (Issue #379). `None` means no timeout.
- **CLI Flag**: Not available (config file only)
- **Default**: `None` (no timeout)
- **Type**: `Optional[int]`
- **Use Cases**: Preventing hung summarization operations, Production reliability, Long transcript handling

- **Example**:

  ```yaml
  summarization_timeout: 1800  # 30 minutes
  ```

**`transcription_queue_size`**

- **Description**: Maximum size of the transcription job queue (Issue #383). When the queue is full, downloads will block until space is available (backpressure). Prevents unbounded memory growth when downloads outpace transcription processing.
- **CLI Flag**: Not available (config file only)
- **Default**: `50`
- **Type**: `int` (minimum: 1)
- **Use Cases**:
  - Preventing memory issues when downloads complete faster than transcription
  - Controlling memory usage for large batch processing
  - Providing backpressure to balance download and transcription stages
- **Technical Details**:
  - Uses Python's `queue.Queue` with `maxsize` parameter
  - When queue is full, `transcription_jobs.put()` blocks until space is available
  - Transcription stage uses `queue.get()` with timeout for efficient polling
  - Queue is thread-safe and replaces the previous list-based approach

- **Example**:

  ```yaml
  transcription_queue_size: 50  # Default: allow up to 50 jobs in queue
  ```

  ```yaml
  transcription_queue_size: 10  # Smaller queue for memory-constrained environments
  ```

  ```yaml
  transcription_queue_size: 100  # Larger queue for high-throughput scenarios
  ```

**`summary_2nd_pass_distill`**

- **Description**: Enable optional 2nd-pass distillation with faithfulness prompt (Issue #387). When enabled, applies an additional distillation pass with a prompt that guides the model to be faithful to the source and reduce hallucinations. Only effective with OpenAI provider (BART/LED models don't use prompts effectively).
- **CLI Flag**: `--summary-2nd-pass-distill`
- **Default**: `false`
- **Type**: `bool`
- **Use Cases**:
  - Hallucination-prone summaries that need additional faithfulness checking
  - Production environments where summary quality is critical
  - Episodes with complex entity relationships that may cause model confusion
- **Note**: This is an optional enhancement that adds processing time. Use when the regular distill phase and faithfulness checks (Segment 4.1) indicate potential hallucinations.

- **Example**:

  ```yaml
  summary_2nd_pass_distill: true
  ```

  ```bash
  python3 -m podcast_scraper.cli https://example.com/feed.xml --summary-2nd-pass-distill
  ```

#### ML Library Configuration (Advanced)

**`HF_HUB_DISABLE_PROGRESS_BARS`**

- **Description**: Disable Hugging Face Hub progress bars to suppress misleading "Downloading" messages when loading models from cache
- **Required**: No (defaults to "1" - disabled, set programmatically)
- **Valid Values**: `"1"` (disabled) or `"0"` (enabled)
- **Priority**: System environment → `.env` file → Programmatic default
- **Use Cases**: Cleaner output when models are cached, Suppress progress bars in production logs
- **Note**: This is set automatically by the application, but can be overridden in `.env` file or system environment

**`OMP_NUM_THREADS`**

- **Description**: Number of OpenMP threads used by PyTorch for CPU operations
- **Required**: No (not set by default - uses all available CPU cores)
- **Valid Values**: Positive integer (e.g., `"1"`, `"4"`, `"8"`)
- **Priority**: System environment → `.env` file → Not set (full CPU utilization)
- **Use Cases**: Docker containers with limited resources, Memory-constrained environments, Performance tuning
- **Note**: Only set this if you want to limit CPU usage. By default, all CPU cores are used for best performance.

**`MKL_NUM_THREADS`**

- **Description**: Number of Intel MKL threads used by PyTorch (if MKL is available)
- **Required**: No (not set by default - uses all available CPU cores)
- **Valid Values**: Positive integer (e.g., `"1"`, `"4"`, `"8"`)
- **Priority**: System environment → `.env` file → Not set (full CPU utilization)
- **Use Cases**: Docker containers with limited resources, Memory-constrained environments, Performance tuning
- **Note**: Only set this if you want to limit CPU usage. By default, all CPU cores are used for best performance.

**`TORCH_NUM_THREADS`**

- **Description**: Number of CPU threads used by PyTorch
- **Required**: No (not set by default - uses all available CPU cores)
- **Valid Values**: Positive integer (e.g., `"1"`, `"4"`, `"8"`)
- **Priority**: System environment → `.env` file → Not set (full CPU utilization)
- **Use Cases**: Docker containers with limited resources, Memory-constrained environments, Performance tuning
- **Note**: Only set this if you want to limit CPU usage. By default, all CPU cores are used for best performance.

### Usage Examples

#### macOS / Linux

**Set for current session**:

```bash
export OPENAI_API_KEY=sk-your-key-here
python3 -m podcast_scraper https://example.com/feed.xml
```

**Inline execution**:

```bash
OPENAI_API_KEY=sk-your-key-here python3 -m podcast_scraper https://example.com/feed.xml
```

**Using .env file**:

```bash
# Create .env file in project root
echo "OPENAI_API_KEY=sk-your-key-here" > .env
python3 -m podcast_scraper https://example.com/feed.xml
```

## Docker

**Using environment variable**:

```bash
docker run -e OPENAI_API_KEY=sk-your-key-here podcast-scraper https://example.com/feed.xml
```

**Using .env file**:

```bash
# Create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Docker Compose automatically loads .env
# In docker-compose.yml:
# env_file:
#   - .env
```

## .env File Setup

### Creating .env File

1. **Copy example template** (if available):

   ```bash
   cp config/examples/.env.example .env
   ```

2. **Create `.env` file** in project root:

   ```bash
   # .env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. **Verify `.env` is in `.gitignore`**:

   ```bash
   # .gitignore should contain:
   .env
   .env.local
   .env.*.local
   ```

### .env File Location

The `.env` file is automatically loaded from:

1. **Project root** (where `config.py` is located): `{project_root}/.env`
2. **Current working directory**: `{cwd}/.env`

The first existing file is used. Project root takes precedence.

#### .env File Format

```bash
# .env file format
# Comments start with #
# Empty lines are ignored
# No spaces around = sign

# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional: Add other variables here
# LOG_LEVEL=DEBUG  # Valid: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Performance Configuration (optional)
# WORKERS=4
# TRANSCRIPTION_PARALLELISM=3
# PROCESSING_PARALLELISM=4
# SUMMARY_BATCH_SIZE=3
# SUMMARY_CHUNK_PARALLELISM=2
# TIMEOUT=60

# Device Configuration
# SUMMARY_DEVICE=cpu
# WHISPER_DEVICE=mps  # For Apple Silicon GPU acceleration
# MPS_EXCLUSIVE=1  # Serialize GPU work to prevent memory contention (default: true)

# ML Library Configuration (Advanced)
# HF_HUB_DISABLE_PROGRESS_BARS=1  # Disable progress bars (default: 1)
# OMP_NUM_THREADS=4  # Limit OpenMP threads
# MKL_NUM_THREADS=4  # Limit MKL threads
# TORCH_NUM_THREADS=4  # Limit PyTorch threads
```

## Best Practices

- **Add `.env` to `.gitignore`** (never commit secrets)
- **Use `config/examples/.env.example` as template** (without real values)
- **Use environment variables in production** (more secure than files)
- **Rotate API keys regularly**
- **Use separate keys for development/production**

## DON'T

- **Never commit `.env` files** with real API keys
- **Never hardcode API keys** in source code
- **Never log API keys** (they're automatically excluded from logs)
- **Never share API keys** in public repositories or chat

### Troubleshooting

#### Environment Variable Not Found

**Problem**: `OPENAI_API_KEY` not found when using OpenAI providers.

**Solutions**:

1. **Check variable name**: Must be exactly `OPENAI_API_KEY` (case-sensitive)
2. **Check `.env` file location**: Should be in project root or current directory
3. **Check `.env` file format**: No spaces around `=`, no quotes needed
4. **Reload shell**: Restart terminal/IDE after setting environment variables
5. **Verify loading**: Check that `python-dotenv` is installed (`pip install python-dotenv`)

**Debug**:

```python
import os
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("API key is set (length: {})".format(len(api_key)))  # Don't print the actual key
else:
    print("API key is not set")
```

## RSS and multi-feed corpus (GitHub #440)

You can target **one** feed with `rss` (string) or **multiple** feeds with **`feeds`** or **`rss_urls`** (YAML list of URL strings). Both list keys normalize to the same internal field (`rss_urls`).

**Rules:**

- **Two or more feeds** require an explicit **`output_dir`** (corpus parent). The CLI and service run **one full pipeline per feed**; each feed’s artifacts live under:

  ```text
  <output_dir>/feeds/rss_<host>_<hash>/
  ```

  (stable per-feed workspace name derived from the feed URL.)

- **Single feed** keeps the existing behavior: `output_dir` can default from env; the pipeline root is typically `output_dir` itself (no `feeds/` segment for the lone feed).

**Unified discovery and batch metadata (GitHub #505 / #506):** With **`vector_search: true`** and FAISS, per-feed runs skip automatic indexing and a **single** vector index is built under **`<output_dir>/search`** after the batch. Semantic **`index`**, **`search`**, **`gi explore --topic`**, and the viewer should use the **same corpus parent** path. The batch also writes **`corpus_manifest.json`**, **`corpus_run_summary.json`**, and a structured log line; inspect with **`corpus-status`**. Advanced: **`skip_auto_vector_index`** (default `false`) suppresses finalize-time indexing when you need to call **`index_corpus`** yourself.

**GI / KG inspect by episode:** **`gi inspect`** and **`kg inspect`** accept **`--output-dir`** as the corpus parent; if the same **`episode_id`** exists in multiple feeds, pass **`--feed-id`** (same value as metadata **`feed.feed_id`**).

**YAML examples:**

```yaml
# Single feed (unchanged)
rss: https://feeds.example.com/podcast.xml
output_dir: ./my_corpus
```

```yaml
# Multi-feed corpus (Planet Money + The Journal pattern)
feeds:
  - https://feeds.npr.org/510289/podcast.xml
  - https://video-api.wsj.com/podcast/rss/wsj/the-journal
output_dir: ./my_corpus  # required when len(feeds) >= 2
```

**Programmatic:**

```python
from podcast_scraper import Config

cfg = Config(
    rss_urls=[
        "https://feeds.npr.org/510289/podcast.xml",
        "https://video-api.wsj.com/podcast/rss/wsj/the-journal",
    ],
    output_dir="./my_corpus",
)
```

See [CLI.md — RSS and multi-feed](CLI.md#rss-and-multi-feed), [SERVICE.md](SERVICE.md), [RFC-063 — Multi-feed corpus](../rfc/RFC-063-multi-feed-corpus-append-resume.md), and checked-in examples:

- `config/examples/config.example.multi-feed.yaml` / `config/examples/config.example.multi-feed.json` (generic placeholder feeds; same provider mix as `config.example.*`)
- `config/acceptance/acceptance_multi_feed_planet_money_journal_openai.yaml` / `acceptance_multi_feed_planet_money_journal_deepseek.yaml` (full-pipeline acceptance presets)
- `config/manual/manual_multi_feed_planet_money_journal_openai.yaml` / `manual_multi_feed_planet_money_journal_deepseek.yaml`

<a id="append-resume-github-444"></a>

## Append / resume (GitHub #444)

Set **`append: true`** in YAML (or CLI **`--append`**) to reuse a **stable** `run_append_*` directory per feed and **skip** episodes that already have valid on-disk metadata (`episode_id` aligned with RSS) plus transcript and any enabled downstream artifacts (summary, GI, KG when those flags are on). Mutually exclusive with **`clean_output`**. `index.json` uses schema **`1.1.0`** and may include **`pipeline_append: true`**. See [PIPELINE_AND_WORKFLOW.md — Run tracking files](../guides/PIPELINE_AND_WORKFLOW.md#run-tracking-files-issue-379-429).

**YAML (single-feed or multi-feed):**

```yaml
rss: https://feeds.example.com/podcast.xml
output_dir: ./my_corpus
append: true
```

```yaml
feeds:
  - https://feeds.npr.org/510289/podcast.xml
  - https://video-api.wsj.com/podcast/rss/wsj/the-journal
output_dir: ./my_corpus
append: true
```

**Checked-in presets (Planet Money + The Journal, full pipeline):**

- **OpenAI:** `config/acceptance/acceptance_multi_feed_planet_money_journal_openai_append.yaml`, `config/manual/manual_multi_feed_planet_money_journal_openai_append.yaml` (use with `make test-acceptance` / CLI)
- **DeepSeek (Whisper + DeepSeek LLM):** `config/acceptance/acceptance_multi_feed_planet_money_journal_deepseek_append.yaml`, `config/manual/manual_multi_feed_planet_money_journal_deepseek_append.yaml`

Re-run the **same** command twice to validate resume: the second run should skip complete episodes under each feed’s `run_append_*` tree.

If the process exits before **finalize** (e.g. crash), **`index.json`** may lag behind what is already on disk; append/resume still prefers **filesystem + metadata** (`episode_id` and artifact checks) over the run index alone.

## Configuration Files

### JSON Example

```json
{
  "rss": "https://example.com/feed.xml",
  "output_dir": "./transcripts",
  "max_episodes": 50,
  "transcribe_missing": true,
  "whisper_model": "base",
  "workers": 8,
  "transcription_parallelism": 1,
  "processing_parallelism": 2,
  "generate_metadata": true,
  "generate_summaries": true,
  "summary_batch_size": 1,
  "summary_chunk_parallelism": 1,
  "preload_models": true,
  "preprocessing_enabled": true,
  "preprocessing_sample_rate": 16000
}
```

### YAML Example

```yaml
max_episodes: 50
transcribe_missing: true
whisper_model: base
workers: 8
transcription_parallelism: 1  # Number of episodes to transcribe in parallel
processing_parallelism: 2  # Number of episodes to process in parallel
generate_metadata: true
generate_summaries: true
summary_batch_size: 1  # Episode-level parallelism
summary_chunk_parallelism: 1  # Chunk-level parallelism
preprocessing_enabled: true  # Enable audio preprocessing
preprocessing_sample_rate: 16000  # Target sample rate
```

## Aliases and Normalization

The configuration system handles various aliases for backward compatibility:

- `rss_url` or `rss` (string) → `rss_url` (single-feed)
- **`feeds` or `rss_urls` (list)** → `rss_urls` (multi-feed; GitHub #440)
- **`append`** → `append` (boolean; stable `run_append_*` workspace and skip-complete semantics; GitHub #444)
- `output_dir` or `output_directory` → `output_dir`
- `screenplay_gap` or `screenplay_gap_s` → `screenplay_gap_s`

### Deprecated fields

The following field names are still accepted for backward compatibility but are deprecated. Use the replacement and expect removal in a future release. Config applies these mappings in `Config._handle_deprecated_fields` (config.py); removal is planned for a future major version.

| Deprecated | Replacement | Notes |
| ---------- | ----------- | ----- |
| `speaker_detector_type` | `speaker_detector_provider` | Value `ner` maps to `spacy`. Set via config or env; a one-time `DeprecationWarning` is emitted when the deprecated field is used. |

## Validation

The `Config` model performs validation on initialization:

```python
from podcast_scraper import Config
from pydantic import ValidationError

try:
    config = Config(rss_url="invalid-url")
except ValidationError as e:
    print(f"Validation failed: {e}")
```
