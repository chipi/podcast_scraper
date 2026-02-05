# Provider Configuration Quick Reference

This guide shows how to configure providers (transcription, speaker detection, summarization)
via CLI, config files, and programmatically.

## Provider Options

### Unified Provider Architecture

The podcast scraper uses a **Unified Provider** pattern where a single class implementation handles multiple capabilities.

1. **`MLProvider` (Local)**: Handles `whisper`, `spacy`, and `transformers`.
2. **`OpenAIProvider` (API)**: Handles all OpenAI-based transcription, speaker detection, and summarization.
3. **`GeminiProvider` (API)**: Handles all Google Gemini-based transcription, speaker detection, and summarization.
4. **`AnthropicProvider` (API)**: Handles Anthropic Claude-based speaker detection and summarization (transcription not supported).
5. **`DeepSeekProvider` (API)**: Handles DeepSeek-based speaker detection and summarization (transcription not supported).
6. **`GrokProvider` (API)**: Handles Grok-based speaker detection and summarization (real-time information access).

### Transcription Providers

- **`whisper`** (default): Local Whisper models (via `MLProvider`)
- **`openai`**: OpenAI Whisper API (via `OpenAIProvider`)
- **`gemini`**: Google Gemini API (via `GeminiProvider`)

### Speaker Detection Providers

- **`spacy`** (default): Local spaCy NER models (via `MLProvider`)
- **`openai`**: OpenAI GPT API (via `OpenAIProvider`)
- **`gemini`**: Google Gemini API (via `GeminiProvider`)
- **`anthropic`**: Anthropic Claude API (via `AnthropicProvider`) - high quality, 200k context
- **`deepseek`**: DeepSeek Chat API (via `DeepSeekProvider`) - 95% cheaper than OpenAI
- **`grok`**: Grok API (via `GrokProvider`) - real-time information access

### Summarization Providers

- **`transformers`** (default): Local HuggingFace Transformers models (via `MLProvider`)
- **`openai`**: OpenAI GPT API (via `OpenAIProvider`)
- **`gemini`**: Google Gemini API (via `GeminiProvider`)
- **`anthropic`**: Anthropic Claude API (via `AnthropicProvider`) - high quality, 200k context
- **`deepseek`**: DeepSeek Chat API (via `DeepSeekProvider`) - 95% cheaper than OpenAI
- **`grok`**: Grok API (via `GrokProvider`) - real-time information access

## Configuration Methods

### 1. Command Line Interface (CLI)

Use `--transcription-provider`, `--speaker-detector-provider`, and `--summary-provider` flags:

```bash

# Use all local ML providers (default)

podcast-scraper --rss https://example.com/feed.xml

# Use OpenAI for transcription

podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider openai \
  --openai-api-key sk-your-key-here

# Use OpenAI for speaker detection

podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider openai \
  --openai-api-key sk-your-key-here

# Use OpenAI for summarization

podcast-scraper --rss https://example.com/feed.xml \
  --summary-provider openai \
  --openai-api-key sk-your-key-here

# Mixed configuration: Whisper transcription + OpenAI speaker detection + Local summarization

podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider whisper \
  --speaker-detector-provider openai \
  --summary-provider transformers \
  --openai-api-key sk-your-key-here

# All OpenAI providers

podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider openai \
  --speaker-detector-provider openai \
  --summary-provider openai \
  --openai-api-key sk-your-key-here

# Use Gemini for transcription

podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider gemini \
  --gemini-api-key your-key-here

# Use Gemini for speaker detection

podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider gemini \
  --gemini-api-key your-key-here

# Use Gemini for summarization

podcast-scraper --rss https://example.com/feed.xml \
  --summary-provider gemini \
  --gemini-api-key your-key-here

# All Gemini providers

podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider gemini \
  --speaker-detector-provider gemini \
  --summary-provider gemini \
  --gemini-api-key your-key-here
```

**OpenAI API Key Options:**

- Set via `--openai-api-key` flag
- Set via `OPENAI_API_KEY` environment variable
- Set via `.env` file in project root

**Custom OpenAI Base URL (for E2E testing):**

```bash
podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider openai \
  --openai-api-base http://localhost:8000/v1 \
  --openai-api-key sk-test123
```

**Gemini API Key Options:**

- Set via `--gemini-api-key` flag
- Set via `GEMINI_API_KEY` environment variable
- Set via `.env` file in project root

**Custom Gemini Base URL (for E2E testing):**

```bash
podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider gemini \
  --gemini-api-base http://localhost:8000/v1beta \
  --gemini-api-key test123
```

**Ollama Setup (No API Key Required):**

Ollama is a local, self-hosted solution. No API key needed, but requires setup:

```bash
# 1. Install Ollama
brew install ollama  # macOS
# Or download from https://ollama.ai

# 2. Start Ollama server (keep running)
ollama serve

# 3. Pull required models
ollama pull llama3.3:latest  # Production
ollama pull llama3.2:latest  # Testing (faster)

# 4. Verify setup
ollama list  # Should show your models
```

**Ollama Configuration Options:**

- Set via `--ollama-api-base` flag (default: `http://localhost:11434/v1`)
- Set via `OLLAMA_API_BASE` environment variable
- Set via `.env` file in project root
- No API key required (local service)

**Custom Ollama Base URL (for remote Ollama server):**

```bash
podcast-scraper --rss https://example.com/feed.xml \
  --speaker-detector-provider ollama \
  --ollama-api-base http://192.168.1.100:11434/v1
```

**Troubleshooting Ollama:**

- If `ollama list` hangs: Server not running - start with `ollama serve`
- If "model not available": Pull model with `ollama pull <model-name>`
- If connection refused: Check server is running: `curl http://localhost:11434/api/tags`

See [Ollama Provider Guide](OLLAMA_PROVIDER_GUIDE.md) for detailed troubleshooting.

## 2. Configuration File (YAML/JSON)

Create a config file (e.g., `config.yaml`) with provider settings:

```yaml

# config.yaml

rss: https://example.com/feed.xml
output_dir: ./transcripts

# Provider configuration

transcription_provider: whisper  # or "openai", "gemini", "mistral"
speaker_detector_provider: spacy  # or "openai", "gemini", "mistral"
summary_provider: transformers  # or "openai", "gemini", "mistral"

# OpenAI configuration (required if using OpenAI providers)

openai_api_key: sk-your-key-here  # Optional: can use OPENAI_API_KEY env var instead
openai_api_base: null  # Optional: custom base URL (e.g., "http://localhost:8000/v1" for E2E testing)

# Gemini configuration (required if using Gemini providers)

gemini_api_key: your-key-here  # Optional: can use GEMINI_API_KEY env var instead
gemini_api_base: null  # Optional: custom base URL (e.g., "http://localhost:8000/v1beta" for E2E testing)

# Mistral configuration (required if using Mistral providers)
# Mistral is a full-stack provider (transcription + speaker detection + summarization) with EU data residency

mistral_api_key: your-key-here  # Optional: can use MISTRAL_API_KEY env var instead
mistral_api_base: null  # Optional: custom base URL (e.g., "http://localhost:8000/v1" for E2E testing)

# Transcription settings (for whisper provider)

transcribe_missing: true
whisper_model: base  # or "tiny", "small", "medium", "large", etc.

# Speaker detection settings (for spacy provider)

auto_speakers: true
ner_model: en_core_web_trf  # spaCy model name. Options: "en_core_web_trf" (default/prod, higher quality), "en_core_web_sm" (dev, fast). Defaults based on environment

# Summarization settings (for local provider)

generate_summaries: true
summary_model: pegasus-cnn  # Transformers model alias. Options: "pegasus-cnn" (default/prod), "bart-small" (dev), "bart-large", "fast", "pegasus", "long", "long-fast"
summary_device: cpu  # or "cuda", "mps"
mps_exclusive: true  # Serialize GPU work on MPS to prevent memory contention (default: true)
```

**JSON format:**

```json
{
  "rss": "https://example.com/feed.xml",
  "output_dir": "./transcripts",
  "transcription_provider": "whisper",
  "speaker_detector_provider": "spacy",
  "summary_provider": "transformers",
  "openai_api_key": "sk-your-key-here",
  "gemini_api_key": "your-key-here",
  "transcribe_missing": true,
  "whisper_model": "base",
  "auto_speakers": true,
  "generate_summaries": true
}
```

**Use config file:**

```bash
podcast-scraper --config config.yaml
```

**Config file with CLI overrides:**

```bash

# Config file sets defaults, CLI flags override

podcast-scraper --config config.yaml --transcription-provider openai
```

## 3. Programmatic (Library API)

Create a `Config` object and pass it to `run_pipeline()`:

```python
from podcast_scraper import Config, run_pipeline

# All local ML providers (default)

cfg = Config(
    rss_url="https://example.com/feed.xml",
    output_dir="./transcripts",
    transcription_provider="whisper",  # default
    speaker_detector_provider="spacy",   # default
    summary_provider="transformers",     # default
)

# OpenAI transcription

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="openai",
    openai_api_key="sk-your-key-here",  # or set OPENAI_API_KEY env var
)

# OpenAI speaker detection

cfg = Config(
    rss_url="https://example.com/feed.xml",
    speaker_detector_provider="openai",
    openai_api_key="sk-your-key-here",
)

# OpenAI summarization

cfg = Config(
    rss_url="https://example.com/feed.xml",
    summary_provider="openai",
    openai_api_key="sk-your-key-here",
    generate_summaries=True,
)

# Gemini transcription

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="gemini",
    gemini_api_key="your-key-here",  # or set GEMINI_API_KEY env var
)

# Gemini speaker detection

cfg = Config(
    rss_url="https://example.com/feed.xml",
    speaker_detector_provider="gemini",
    gemini_api_key="your-key-here",
)

# Gemini summarization

cfg = Config(
    rss_url="https://example.com/feed.xml",
    summary_provider="gemini",
    gemini_api_key="your-key-here",
    generate_summaries=True,
)

# Mixed configuration

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="whisper",      # MLProvider
    speaker_detector_provider="openai",    # OpenAIProvider
    summary_provider="transformers",      # MLProvider
    openai_api_key="sk-your-key-here",
)

# All OpenAI providers

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="openai",
    speaker_detector_provider="openai",
    summary_provider="openai",
    openai_api_key="sk-your-key-here",
    transcribe_missing=True,
    auto_speakers=True,
    generate_summaries=True,
)

# All Gemini providers

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="gemini",
    speaker_detector_provider="gemini",
    summary_provider="gemini",
    gemini_api_key="your-key-here",
    transcribe_missing=True,
    auto_speakers=True,
    generate_summaries=True,
)

# Run pipeline

count, summary = run_pipeline(cfg)
print(f"Processed {count} episodes: {summary}")
```python

**Load from config file programmatically:**

```python
from podcast_scraper import Config, load_config_file

# Load config file

config_dict = load_config_file("config.yaml")
cfg = Config(**config_dict)

# Override provider settings

cfg = Config(
    **config_dict,
    transcription_provider="openai",  # Override from file
    openai_api_key="sk-your-key-here",
)

# Run pipeline

count, summary = run_pipeline(cfg)
```

**Custom OpenAI base URL (for E2E testing):**

```python
cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="openai",
    openai_api_base="http://localhost:8000/v1",  # E2E server
    openai_api_key="sk-test123",
)
```

## Configuration Priority

When using multiple methods, priority is:

1. **CLI arguments** (highest priority)
2. **Config file**
3. **Environment variables**
4. **Defaults** (lowest priority)

Example:

```bash

# config.yaml has: transcription_provider: whisper
# CLI has: --transcription-provider openai
# Result: openai (CLI overrides config file)

podcast-scraper --config config.yaml --transcription-provider openai
```

## Environment Variables

You can also set provider-related settings via environment variables:

```bash

# OpenAI API key

export OPENAI_API_KEY=sk-your-key-here

# OpenAI API base URL (for E2E testing)

export OPENAI_API_BASE=http://localhost:8000/v1

# Gemini API key

export GEMINI_API_KEY=your-key-here

# Gemini API base URL (for E2E testing)

export GEMINI_API_BASE=http://localhost:8000/v1beta

# Mistral API key

export MISTRAL_API_KEY=your-key-here

# Mistral API base URL (for E2E testing)

export MISTRAL_API_BASE=http://localhost:8000/v1

# Then use in CLI or config

podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider openai

# Or with Gemini

podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider gemini

# Or with Mistral

podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider mistral
```

## Common Configuration Patterns

### Pattern 1: All Local (Default)

```yaml
transcription_provider: whisper
speaker_detector_provider: spacy
summary_provider: transformers
```

- Fast, no API costs
- Requires ML models to be installed/cached
- Works offline

### Pattern 2: All OpenAI

```yaml
transcription_provider: openai
speaker_detector_provider: openai
summary_provider: openai
openai_api_key: sk-your-key-here
```

- No local ML models needed
- API costs per request
- Requires internet connection

### Pattern 2b: All Gemini

```yaml
transcription_provider: gemini
speaker_detector_provider: gemini
summary_provider: gemini
gemini_api_key: your-key-here
```

### Pattern 2c: All Mistral

```yaml
transcription_provider: mistral
speaker_detector_provider: mistral
summary_provider: mistral
mistral_api_key: your-key-here
```

- Full-stack provider (all three capabilities)
- EU data residency (compliance-friendly)
- Competitive pricing
- Requires internet connection

- No local ML models needed
- API costs per request
- Requires internet connection
- Alternative to OpenAI

### Pattern 3: Hybrid (Local + API)

```yaml
transcription_provider: whisper      # Local (fast, free)
speaker_detector_provider: openai    # API (accurate)
summary_provider: transformers       # Local (fast, free)
openai_api_key: sk-your-key-here
```

- Balance between cost and performance
- Use local for heavy operations, API for accuracy

### Pattern 4: All Ollama (Local Self-Hosted)

```yaml
speaker_detector_provider: ollama
summary_provider: ollama
ollama_api_base: http://localhost:11434/v1  # Default, can be omitted
ollama_speaker_model: llama3.3:latest
ollama_summary_model: llama3.3:latest
```

- Zero API costs (all processing on local hardware)
- Complete privacy (data never leaves your machine)
- Works offline/air-gapped
- Requires Ollama installed and models pulled
- Note: Ollama does NOT support transcription (use `whisper` or `openai`)

**Prerequisites:**

1. Install Ollama: `brew install ollama` (macOS) or [download](https://ollama.ai)
2. Start server: `ollama serve` (keep running)
3. Pull models:
   - Limited RAM (6-8GB): `ollama pull llama3.1:8b`
   - Standard (8-12GB): `ollama pull llama3.2:latest`
   - Production (12-16GB): `ollama pull llama3.3:latest` (fits under 16GB)
4. Verify: `ollama list` should show your models

See [Ollama Provider Guide](OLLAMA_PROVIDER_GUIDE.md) for detailed installation and troubleshooting.

### Pattern 5: Transcription Only

```yaml
transcription_provider: whisper
transcribe_missing: true

# No speaker detection or summarization

auto_speakers: false
generate_summaries: false
```

## Validation

Invalid provider types will raise `ValueError`:

```python

# ❌ Invalid - will raise ValueError

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="invalid",  # Not "whisper", "openai", or "gemini"
)

# ✅ Valid

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="whisper",  # Valid option
)
```

Missing API key when using API providers will raise `ValueError`:

```python

# ❌ Invalid - will raise ValueError

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="openai",
    # Missing openai_api_key
)

# ✅ Valid

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="openai",
    openai_api_key="sk-your-key-here",  # Required
)

# ❌ Invalid - will raise ValueError

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="gemini",
    # Missing gemini_api_key
)

# ✅ Valid

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="gemini",
    gemini_api_key="your-key-here",  # Required
)
```

## Quick Examples

### Example 1: Basic Local Setup

```bash
podcast-scraper --rss https://example.com/feed.xml \
  --transcribe-missing \
  --auto-speakers \
  --generate-summaries
```

### Example 2: OpenAI Transcription Only

```bash
export OPENAI_API_KEY=sk-your-key-here
podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider openai \
  --transcribe-missing
```

### Example 2b: Gemini Transcription Only

```bash
export GEMINI_API_KEY=your-key-here
podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider gemini \
  --transcribe-missing
```

### Example 3: Mixed Providers (Config File)

```yaml

# config.yaml

rss: https://example.com/feed.xml
transcription_provider: whisper
speaker_detector_provider: openai
summary_provider: transformers
openai_api_key: sk-your-key-here

```

podcast-scraper --config config.yaml

```bash

## Example 4: Programmatic Mixed Providers

```python

from podcast_scraper import Config, run_pipeline

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="whisper",
    speaker_detector_provider="openai",
    summary_provider="transformers",
    openai_api_key="sk-your-key-here",
    transcribe_missing=True,
    auto_speakers=True,
    generate_summaries=True,
)

count, summary = run_pipeline(cfg)

```

## See Also

- [Provider Implementation Guide](../guides/PROVIDER_IMPLEMENTATION_GUIDE.md) - How providers work internally
- [Configuration API Reference](../api/CONFIGURATION.md) - Full configuration options
- [Development Guide](../guides/DEVELOPMENT_GUIDE.md) - Development setup
