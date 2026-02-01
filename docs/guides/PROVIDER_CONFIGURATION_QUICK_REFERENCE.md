# Provider Configuration Quick Reference

This guide shows how to configure providers (transcription, speaker detection, summarization)
via CLI, config files, and programmatically.

## Provider Options

### Unified Provider Architecture

The podcast scraper uses a **Unified Provider** pattern where a single class implementation handles multiple capabilities.

1. **`MLProvider` (Local)**: Handles `whisper`, `spacy`, and `transformers`.
2. **`OpenAIProvider` (API)**: Handles all OpenAI-based transcription, speaker detection, and summarization.

### Transcription Providers

- **`whisper`** (default): Local Whisper models (via `MLProvider`)
- **`openai`**: OpenAI Whisper API (via `OpenAIProvider`)

### Speaker Detection Providers

- **`spacy`** (default): Local spaCy NER models (via `MLProvider`)
- **`openai`**: OpenAI GPT API (via `OpenAIProvider`)

### Summarization Providers

- **`transformers`** (default): Local HuggingFace Transformers models (via `MLProvider`)
- **`openai`**: OpenAI GPT API (via `OpenAIProvider`)

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

## 2. Configuration File (YAML/JSON)

Create a config file (e.g., `config.yaml`) with provider settings:

```yaml

# config.yaml

rss: https://example.com/feed.xml
output_dir: ./transcripts

# Provider configuration

transcription_provider: whisper  # or "openai"
speaker_detector_provider: spacy  # or "openai"
summary_provider: transformers  # or "openai"

# OpenAI configuration (required if using OpenAI providers)

openai_api_key: sk-your-key-here  # Optional: can use OPENAI_API_KEY env var instead
openai_api_base: null  # Optional: custom base URL (e.g., "http://localhost:8000/v1" for E2E testing)

# Transcription settings (for whisper provider)

transcribe_missing: true
whisper_model: base  # or "tiny", "small", "medium", "large", etc.

# Speaker detection settings (for spacy provider)

auto_speakers: true
ner_model: en_core_web_sm  # spaCy model name

# Summarization settings (for local provider)

generate_summaries: true
summary_model: bart-large  # Transformers model alias (options: bart-large, bart-small)
summary_device: cpu  # or "cuda", "mps"
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

# Then use in CLI or config

podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider openai
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

### Pattern 3: Hybrid (Local + OpenAI)

```yaml
transcription_provider: whisper      # Local (fast, free)
speaker_detector_provider: openai    # API (accurate)
summary_provider: transformers       # Local (fast, free)
openai_api_key: sk-your-key-here
```

- Balance between cost and performance
- Use local for heavy operations, API for accuracy

### Pattern 4: Transcription Only

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
    transcription_provider="invalid",  # Not "whisper" or "openai"
)

# ✅ Valid

cfg = Config(
    rss_url="https://example.com/feed.xml",
    transcription_provider="whisper",  # Valid option
)
```

Missing OpenAI API key when using OpenAI providers will raise `ValueError`:

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
