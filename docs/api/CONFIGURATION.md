# Configuration API

The `Config` class is the central configuration model for podcast_scraper, built on Pydantic for validation and type safety.

## Overview

All runtime options flow through the `Config` model:

````python
from podcast_scraper import Config

cfg = Config(
    rss="https://example.com/feed.xml",
    output_dir="./transcripts",
    max_episodes=50,
    transcribe_missing=True,  # Default: True (automatically transcribe missing transcripts)
    whisper_model="base.en",
    workers=8
)
```python

## Environment Variables

Many configuration options can be set via environment variables for flexible deployment. Environment variables can be set:

1. **System environment variables** (highest priority among env vars)
2. **`.env` file** (loaded automatically from project root or current directory, lower priority than system env)

Environment variables are automatically loaded when the `podcast_scraper.config` module is imported using `python-dotenv`.

### Priority Order

**General Rule** (for each configuration field):

1. **CLI arguments** (highest priority)
2. **Config file field** - if the field is set in the config file and not `null`/empty, it takes precedence over env vars
3. **Environment variable** - only used if the field is not set in CLI or config file
4. **Default value** - used if no other source is set

**Exception**: `LOG_LEVEL` environment variable takes precedence over config file (allows easy runtime log level control without modifying config files).

### Supported Environment Variables

#### Provider Configuration

| Variable | Description | Default |
| ---------- | ------------- | --------- |
| `TRANSCRIPTION_PROVIDER` | `whisper` or `openai` | `whisper` |
| `SPEAKER_DETECTOR_PROVIDER` | `spacy` or `openai` | `spacy` |
| `SUMMARY_PROVIDER` | `transformers` or `openai` | `transformers` |
| `TRANSCRIPTION_PARALLELISM` | Episodes to transcribe in parallel (Whisper: 1, OpenAI: N) | `1` |
| `PROCESSING_PARALLELISM` | Episodes to process (metadata/summaries) in parallel | `2` |
| `METRICS_OUTPUT` | Path to save pipeline metrics JSON | `{output_dir}/metrics.json` |

#### OpenAI API Configuration

Many fields support [prompt_store](../guides/PROTOCOL_EXTENSION_GUIDE.md#prompt-management) for versioned prompts.

| Variable | Description | Default |
| ---------- | ------------- | --------- |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `OPENAI_API_BASE` | Custom API base URL | None |
| `OPENAI_TRANSCRIPTION_MODEL` | Model for transcription | `whisper-1` |
| `OPENAI_SPEAKER_MODEL` | Model for speaker detection | `gpt-4o-mini` |
| `OPENAI_SUMMARY_MODEL` | Model for summarization | `gpt-4o-mini` (test) / `gpt-4o` (prod) |
| `OPENAI_TEMPERATURE` | Generation temperature (0.0-2.0) | `0.3` |
| `OPENAI_MAX_TOKENS` | Max tokens for generation | None |
| `OPENAI_SUMMARY_SYSTEM_PROMPT` | System prompt name for summaries | None |
| `OPENAI_SUMMARY_USER_PROMPT` | User prompt name for summaries | `summarization/long_v1` |
| `OPENAI_SPEAKER_SYSTEM_PROMPT` | System prompt name for NER | None |
| `OPENAI_SPEAKER_USER_PROMPT` | User prompt name for NER | `ner/guest_host_v1` |

#### Summarization Configuration (Local)

| Variable | Description | Default |
| ---------- | ------------- | --------- |
| `SUMMARY_MODEL` | Model for MAP phase (e.g., BART) | `facebook/bart-large-cnn` |
| `SUMMARY_REDUCE_MODEL` | Model for REDUCE phase (e.g., LED) | `allenai/led-large-16384` |
| `SUMMARY_DEVICE` | Device for execution (`cpu`, `cuda`, `mps`) | `auto` |
| `SUMMARY_MAX_LENGTH` | Max summary length (tokens) | `160` |
| `SUMMARY_MIN_LENGTH` | Min summary length (tokens) | `60` |
| `SUMMARY_BATCH_SIZE` | Episodes to process in parallel | `1` |
| `SUMMARY_CHUNK_PARALLELISM` | Chunks to process in parallel (local only) | `1` |
| `SUMMARY_CHUNK_SIZE` | Token chunk size for long transcripts | `2048` |
| `SUMMARY_WORD_CHUNK_SIZE` | Word-based chunk size | `900` |
| `SUMMARY_WORD_OVERLAP` | Word-based overlap | `150` |
| `SAVE_CLEANED_TRANSCRIPT` | Save .cleaned.txt file | `true` |

#### Logging Configuration

**`LOG_LEVEL`**

- **Description**: Default logging level for the application
- **Valid Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Priority**: Takes precedence over config file `log_level` field

#### Performance Configuration

| Variable | Description | Default |
| ---------- | ------------- | --------- |
| `WORKERS` | Number of parallel download workers | CPU count (1-8) |
| `TRANSCRIPTION_PARALLELISM` | Number of episodes to transcribe in parallel | `1` |
| `PROCESSING_PARALLELISM` | Number of episodes to process in parallel | `2` |
| `SUMMARY_BATCH_SIZE` | Number of episodes to summarize in parallel | `1` |
| `SUMMARY_CHUNK_PARALLELISM` | Number of chunks to process in parallel (local only) | `1` |
| `TIMEOUT` | Request timeout in seconds | `20` |
| `SUMMARY_DEVICE` | Device for summarization (`cpu`, `cuda`, `mps`) | `auto` |
| `WHISPER_DEVICE` | Device for Whisper (`cpu`, `cuda`, `mps`) | `auto` |

#### Control Flags

| Variable | Description | Default |
| ---------- | ------------- | --------- |
| `SKIP_EXISTING` | Skip episodes with existing output files | `false` |
| `CLEAN_OUTPUT` | Remove output directory before processing | `false` |
| `REUSE_MEDIA` | Reuse existing media files instead of re-downloading | `false` |
| `DRY_RUN` | Preview planned work without saving files | `false` |

#### ML Library Configuration (Advanced)

**`CACHE_DIR`**

- **Description**: Base directory for all ML model caches
- **Default**: `~/.cache` (platform dependent)

**`SUMMARY_CACHE_DIR`**

- **Description**: Custom cache directory for transformer models
- **Default**: Derived from `CACHE_DIR`

**`HF_HUB_DISABLE_PROGRESS_BARS`**

- **Description**: Disable Hugging Face Hub progress bars
- **Default**: `1` (disabled)

**`TORCH_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`**

- **Description**: CPU thread limits for PyTorch/OpenMP/MKL
- **Default**: Not set (uses all cores)

## Configuration Files

### JSON Example

```json
{
  "rss": "https://example.com/feed.xml",
  "output_dir": "./output",
  "max_episodes": 50,
  "transcribe_missing": true,
  "whisper_model": "base.en",
  "workers": 8,
  "generate_metadata": true,
  "generate_summaries": true,
  "summary_provider": "transformers",
  "skip_existing": true
}
```

### YAML Example

```yaml
rss: https://example.com/feed.xml
output_dir: ./output
max_episodes: 50
transcribe_missing: true
whisper_model: base.en
workers: 8
generate_metadata: true
generate_summaries: true
summary_provider: transformers
skip_existing: true
```

## Validation

The `Config` model performs strict validation on initialization using Pydantic. Invalid values will raise `ValidationError`.

### Mutual Exclusivity

- `clean_output` and `skip_existing` are mutually exclusive.
- `clean_output` and `reuse_media` are mutually exclusive.
- `generate_summaries=True` requires `generate_metadata=True`.
- `transcribe_missing=True` requires a valid `whisper_model`.
- `summary_max_length` must be greater than `summary_min_length`.
- `summary_word_overlap` must be less than `summary_word_chunk_size`.
