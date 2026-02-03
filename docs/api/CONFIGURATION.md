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
| `openai_transcription_model` | `--openai-transcription-model` | `whisper-1` | OpenAI model for transcription |
| `openai_speaker_model` | `--openai-speaker-model` | `gpt-4o-mini` | OpenAI model for speaker detection |
| `openai_summary_model` | `--openai-summary-model` | `gpt-4o-mini` | OpenAI model for summarization |
| `openai_temperature` | `--openai-temperature` | `0.3` | Temperature for generation (0.0-2.0) |

**Note on Transcription Limits**: The OpenAI Whisper API has a **25 MB file size limit**. The system proactively checks file sizes via HTTP HEAD requests before downloading or attempting transcription with the OpenAI provider. Episodes exceeding this limit will be skipped to avoid API errors and unnecessary bandwidth usage.

**Recommended Models by Environment**:

| Purpose | Test/Dev | Production | Notes |
| --------- | ---------- | ------------ | ------- |
| Transcription | `whisper-1` | `whisper-1` | Only OpenAI option |
| Speaker Detection | `gpt-4o-mini` | `gpt-4o` | Mini is fast/cheap; 4o is more accurate |
| Summarization | `gpt-4o-mini` | `gpt-4o` | Mini is fast/cheap; 4o produces better summaries |

**Example** (config file):

```yaml
transcription_provider: openai
speaker_detector_provider: openai
summary_provider: openai
openai_transcription_model: whisper-1
openai_speaker_model: gpt-4o      # Production: higher quality
openai_summary_model: gpt-4o      # Production: better summaries
openai_temperature: 0.3           # Lower = more deterministic
```

**Example** (CLI):

```bash
podcast-scraper --rss https://example.com/feed.xml \
  --transcription-provider openai \
  --speaker-detector-provider openai \
  --summary-provider openai \
  --openai-speaker-model gpt-4o \
  --openai-summary-model gpt-4o
```

#### Logging Configuration

**`LOG_LEVEL`**

- **Description**: Default logging level for the application
- **Required**: No (defaults to "INFO" if not specified)
- **Valid Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Priority**: Takes precedence over config file `log_level` field (unlike other variables)

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
  - **Local Whisper**: Always uses sequential processing (parallelism=1) regardless of configured value. The pipeline logs a debug message when configured parallelism is ignored.
  - **OpenAI provider**: Uses configured parallelism for parallel API calls.
- **Use Cases**: OpenAI provider (`TRANSCRIPTION_PARALLELISM=3` for parallel API calls), Rate limit tuning
- **Logging**: When using Whisper, debug logs will show `"Whisper provider: Using sequential processing (parallelism=X ignored)"` to indicate that the configured parallelism was ignored due to provider limitations.

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

**`MPS_EXCLUSIVE`**

- **Description**: Serialize GPU work on MPS to prevent memory contention between Whisper transcription and summarization
- **Required**: No (defaults to `true` for safer behavior)
- **Valid Values**: `1`, `true`, `yes`, `on` (enable) or `0`, `false`, `no`, `off` (disable)
- **Priority**: Config file → Environment variable → Default
- **Use Cases**:
  - Apple Silicon with limited GPU memory: `MPS_EXCLUSIVE=1` (default) prevents both models from competing for GPU memory
  - Systems with sufficient GPU memory: `MPS_EXCLUSIVE=0` allows concurrent GPU operations for better throughput
- **Behavior**: When enabled and both Whisper and summarization use MPS, transcription completes before summarization starts. I/O operations (downloads, parsing) remain parallel.
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
- **API Compatibility**: Ensures files fit within 25 MB limit for OpenAI/Groq
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

**Note**: Preprocessing happens at the pipeline level before any transcription provider receives the audio. All providers (Whisper, OpenAI, future providers) benefit from optimized audio.

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
   cp examples/.env.example .env
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
- **Use `examples/.env.example` as template** (without real values)
- **Use environment variables in production** (more secure than files)
- **Rotate API keys regularly**
- **Use separate keys for development/production**

## ❌ DON'T

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
print(os.getenv("OPENAI_API_KEY"))  # Should print your key (or None)
```

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

- `rss_url` or `rss` → `rss_url`
- `output_dir` or `output_directory` → `output_dir`
- `screenplay_gap` or `screenplay_gap_s` → `screenplay_gap_s`

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
