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
    transcribe_missing=True,
    whisper_model="base",
    workers=8
)
```text

    options:

```
## Helper Functions

::: podcast_scraper.config.load_config_file
    options:

```python

## Environment Variables

Many configuration options can be set via environment variables for flexible deployment. Environment variables can be set:

1. **System environment variables** (highest priority)
2. **`.env` file** (loaded automatically from project root or current directory)
3. **Config file fields** (lowest priority, used as fallback)

Environment variables are automatically loaded when the `podcast_scraper.config` module is imported using `python-dotenv`.

### Priority Order

**General Rule**:

1. Config file field (highest priority)
2. Environment variable
3. Default value

**Exception**: `LOG_LEVEL` environment variable takes precedence over config file (allows easy runtime log level control).

### Supported Environment Variables

#### OpenAI API Configuration

**`OPENAI_API_KEY`**

- **Description**: OpenAI API key for OpenAI-based providers (transcription, speaker detection, summarization)
- **Required**: Yes, when using OpenAI providers (`transcription_provider=openai`, `speaker_detector_provider=openai`, or `summary_provider=openai`)
- **Example**: `export OPENAI_API_KEY=sk-your-actual-api-key-here`
- **Security**: Never commit `.env` files containing API keys. API keys are never logged or exposed in error messages.

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
- **Note**: Local Whisper ignores values > 1 (sequential only). OpenAI provider uses this for parallel API calls.
- **Use Cases**: OpenAI provider (`TRANSCRIPTION_PARALLELISM=3` for parallel API calls), Rate limit tuning

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

- **Description**: Device for model execution (CPU, CUDA, MPS, or None for auto-detection)
- **Required**: No (defaults to None for auto-detection)
- **Valid Values**: `cpu`, `cuda`, `mps`, or empty string (for None/auto-detect)
- **Priority**: Config file → Environment variable → Default
- **Use Cases**: Docker containers (`SUMMARY_DEVICE=cpu`), CI/CD (`SUMMARY_DEVICE=cpu`), NVIDIA GPU (`SUMMARY_DEVICE=cuda` or auto-detect), Apple Silicon (`SUMMARY_DEVICE=mps` or auto-detect)

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
python3 -m podcast_scraper <https://example.com/feed.xml>

```text

OPENAI_API_KEY=sk-your-key-here python3 -m podcast_scraper <https://example.com/feed.xml>

````

```bash

# Create .env file in project root

echo "OPENAI_API_KEY=sk-your-key-here" > .env
python3 -m podcast_scraper <https://example.com/feed.xml>

```

## Docker

**Using environment variable**:

```bash

docker run -e OPENAI_API_KEY=sk-your-key-here podcast-scraper https://example.com/feed.xml

````

```bash

# Create .env file

echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Docker Compose automatically loads .env

```yaml

      - ./.env:/app/.env:ro

```text

## .env File Setup

### Creating .env File

1. **Copy example template** (if available):

   ```bash

   cp examples/.env.example .env

````

1. **Create `.env` file** in project root:

   ```bash

   # .env

   OPENAI_API_KEY=sk-your-actual-api-key-here

   ```

2. **Verify `.env` is in `.gitignore`**:

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

````bash

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

# SUMMARY_DEVICE=cpu

# ML Library Configuration (Advanced)
# HF_HUB_DISABLE_PROGRESS_BARS=1  # Disable progress bars (default: 1)
# OMP_NUM_THREADS=4  # Limit OpenMP threads (optional, not set by default)
# MKL_NUM_THREADS=4  # Limit MKL threads (optional, not set by default)
# TORCH_NUM_THREADS=4  # Limit PyTorch threads (optional, not set by default)

```text

- **Add `.env` to `.gitignore`** (never commit secrets)
- **Use `examples/.env.example` as template** (without real values)
- **Use environment variables in production** (more secure than files)
- **Rotate API keys regularly**
- **Use separate keys for development/production**
- **Restrict API key permissions** (if supported by provider)

## ❌ DON'T

- **Never commit `.env` files** with real API keys
- **Never hardcode API keys** in source code
- **Never log API keys** (they're automatically excluded from logs)
- **Never share API keys** in public repositories or chat
- **Never use production keys** in development

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

```text

1. **Check file location**: Must be in project root (where `config.py` is) or current directory
2. **Check file name**: Must be exactly `.env` (not `.env.txt` or `env`)
3. **Check file permissions**: Must be readable
4. **Check file format**: No syntax errors, proper `KEY=value` format
5. **Verify `python-dotenv` installed**: `pip install python-dotenv`

**Debug**:

```python

from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
print(f"Looking for .env at: {env_path}")
print(f"File exists: {env_path.exists()}")

if env_path.exists():

```python

    load_dotenv(env_path, override=False)
    import os
    print(f"OPENAI_API_KEY loaded: {bool(os.getenv('OPENAI_API_KEY'))}")

```python

1. Config file field (`openai_api_key`)
2. System environment variable (`OPENAI_API_KEY`)
3. `.env` file (`OPENAI_API_KEY`)

**Solution**: Remove `openai_api_key` from config file to use environment variable.

**Exception**: `LOG_LEVEL` environment variable takes precedence over config file (allows easy runtime log level control).

### Future Environment Variables

The following environment variables may be added in future versions:

- `OPENAI_ORGANIZATION` - OpenAI organization ID (for multi-org accounts)
- `OPENAI_API_BASE` - Custom API base URL (for proxies)
- `DRY_RUN` - Testing/debugging flag
- `SKIP_EXISTING` - Resumption convenience flag
- `CLEAN_OUTPUT` - Safety control flag

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
  "summary_chunk_parallelism": 1
}

```text

max_episodes: 50
transcribe_missing: true
whisper_model: base
workers: 8
transcription_parallelism: 1  # Number of episodes to transcribe in parallel (Whisper ignores >1, OpenAI uses for parallel API calls)
processing_parallelism: 2  # Number of episodes to process (metadata/summarization) in parallel
generate_metadata: true
generate_summaries: true
summary_batch_size: 1  # Episode-level parallelism: Number of episodes to summarize in parallel
summary_chunk_parallelism: 1  # Chunk-level parallelism: Number of chunks to process in parallel within a single episode

```text

- `rss_url` or `rss` → `rss_url`
- `output_dir` or `output_directory` → `output_dir`
- `screenplay_gap` or `screenplay_gap_s` → `screenplay_gap_s`
- And more...

## Validation

The `Config` model performs validation on initialization:

```python

from podcast_scraper import Config
from pydantic import ValidationError

try:
    pass  # Your code here

```
