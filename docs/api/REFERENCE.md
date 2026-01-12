# API Reference

Complete reference documentation for the `podcast_scraper` public API.

**API Version**: `2.3.0` (tied to module version)

## Table of Contents

- Core API
  - run_pipeline
  - Config
  - load_config_file
- Service API
  - ServiceResult
  - service.run
  - service.run_from_config_file
- Version Information
- Quick Start Examples

---

## Core API

### run_pipeline {: #run_pipeline }

Execute the main podcast scraping pipeline.

```python
def run_pipeline(cfg: Config) -> Tuple[int, str]
```

1. Setup output directory
2. Fetch and parse RSS feed
3. Process episodes (download transcripts or queue for transcription)
4. Preprocess audio files if enabled (reduce size, remove silence, normalize)
5. Transcribe media files using Whisper if needed
6. Generate metadata and summaries if enabled
7. Clean up temporary files

**Parameters:**

- `cfg` (`Config`): Configuration object with all settings. See [Config](#config) for details.

**Returns:**

- `Tuple[int, str]`: A tuple containing:
  - `count` (int): Number of transcripts saved/planned
  - `summary` (str): Human-readable summary message

**Raises:**

- `RuntimeError`: If output directory cleanup fails
- `ValueError`: If RSS fetch or parse fails
- `FileNotFoundError`: If required files are missing
- `OSError`: If file system operations fail

**Example:**

```python
from podcast_scraper import Config, run_pipeline

config = Config(
    rss_url="https://example.com/feed.xml",
    output_dir="./transcripts",
    max_episodes=10,
    transcribe_missing=True,
)

count, summary = run_pipeline(config)
print(f"Processed {count} episodes")
print(summary)
```

- [Config](#config) - Configuration options
- [service.run](#service_run) - Service API with structured results

---

### Config {: #config }

Configuration model for podcast scraping pipeline.

```python
class Config(BaseModel)
```

**Fields:**

#### RSS Feed Configuration

- `rss_url` (`Optional[str]`, alias: `"rss"`): RSS feed URL to scrape. Required unless loading from config file.
- `max_episodes` (`Optional[int]`, alias: `"max_episodes"`): Maximum number of episodes to process. If `None`, processes all episodes.

#### Output Configuration

- `output_dir` (`Optional[str]`, alias: `"output_dir"`): Output directory for transcripts. If `None`, auto-generated from RSS URL.
- `run_id` (`Optional[str]`, alias: `"run_id"`): Optional run identifier. Use `"auto"` for timestamp-based ID.
- `skip_existing` (`bool`, default: `False`, alias: `"skip_existing"`): Skip episodes whose output already exists.
- `clean_output` (`bool`, default: `False`, alias: `"clean_output"`): Remove output directory before processing.
- `reuse_media` (`bool`, default: `False`, alias: `"reuse_media"`): Reuse existing media files instead of re-downloading (for faster testing).

#### HTTP Configuration

- `user_agent` (`str`, default: `DEFAULT_USER_AGENT`, alias: `"user_agent"`): User-Agent header for HTTP requests.
- `timeout` (`int`, default: `20`, alias: `"timeout"`): Request timeout in seconds (minimum: 1).
- `delay_ms` (`int`, default: `0`, alias: `"delay_ms"`): Delay between requests in milliseconds.
- `prefer_types` (`List[str]`, default: `[]`, alias: `"prefer_type"`): Preferred transcript types or extensions (e.g., `["text/vtt", ".srt"]`).

#### Transcription Configuration

- `transcribe_missing` (`bool`, default: `True`, alias: `"transcribe_missing"`): Enable Whisper transcription for episodes without transcripts. Set to `False` to only download existing transcripts.
- `whisper_model` (`str`, default: `"base.en"`, alias: `"whisper_model"`): Whisper model to use. Valid values: `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large"`, `"large-v2"`, `"large-v3"`, or language-specific variants (e.g., `"base.en"`). For English (the default language), the default is `"base.en"` which matches the actual model used at runtime. The conversion logic still works for backward compatibility if you explicitly specify `"base"`.
- `language` (`str`, default: `"en"`, alias: `"language"`): Language code for transcription (e.g., `"en"`, `"fr"`, `"de"`).

#### Screenplay Formatting

- `screenplay` (`bool`, default: `False`, alias: `"screenplay"`): Format transcripts as screenplay with speaker labels.
- `screenplay_gap_s` (`float`, default: `1.25`, alias: `"screenplay_gap"`): Minimum gap in seconds between speaker segments.
- `screenplay_num_speakers` (`int`, default: `2`, alias: `"num_speakers"`): Number of speakers for Whisper diarization (minimum: 1).
- `screenplay_speaker_names` (`List[str]`, default: `[]`, alias: `"speaker_names"`): Manual speaker names (overrides auto-detection). Format: `["Host", "Guest"]`.

#### Speaker Detection

- `auto_speakers` (`bool`, default: `True`, alias: `"auto_speakers"`): Enable automatic speaker name detection using NER.
- `cache_detected_hosts` (`bool`, default: `True`, alias: `"cache_detected_hosts"`): Cache detected host names across episodes.
- `ner_model` (`Optional[str]`, default: `None`, alias: `"ner_model"`): spaCy NER model name (e.g., `"en_core_web_sm"`). If `None`, uses default.

#### Metadata Generation

- `generate_metadata` (`bool`, default: `False`, alias: `"generate_metadata"`): Generate per-episode metadata documents.
- `metadata_format` (`Literal["json", "yaml"]`, default: `"json"`, alias: `"metadata_format"`): Metadata file format.
- `metadata_subdirectory` (`Optional[str]`, default: `None`, alias: `"metadata_subdirectory"`): Subdirectory for metadata files. If `None`, stored alongside transcripts.

#### Summarization

- `generate_summaries` (`bool`, default: `False`, alias: `"generate_summaries"`): Generate episode summaries.
- `summary_provider` (`Literal["transformers", "local", "openai"]`, default: `"transformers"`, alias: `"summary_provider"`): Summary generation provider. Deprecated: `"local"` is accepted as alias for `"transformers"`.
- `summary_model` (`Optional[str]`, default: `None`, alias: `"summary_model"`): MAP-phase model identifier (e.g., `"bart-large"`, `"facebook/bart-large-cnn"`). Defaults to `"bart-large"` for fast chunk summarization.
- `summary_reduce_model` (`Optional[str]`, default: `None`, alias: `"summary_reduce_model"`): REDUCE-phase model identifier (e.g., `"long-fast"`, `"allenai/led-base-16384"`). Defaults to `"long-fast"` (LED) for accurate, long-context final combine. If not set, uses LED instead of falling back to MAP model.
- `summary_max_length` (`int`, default: `150`, alias: `"summary_max_length"`): Maximum summary length in tokens.
- `summary_min_length` (`int`, default: `30`, alias: `"summary_min_length"`): Minimum summary length in tokens.
- `summary_device` (`Optional[str]`, default: `None`, alias: `"summary_device"`): Device for model execution (`"cpu"`, `"cuda"`, `"mps"`, or `None` for auto-detect).
- `summary_batch_size` (`int`, default: `1`, alias: `"summary_batch_size"`): Batch size for parallel processing (CPU only).
- `summary_chunk_size` (`Optional[int]`, default: `None`, alias: `"summary_chunk_size"`): Chunk size in tokens for long transcripts. If `None`, auto-detected from model.
- `summary_cache_dir` (`Optional[str]`, default: `None`, alias: `"summary_cache_dir"`): Custom cache directory for transformer models. If `None`, uses default Hugging Face cache.

#### Processing Options

- `workers` (`int`, default: `min(8, cpu_count())`, alias: `"workers"`): Number of parallel download workers.
- `dry_run` (`bool`, default: `False`, alias: `"dry_run"`): Preview planned work without saving files.

#### Logging

- `log_level` (`str`, default: `"INFO"`, alias: `"log_level"`): Logging level (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`).
- `log_file` (`Optional[str]`, default: `None`, alias: `"log_file"`): Path to log file. If provided, logs are written to both console and file.

**Validation:**

- `whisper_model` must be one of the valid Whisper models
- `timeout` must be at least 1 second
- `screenplay_num_speakers` must be at least 1
- `log_level` must be a valid logging level
- `metadata_format` must be `"json"` or `"yaml"`
- `summary_provider` must be `"transformers"` or `"openai"` (deprecated: `"local"` accepted as alias)

**Example:**

```python
from podcast_scraper import Config

# Minimal configuration

config = Config(rss_url="https://example.com/feed.xml")

# Full configuration

config = Config(
    rss_url="https://example.com/feed.xml",
    output_dir="./transcripts",
    max_episodes=50,
    transcribe_missing=True,
    whisper_model="base",
    generate_metadata=True,
    metadata_format="yaml",
    generate_summaries=True,
    summary_model="facebook/bart-base",
    summary_device="mps",  # Apple Silicon
    log_level="DEBUG",
    log_file="scraper.log",
)
```

- [load_config_file](#load_config_file) - Load configuration from file
- [Configuration Examples](CONFIGURATION.md) - Example configuration files

---

## load_config_file {: #load_config_file }

Load configuration from a JSON or YAML file.

```python
def load_config_file(path: str) -> Dict[str, Any]
```

**Note:** This function returns a dictionary, not a `Config` object. To create a `Config` object from the loaded dictionary, pass it to the `Config` constructor.

**Parameters:**

- `path` (`str`): Path to configuration file (JSON or YAML).

**Returns:**

- `Dict[str, Any]`: Dictionary containing configuration values from the file.

**Raises:**

- `ValueError`: If the configuration file doesn't exist, format is invalid, or parsing fails.
- `yaml.YAMLError`: If YAML parsing fails.
- `json.JSONDecodeError`: If JSON parsing fails.

**Example:**

```python
from podcast_scraper import Config, load_config_file, run_pipeline

# Load from YAML and create Config object

config_dict = load_config_file("config.yaml")
config = Config(**config_dict)

# Load from JSON and create Config object

config_dict = load_config_file("config.json")
config = Config(**config_dict)

# Use the config

count, summary = run_pipeline(config)
```

```python
from podcast_scraper import Config, load_config_file

# One-liner: load and create Config

config = Config(**load_config_file("config.yaml"))
```

```json

{
  "rss": "https://example.com/feed.xml",
  "output_dir": "./transcripts",
  "max_episodes": 10,
  "transcribe_missing": true,
  "whisper_model": "base",
  "generate_metadata": true,
  "generate_summaries": true,
  "log_level": "INFO"
}

```

```yaml
max_episodes: 10
transcribe_missing: true
whisper_model: "base"
generate_metadata: true
generate_summaries: true
log_level: "INFO"
```

- [Config](#config) - Configuration model
- [Configuration Examples](CONFIGURATION.md) - Example configuration files

---

## Service API

The service API provides a programmatic interface optimized for non-interactive use, such as running as a daemon or service.

### ServiceResult {: #serviceresult }

Result of a service run.

```python
@dataclass
class ServiceResult:
    episodes_processed: int
    summary: str
    success: bool = True
    error: Optional[str] = None
```

- `episodes_processed` (`int`): Number of episodes processed (transcripts saved/planned).
- `summary` (`str`): Human-readable summary message with processing statistics.
- `success` (`bool`): Whether the run completed successfully. Default: `True`.
- `error` (`Optional[str]`): Error message if `success` is `False`, `None` otherwise.

**Example:**

```python
from podcast_scraper import service

result = service.run_from_config_file("config.yaml")

if result.success:
    print(f"Successfully processed {result.episodes_processed} episodes")
    print(f"Summary: {result.summary}")
else:
    print(f"Error: {result.error}")
    sys.exit(1)
```

---

### service.run {: #service_run }

Run the pipeline with a Config object.

```python
def run(cfg: Config) -> ServiceResult
```

**Parameters:**

- `cfg` (`Config`): Configuration object (can be created from `Config()` or `Config(**load_config_file())`).

**Returns:**

- `ServiceResult`: Structured result with processing outcomes.

**Example:**

```python
from podcast_scraper import service, config

# Create config programmatically

cfg = config.Config(rss_url="https://example.com/feed.xml")

# Run pipeline

result = service.run(cfg)

if result.success:
    print(f"Processed {result.episodes_processed} episodes")
    print(f"Summary: {result.summary}")
else:
    print(f"Error: {result.error}")
```

- [Config](#config) - Configuration options

---

## service.run_from_config_file {: #service_run_from_config_file }

Run the pipeline from a configuration file.

```python
def run_from_config_file(config_path: str | Path) -> ServiceResult
```

- `config_path` (`str | Path`): Path to configuration file (JSON or YAML).

**Returns:**

- `ServiceResult`: Structured result with processing outcomes.

**Raises:**

- `FileNotFoundError`: If config file doesn't exist (handled internally, returns failed `ServiceResult`).
- `ValueError`: If config file is invalid (handled internally, returns failed `ServiceResult`).

**Example:**

```python
from podcast_scraper import service

# Run from config file

result = service.run_from_config_file("config.yaml")

if not result.success:
    print(f"Failed: {result.error}")
    sys.exit(1)

print(f"Success: {result.summary}")
```

```bash
# Command-line entry point
python -m podcast_scraper.service --config config.yaml

# Exit codes: 0 = success, 1 = failure
```

- [load_config_file](#load_config_file) - Load configuration from file
- [Service Examples](SERVICE.md) - Supervisor and systemd configuration examples

---

## Version Information

### **api_version** {: #api_version }

API version string following semantic versioning.

```python
import podcast_scraper

api_version = podcast_scraper.__api_version__  # "2.3.0"
```

- **Major version (X.y.z)**: Breaking API changes (function signatures, return types, required parameters)
- **Minor version (x.Y.z)**: New features, backward compatible (new functions, optional parameters)
- **Patch version (x.y.Z)**: Bug fixes, backward compatible (no API changes)

**Example:**

```python
import podcast_scraper

# Check API version

if podcast_scraper.__api_version__.startswith("2."):

    # Use v2 API

    config = podcast_scraper.Config(...)
else:

    # Handle older versions

    pass
```

- [API Migration Guide](MIGRATION_GUIDE.md) - Migration between major versions

---

## Quick Start Examples

### Basic Usage

```python

from podcast_scraper import Config, run_pipeline

# Minimal configuration

config = Config(rss_url="https://example.com/feed.xml")
count, summary = run_pipeline(config)
print(f"Downloaded {count} transcripts")

```

### Transcription Options

```python
config = Config(
    rss_url="https://example.com/feed.xml",
    transcribe_missing=True,
    whisper_model="base",
    language="en",
)
count, summary = run_pipeline(config)
```

### Metadata and Summaries

```python
config = Config(
    rss_url="https://example.com/feed.xml",
    generate_metadata=True,
    metadata_format="yaml",
    generate_summaries=True,
    summary_model="facebook/bart-base",
    summary_device="mps",  # Apple Silicon
)
count, summary = run_pipeline(config)

```

### Service API Usage

```python
# Recommended for automation/daemon use
result = service.run_from_config_file("config.yaml")

if result.success:
    print(f"Processed {result.episodes_processed} episodes")
    print(f"Summary: {result.summary}")
else:
    print(f"Error: {result.error}")
    sys.exit(1)
```

### Loading Configuration

```python
# Load configuration
config = load_config_file("config.yaml")

# Run pipeline
count, summary = run_pipeline(config)
```

### Error Handling

```python
try:
    config = Config(rss_url="https://example.com/feed.xml")
    count, summary = run_pipeline(config)
    print(f"Success: {summary}")
except ValueError as e:
    print(f"Configuration error: {e}", file=sys.stderr)
    sys.exit(1)
except RuntimeError as e:
    print(f"Runtime error: {e}", file=sys.stderr)
    sys.exit(1)
```

### Type Hints Example

```python

from podcast_scraper import Config, run_pipeline
from typing import Tuple

# Type hints are available

def process_podcast(rss_url: str) -> Tuple[int, str]:
    config = Config(rss_url=rss_url)
    return run_pipeline(config)  # Returns Tuple[int, str]

```

## See Also

- [Architecture](../ARCHITECTURE.md) - System architecture overview
- [Configuration Examples](CONFIGURATION.md) - Example configuration files
- [Service Examples](SERVICE.md) - Supervisor and systemd configurations
